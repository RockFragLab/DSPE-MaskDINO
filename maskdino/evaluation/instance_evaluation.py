# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import defaultdict, OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table


_EXTRA_METRIC_SCORE_THRESH = 0.5


def _decode_coco_mask(segmentation):
    mask = mask_util.decode(segmentation)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def _mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _binary_dilation(mask, iterations):
    dilated = mask.astype(bool)
    for _ in range(max(1, iterations)):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        dilated = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return dilated


def _mask_boundary(mask):
    if mask.sum() == 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    eroded = (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )
    return mask.astype(bool) & ~eroded


def _boundary_metrics(gt_mask, pred_mask):
    gt_boundary = _mask_boundary(gt_mask)
    pred_boundary = _mask_boundary(pred_mask)
    if gt_boundary.sum() == 0 and pred_boundary.sum() == 0:
        return 1.0, 1.0
    if gt_boundary.sum() == 0 or pred_boundary.sum() == 0:
        return 0.0, 0.0

    height, width = gt_mask.shape
    tolerance = max(1, int(round(0.002 * np.sqrt(height * height + width * width))))
    gt_dilated = _binary_dilation(gt_boundary, tolerance)
    pred_dilated = _binary_dilation(pred_boundary, tolerance)

    boundary_iou = _mask_iou(gt_dilated, pred_dilated)
    precision = np.logical_and(pred_boundary, gt_dilated).sum() / max(pred_boundary.sum(), 1)
    recall = np.logical_and(gt_boundary, pred_dilated).sum() / max(gt_boundary.sum(), 1)
    boundary_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return boundary_iou, boundary_f1


def _greedy_match(gts, preds, iou_thr):
    if len(gts) == 0 or len(preds) == 0:
        return [], set(), set()

    candidates = []
    for gt_idx, gt_mask in enumerate(gts):
        for pred_idx, pred_mask in enumerate(preds):
            iou = _mask_iou(gt_mask, pred_mask)
            if iou > iou_thr:
                candidates.append((iou, gt_idx, pred_idx))
    candidates.sort(reverse=True)

    matches = []
    used_gts = set()
    used_preds = set()
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gts or pred_idx in used_preds:
            continue
        matches.append((gt_idx, pred_idx, iou))
        used_gts.add(gt_idx)
        used_preds.add(pred_idx)
    return matches, used_gts, used_preds


def _empty_extra_stats():
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "iou_sum": 0.0,
        "aji_intersection": 0.0,
        "aji_union": 0.0,
        "boundary_iou_sum": 0.0,
        "boundary_f1_sum": 0.0,
        "boundary_count": 0,
    }


# modified from COCOEvaluator for instance segmetnat
class InstanceSegEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)
        if coco_eval is None:
            return results

        per_category_results = self._derive_per_category_ap50_ap75(coco_eval)
        if per_category_results:
            results.update(per_category_results)
        return results

    def _derive_per_category_ap50_ap75(self, coco_eval):
        precisions = coco_eval.eval.get("precision")
        if precisions is None:
            return OrderedDict()

        category_ids = list(coco_eval.params.catIds)
        if len(category_ids) == 0:
            return OrderedDict()

        area_index = 0
        if hasattr(coco_eval.params, "areaRngLbl") and "all" in coco_eval.params.areaRngLbl:
            area_index = coco_eval.params.areaRngLbl.index("all")
        max_det_index = len(coco_eval.params.maxDets) - 1

        iou_thresholds = np.array(coco_eval.params.iouThrs)
        metric_specs = [("AP50", 0.50), ("AP75", 0.75)]
        results = OrderedDict()
        table = []

        for category_index, category_id in enumerate(category_ids):
            category_name = self._category_name(category_id)
            row = [category_name]

            for metric_name, iou_threshold in metric_specs:
                threshold_matches = np.where(np.isclose(iou_thresholds, iou_threshold))[0]
                if len(threshold_matches) == 0:
                    ap = float("nan")
                else:
                    precision = precisions[
                        threshold_matches[0],
                        :,
                        category_index,
                        area_index,
                        max_det_index,
                    ]
                    precision = precision[precision > -1]
                    ap = np.mean(precision) * 100 if precision.size else float("nan")

                results[f"{metric_name}-{category_name}"] = ap
                row.append(ap)

            table.append(row)

        self._logger.info(
            "Per-category {} AP50/AP75:\n".format(coco_eval.params.iouType)
            + tabulate(
                table,
                headers=["category", "AP50", "AP75"],
                tablefmt="pipe",
                floatfmt=".3f",
            )
        )
        return results

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            # all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            # num_classes = len(all_contiguous_ids)
            # assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                # assert category_id < num_classes, (
                #     f"A prediction has class={category_id}, "
                #     f"but the dataset only has {num_classes} classes and "
                #     f"predicted class id should be in [0, {num_classes - 1}]."
                # )
                assert category_id in reverse_id_mapping, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has class ids in {dataset_id_to_contiguous_id}."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    cocoeval_fn=COCOeval_opt if self._use_fast_impl else COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            if task == "segm" and coco_eval is not None:
                res.update(self._derive_extra_segmentation_results(coco_results))
            self._results[task] = res

    def _derive_extra_segmentation_results(self, coco_results):
        stats = defaultdict(_empty_extra_stats)
        pred_by_image_category = defaultdict(list)

        for result in coco_results:
            if "segmentation" not in result:
                continue
            if result.get("score", 0.0) < _EXTRA_METRIC_SCORE_THRESH:
                continue
            pred_by_image_category[(result["image_id"], result["category_id"])].append(
                (result.get("score", 0.0), _decode_coco_mask(result["segmentation"]))
            )

        gt_by_image_category = defaultdict(list)
        category_ids = set()
        for ann in self._coco_api.dataset.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            category_id = ann["category_id"]
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
                if category_id not in self._metadata.thing_dataset_id_to_contiguous_id:
                    continue
            category_ids.add(category_id)
            gt_by_image_category[(ann["image_id"], category_id)].append(
                self._coco_api.annToMask(ann).astype(bool)
            )

        for image_id in self._coco_api.getImgIds():
            image_category_ids = set(category_ids)
            image_category_ids.update(
                category_id
                for pred_image_id, category_id in pred_by_image_category.keys()
                if pred_image_id == image_id
            )
            for category_id in image_category_ids:
                gts = gt_by_image_category.get((image_id, category_id), [])
                preds = [
                    pred_mask
                    for _, pred_mask in sorted(
                        pred_by_image_category.get((image_id, category_id), []),
                        key=lambda item: item[0],
                        reverse=True,
                    )
                ]

                pq_matches, used_gts, used_preds = _greedy_match(gts, preds, 0.5)
                cat_stats = stats[category_id]
                cat_stats["tp"] += len(pq_matches)
                cat_stats["fp"] += len(preds) - len(used_preds)
                cat_stats["fn"] += len(gts) - len(used_gts)
                cat_stats["iou_sum"] += sum(iou for _, _, iou in pq_matches)

                for gt_idx, pred_idx, _ in pq_matches:
                    boundary_iou, boundary_f1 = _boundary_metrics(gts[gt_idx], preds[pred_idx])
                    cat_stats["boundary_iou_sum"] += boundary_iou
                    cat_stats["boundary_f1_sum"] += boundary_f1
                    cat_stats["boundary_count"] += 1

                aji_matches, aji_used_gts, aji_used_preds = _greedy_match(gts, preds, 0.0)
                for gt_idx, pred_idx, _ in aji_matches:
                    intersection = np.logical_and(gts[gt_idx], preds[pred_idx]).sum()
                    union = np.logical_or(gts[gt_idx], preds[pred_idx]).sum()
                    cat_stats["aji_intersection"] += intersection
                    cat_stats["aji_union"] += union
                for gt_idx, gt_mask in enumerate(gts):
                    if gt_idx not in aji_used_gts:
                        cat_stats["aji_union"] += gt_mask.sum()
                for pred_idx, pred_mask in enumerate(preds):
                    if pred_idx not in aji_used_preds:
                        cat_stats["aji_union"] += pred_mask.sum()

        total_stats = _empty_extra_stats()
        for category_id in category_ids:
            for key, value in stats[category_id].items():
                total_stats[key] += value

        results = self._summarize_extra_stats(total_stats)
        self._logger.info(
            "Additional segm metrics (score >= {:.2f}):\n".format(_EXTRA_METRIC_SCORE_THRESH)
            + create_small_table(results)
        )

        class_names = self._metadata.get("thing_classes")
        if class_names:
            table = []
            for category_id in sorted(category_ids):
                category_results = self._summarize_extra_stats(stats[category_id])
                table.append(
                    [
                        self._category_name(category_id),
                        category_results["Precision"],
                        category_results["Recall"],
                        category_results["Acc"],
                        category_results["F1Score"],
                        category_results["PQ"],
                        category_results["AJI"],
                        category_results["BoundaryIoU"],
                        category_results["BoundaryF1"],
                    ]
                )
            self._logger.info(
                "Per-category extra segm metrics:\n"
                + tabulate(
                    table,
                    headers=[
                        "category",
                        "P",
                        "R",
                        "Acc",
                        "F1Score",
                        "PQ",
                        "AJI",
                        "BoundaryIoU",
                        "BoundaryF1",
                    ],
                    tablefmt="pipe",
                    floatfmt=".3f",
                )
            )

        return results

    def _summarize_extra_stats(self, stats):
        denom = stats["tp"] + 0.5 * stats["fp"] + 0.5 * stats["fn"]
        pq = stats["iou_sum"] / denom if denom > 0 else float("nan")
        sq = stats["iou_sum"] / stats["tp"] if stats["tp"] > 0 else float("nan")
        rq = stats["tp"] / denom if denom > 0 else float("nan")
        aji = (
            stats["aji_intersection"] / stats["aji_union"]
            if stats["aji_union"] > 0
            else float("nan")
        )
        boundary_iou = (
            stats["boundary_iou_sum"] / stats["boundary_count"]
            if stats["boundary_count"] > 0
            else float("nan")
        )
        boundary_f1 = (
            stats["boundary_f1_sum"] / stats["boundary_count"]
            if stats["boundary_count"] > 0
            else float("nan")
        )
        precision = (
            stats["tp"] / (stats["tp"] + stats["fp"])
            if stats["tp"] + stats["fp"] > 0
            else float("nan")
        )
        recall = (
            stats["tp"] / (stats["tp"] + stats["fn"])
            if stats["tp"] + stats["fn"] > 0
            else float("nan")
        )
        accuracy = (
            stats["tp"] / (stats["tp"] + stats["fp"] + stats["fn"])
            if stats["tp"] + stats["fp"] + stats["fn"] > 0
            else float("nan")
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else float("nan")
        )
        return OrderedDict(
            [
                ("Precision", precision * 100),
                ("Recall", recall * 100),
                ("Acc", accuracy * 100),
                ("F1Score", f1_score * 100),
                ("PQ", pq * 100),
                ("SQ", sq * 100),
                ("RQ", rq * 100),
                ("AJI", aji * 100),
                ("BoundaryIoU", boundary_iou * 100),
                ("BoundaryF1", boundary_f1 * 100),
            ]
        )

    def _category_name(self, category_id):
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id.get(category_id)
            class_names = self._metadata.get("thing_classes")
            if contiguous_id is not None and class_names and contiguous_id < len(class_names):
                return class_names[contiguous_id]
        return self._coco_api.cats.get(category_id, {}).get("name", str(category_id))
