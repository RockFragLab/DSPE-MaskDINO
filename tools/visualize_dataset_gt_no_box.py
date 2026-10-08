import argparse
import os
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import annotations_to_instances, read_image
from detectron2.structures import Instances

import maskdino  # noqa: F401 - registers local datasets
from predict_one_no_box import overlay_masks


def iter_dataset_records(dataset_name):
    records = DatasetCatalog.get(dataset_name)
    return sorted(records, key=lambda r: os.path.basename(r["file_name"]))


def build_gt_instances(record, image_shape, show_score=False):
    annos = [anno for anno in record.get("annotations", []) if anno.get("iscrowd", 0) == 0]
    gt = annotations_to_instances(annos, image_shape, mask_format="bitmask")

    instances = Instances(image_shape)
    if len(gt) == 0:
        instances.pred_masks = torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.bool)
        instances.pred_classes = torch.zeros((0,), dtype=torch.int64)
        if show_score:
            instances.scores = torch.zeros((0,), dtype=torch.float32)
        return instances

    instances.pred_masks = gt.gt_masks.tensor
    instances.pred_classes = gt.gt_classes
    if show_score:
        instances.scores = torch.ones((len(gt),), dtype=torch.float32)
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="date2_val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--instance-colors", action="store_true")
    parser.add_argument("--draw-labels", action="store_true")
    parser.add_argument("--show-score", action="store_true", help="Draw GT labels as class 1.00.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata = MetadataCatalog.get(args.dataset)
    records = iter_dataset_records(args.dataset)
    if args.limit > 0:
        records = records[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, record in enumerate(records, start=1):
        image_path = record["file_name"]
        image_bgr = read_image(image_path, format="BGR")
        instances = build_gt_instances(record, image_bgr.shape[:2], show_score=args.show_score)
        visualized, kept = overlay_masks(
            image_bgr,
            instances,
            metadata,
            threshold=0.0,
            alpha=args.alpha,
            class_names=args.classes,
            instance_colors=args.instance_colors,
            draw_labels=args.draw_labels,
        )

        stem = Path(image_path).stem
        out_path = os.path.join(args.output_dir, f"{idx:04d}_{stem}.png")
        cv2.imwrite(out_path, visualized)
        print(f"[{idx}/{len(records)}] gt={kept} output={out_path}", flush=True)


if __name__ == "__main__":
    main()
