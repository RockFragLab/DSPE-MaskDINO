# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskDINO Training Script based on Mask2Former.
"""
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import json
import logging
import math
import os
import re
import time

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import create_small_table
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

# MaskDINO
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref

import numpy as np
from PIL import Image


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class _TimedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.total_inference_time = 0.0
        self.num_images = 0

    def forward(self, inputs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = self.model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_inference_time += time.perf_counter() - start
        self.num_images += len(inputs)
        return outputs


def _convert_epoch_schedule_to_iter(cfg):
    if cfg.SOLVER.MAX_EPOCH <= 0:
        return None

    if len(cfg.DATASETS.TRAIN) == 0:
        raise ValueError("SOLVER.MAX_EPOCH is set, but DATASETS.TRAIN is empty.")

    dataset_sizes = {name: len(DatasetCatalog.get(name)) for name in cfg.DATASETS.TRAIN}
    train_images = sum(dataset_sizes.values())
    if train_images <= 0:
        raise ValueError(f"Training datasets are empty: {dataset_sizes}")

    iters_per_epoch = int(math.ceil(train_images / cfg.SOLVER.IMS_PER_BATCH))
    max_iter = int(round(cfg.SOLVER.MAX_EPOCH * iters_per_epoch))

    cfg.SOLVER.TRAIN_IMAGES = train_images
    cfg.SOLVER.ITER_PER_EPOCH = iters_per_epoch
    cfg.SOLVER.MAX_ITER = max_iter

    if len(cfg.SOLVER.STEP_EPOCHS) > 0:
        cfg.SOLVER.STEPS = tuple(
            max(1, int(round(float(epoch) * iters_per_epoch)))
            for epoch in cfg.SOLVER.STEP_EPOCHS
        )

    if cfg.SOLVER.CHECKPOINT_EPOCH > 0:
        cfg.SOLVER.CHECKPOINT_PERIOD = max(
            1, int(round(float(cfg.SOLVER.CHECKPOINT_EPOCH) * iters_per_epoch))
        )

    if cfg.TEST.EVAL_EPOCH > 0:
        cfg.TEST.EVAL_PERIOD = max(1, int(round(float(cfg.TEST.EVAL_EPOCH) * iters_per_epoch)))

    if cfg.TEST.EVAL_START_EPOCH > 0:
        cfg.TEST.EVAL_START_ITER = max(
            1, int(round(float(cfg.TEST.EVAL_START_EPOCH) * iters_per_epoch))
        )
        if cfg.TEST.EVAL_START_ITER > cfg.SOLVER.MAX_ITER:
            raise ValueError(
                "TEST.EVAL_START_EPOCH must not exceed SOLVER.MAX_EPOCH: "
                f"{cfg.TEST.EVAL_START_EPOCH} > {cfg.SOLVER.MAX_EPOCH}"
            )

    return {
        "datasets": dataset_sizes,
        "images": train_images,
        "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
        "iters_per_epoch": iters_per_epoch,
        "max_epoch": cfg.SOLVER.MAX_EPOCH,
        "max_iter": cfg.SOLVER.MAX_ITER,
        "steps": tuple(cfg.SOLVER.STEPS),
        "checkpoint_period": cfg.SOLVER.CHECKPOINT_PERIOD,
        "eval_period": cfg.TEST.EVAL_PERIOD,
        "eval_start_iter": cfg.TEST.EVAL_START_ITER,
    }


def _sanitize_output_part(value):
    value = str(value).strip().replace("\\", "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "run"


def _build_auto_output_dir(cfg):
    parent = os.path.dirname(cfg.OUTPUT_DIR) or "output"
    dataset = cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "dataset"
    for suffix in ("_train", "-train"):
        if dataset.endswith(suffix):
            dataset = dataset[: -len(suffix)]

    meta_arch = getattr(cfg.MODEL, "META_ARCHITECTURE", "model").lower()
    if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
        model_tag = f"{meta_arch}_R{cfg.MODEL.RESNETS.DEPTH}"
    else:
        model_tag = f"{meta_arch}_{cfg.MODEL.BACKBONE.NAME}"

    if cfg.SOLVER.MAX_EPOCH > 0:
        schedule_tag = f"{cfg.SOLVER.MAX_EPOCH}ep"
    else:
        schedule_tag = f"{cfg.SOLVER.MAX_ITER}iter"

    enhancer_tag = ""
    if cfg.MODEL.FEATURE_ENHANCER.ENABLED:
        enhancer_tag = f"{cfg.MODEL.FEATURE_ENHANCER.NAME.lower()}_"
    if cfg.MODEL.PIXEL_DECODER_ENHANCER.ENABLED:
        enhancer_tag += f"{cfg.MODEL.PIXEL_DECODER_ENHANCER.NAME.lower()}_"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = "_".join(
        [
            _sanitize_output_part(dataset),
            _sanitize_output_part(f"{enhancer_tag}{model_tag}"),
            f"bs{cfg.SOLVER.IMS_PER_BATCH}",
            _sanitize_output_part(schedule_tag),
            timestamp,
        ]
    )
    output_dir = os.path.join(parent, run_name)
    if not os.path.exists(output_dir):
        return output_dir

    index = 2
    while os.path.exists(f"{output_dir}_{index}"):
        index += 1
    return f"{output_dir}_{index}"


def _maybe_update_output_dir(cfg, args):
    if not cfg.OUTPUT_DIR_AUTO or args.resume or args.eval_only:
        return None

    output_dir = _build_auto_output_dir(cfg)
    cfg.OUTPUT_DIR = output_dir
    return output_dir


class EpochMetricHook(hooks.HookBase):
    def __init__(self, iters_per_epoch):
        self._iters_per_epoch = iters_per_epoch

    def after_step(self):
        epoch = (self.trainer.iter + 1) / self._iters_per_epoch
        self.trainer.storage.put_scalar("epoch", epoch, smoothing_hint=False)


class DelayedEvalHook(hooks.EvalHook):
    def __init__(self, eval_period, eval_function, start_iter, eval_after_train=True):
        super().__init__(eval_period, eval_function, eval_after_train=eval_after_train)
        self._start_iter = int(start_iter)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter < self._start_iter:
            return
        if (
            self._period > 0
            and (next_iter - self._start_iter) % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._do_eval()


class PersistentBestCheckpointer(hooks.BestCheckpointer):
    def __init__(self, eval_period, checkpointer, metric_name, state_path, start_iter=0):
        super().__init__(eval_period, checkpointer, metric_name, mode="max")
        self.state_path = state_path
        self.start_iter = int(start_iter)
        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            if state["metric_name"] != metric_name:
                raise ValueError(f"Best checkpoint metric mismatch: {state['metric_name']} != {metric_name}")
            self.best_metric = float(state["metric"])
            self.best_iter = int(state["iteration"])

    def _best_checking(self):
        previous = (self.best_metric, self.best_iter)
        super()._best_checking()
        if (self.best_metric, self.best_iter) != previous:
            state = {}
            if os.path.isfile(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as handle:
                    state = json.load(handle)
            state.update(
                metric_name=self._val_metric,
                metric=self.best_metric,
                iteration=self.best_iter,
                source="validation",
            )
            temporary_path = self.state_path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_path, self.state_path)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter < self.start_iter:
            return
        if (
            self._period > 0
            and (next_iter - self.start_iter) % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._best_checking()


class TrainingVisualizationHook(hooks.HookBase):
    def __init__(self, cfg):
        self._enabled = cfg.VISUALIZATION.ENABLED
        self._period = int(cfg.VISUALIZATION.PERIOD)
        self._num_images = int(cfg.VISUALIZATION.NUM_IMAGES)
        self._score_thresh = float(cfg.VISUALIZATION.SCORE_THRESH)
        self._draw_gt = bool(cfg.VISUALIZATION.DRAW_GT)
        self._draw_curves = bool(cfg.VISUALIZATION.DRAW_CURVES)
        self._output_dir = os.path.join(cfg.OUTPUT_DIR, "visualization")
        self._image_dir = os.path.join(self._output_dir, "predictions")
        self._curve_dir = os.path.join(self._output_dir, "curves")
        self._dataset_name = cfg.VISUALIZATION.DATASET
        if not self._dataset_name and len(cfg.DATASETS.TEST):
            self._dataset_name = cfg.DATASETS.TEST[0]
        elif not self._dataset_name and len(cfg.DATASETS.TRAIN):
            self._dataset_name = cfg.DATASETS.TRAIN[0]
        self._input_format = cfg.INPUT.FORMAT
        self._samples = []
        self._metadata = None

        if self._enabled and self._dataset_name:
            dataset = DatasetCatalog.get(self._dataset_name)
            self._samples = dataset[: self._num_images]
            self._metadata = MetadataCatalog.get(self._dataset_name)

    def _should_run(self):
        if not self._enabled or self._period <= 0 or not comm.is_main_process():
            return False
        next_iter = self.trainer.iter + 1
        return next_iter == self.trainer.max_iter or next_iter % self._period == 0

    def _draw_curves_from_metrics(self, iteration):
        if not self._draw_curves:
            return

        metrics_path = os.path.join(self.trainer.cfg.OUTPUT_DIR, "metrics.json")
        if not os.path.isfile(metrics_path):
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logging.getLogger("detectron2.trainer").warning(
                "Could not import matplotlib for training curves: %s", exc
            )
            return

        rows = []
        with open(metrics_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            return

        keys = [
            key
            for key in ("total_loss", "loss_ce", "loss_mask", "loss_dice", "loss_bbox", "loss_giou", "lr")
            if any(key in row for row in rows)
        ]
        if not keys:
            return

        os.makedirs(self._curve_dir, exist_ok=True)
        cols = 2
        rows_count = int(math.ceil(len(keys) / cols))
        fig, axes = plt.subplots(rows_count, cols, figsize=(12, 4 * rows_count), squeeze=False)
        for axis, key in zip(axes.flatten(), keys):
            xs = [row["iteration"] for row in rows if key in row and "iteration" in row]
            ys = [row[key] for row in rows if key in row and "iteration" in row]
            axis.plot(xs, ys, linewidth=1.2)
            axis.set_title(key)
            axis.set_xlabel("iteration")
            axis.grid(True, alpha=0.25)
        for axis in axes.flatten()[len(keys):]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(self._curve_dir, f"curves_iter_{iteration:07d}.svg"), format="svg")
        fig.savefig(os.path.join(self._curve_dir, "latest.svg"), format="svg")
        plt.close(fig)

    def _draw_predictions(self, iteration):
        if not self._samples or self._metadata is None:
            return

        os.makedirs(self._image_dir, exist_ok=True)
        model = self.trainer.model
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for idx, record in enumerate(self._samples):
                    image = utils.read_image(record["file_name"], format=self._input_format)
                    height, width = image.shape[:2]
                    tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                    outputs = model([{"image": tensor, "height": height, "width": width}])[0]

                    pred_vis = image[:, :, ::-1].copy() if self._input_format == "BGR" else image.copy()
                    if "instances" in outputs:
                        instances = outputs["instances"].to("cpu")
                        if instances.has("scores"):
                            instances = instances[instances.scores >= self._score_thresh]
                        pred_vis = Visualizer(pred_vis, self._metadata).draw_instance_predictions(
                            instances
                        ).get_image()

                    if self._draw_gt:
                        gt_image = image[:, :, ::-1].copy() if self._input_format == "BGR" else image.copy()
                        gt_vis = Visualizer(gt_image, self._metadata).draw_dataset_dict(record).get_image()
                        canvas = np.concatenate([gt_vis, pred_vis], axis=1)
                    else:
                        canvas = pred_vis

                    output_path = os.path.join(
                        self._image_dir, f"iter_{iteration:07d}_sample_{idx:02d}.jpg"
                    )
                    Image.fromarray(canvas).save(output_path, quality=92)
        finally:
            if was_training:
                model.train()

    def after_step(self):
        if not self._should_run():
            return
        iteration = self.trainer.iter + 1
        self._draw_predictions(iteration)
        self._draw_curves_from_metrics(iteration)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training

    def build_hooks(self):
        hook_list = super().build_hooks()
        eval_start_iter = int(self.cfg.TEST.EVAL_START_ITER)
        if eval_start_iter > 0:
            for idx, hook in enumerate(hook_list):
                if isinstance(hook, hooks.EvalHook):
                    hook_list[idx] = DelayedEvalHook(
                        hook._period,
                        hook._func,
                        eval_start_iter,
                        eval_after_train=hook._eval_after_train,
                    )
                    break
            else:
                raise RuntimeError("No evaluation hook available for delayed evaluation")

        metric_name = self.cfg.TEST.BEST_CHECKPOINT_METRIC
        if metric_name and comm.is_main_process():
            if self.cfg.TEST.EVAL_PERIOD <= 0:
                raise ValueError("Best checkpoint selection requires TEST.EVAL_PERIOD > 0")
            best_dir = os.path.join(self.cfg.OUTPUT_DIR, "best")
            os.makedirs(best_dir, exist_ok=True)
            best_checkpointer = DetectionCheckpointer(self.model, save_dir=best_dir)
            best_hook = PersistentBestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                best_checkpointer,
                metric_name,
                os.path.join(best_dir, "best_state.json"),
                start_iter=eval_start_iter,
            )
            for idx, hook in enumerate(hook_list):
                if isinstance(hook, hooks.EvalHook):
                    hook_list.insert(idx + 1, best_hook)
                    break
            else:
                raise RuntimeError("No evaluation hook available for best checkpoint selection")
        if self.cfg.VISUALIZATION.ENABLED:
            hook_list.append(TrainingVisualizationHook(self.cfg))
        if self.cfg.SOLVER.ITER_PER_EPOCH > 0:
            epoch_hook = EpochMetricHook(self.cfg.SOLVER.ITER_PER_EPOCH)
            for idx, hook in enumerate(hook_list):
                if isinstance(hook, hooks.PeriodicWriter):
                    hook_list.insert(idx, epoch_hook)
                    break
            else:
                hook_list.append(epoch_hook)
        return hook_list

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_instance_extra":
            evaluator = COCOEvaluator if cfg.TEST.BEST_CHECKPOINT_METRIC else InstanceSegEvaluator
            evaluator_list.append(evaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def _log_model_complexity(cls, cfg, model, data_loader):
        logger = logging.getLogger("detectron2.trainer")
        raw_model = _unwrap_model(model)
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        complexity = OrderedDict(
            [
                ("Params(M)", total_params / 1e6),
                ("TrainableParams(M)", trainable_params / 1e6),
            ]
        )

        try:
            from detectron2.utils.analysis import FlopCountAnalysis, parameter_count_table

            logger.info("Parameter count table:\n" + parameter_count_table(raw_model, max_depth=5))
            data = next(iter(data_loader))
            with torch.no_grad():
                flops = FlopCountAnalysis(raw_model, data)
                flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
                flops.uncalled_modules_warnings(False)
                complexity["GFlops"] = flops.total() / 1e9
        except Exception as exc:
            logger.warning("Could not compute FLOPs automatically: %s", exc)

        logger.info("Model complexity:\n" + create_small_table(complexity))
        return complexity

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger("detectron2.trainer")
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        model_complexity = None
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if comm.is_main_process() and model_complexity is None:
                model_complexity = cls._log_model_complexity(cfg, model, data_loader)

            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            timed_model = _TimedModel(model)
            results_i = inference_on_dataset(timed_model, data_loader, evaluator)
            if comm.is_main_process():
                seconds_per_image = (
                    timed_model.total_inference_time / timed_model.num_images
                    if timed_model.num_images > 0
                    else float("nan")
                )
                fps = (
                    timed_model.num_images / timed_model.total_inference_time
                    if timed_model.total_inference_time > 0
                    else float("nan")
                )
                speed = OrderedDict(
                    [
                        ("Images", timed_model.num_images),
                        ("InferenceTime(s)", timed_model.total_inference_time),
                        ("SecondsPerImage", seconds_per_image),
                        ("FPS", fps),
                    ]
                )
                logger.info("Model inference speed:\n" + create_small_table(speed))
                results_i["speed"] = speed
                if model_complexity is not None:
                    results_i["model_complexity"] = model_complexity
                results[dataset_name] = results_i
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_train_loader(cls, cfg):
        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    epoch_schedule = _convert_epoch_schedule_to_iter(cfg)
    auto_output_dir = _maybe_update_output_dir(cfg, args)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    if auto_output_dir is not None:
        logging.getLogger("detectron2").info("Auto OUTPUT_DIR: %s", auto_output_dir)
    if epoch_schedule is not None:
        logger = logging.getLogger("detectron2")
        logger.info(
            "Epoch schedule: datasets=%s, images=%s, ims_per_batch=%s, "
            "iters_per_epoch=%s, max_epoch=%s -> max_iter=%s, steps=%s, "
            "checkpoint_period=%s, eval_period=%s, eval_start_iter=%s",
            epoch_schedule["datasets"],
            epoch_schedule["images"],
            epoch_schedule["ims_per_batch"],
            epoch_schedule["iters_per_epoch"],
            epoch_schedule["max_epoch"],
            epoch_schedule["max_iter"],
            epoch_schedule["steps"],
            epoch_schedule["checkpoint_period"],
            epoch_schedule["eval_period"],
            epoch_schedule["eval_start_iter"],
        )
    return cfg


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.SOLVER.RESTART_LR_EPOCH > 0:
        restart_iter = cfg.SOLVER.RESTART_LR_EPOCH * cfg.SOLVER.ITER_PER_EPOCH
        if args.resume and trainer.start_iter == restart_iter:
            for group in trainer.optimizer.param_groups:
                group["lr"] = group["initial_lr"]
            logging.getLogger("detectron2").info(
                "Restarted learning rates at epoch %s from optimizer initial_lr values",
                cfg.SOLVER.RESTART_LR_EPOCH,
            )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
