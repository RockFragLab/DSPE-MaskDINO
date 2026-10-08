import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config


def setup_cfg(config_file, weights, dataset, threshold, device):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TEST = (dataset,)

    if hasattr(cfg.MODEL, "CONVNEXT") and cfg.MODEL.CONVNEXT.WEIGHTS:
        convnext_weights = cfg.MODEL.CONVNEXT.WEIGHTS
        if not os.path.isfile(convnext_weights):
            fallback = Path(__file__).resolve().parents[1] / "weights" / "convnextv2_tiny.pth"
            if fallback.is_file():
                cfg.MODEL.CONVNEXT.WEIGHTS = str(fallback)

    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    if hasattr(cfg.MODEL, "PANOPTIC_FPN"):
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
    if hasattr(cfg.MODEL, "MaskDINO"):
        cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = threshold

    cfg.freeze()
    return cfg


def choose_image(dataset_name, image_path=None):
    if image_path:
        return image_path

    records = DatasetCatalog.get(dataset_name)
    if not records:
        raise RuntimeError(f"No images found in dataset: {dataset_name}")

    return records[0]["file_name"]


def make_colors(num_classes, num_instances, by_instance=False):
    base = [
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
        (0, 172, 193),
    ]
    rng = random.Random(2026)
    colors = list(base)
    while len(colors) < max(num_classes, num_instances, 1):
        colors.append(tuple(rng.randint(40, 230) for _ in range(3)))
    if by_instance:
        rng.shuffle(colors)
    return colors


def _mask_center(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(np.median(xs)), int(np.median(ys))


def _display_class_name(name):
    if name == "shone":
        return "stone"
    return name


def _draw_reference_label(image, label, center, color_bgr):
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.32
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        pad_x, pad_y = 3, 2
        x = int(max(0, min(center[0] - tw // 2, image.shape[1] - tw - 2 * pad_x - 1)))
        y = int(max(th + baseline + pad_y, min(center[1], image.shape[0] - baseline - pad_y - 1)))

        x1 = int(max(0, x - pad_x))
        y1 = int(max(0, y - th - pad_y))
        x2 = int(min(image.shape[1], x + tw + pad_x))
        y2 = int(min(image.shape[0], y + baseline + pad_y))
        if x2 > x1 and y2 > y1:
            fill = np.full_like(image[y1:y2, x1:x2], 245, dtype=np.uint8)
            tint = np.zeros_like(image[y1:y2, x1:x2], dtype=np.uint8)
            tint[:, :] = np.asarray(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
            label_bg = cv2.addWeighted(fill, 0.78, tint, 0.22, 0)
            if label_bg is not None:
                blended = cv2.addWeighted(image[y1:y2, x1:x2], 0.25, label_bg, 0.75, 0)
                if blended is not None:
                    image[y1:y2, x1:x2] = blended

        cv2.putText(image, label, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    except Exception:
        return


def overlay_masks(
    image_bgr,
    instances,
    metadata,
    threshold,
    alpha,
    class_names,
    instance_colors=False,
    draw_labels=False,
):
    output = image_bgr.copy()
    if not instances.has("pred_masks") or len(instances) == 0:
        return output, 0

    scores = instances.scores.cpu().numpy() if instances.has("scores") else np.ones(len(instances))
    pred_classes = (
        instances.pred_classes.cpu().numpy()
        if instances.has("pred_classes")
        else np.zeros(len(instances), dtype=np.int64)
    )
    masks = instances.pred_masks.cpu().numpy().astype(bool)
    thing_classes = list(getattr(metadata, "thing_classes", []))

    keep = scores >= threshold
    if class_names:
        wanted = set(class_names)
        keep = keep & np.array(
            [
                (thing_classes[c] if c < len(thing_classes) else str(c)) in wanted
                for c in pred_classes
            ],
            dtype=bool,
        )

    colors = make_colors(len(thing_classes), len(instances), by_instance=instance_colors)
    kept_indices = np.where(keep)[0]
    # Draw larger masks first so smaller foreground instances remain visible.
    kept_indices = sorted(kept_indices, key=lambda i: masks[i].sum(), reverse=True)

    for draw_order, idx in enumerate(kept_indices):
        cls = int(pred_classes[idx])
        color_idx = draw_order if instance_colors else cls
        color_rgb = colors[color_idx % len(colors)]
        color_bgr = np.array(color_rgb[::-1], dtype=np.uint8)
        mask = masks[idx]

        colored = np.zeros_like(output, dtype=np.uint8)
        colored[mask] = color_bgr
        output[mask] = cv2.addWeighted(output, 1.0 - alpha, colored, alpha, 0)[mask]

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color_bgr.tolist(), thickness=2)

        if draw_labels:
            center = _mask_center(mask)
            if center is not None:
                name = thing_classes[cls] if cls < len(thing_classes) else str(cls)
                name = _display_class_name(name)
                label = f"{name} {scores[idx]:.2f}" if instances.has("scores") else name
                _draw_reference_label(output, label, center, color_bgr)

    return output, int(keep.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="date2_val")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--classes", nargs="*", default=None, help="Optional class names to visualize.")
    parser.add_argument("--instance-colors", action="store_true", help="Use a different color for each instance.")
    parser.add_argument("--draw-labels", action="store_true", help="Draw small class/score labels without boxes.")
    args = parser.parse_args()

    cfg = setup_cfg(args.config, args.weights, args.dataset, args.threshold, args.device)
    image_path = choose_image(args.dataset, args.image)
    image_bgr = read_image(image_path, format="BGR")

    predictor = DefaultPredictor(cfg)
    predictions = predictor(image_bgr)
    instances = predictions["instances"].to("cpu")
    metadata = MetadataCatalog.get(args.dataset)

    visualized, kept = overlay_masks(
        image_bgr,
        instances,
        metadata,
        threshold=args.threshold,
        alpha=args.alpha,
        class_names=args.classes,
        instance_colors=args.instance_colors,
        draw_labels=args.draw_labels,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, visualized)
    print(f"image={image_path}")
    print(f"output={args.output}")
    print(f"instances_after_filter={kept}")


if __name__ == "__main__":
    main()
