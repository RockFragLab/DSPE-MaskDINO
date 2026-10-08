import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor

from predict_one_no_box import (
    _display_class_name,
    _draw_reference_label,
    _mask_center,
    make_colors,
    setup_cfg,
)


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def imwrite_unicode(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    data.tofile(path)


def add_window_params_to_output(path, tile_size, overlap):
    root, ext = os.path.splitext(path)
    ext = ext or ".png"
    root = re.sub(r"_sliding\d+_merged$", "", root)
    root = re.sub(r"_sliding\d+$", "", root)
    root = re.sub(r"_merged$", "", root)
    suffix = f"_tile{tile_size}_overlap{overlap}"
    if root.endswith(suffix):
        return root + ext
    return root + suffix + ext


def to_gray_bgr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def center_in_valid_region(center, x0, y0, tile_w, tile_h, full_w, full_h, margin):
    cx, cy = center
    left = 0 if x0 == 0 else margin
    top = 0 if y0 == 0 else margin
    right = tile_w if x0 + tile_w >= full_w else tile_w - margin
    bottom = tile_h if y0 + tile_h >= full_h else tile_h - margin
    local_x = cx - x0
    local_y = cy - y0
    return left <= local_x < right and top <= local_y < bottom


def valid_region_mask(tile_h, tile_w, x0, y0, full_w, full_h, margin):
    valid = np.zeros((tile_h, tile_w), dtype=bool)
    left = 0 if x0 == 0 else margin
    top = 0 if y0 == 0 else margin
    right = tile_w if x0 + tile_w >= full_w else tile_w - margin
    bottom = tile_h if y0 + tile_h >= full_h else tile_h - margin
    valid[top:bottom, left:right] = True
    return valid


def draw_instance(output, mask, cls, score, thing_classes, color_bgr, alpha, draw_labels):
    colored = np.zeros_like(output, dtype=np.uint8)
    colored[mask] = color_bgr
    output[mask] = cv2.addWeighted(output, 1.0 - alpha, colored, alpha, 0)[mask]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color_bgr.tolist(), thickness=2)

    if draw_labels:
        center = _mask_center(mask)
        if center is not None:
            name = thing_classes[cls] if cls < len(thing_classes) else str(cls)
            label = f"{_display_class_name(name)} {score:.2f}"
            _draw_reference_label(output, label, center, color_bgr)


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compact_instance(local_mask, x0, y0, cls, score):
    bbox = mask_bbox(local_mask)
    if bbox is None:
        return None
    bx0, by0, bx1, by1 = bbox
    mask = local_mask[by0:by1, bx0:bx1].copy()
    return {
        "x0": x0 + bx0,
        "y0": y0 + by0,
        "x1": x0 + bx1,
        "y1": y0 + by1,
        "mask": mask,
        "cls": int(cls),
        "score": float(score),
    }


def instance_area(inst):
    return int(inst["mask"].sum())


def instance_intersection(a, b):
    ix0 = max(a["x0"], b["x0"])
    iy0 = max(a["y0"], b["y0"])
    ix1 = min(a["x1"], b["x1"])
    iy1 = min(a["y1"], b["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    a_crop = a["mask"][iy0 - a["y0"] : iy1 - a["y0"], ix0 - a["x0"] : ix1 - a["x0"]]
    b_crop = b["mask"][iy0 - b["y0"] : iy1 - b["y0"], ix0 - b["x0"] : ix1 - b["x0"]]
    return int(np.logical_and(a_crop, b_crop).sum())


def merge_instances(a, b):
    x0 = min(a["x0"], b["x0"])
    y0 = min(a["y0"], b["y0"])
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    merged = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    merged[a["y0"] - y0 : a["y1"] - y0, a["x0"] - x0 : a["x1"] - x0] |= a["mask"]
    merged[b["y0"] - y0 : b["y1"] - y0, b["x0"] - x0 : b["x1"] - x0] |= b["mask"]
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "mask": merged,
        "cls": a["cls"],
        "score": max(a["score"], b["score"]),
    }


def recrop_instance(inst):
    bbox = mask_bbox(inst["mask"])
    if bbox is None:
        return None
    bx0, by0, bx1, by1 = bbox
    return {
        "x0": inst["x0"] + bx0,
        "y0": inst["y0"] + by0,
        "x1": inst["x0"] + bx1,
        "y1": inst["y0"] + by1,
        "mask": inst["mask"][by0:by1, bx0:bx1].copy(),
        "cls": inst["cls"],
        "score": inst["score"],
    }


def subtract_overlap(target, keeper, min_area):
    ix0 = max(target["x0"], keeper["x0"])
    iy0 = max(target["y0"], keeper["y0"])
    ix1 = min(target["x1"], keeper["x1"])
    iy1 = min(target["y1"], keeper["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return target
    target_crop = target["mask"][
        iy0 - target["y0"] : iy1 - target["y0"],
        ix0 - target["x0"] : ix1 - target["x0"],
    ]
    keeper_crop = keeper["mask"][
        iy0 - keeper["y0"] : iy1 - keeper["y0"],
        ix0 - keeper["x0"] : ix1 - keeper["x0"],
    ]
    target_crop &= ~keeper_crop
    if int(target["mask"].sum()) < min_area:
        return None
    return recrop_instance(target)


def refine_instances(instances, merge_thresh, resolve_thresh, min_area):
    instances = [inst for inst in instances if instance_area(inst) >= min_area]

    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(instances)
        for i, inst in enumerate(instances):
            if used[i]:
                continue
            cur = inst
            used[i] = True
            for j in range(i + 1, len(instances)):
                if used[j] or cur["cls"] != instances[j]["cls"]:
                    continue
                inter = instance_intersection(cur, instances[j])
                if inter == 0:
                    continue
                ratio = inter / float(max(1, min(instance_area(cur), instance_area(instances[j]))))
                if ratio >= merge_thresh:
                    cur = merge_instances(cur, instances[j])
                    used[j] = True
                    changed = True
            merged.append(cur)
        instances = merged

    instances = sorted(instances, key=lambda item: item["score"], reverse=True)
    resolved = []
    for inst in instances:
        current = inst
        for keeper in resolved:
            if current is None or current["cls"] != keeper["cls"]:
                continue
            inter = instance_intersection(current, keeper)
            if inter == 0:
                continue
            ratio = inter / float(max(1, min(instance_area(current), instance_area(keeper))))
            if ratio <= resolve_thresh:
                current = subtract_overlap(current, keeper, min_area)
        if current is not None and instance_area(current) >= min_area:
            resolved.append(current)

    return resolved


def draw_compact_instance(output, inst, thing_classes, color_bgr, alpha, draw_labels):
    full_mask = np.zeros(output.shape[:2], dtype=bool)
    full_mask[inst["y0"] : inst["y1"], inst["x0"] : inst["x1"]] = inst["mask"]
    draw_instance(output, full_mask, inst["cls"], inst["score"], thing_classes, color_bgr, alpha, draw_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="date1_val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min-size-test", type=int, default=0)
    parser.add_argument("--max-size-test", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--max-local-mask-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-local-bbox-area-ratio", type=float, default=0.0)
    parser.add_argument("--gray-input", action="store_true")
    parser.add_argument("--gray-background", action="store_true")
    parser.add_argument("--draw-labels", action="store_true")
    parser.add_argument("--merge-masks", action="store_true")
    parser.add_argument("--merge-overlap-thresh", type=float, default=0.2)
    parser.add_argument("--resolve-overlap-thresh", type=float, default=0.05)
    parser.add_argument(
        "--clip-to-valid-region",
        action="store_true",
        help="For sliding-window inference, draw only the non-overlap center region of each tile.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.overlap >= args.tile_size:
        raise ValueError("--overlap must be smaller than --tile-size")

    args.output = add_window_params_to_output(args.output, args.tile_size, args.overlap)

    image_bgr = imread_unicode(args.image)
    inference_image = to_gray_bgr(image_bgr) if args.gray_input else image_bgr
    output = to_gray_bgr(image_bgr) if args.gray_background else image_bgr.copy()

    cfg = setup_cfg(args.config, args.weights, args.dataset, args.threshold, args.device)
    if args.min_size_test > 0 or args.max_size_test > 0:
        cfg.defrost()
        if args.min_size_test > 0:
            cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
        if args.max_size_test > 0:
            cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
        cfg.freeze()
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(args.dataset)
    thing_classes = list(getattr(metadata, "thing_classes", []))

    height, width = image_bgr.shape[:2]
    stride = args.tile_size - args.overlap
    xs = tile_starts(width, args.tile_size, stride)
    ys = tile_starts(height, args.tile_size, stride)
    margin = args.overlap // 2
    colors = make_colors(len(thing_classes), len(xs) * len(ys) * 64, by_instance=True)

    kept = 0
    collected_instances = []
    total_tiles = len(xs) * len(ys)
    tile_id = 0

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                tile_id += 1
                x1 = min(x0 + args.tile_size, width)
                y1 = min(y0 + args.tile_size, height)
                tile = inference_image[y0:y1, x0:x1]
                predictions = predictor(tile)
                instances = predictions["instances"].to("cpu")
                if len(instances) == 0 or not instances.has("pred_masks"):
                    print(f"[{tile_id}/{total_tiles}] kept=0 total={kept}", flush=True)
                    continue

                scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(instances))
                classes = (
                    instances.pred_classes.numpy()
                    if instances.has("pred_classes")
                    else np.zeros(len(instances), dtype=np.int64)
                )
                masks = instances.pred_masks.numpy().astype(bool)
                order = sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True)
                tile_kept = 0
                valid_mask = valid_region_mask(y1 - y0, x1 - x0, x0, y0, width, height, margin)

                for idx in order:
                    score = float(scores[idx])
                    if score < args.threshold:
                        continue
                    local_mask = masks[idx]
                    if int(local_mask.sum()) < args.min_area:
                        continue

                    tile_area = float((y1 - y0) * (x1 - x0))
                    local_area_ratio = float(local_mask.sum()) / tile_area
                    bbox = mask_bbox(local_mask)
                    bbox_area_ratio = 0.0
                    if bbox is not None:
                        bx0, by0, bx1, by1 = bbox
                        bbox_area_ratio = float((bx1 - bx0) * (by1 - by0)) / tile_area
                    if args.max_local_mask_area_ratio > 0 and local_area_ratio > args.max_local_mask_area_ratio:
                        continue
                    if args.max_local_bbox_area_ratio > 0 and bbox_area_ratio > args.max_local_bbox_area_ratio:
                        continue

                    center_local = _mask_center(local_mask)
                    if center_local is None:
                        continue
                    center_global = (center_local[0] + x0, center_local[1] + y0)
                    if not center_in_valid_region(
                        center_global, x0, y0, x1 - x0, y1 - y0, width, height, margin
                    ):
                        continue

                    if args.clip_to_valid_region:
                        local_mask = local_mask & valid_mask
                        if int(local_mask.sum()) < args.min_area:
                            continue

                    if args.merge_masks:
                        compact = compact_instance(local_mask[: y1 - y0, : x1 - x0], x0, y0, int(classes[idx]), score)
                        if compact is None:
                            continue
                        collected_instances.append(compact)
                    else:
                        global_mask = np.zeros((height, width), dtype=bool)
                        global_mask[y0:y1, x0:x1] = local_mask[: y1 - y0, : x1 - x0]

                        color_rgb = colors[kept % len(colors)]
                        color_bgr = np.array(color_rgb[::-1], dtype=np.uint8)
                        draw_instance(
                            output,
                            global_mask,
                            int(classes[idx]),
                            score,
                            thing_classes,
                            color_bgr,
                            args.alpha,
                            args.draw_labels,
                        )
                    kept += 1
                    tile_kept += 1

                print(f"[{tile_id}/{total_tiles}] kept={tile_kept} total={kept}", flush=True)

    if args.merge_masks:
        before_merge = len(collected_instances)
        collected_instances = refine_instances(
            collected_instances,
            merge_thresh=args.merge_overlap_thresh,
            resolve_thresh=args.resolve_overlap_thresh,
            min_area=args.min_area,
        )
        colors = make_colors(len(thing_classes), len(collected_instances), by_instance=True)
        for idx, inst in enumerate(sorted(collected_instances, key=instance_area, reverse=True)):
            color_rgb = colors[idx % len(colors)]
            color_bgr = np.array(color_rgb[::-1], dtype=np.uint8)
            draw_compact_instance(output, inst, thing_classes, color_bgr, args.alpha, args.draw_labels)
        kept = len(collected_instances)
        print(f"instances_before_merge={before_merge}")
        print(f"instances_after_merge={kept}")

    imwrite_unicode(args.output, output)
    print(f"image={args.image}")
    print(f"shape={width}x{height}")
    print(f"output={args.output}")
    print(f"instances_after_filter={kept}")


if __name__ == "__main__":
    main()
