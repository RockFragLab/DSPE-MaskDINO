import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_STONE_NAMES = ("shone", "stone", "rock", "rocks")
DEFAULT_MARKER_NAMES = ("basketball",)


def parse_csv_floats(text):
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_ints(text):
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def default_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def normalize_name(name):
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def class_name(cls_id, class_names):
    if 0 <= int(cls_id) < len(class_names):
        return class_names[int(cls_id)]
    return str(cls_id)


def class_matches(cls_id, class_names, wanted_names, wanted_ids):
    if int(cls_id) in wanted_ids:
        return True
    name = normalize_name(class_name(cls_id, class_names))
    return name in {normalize_name(item) for item in wanted_names}


def save_instances_npz(path, instances, class_names, image_shape):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    masks = np.empty(len(instances), dtype=object)
    for idx, inst in enumerate(instances):
        masks[idx] = inst["mask"].astype(np.uint8)

    np.savez_compressed(
        path,
        x0=np.asarray([inst["x0"] for inst in instances], dtype=np.int32),
        y0=np.asarray([inst["y0"] for inst in instances], dtype=np.int32),
        x1=np.asarray([inst["x1"] for inst in instances], dtype=np.int32),
        y1=np.asarray([inst["y1"] for inst in instances], dtype=np.int32),
        cls=np.asarray([inst["cls"] for inst in instances], dtype=np.int32),
        score=np.asarray([inst["score"] for inst in instances], dtype=np.float32),
        masks=masks,
        class_names=np.asarray(class_names, dtype=object),
        image_shape=np.asarray(image_shape, dtype=np.int32),
    )


def load_instances_npz(path):
    data = np.load(path, allow_pickle=True)
    instances = []
    for idx, mask in enumerate(data["masks"]):
        instances.append(
            {
                "x0": int(data["x0"][idx]),
                "y0": int(data["y0"][idx]),
                "x1": int(data["x1"][idx]),
                "y1": int(data["y1"][idx]),
                "mask": mask.astype(bool),
                "cls": int(data["cls"][idx]),
                "score": float(data["score"][idx]),
            }
        )
    class_names = [str(item) for item in data.get("class_names", [])]
    image_shape = tuple(int(v) for v in data.get("image_shape", np.asarray([0, 0])))
    return instances, class_names, image_shape


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def imwrite_unicode(path, image):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    data.tofile(path)


def instance_color(index, cls_id, class_names):
    name = normalize_name(class_name(cls_id, class_names))
    if name in {normalize_name(item) for item in DEFAULT_MARKER_NAMES}:
        return np.asarray([0, 215, 255], dtype=np.uint8)
    base = int((index + 1) * 2654435761 + int(cls_id) * 97531) & 0xFFFFFF
    return np.asarray(
        [
            80 + (base & 0x7F),
            80 + ((base >> 8) & 0x7F),
            80 + ((base >> 16) & 0x7F),
        ],
        dtype=np.uint8,
    )


def draw_segmentation_overlay(image_path, instances, class_names, output_path, alpha=0.58, draw_labels=False):
    output = imread_unicode(image_path).copy()
    height, width = output.shape[:2]

    for idx, inst in enumerate(instances):
        x0 = max(0, int(inst["x0"]))
        y0 = max(0, int(inst["y0"]))
        x1 = min(width, int(inst["x1"]))
        y1 = min(height, int(inst["y1"]))
        if x1 <= x0 or y1 <= y0:
            continue

        mask = inst["mask"].astype(bool)
        mask = mask[: y1 - y0, : x1 - x0]
        if not mask.any():
            continue

        color = instance_color(idx, inst["cls"], class_names)
        roi = output[y0:y1, x0:x1]
        colored = np.zeros_like(roi, dtype=np.uint8)
        colored[mask] = color
        roi[mask] = cv2.addWeighted(roi, 1.0 - alpha, colored, alpha, 0)[mask]

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            shifted = [contour + np.asarray([[[x0, y0]]], dtype=contour.dtype) for contour in contours]
            cv2.drawContours(output, shifted, -1, color.tolist(), thickness=2)

        if draw_labels:
            geom = mask_geometry(inst)
            label = f"{class_name(inst['cls'], class_names)} {float(inst['score']):.2f}"
            pos = (int(round(geom["cx"])), int(round(geom["cy"])))
            cv2.putText(output, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color.tolist(), 1, cv2.LINE_AA)

    imwrite_unicode(output_path, output)


def run_sliding_prediction(args):
    import torch
    from detectron2.data import MetadataCatalog
    from detectron2.engine.defaults import DefaultPredictor
    from predict_large_sliding_no_box import (
        center_in_valid_region,
        compact_instance,
        imread_unicode,
        mask_bbox,
        refine_instances,
        setup_cfg,
        tile_starts,
        valid_region_mask,
    )

    image_bgr = imread_unicode(args.image)
    if args.gray_input:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        inference_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        inference_image = image_bgr
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
    class_names = list(getattr(metadata, "thing_classes", []))

    height, width = image_bgr.shape[:2]
    stride = args.tile_size - args.overlap
    xs = tile_starts(width, args.tile_size, stride)
    ys = tile_starts(height, args.tile_size, stride)
    margin = args.overlap // 2

    collected = []
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
                detected = predictions["instances"].to("cpu")
                if len(detected) == 0 or not detected.has("pred_masks"):
                    print(f"[{tile_id}/{total_tiles}] kept=0 total={len(collected)}", flush=True)
                    continue

                scores = detected.scores.numpy() if detected.has("scores") else np.ones(len(detected))
                classes = (
                    detected.pred_classes.numpy()
                    if detected.has("pred_classes")
                    else np.zeros(len(detected), dtype=np.int64)
                )
                masks = detected.pred_masks.numpy().astype(bool)
                order = sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True)
                valid_mask = valid_region_mask(y1 - y0, x1 - x0, x0, y0, width, height, margin)
                tile_kept = 0

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

                    ys_local, xs_local = np.where(local_mask)
                    if len(xs_local) == 0:
                        continue
                    center_global = (float(np.median(xs_local)) + x0, float(np.median(ys_local)) + y0)
                    if not center_in_valid_region(
                        center_global, x0, y0, x1 - x0, y1 - y0, width, height, margin
                    ):
                        continue

                    if args.clip_to_valid_region:
                        local_mask = local_mask & valid_mask
                        if int(local_mask.sum()) < args.min_area:
                            continue

                    compact = compact_instance(local_mask[: y1 - y0, : x1 - x0], x0, y0, int(classes[idx]), score)
                    if compact is None:
                        continue
                    collected.append(compact)
                    tile_kept += 1

                print(f"[{tile_id}/{total_tiles}] kept={tile_kept} total={len(collected)}", flush=True)

    before_merge = len(collected)
    if args.merge_masks:
        collected = refine_instances(
            collected,
            merge_thresh=args.merge_overlap_thresh,
            resolve_thresh=args.resolve_overlap_thresh,
            min_area=args.min_area,
        )
    print(f"instances_before_merge={before_merge}")
    print(f"instances_after_merge={len(collected)}")
    return collected, class_names, (height, width)


def mask_geometry(inst):
    mask = inst["mask"].astype(np.uint8)
    area = float(mask.sum())
    if area <= 0:
        raise ValueError("empty mask")

    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] > 0:
        cx_local = moments["m10"] / moments["m00"]
        cy_local = moments["m01"] / moments["m00"]
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        cov = np.asarray([[mu20, mu11], [mu11, mu02]], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        minor_px = 4.0 * math.sqrt(float(eigvals[0]))
        major_px = 4.0 * math.sqrt(float(eigvals[1]))
    else:
        ys, xs = np.where(mask)
        cx_local = float(xs.mean())
        cy_local = float(ys.mean())
        major_px = 0.0
        minor_px = 0.0

    if major_px <= 0 or minor_px <= 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
            w, h = rect[1]
            major_px = max(float(w), float(h), major_px)
            minor_px = max(min(float(w), float(h)), 1.0)

    return {
        "cx": float(inst["x0"]) + float(cx_local),
        "cy": float(inst["y0"]) + float(cy_local),
        "area_px": area,
        "major_px": float(max(major_px, minor_px)),
        "minor_px": float(max(min(major_px, minor_px), 1e-6)),
    }


def marker_pixel_size(geom, mode):
    if mode == "equiv_diameter":
        return math.sqrt(4.0 * geom["area_px"] / math.pi)
    if mode == "major":
        return geom["major_px"]
    if mode == "minor":
        return geom["minor_px"]
    if mode == "sqrt_axes":
        return math.sqrt(geom["major_px"] * geom["minor_px"])
    raise ValueError(f"unknown marker measure: {mode}")


class LocalScaleModel:
    def __init__(self, markers, mode="auto", clip=True):
        self.markers = markers
        self.mode = mode
        self.clip = clip
        self.coeff = None
        self.base = None

        if not markers:
            raise ValueError("No calibration marker was found.")
        if mode == "constant" or len(markers) == 1:
            self.mode = "constant"
            self.base = float(np.mean([m["px_per_unit"] for m in markers]))
        elif mode == "two_marker" or (mode == "auto" and len(markers) == 2):
            self.mode = "two_marker"
            self.base = (markers[0], markers[1])
        elif mode == "plane" or mode == "auto":
            self.mode = "plane"
            a = np.asarray([[m["cx"], m["cy"], 1.0] for m in markers], dtype=np.float64)
            b = np.asarray([m["px_per_unit"] for m in markers], dtype=np.float64)
            self.coeff = np.linalg.lstsq(a, b, rcond=None)[0]
        else:
            raise ValueError(f"unsupported scale mode: {mode}")

    def __call__(self, x, y):
        if self.mode == "constant":
            return max(float(self.base), 1e-9)

        if self.mode == "two_marker":
            m0, m1 = self.base
            vx = m1["cx"] - m0["cx"]
            vy = m1["cy"] - m0["cy"]
            denom = vx * vx + vy * vy
            if denom <= 1e-9:
                return max(float(m0["px_per_unit"]), 1e-9)
            t = ((x - m0["cx"]) * vx + (y - m0["cy"]) * vy) / denom
            if self.clip:
                t = max(0.0, min(1.0, t))
            value = m0["px_per_unit"] + t * (m1["px_per_unit"] - m0["px_per_unit"])
            return max(float(value), 1e-9)

        value = self.coeff[0] * x + self.coeff[1] * y + self.coeff[2]
        if self.clip:
            lo = min(m["px_per_unit"] for m in self.markers)
            hi = max(m["px_per_unit"] for m in self.markers)
            value = max(lo, min(hi, value))
        return max(float(value), 1e-9)


def split_desktop_sieve_size(major, minor):
    a = max(float(major), float(minor))
    b = max(min(float(major), float(minor)), 1e-9)
    return 1.16 * b * math.sqrt((a / b) / 1.35)


def select_instances(instances, class_names, names, ids):
    return [inst for inst in instances if class_matches(inst["cls"], class_names, names, ids)]


def compute_fragmentation(instances, class_names, args):
    stone_ids = set(parse_csv_ints(args.stone_class_ids))
    marker_ids = set(parse_csv_ints(args.marker_class_ids))
    stone_names = args.stone_class_names or list(DEFAULT_STONE_NAMES)
    marker_names = args.marker_class_names or list(DEFAULT_MARKER_NAMES)

    stones = select_instances(instances, class_names, stone_names, stone_ids)
    markers = select_instances(instances, class_names, marker_names, marker_ids)

    if not stones:
        raise RuntimeError("No stone instances matched the requested stone classes.")
    if not markers:
        raise RuntimeError("No basketball calibration marker matched the requested marker classes.")

    marker_rows = []
    for idx, inst in enumerate(markers):
        geom = mask_geometry(inst)
        marker_px = marker_pixel_size(geom, args.marker_measure)
        marker_rows.append(
            {
                "id": idx,
                "class_id": int(inst["cls"]),
                "class_name": class_name(inst["cls"], class_names),
                "score": float(inst["score"]),
                "cx": geom["cx"],
                "cy": geom["cy"],
                "area_px": geom["area_px"],
                "major_px": geom["major_px"],
                "minor_px": geom["minor_px"],
                "marker_px": marker_px,
                "px_per_unit": marker_px / args.marker_real_size,
            }
        )

    marker_rows = sorted(marker_rows, key=lambda row: row["score"], reverse=True)
    if args.max_markers > 0:
        marker_rows = marker_rows[: args.max_markers]
    scale_model = LocalScaleModel(marker_rows, mode=args.scale_mode, clip=not args.no_clip_scale)

    particle_rows = []
    for idx, inst in enumerate(stones):
        geom = mask_geometry(inst)
        px_per_unit = scale_model(geom["cx"], geom["cy"])
        major = geom["major_px"] / px_per_unit
        minor = geom["minor_px"] / px_per_unit
        area = geom["area_px"] / (px_per_unit * px_per_unit)
        sieve = split_desktop_sieve_size(major, minor)
        weight = area if args.weight_mode == "area" else 1.0
        particle_rows.append(
            {
                "id": idx,
                "class_id": int(inst["cls"]),
                "class_name": class_name(inst["cls"], class_names),
                "score": float(inst["score"]),
                "cx": geom["cx"],
                "cy": geom["cy"],
                "px_per_unit": px_per_unit,
                "area_px": geom["area_px"],
                f"area_{args.unit}2": area,
                "major_px": geom["major_px"],
                "minor_px": geom["minor_px"],
                f"major_{args.unit}": major,
                f"minor_{args.unit}": minor,
                f"sieve_size_{args.unit}": sieve,
                "weight": weight,
            }
        )

    return particle_rows, marker_rows, scale_model.mode


def split_desktop_size_column(unit):
    return f"Size [{unit}]"


def split_desktop_percentile_label(percent):
    value = float(percent)
    if 0.0 < value <= 90.0 and abs(value - round(value)) < 1e-9:
        return f"F{int(round(value))}"
    return f"P{value:g}"


def build_curve(particles, sieve_sizes, unit, weight_mode, split_desktop_columns=True):
    size_key = f"sieve_size_{unit}"
    sizes = np.asarray([p[size_key] for p in particles], dtype=np.float64)
    weights = np.asarray([p["weight"] for p in particles], dtype=np.float64)
    valid = np.isfinite(sizes) & np.isfinite(weights) & (weights > 0)
    sizes = sizes[valid]
    weights = weights[valid]
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("No positive particle weights.")

    if not sieve_sizes:
        lo = max(float(np.nanmin(sizes)), 1e-6)
        hi = max(float(np.nanmax(sizes)), lo * 1.01)
        sieve_sizes = np.geomspace(lo, hi, num=32).tolist()

    rows = []
    for sieve in sorted(sieve_sizes):
        passing = float(weights[sizes <= sieve].sum() / total * 100.0)
        if split_desktop_columns:
            rows.append({split_desktop_size_column(unit): float(sieve), "% Passing": passing, "weight_mode": weight_mode})
        else:
            rows.append({f"sieve_size_{unit}": float(sieve), "percent_passing": passing, "weight_mode": weight_mode})
    return rows


def size_at_percent(particles, percents, unit):
    size_key = f"sieve_size_{unit}"
    sizes = np.asarray([p[size_key] for p in particles], dtype=np.float64)
    weights = np.asarray([p["weight"] for p in particles], dtype=np.float64)
    valid = np.isfinite(sizes) & np.isfinite(weights) & (weights > 0)
    sizes = sizes[valid]
    weights = weights[valid]
    order = np.argsort(sizes)
    sizes = sizes[order]
    weights = weights[order]
    cum = np.cumsum(weights) / weights.sum() * 100.0

    result = {}
    for percent in percents:
        label = split_desktop_percentile_label(percent)
        result[f"{label}_{unit}"] = float(np.interp(percent, cum, sizes))
    return result


def build_f_values(particles, percents, unit):
    size_key = f"sieve_size_{unit}"
    sizes = np.asarray([p[size_key] for p in particles], dtype=np.float64)
    weights = np.asarray([p["weight"] for p in particles], dtype=np.float64)
    valid = np.isfinite(sizes) & np.isfinite(weights) & (weights > 0)
    sizes = sizes[valid]
    weights = weights[valid]
    order = np.argsort(sizes)
    sizes = sizes[order]
    weights = weights[order]
    cum = np.cumsum(weights) / weights.sum() * 100.0

    rows = []
    for percent in percents:
        rows.append(
            {
                "Name": split_desktop_percentile_label(percent),
                "% Passing": float(percent),
                split_desktop_size_column(unit): float(np.interp(percent, cum, sizes)),
            }
        )
    return rows


def particle_distribution_arrays(particles, unit):
    size_key = f"sieve_size_{unit}"
    sizes = np.asarray([p[size_key] for p in particles], dtype=np.float64)
    weights = np.asarray([p["weight"] for p in particles], dtype=np.float64)
    valid = np.isfinite(sizes) & np.isfinite(weights) & (sizes > 0) & (weights > 0)
    sizes = sizes[valid]
    weights = weights[valid]
    if sizes.size < 5:
        raise RuntimeError("At least 5 positive particles are required for Swebrec fitting.")
    order = np.argsort(sizes)
    sizes = sizes[order]
    weights = weights[order]
    passing_fraction = np.cumsum(weights) / weights.sum()
    return sizes, weights, passing_fraction


def expanded_swebrec_function(x, a, b, c, x50, xmax):
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-12
    x = np.clip(x, eps, xmax * (1.0 - 1e-12))
    x50 = np.clip(float(x50), eps, xmax * (1.0 - 1e-12))
    a = np.clip(float(a), 1e-6, 1.0 - 1e-6)
    log_den = max(math.log(xmax / x50), eps)
    lin_den = max(xmax / x50 - 1.0, eps)
    r_log = np.maximum(np.log(xmax / x) / log_den, eps)
    r_lin = np.maximum((xmax / x - 1.0) / lin_den, eps)
    return 1.0 / (1.0 + a * np.power(r_log, b) + (1.0 - a) * np.power(r_lin, c))


def empirical_size_at_fraction(sizes, weights, fraction):
    order = np.argsort(sizes)
    sizes = sizes[order]
    weights = weights[order]
    cum = np.cumsum(weights) / weights.sum()
    return float(np.interp(float(fraction), cum, sizes))


def fit_expanded_swebrec(particles, unit, percents, curve_sizes, max_multiplier=5.0, curve_points=120):
    try:
        from scipy.optimize import brentq, curve_fit
    except Exception as exc:
        raise RuntimeError("Swebrec fitting requires scipy in the active Python environment.") from exc

    sizes, weights, passing_fraction = particle_distribution_arrays(particles, unit)
    x_min = float(np.min(sizes))
    x_max_observed = float(np.max(sizes))
    x50_empirical = empirical_size_at_fraction(sizes, weights, 0.5)
    upper_xmax = max(x_max_observed * max(float(max_multiplier), 1.01), x_max_observed * 1.02)
    lower_xmax = x_max_observed * 1.0001
    x50_lower = max(x_min * 0.25, 1e-9)
    x50_upper = max(x_max_observed * 0.999, x50_lower * 1.01)
    p0 = [0.5, 2.0, 2.0, min(max(x50_empirical, x50_lower), x50_upper), x_max_observed * 1.05]
    bounds = (
        [1e-4, 0.05, 0.05, x50_lower, lower_xmax],
        [0.9999, 20.0, 20.0, x50_upper, upper_xmax],
    )

    params, _ = curve_fit(
        expanded_swebrec_function,
        sizes,
        passing_fraction,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    a, b, c, x50, xmax = [float(value) for value in params]
    fitted = expanded_swebrec_function(sizes, a, b, c, x50, xmax)
    residual = passing_fraction - fitted
    sse = float(np.sum(residual * residual))
    sst = float(np.sum((passing_fraction - float(np.mean(passing_fraction))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    rmse = math.sqrt(sse / max(len(sizes), 1))

    fit_rows = []
    for percent in percents:
        fraction = float(percent) / 100.0

        def root_func(value):
            return float(expanded_swebrec_function(value, a, b, c, x50, xmax) - fraction)

        lo = max(x_min * 1e-3, 1e-9)
        hi = xmax * (1.0 - 1e-9)
        try:
            size_value = float(brentq(root_func, lo, hi, maxiter=200))
        except ValueError:
            grid = np.geomspace(lo, hi, num=10000)
            y = expanded_swebrec_function(grid, a, b, c, x50, xmax)
            size_value = float(np.interp(fraction, y, grid))
        fit_rows.append(
            {
                "Name": split_desktop_percentile_label(percent),
                "% Passing": float(percent),
                split_desktop_size_column(unit): size_value,
            }
        )

    if curve_sizes:
        curve_x = np.asarray(sorted(curve_sizes), dtype=np.float64)
    else:
        lo = max(x_min, 1e-6)
        hi = xmax * (1.0 - 1e-6)
        curve_x = np.geomspace(lo, hi, num=max(int(curve_points), 16))
    curve_y = expanded_swebrec_function(curve_x, a, b, c, x50, xmax) * 100.0
    curve_rows = [
        {split_desktop_size_column(unit): float(x), "% Passing": float(y), "model": "expanded_swebrec"}
        for x, y in zip(curve_x, curve_y)
    ]

    params_row = {
        "model": "expanded_swebrec",
        "A": a,
        "B": b,
        "C": c,
        f"x50_{unit}": x50,
        f"xmax_{unit}": xmax,
        f"xmax_observed_{unit}": x_max_observed,
        "R2": r2,
        "RMSE": rmse,
        "particle_count": int(len(sizes)),
    }
    return params_row, fit_rows, curve_rows


def build_size_bins(particles, bin_edges, unit, weight_mode):
    if not bin_edges:
        return [], {}

    edges = sorted(float(edge) for edge in bin_edges)
    size_key = f"sieve_size_{unit}"
    sizes = np.asarray([p[size_key] for p in particles], dtype=np.float64)
    weights = np.asarray([p["weight"] for p in particles], dtype=np.float64)
    valid = np.isfinite(sizes) & np.isfinite(weights) & (weights > 0)
    sizes = sizes[valid]
    weights = weights[valid]
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("No positive particle weights.")

    intervals = []
    intervals.append((f"<{edges[0]:g}", -np.inf, edges[0]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        intervals.append((f"{lo:g}~{hi:g}", lo, hi))
    intervals.append((f">{edges[-1]:g}", edges[-1], np.inf))

    rows = []
    wide = {}
    for idx, (label, lo, hi) in enumerate(intervals):
        if idx == 0:
            mask = sizes < hi
        elif idx == len(intervals) - 1:
            mask = sizes >= lo
        else:
            mask = (sizes >= lo) & (sizes < hi)
        weight = float(weights[mask].sum())
        percent = weight / total * 100.0
        row = {
            "bin": label,
            f"lower_{unit}": "" if math.isinf(lo) else lo,
            f"upper_{unit}": "" if math.isinf(hi) else hi,
            "percent": percent,
            "weight": weight,
            "count": int(mask.sum()),
            "weight_mode": weight_mode,
        }
        rows.append(row)
        wide[label] = percent
    return rows, wide


def write_wide_csv(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute Split-Desktop-like rock fragmentation from stone/basketball masks.")
    parser.add_argument("--image", help="Input image. Required when running prediction.")
    parser.add_argument("--config", help="MaskDINO config. Required unless --instances-npz is used.")
    parser.add_argument("--weights", help="MaskDINO weights. Required unless --instances-npz is used.")
    parser.add_argument("--dataset", default="date1_val")
    parser.add_argument("--instances-npz", help="Optional saved compact instance masks from a previous run.")
    parser.add_argument("--save-instances-npz", help="Optional path to save compact instance masks.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--segmentation-output",
        default="",
        help="Optional segmentation overlay image path. Defaults to segmentation_overlay.png in --output-dir when --image is provided.",
    )
    parser.add_argument("--segmentation-alpha", type=float, default=0.58)
    parser.add_argument("--draw-labels", action="store_true")
    parser.add_argument("--unit", default="mm", help="Real-world unit for marker size and outputs.")
    parser.add_argument("--marker-real-size", type=float, required=True, help="Basketball real diameter in --unit.")
    parser.add_argument("--marker-measure", default="equiv_diameter", choices=["equiv_diameter", "major", "minor", "sqrt_axes"])
    parser.add_argument("--scale-mode", default="auto", choices=["auto", "constant", "two_marker", "plane"])
    parser.add_argument("--no-clip-scale", action="store_true")
    parser.add_argument("--max-markers", type=int, default=2, help="Use top-N basketball markers by score. 0 keeps all.")
    parser.add_argument("--stone-class-names", nargs="*", default=list(DEFAULT_STONE_NAMES))
    parser.add_argument("--marker-class-names", nargs="*", default=list(DEFAULT_MARKER_NAMES))
    parser.add_argument("--stone-class-ids", default="")
    parser.add_argument("--marker-class-ids", default="")
    parser.add_argument("--weight-mode", choices=["area", "count"], default="area")
    parser.add_argument("--sieve-sizes", default="", help="Comma-separated sieve sizes in --unit. Empty = auto log-spaced.")
    parser.add_argument(
        "--size-bin-edges",
        default="",
        help="Comma-separated custom size bin edges in --unit, e.g. 4.75,10,20,40,100,200,300,400,500,600,800.",
    )
    parser.add_argument("--percentiles", default="10,20,30,40,50,60,70,80,90,95,99.5")
    parser.add_argument("--no-swebrec-fit", action="store_true", help="Disable expanded Swebrec fitting outputs.")
    parser.add_argument("--swebrec-max-multiplier", type=float, default=5.0, help="Upper bound multiplier for fitted xmax.")
    parser.add_argument("--swebrec-curve-points", type=int, default=120, help="Number of points in auto-generated Swebrec curve.")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min-size-test", type=int, default=0)
    parser.add_argument("--max-size-test", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--max-local-mask-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-local-bbox-area-ratio", type=float, default=0.0)
    parser.add_argument("--gray-input", action="store_true")
    parser.add_argument("--clip-to-valid-region", action="store_true")
    parser.add_argument("--merge-masks", action="store_true", default=True)
    parser.add_argument("--merge-overlap-thresh", type=float, default=0.2)
    parser.add_argument("--resolve-overlap-thresh", type=float, default=0.05)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    if args.instances_npz:
        instances, class_names, image_shape = load_instances_npz(args.instances_npz)
    else:
        if not args.image or not args.config or not args.weights:
            raise ValueError("--image, --config and --weights are required unless --instances-npz is provided.")
        if args.overlap >= args.tile_size:
            raise ValueError("--overlap must be smaller than --tile-size.")
        instances, class_names, image_shape = run_sliding_prediction(args)

    if args.save_instances_npz:
        save_instances_npz(args.save_instances_npz, instances, class_names, image_shape)

    particles, markers, scale_mode_used = compute_fragmentation(instances, class_names, args)
    sieve_sizes = parse_csv_floats(args.sieve_sizes)
    bin_edges = parse_csv_floats(args.size_bin_edges)
    if not sieve_sizes and bin_edges:
        sieve_sizes = bin_edges
    percentile_values = parse_csv_floats(args.percentiles)
    curve = build_curve(particles, sieve_sizes, args.unit, args.weight_mode, split_desktop_columns=True)
    legacy_curve = build_curve(particles, sieve_sizes, args.unit, args.weight_mode, split_desktop_columns=False)
    bins, bins_wide = build_size_bins(particles, bin_edges, args.unit, args.weight_mode)
    percentiles = size_at_percent(particles, percentile_values, args.unit)
    f_values = build_f_values(particles, percentile_values, args.unit)
    swebrec_params = None
    swebrec_f_values = []
    swebrec_curve = []
    swebrec_error = None
    if not args.no_swebrec_fit:
        try:
            swebrec_params, swebrec_f_values, swebrec_curve = fit_expanded_swebrec(
                particles,
                args.unit,
                percentile_values,
                sieve_sizes,
                max_multiplier=args.swebrec_max_multiplier,
                curve_points=args.swebrec_curve_points,
            )
        except Exception as exc:
            swebrec_error = str(exc)

    os.makedirs(args.output_dir, exist_ok=True)
    particles_csv = os.path.join(args.output_dir, "fragmentation_particles.csv")
    markers_csv = os.path.join(args.output_dir, "fragmentation_markers.csv")
    curve_csv = os.path.join(args.output_dir, "fragmentation_curve.csv")
    legacy_curve_csv = os.path.join(args.output_dir, "fragmentation_curve_legacy.csv")
    f_values_csv = os.path.join(args.output_dir, "fragmentation_f_values.csv")
    swebrec_params_csv = os.path.join(args.output_dir, "fragmentation_swebrec_parameters.csv")
    swebrec_f_values_csv = os.path.join(args.output_dir, "fragmentation_swebrec_f_values.csv")
    swebrec_curve_csv = os.path.join(args.output_dir, "fragmentation_swebrec_curve.csv")
    bins_csv = os.path.join(args.output_dir, "fragmentation_bins.csv")
    bins_wide_csv = os.path.join(args.output_dir, "fragmentation_bins_wide.csv")
    summary_json = os.path.join(args.output_dir, "fragmentation_summary.json")
    segmentation_output = None

    if args.image:
        segmentation_output = args.segmentation_output or os.path.join(args.output_dir, "segmentation_overlay.png")
        draw_segmentation_overlay(
            args.image,
            instances,
            class_names,
            segmentation_output,
            alpha=args.segmentation_alpha,
            draw_labels=args.draw_labels,
        )

    write_csv(particles_csv, particles)
    write_csv(markers_csv, markers)
    write_csv(curve_csv, curve)
    write_csv(legacy_curve_csv, legacy_curve)
    write_csv(f_values_csv, f_values)
    if swebrec_params:
        write_csv(swebrec_params_csv, [swebrec_params])
        write_csv(swebrec_f_values_csv, swebrec_f_values)
        write_csv(swebrec_curve_csv, swebrec_curve)
    write_csv(bins_csv, bins)
    if bins_wide:
        write_wide_csv(bins_wide_csv, bins_wide)

    summary = {
        "image": args.image,
        "instances_npz": args.instances_npz,
        "image_shape": list(image_shape),
        "class_names": class_names,
        "stone_count": len(particles),
        "marker_count": len(markers),
        "marker_real_size": args.marker_real_size,
        "unit": args.unit,
        "marker_measure": args.marker_measure,
        "scale_mode_used": scale_mode_used,
        "weight_mode": args.weight_mode,
        **percentiles,
        "particles_csv": particles_csv,
        "markers_csv": markers_csv,
        "curve_csv": curve_csv,
        "legacy_curve_csv": legacy_curve_csv,
        "f_values_csv": f_values_csv,
        "swebrec_fit_enabled": not args.no_swebrec_fit,
        "swebrec_fit_error": swebrec_error,
        "swebrec_parameters_csv": swebrec_params_csv if swebrec_params else None,
        "swebrec_f_values_csv": swebrec_f_values_csv if swebrec_params else None,
        "swebrec_curve_csv": swebrec_curve_csv if swebrec_params else None,
        "bins_csv": bins_csv if bins else None,
        "bins_wide_csv": bins_wide_csv if bins_wide else None,
        "segmentation_output": segmentation_output,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"stone_count={len(particles)}")
    print(f"marker_count={len(markers)}")
    print(f"scale_mode_used={scale_mode_used}")
    for key, value in percentiles.items():
        print(f"{key}={value:.6g}")
    print(f"particles_csv={particles_csv}")
    print(f"markers_csv={markers_csv}")
    print(f"curve_csv={curve_csv}")
    print(f"legacy_curve_csv={legacy_curve_csv}")
    print(f"f_values_csv={f_values_csv}")
    if swebrec_params:
        print(f"swebrec_parameters_csv={swebrec_params_csv}")
        print(f"swebrec_f_values_csv={swebrec_f_values_csv}")
        print(f"swebrec_curve_csv={swebrec_curve_csv}")
        print(f"swebrec_R2={swebrec_params['R2']:.6g}")
    elif swebrec_error:
        print(f"swebrec_fit_error={swebrec_error}")
    if bins:
        print(f"bins_csv={bins_csv}")
        print(f"bins_wide_csv={bins_wide_csv}")
    if segmentation_output:
        print(f"segmentation_output={segmentation_output}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
