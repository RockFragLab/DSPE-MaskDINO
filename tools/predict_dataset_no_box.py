import argparse
import os
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor

from predict_one_no_box import overlay_masks, setup_cfg


def iter_dataset_images(dataset_name):
    records = DatasetCatalog.get(dataset_name)
    records = sorted(records, key=lambda r: os.path.basename(r["file_name"]))
    for record in records:
        yield record["file_name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="date2_val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--instance-colors", action="store_true")
    parser.add_argument("--draw-labels", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = setup_cfg(args.config, args.weights, args.dataset, args.threshold, args.device)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = list(iter_dataset_images(args.dataset))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    for idx, image_path in enumerate(image_paths, start=1):
        image_bgr = read_image(image_path, format="BGR")
        predictions = predictor(image_bgr)
        instances = predictions["instances"].to("cpu")
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

        stem = Path(image_path).stem
        out_path = os.path.join(args.output_dir, f"{idx:04d}_{stem}.png")
        cv2.imwrite(out_path, visualized)
        print(f"[{idx}/{len(image_paths)}] kept={kept} output={out_path}", flush=True)


if __name__ == "__main__":
    main()
