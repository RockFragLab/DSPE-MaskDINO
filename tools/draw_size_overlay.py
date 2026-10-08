import argparse
import csv
import os

import cv2
import numpy as np


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def imwrite_unicode(path, image):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok, data = cv2.imencode(os.path.splitext(path)[1] or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    data.tofile(path)


def draw_text(img, text, xy, font_scale, color, thickness):
    x, y = xy
    outline = thickness + 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Draw particle size labels on a segmentation overlay.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--base-image", default="", help="Defaults to segmentation_overlay.png in --result-dir.")
    parser.add_argument("--particles-csv", default="", help="Defaults to fragmentation_particles.csv in --result-dir.")
    parser.add_argument("--markers-csv", default="", help="Defaults to fragmentation_markers.csv in --result-dir.")
    parser.add_argument("--output", default="", help="Defaults to segmentation_size_overlay.png in --result-dir.")
    parser.add_argument("--marker-real-size", type=float, required=True)
    parser.add_argument("--min-label-size", type=float, default=0.0, help="Only label stones with sieve size >= this value.")
    args = parser.parse_args()

    result_dir = args.result_dir
    base_image = args.base_image or os.path.join(result_dir, "segmentation_overlay.png")
    particles_csv = args.particles_csv or os.path.join(result_dir, "fragmentation_particles.csv")
    markers_csv = args.markers_csv or os.path.join(result_dir, "fragmentation_markers.csv")
    output = args.output or os.path.join(result_dir, "segmentation_size_overlay.png")

    img = imread_unicode(base_image)
    h, w = img.shape[:2]
    scale = max(w, h) / 4096.0
    font_scale = max(0.38, 0.48 * scale)
    thickness = max(1, int(round(1.5 * scale)))

    rows = []
    with open(particles_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            size = float(row["sieve_size_mm"])
            if size < args.min_label_size:
                continue
            rows.append(
                {
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "size": size,
                }
            )
    rows.sort(key=lambda item: item["size"], reverse=True)

    for row in rows:
        x = int(round(row["cx"])) + 4
        y = int(round(row["cy"])) - 4
        draw_text(img, f"{row['size']:.0f} mm", (x, y), font_scale, (255, 255, 255), thickness)

    if os.path.exists(markers_csv):
        with open(markers_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                x = int(round(float(row["cx"])))
                y = int(round(float(row["cy"])))
                cv2.circle(img, (x, y), max(7, int(8 * scale)), (0, 215, 255), -1, cv2.LINE_AA)
                draw_text(
                    img,
                    f"basketball {args.marker_real_size:g} mm",
                    (x + 12, y + 12),
                    font_scale * 1.15,
                    (0, 215, 255),
                    thickness + 1,
                )

    imwrite_unicode(output, img)
    print(output)


if __name__ == "__main__":
    main()
