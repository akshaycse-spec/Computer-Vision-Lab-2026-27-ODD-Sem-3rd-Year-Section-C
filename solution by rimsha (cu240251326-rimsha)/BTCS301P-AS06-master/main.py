"""Run and compare classical image-segmentation techniques."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from time import perf_counter

import cv2

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "segmentation-mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import evaluate_mask
from src.io_utils import create_demo_scene, load_color_image, load_mask
from src.segmenters import (
    adaptive_threshold,
    global_threshold,
    kmeans_segmentation,
    otsu_threshold,
    preprocess,
    watershed_segmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Input image; uses a demo scene if omitted")
    parser.add_argument("--ground-truth", type=Path, help="Optional binary ground-truth mask")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--clusters", type=int, default=4, help="Number of K-means color clusters")
    return parser.parse_args()


def timed(function, *args, **kwargs):
    start = perf_counter()
    value = function(*args, **kwargs)
    return value, (perf_counter() - start) * 1000.0


def save_comparison(images: list[tuple[str, object]], destination: Path) -> None:
    columns = 3
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    for axis, (title, image) in zip(axes.flat, images):
        if len(image.shape) == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(images) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2")

    if args.image:
        image = load_color_image(args.image)
        truth = load_mask(args.ground_truth, image.shape[:2]) if args.ground_truth else None
        source_name = args.image.name
    else:
        image, truth = create_demo_scene()
        source_name = "generated benchmark scene"

    args.output.mkdir(parents=True, exist_ok=True)
    gray = preprocess(image)

    global_mask, global_ms = timed(global_threshold, gray)
    (otsu_mask, otsu_value), otsu_ms = timed(otsu_threshold, gray)
    adaptive_mask, adaptive_ms = timed(adaptive_threshold, gray)
    (watershed_mask, watershed_view), watershed_ms = timed(watershed_segmentation, image)
    (kmeans_view, kmeans_mask), kmeans_ms = timed(kmeans_segmentation, image, args.clusters)

    results = {
        "global": (global_mask, global_ms),
        "otsu": (otsu_mask, otsu_ms),
        "adaptive": (adaptive_mask, adaptive_ms),
        "watershed": (watershed_mask, watershed_ms),
        "kmeans": (kmeans_mask, kmeans_ms),
    }

    cv2.imwrite(str(args.output / "input.png"), image)
    cv2.imwrite(str(args.output / "preprocessed.png"), gray)
    cv2.imwrite(str(args.output / "watershed.png"), watershed_view)
    cv2.imwrite(str(args.output / "kmeans_color.png"), kmeans_view)
    for name, (mask, _) in results.items():
        cv2.imwrite(str(args.output / f"{name}_mask.png"), mask)

    rows = []
    for name, (mask, elapsed_ms) in results.items():
        row = {"method": name, "milliseconds": round(elapsed_ms, 3)}
        row.update(evaluate_mask(mask, truth))
        rows.append(row)

    fields = ["method", "milliseconds", "foreground_ratio", "accuracy", "iou", "dice", "boundary_f1"]
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    save_comparison(
        [
            ("Input", image),
            ("Preprocessed", gray),
            ("Global", global_mask),
            (f"Otsu (T={otsu_value:.1f})", otsu_mask),
            ("Adaptive", adaptive_mask),
            ("Watershed", watershed_view),
            (f"K-means (K={args.clusters})", kmeans_view),
        ],
        args.output / "comparison.png",
    )

    print(f"Processed {source_name}. Results written to {args.output.resolve()}")
    for row in rows:
        score = "N/A" if row["iou"] == "N/A" else f"{row['iou']:.3f}"
        print(f"{row['method']:10s} {row['milliseconds']:9.3f} ms  IoU: {score}")


if __name__ == "__main__":
    main()
