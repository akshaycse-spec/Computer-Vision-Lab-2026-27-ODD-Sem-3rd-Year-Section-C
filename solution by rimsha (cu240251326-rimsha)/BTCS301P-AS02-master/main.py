"""
BTCS301P-AS02 - Experiment No. 2
Contrast Enhancement and Histogram-Based Image Processing using Python and OpenCV

This program implements the complete experiment workflow:
1. Load / create a low-contrast image.
2. Convert it to grayscale.
3. Plot the original histogram.
4. Apply contrast stretching.
5. Apply global histogram equalization.
6. Apply CLAHE.
7. Compare histograms.
8. Measure brightness, contrast and feature visibility.
9. Compare enhancement techniques.
10. Save all results and observations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "sample_input.png"
DEFAULT_OUTPUT = BASE_DIR / "output"


def create_sample_low_contrast_image(path: Path, width: int = 900, height: int = 600) -> None:
    """Create a deterministic low-contrast test image so the project runs out-of-the-box."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]

    # Deliberately narrow intensity range (~90..155) to simulate poor contrast.
    base = 108 + 28 * x + 12 * y
    image = np.clip(base, 0, 255).astype(np.uint8)

    # Add low-contrast objects/details.
    cv2.rectangle(image, (80, 90), (350, 260), 128, -1)
    cv2.circle(image, (620, 185), 110, 142, -1)
    cv2.line(image, (70, 420), (820, 360), 98, 7)
    cv2.putText(
        image,
        "LOW CONTRAST",
        (150, 520),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        148,
        4,
        cv2.LINE_AA,
    )

    # Add small deterministic noise to imitate a real acquisition.
    rng = np.random.default_rng(30102)
    noise = rng.normal(0, 2.0, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load an image and return an 8-bit grayscale image."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    """Linearly stretch the minimum/maximum intensities to the full [0, 255] range."""
    min_val = float(np.min(image))
    max_val = float(np.max(image))

    if max_val <= min_val:
        return image.copy()

    stretched = (image.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply OpenCV global histogram equalization."""
    return cv2.equalizeHist(image)


def clahe_enhancement(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def histogram(image: np.ndarray) -> np.ndarray:
    """Return a 256-bin grayscale histogram."""
    return cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()


def entropy(image: np.ndarray) -> float:
    """Compute Shannon entropy of an 8-bit grayscale image."""
    hist = histogram(image)
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def image_metrics(image: np.ndarray) -> Dict[str, float]:
    """Compute objective indicators useful for the experiment comparison."""
    brightness = float(np.mean(image))
    contrast_std = float(np.std(image))
    dynamic_range = float(np.max(image) - np.min(image))
    laplacian_variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
    edges = cv2.Canny(image, 60, 140)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    return {
        "Brightness (mean)": brightness,
        "Contrast (std dev)": contrast_std,
        "Dynamic range": dynamic_range,
        "Entropy (bits)": entropy(image),
        "Feature visibility (Laplacian variance)": laplacian_variance,
        "Edge density": edge_density,
    }


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def save_comparison_figure(images: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save a 2x2 visual comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, image) in zip(axes.ravel(), images.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Contrast Enhancement Comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_histogram_figure(images: Dict[str, np.ndarray], output_path: Path) -> None:
    """Plot histograms of all compared images on one graph."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, image in images.items():
        ax.plot(histogram(image), label=name, linewidth=1.3)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_individual_histograms(images: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Save a separate histogram for each technique."""
    for name, image in images.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(histogram(image))
        ax.set_xlim(0, 255)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram - {name}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{safe_name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_metrics_csv(metrics: Dict[str, Dict[str, float]], path: Path) -> None:
    metric_names = list(next(iter(metrics.values())).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Technique", *metric_names])
        for technique, values in metrics.items():
            writer.writerow([technique, *[f"{values[m]:.6f}" for m in metric_names]])


def save_observations(metrics: Dict[str, Dict[str, float]], path: Path) -> None:
    """Create a concise experiment observation file using computed values."""
    best_contrast = max(metrics, key=lambda k: metrics[k]["Contrast (std dev)"])
    best_range = max(metrics, key=lambda k: metrics[k]["Dynamic range"])
    best_edges = max(metrics, key=lambda k: metrics[k]["Edge density"])

    lines = [
        "BTCS301P-AS02 - COMPUTED OBSERVATIONS",
        "=" * 48,
        "",
        "The program compared the original grayscale image with contrast stretching,",
        "global histogram equalization, and CLAHE.",
        "",
    ]

    for name, values in metrics.items():
        lines.append(f"{name}:")
        for metric, value in values.items():
            lines.append(f"  - {metric}: {value:.4f}")
        lines.append("")

    lines.extend(
        [
            f"Highest measured standard-deviation contrast: {best_contrast}",
            f"Largest measured dynamic range: {best_range}",
            f"Highest measured edge density: {best_edges}",
            "",
            "Interpretation:",
            "- Contrast stretching expands the occupied intensity range approximately linearly.",
            "- Global histogram equalization redistributes intensities using the image-wide CDF.",
            "- CLAHE operates on local tiles and clips excessive histogram peaks, so it is usually",
            "  more suitable when illumination varies across an image and when noise amplification",
            "  must be limited.",
            "- A visually useful result should improve detail visibility without introducing excessive",
            "  noise, over-enhancement, or unnatural brightness.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    print("\nIMAGE QUALITY / VISIBILITY METRICS")
    print("-" * 95)
    print(
        f"{'Technique':<25} {'Mean':>9} {'StdDev':>9} {'Range':>9} "
        f"{'Entropy':>9} {'LapVar':>12} {'EdgeDen':>10}"
    )
    print("-" * 95)
    for name, m in metrics.items():
        print(
            f"{name:<25} "
            f"{m['Brightness (mean)']:>9.2f} "
            f"{m['Contrast (std dev)']:>9.2f} "
            f"{m['Dynamic range']:>9.2f} "
            f"{m['Entropy (bits)']:>9.3f} "
            f"{m['Feature visibility (Laplacian variance)']:>12.2f} "
            f"{m['Edge density']:>10.4f}"
        )
    print("-" * 95)


def run(input_path: Path, output_dir: Path, clip_limit: float, tile_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        if input_path == DEFAULT_INPUT:
            print(f"Default sample image not found. Creating: {input_path.name}")
            create_sample_low_contrast_image(input_path)
        else:
            raise FileNotFoundError(f"Input image does not exist: {input_path}")

    original = load_grayscale_image(input_path)
    stretched = contrast_stretch(original)
    equalized = histogram_equalization(original)
    clahe = clahe_enhancement(original, clip_limit=clip_limit, tile_grid_size=(tile_size, tile_size))

    images = {
        "Original": original,
        "Contrast Stretching": stretched,
        "Histogram Equalization": equalized,
        "CLAHE": clahe,
    }

    save_image(output_dir / "01_original_grayscale.png", original)
    save_image(output_dir / "02_contrast_stretched.png", stretched)
    save_image(output_dir / "03_histogram_equalized.png", equalized)
    save_image(output_dir / "04_clahe_enhanced.png", clahe)

    save_comparison_figure(images, output_dir / "05_image_comparison.png")
    save_histogram_figure(images, output_dir / "06_histogram_comparison.png")
    save_individual_histograms(images, output_dir)

    metrics = {name: image_metrics(image) for name, image in images.items()}
    save_metrics_csv(metrics, output_dir / "metrics.csv")
    save_observations(metrics, output_dir / "observations.txt")

    print(f"\nInput image : {input_path}")
    print(f"Output dir  : {output_dir}")
    print_metrics(metrics)
    print("\nExperiment completed successfully.")
    print("Open output/05_image_comparison.png and output/06_histogram_comparison.png for comparison.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contrast Enhancement and Histogram-Based Image Processing using OpenCV"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to a source image. Default: sample_input.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for generated results. Default: output",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit. Default: 2.0",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=8,
        help="CLAHE tile grid width/height. Default: 8",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input.resolve(), args.output.resolve(), args.clip_limit, args.tile_size)
