"""
BTCS301P-AS01 - Experiment No. 1
Implementation of Fundamental Image Processing Operations using Python and OpenCV.

Run:
    python main.py
    python main.py --show
    python main.py --image path/to/your/image.jpg --show

All processed images and observations are saved in the output/ folder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib
import numpy as np

# Use a non-interactive backend by default so the program also runs on systems
# without a desktop/GUI. With --show, pyplot windows can still be requested on
# normal desktop Python installations by changing backend externally if needed.
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_sample_image(path: Path, width: int = 960, height: int = 640) -> None:
    """Create a deterministic color image so the assignment runs immediately."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Smooth color gradients in BGR order.
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    canvas[:, :, 0] = x[None, :]                 # Blue channel gradient
    canvas[:, :, 1] = y[:, None]                 # Green channel gradient
    canvas[:, :, 2] = 180                        # Red channel constant

    # Add shapes/text to make transformations and ROI visibly meaningful.
    cv2.rectangle(canvas, (70, 80), (350, 300), (30, 220, 245), -1)
    cv2.circle(canvas, (650, 220), 120, (220, 60, 60), -1)
    cv2.line(canvas, (90, 500), (850, 420), (255, 255, 255), 10)
    cv2.putText(
        canvas,
        "BTCS301P-AS01",
        (250, 575),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)


def ensure_input_image(image_path: Path) -> Path:
    """Ensure there is an input image; create the included sample if necessary."""
    if image_path.exists():
        return image_path

    default_sample = Path(__file__).resolve().parent / "sample_input.jpg"
    if image_path.name == "sample_input.jpg" and image_path.parent.resolve() == default_sample.parent:
        create_sample_image(default_sample)
        return default_sample

    raise FileNotFoundError(
        f"Input image not found: {image_path}\n"
        "Provide a valid file with --image, e.g. python main.py --image myphoto.jpg"
    )


def load_image(image_path: Path) -> np.ndarray:
    """Load a color image using OpenCV (BGR channel order)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode the image: {image_path}")
    return image


def image_properties(image: np.ndarray) -> Dict[str, object]:
    """Return dimensions, resolution, channels and data type."""
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "dimensions": tuple(image.shape),
        "width": width,
        "height": height,
        "resolution": f"{width} x {height}",
        "total_pixels": width * height,
        "channels": channels,
        "dtype": str(image.dtype),
    }


def save_format_comparison(image: np.ndarray, output_dir: Path) -> Dict[str, object]:
    """Save JPEG and PNG versions and compare file sizes."""
    jpg_path = output_dir / "01_saved_as_jpeg.jpg"
    png_path = output_dir / "02_saved_as_png.png"

    # JPEG quality 95: lossy compression. PNG: lossless compression.
    cv2.imwrite(str(jpg_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(png_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    jpg_size = jpg_path.stat().st_size
    png_size = png_path.stat().st_size

    return {
        "jpeg_path": jpg_path,
        "png_path": png_path,
        "jpeg_bytes": jpg_size,
        "png_bytes": png_size,
        "smaller_file": "JPEG" if jpg_size < png_size else "PNG",
    }


def convert_color_spaces(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert BGR image to RGB, Grayscale, HSV and LAB."""
    return {
        "BGR": image_bgr,
        "RGB": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "Grayscale": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        "HSV": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV),
        "LAB": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB),
    }


def geometric_transformations(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Resize, rotate, and flip the input image."""
    height, width = image.shape[:2]

    resized = cv2.resize(
        image,
        (width // 2, height // 2),
        interpolation=cv2.INTER_AREA,
    )

    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    horizontal_flip = cv2.flip(image, 1)
    vertical_flip = cv2.flip(image, 0)

    return {
        "resized_50_percent": resized,
        "rotated_45_degrees": rotated,
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
    }


def image_negative(image: np.ndarray) -> np.ndarray:
    """Generate the complement/negative: for uint8, output = 255 - input."""
    return cv2.bitwise_not(image)


def crop_and_analyze_roi(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """Crop the central 50% region and return simple numerical characteristics."""
    height, width = image.shape[:2]
    x1, x2 = width // 4, (3 * width) // 4
    y1, y2 = height // 4, (3 * height) // 4

    roi = image[y1:y2, x1:x2].copy()
    mean_bgr = cv2.mean(roi)[:3]

    analysis = {
        "coordinates": f"x={x1}:{x2}, y={y1}:{y2}",
        "shape": tuple(roi.shape),
        "width": roi.shape[1],
        "height": roi.shape[0],
        "mean_B": round(mean_bgr[0], 2),
        "mean_G": round(mean_bgr[1], 2),
        "mean_R": round(mean_bgr[2], 2),
        "minimum_intensity": int(roi.min()),
        "maximum_intensity": int(roi.max()),
        "standard_deviation": round(float(roi.std()), 2),
    }
    return roi, analysis


def save_bgr(path: Path, image: np.ndarray) -> None:
    """Save an OpenCV image safely."""
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save image to {path}")


def save_processing_results(
    output_dir: Path,
    color_spaces: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    negative: np.ndarray,
    roi: np.ndarray,
) -> None:
    """Save each processed result as an image file."""
    save_bgr(output_dir / "03_grayscale.png", color_spaces["Grayscale"])
    save_bgr(output_dir / "04_hsv.png", color_spaces["HSV"])
    save_bgr(output_dir / "05_lab.png", color_spaces["LAB"])
    save_bgr(output_dir / "06_resized_50_percent.png", transforms["resized_50_percent"])
    save_bgr(output_dir / "07_rotated_45_degrees.png", transforms["rotated_45_degrees"])
    save_bgr(output_dir / "08_horizontal_flip.png", transforms["horizontal_flip"])
    save_bgr(output_dir / "09_vertical_flip.png", transforms["vertical_flip"])
    save_bgr(output_dir / "10_negative.png", negative)
    save_bgr(output_dir / "11_roi.png", roi)


def _to_rgb_for_plot(image: np.ndarray, title: str) -> Tuple[np.ndarray, str | None]:
    """Prepare an OpenCV image for correct Matplotlib visualization."""
    if image.ndim == 2:
        return image, "gray"
    if title == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB), None
    if title == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB), None
    # All other color images here are BGR unless explicitly RGB.
    if title == "RGB":
        return image, None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None


def create_comparison_figure(
    original: np.ndarray,
    color_spaces: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    negative: np.ndarray,
    roi: np.ndarray,
    output_dir: Path,
    show: bool,
) -> Path:
    """Display/save original and processed images together using Matplotlib."""
    items = [
        ("Original (BGR shown as RGB)", original),
        ("Grayscale", color_spaces["Grayscale"]),
        ("HSV", color_spaces["HSV"]),
        ("LAB", color_spaces["LAB"]),
        ("Resized 50%", transforms["resized_50_percent"]),
        ("Rotated 45 degrees", transforms["rotated_45_degrees"]),
        ("Horizontal Flip", transforms["horizontal_flip"]),
        ("Vertical Flip", transforms["vertical_flip"]),
        ("Negative", negative),
        ("ROI", roi),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    for ax, (title, img) in zip(axes, items):
        plot_key = title
        if title == "HSV":
            display_img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            cmap = None
        elif title == "LAB":
            display_img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
            cmap = None
        elif img.ndim == 2:
            display_img = img
            cmap = "gray"
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None
        ax.imshow(display_img, cmap=cmap)
        ax.set_title(plot_key)
        ax.axis("off")

    fig.suptitle("Fundamental Image Processing Operations - Comparison", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    comparison_path = output_dir / "12_all_operations_comparison.png"
    fig.savefig(comparison_path, dpi=160, bbox_inches="tight")

    if show:
        # On a desktop, this will display when an interactive backend is active.
        try:
            plt.show()
        except Exception as exc:
            print(f"Matplotlib display skipped: {exc}")
    plt.close(fig)
    return comparison_path


def display_with_opencv(image: np.ndarray, show: bool) -> None:
    """Display original image in an OpenCV window when --show is requested."""
    if not show:
        return
    try:
        cv2.imshow("Original Image - OpenCV", image)
        print("OpenCV window opened. Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        print("OpenCV GUI display is unavailable in this environment.")
        print(f"Reason: {exc}")


def save_observations(
    output_dir: Path,
    props: Dict[str, object],
    formats: Dict[str, object],
    roi_analysis: Dict[str, object],
) -> Path:
    """Write lab observations and preprocessing significance to a text file."""
    text = f"""BTCS301P-AS01 - EXPERIMENT 1 OBSERVATIONS
================================================

1. Image Properties
- Dimensions (NumPy shape): {props['dimensions']}
- Resolution: {props['resolution']}
- Total pixels: {props['total_pixels']}
- Number of channels: {props['channels']}
- Data type: {props['dtype']}

2. JPEG vs PNG
- JPEG size: {formats['jpeg_bytes']} bytes
- PNG size: {formats['png_bytes']} bytes
- Smaller file for this input: {formats['smaller_file']}
- JPEG normally uses lossy compression, whereas PNG is lossless. Therefore, JPEG often
  produces a smaller file for natural images, while PNG better preserves exact pixel values.

3. Color Space Conversion
- Grayscale reduces a three-channel color image to one intensity channel.
- HSV separates hue/saturation from brightness, which is useful for color-based segmentation.
- LAB separates perceptual lightness (L) from color components (A and B), which is useful
  when illumination and color information need to be handled separately.

4. Geometric Transformations
- Resizing changes spatial dimensions and may remove or interpolate information.
- Rotation changes orientation and introduces border regions when the output canvas is fixed.
- Horizontal/vertical flipping reflects the image and preserves its dimensions.

5. Negative / Complement
- The negative reverses the intensity relationship: dark pixels become bright and bright
  pixels become dark. For an 8-bit image, each channel is transformed approximately as 255-I.

6. Region of Interest (ROI)
- Coordinates: {roi_analysis['coordinates']}
- ROI shape: {roi_analysis['shape']}
- Mean BGR: ({roi_analysis['mean_B']}, {roi_analysis['mean_G']}, {roi_analysis['mean_R']})
- Min intensity: {roi_analysis['minimum_intensity']}
- Max intensity: {roi_analysis['maximum_intensity']}
- Standard deviation: {roi_analysis['standard_deviation']}

7. Significance of Preprocessing
Preprocessing standardizes image size and representation, reduces unnecessary information,
improves contrast or color separability, isolates relevant regions, and reduces variation caused
by orientation, illumination, or acquisition conditions. This makes subsequent feature
extraction, segmentation, object detection, classification, and deep-learning pipelines more
stable and computationally efficient.
"""
    path = output_dir / "observations.txt"
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="BTCS301P-AS01: Fundamental Image Processing Operations using OpenCV"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=project_dir / "sample_input.jpg",
        help="Path to the color input image (default: included sample_input.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "output",
        help="Directory in which processed images will be stored",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Attempt OpenCV/Matplotlib GUI display in addition to saving results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = ensure_input_image(args.image.resolve())
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BTCS301P-AS01 - Fundamental Image Processing Operations")
    print("=" * 70)
    print(f"Input image : {image_path}")
    print(f"Output dir  : {output_dir}\n")

    # Step 2: Load and display the original image.
    image = load_image(image_path)
    display_with_opencv(image, args.show)

    # Step 3: Examine image properties.
    props = image_properties(image)
    print("IMAGE PROPERTIES")
    for key, value in props.items():
        print(f"  {key:14}: {value}")

    # Step 4: Save in JPEG and PNG and compare.
    formats = save_format_comparison(image, output_dir)
    print("\nFORMAT COMPARISON")
    print(f"  JPEG: {formats['jpeg_bytes']} bytes")
    print(f"  PNG : {formats['png_bytes']} bytes")
    print(f"  Smaller file for this image: {formats['smaller_file']}")

    # Step 5: Color-space conversions.
    color_spaces = convert_color_spaces(image)

    # Step 6: Geometric transformations.
    transforms = geometric_transformations(image)

    # Step 7: Image complement / negative.
    negative = image_negative(image)

    # Step 8: Crop and analyze ROI.
    roi, roi_analysis = crop_and_analyze_roi(image)
    print("\nROI ANALYSIS")
    for key, value in roi_analysis.items():
        print(f"  {key:20}: {value}")

    # Step 9: Save/display processed images for comparison.
    save_processing_results(output_dir, color_spaces, transforms, negative, roi)
    comparison_path = create_comparison_figure(
        image,
        color_spaces,
        transforms,
        negative,
        roi,
        output_dir,
        args.show,
    )

    # Step 10: Document observations/significance.
    observations_path = save_observations(output_dir, props, formats, roi_analysis)

    print("\nCOMPLETED SUCCESSFULLY")
    print(f"  Comparison image : {comparison_path}")
    print(f"  Observations     : {observations_path}")
    print("  All individual processed images are saved in the output folder.")


if __name__ == "__main__":
    main()
