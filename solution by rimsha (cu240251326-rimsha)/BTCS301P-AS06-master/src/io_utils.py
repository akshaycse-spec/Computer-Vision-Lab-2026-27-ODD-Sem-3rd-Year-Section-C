"""Input helpers and a deterministic scene with a known foreground mask."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read ground-truth mask: {path}")
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {expected_shape}")
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def create_demo_scene(width: int = 640, height: int = 420) -> tuple[np.ndarray, np.ndarray]:
    """Create a noisy, unevenly lit color scene and its binary object mask."""
    x_gradient = np.linspace(25, 115, width, dtype=np.float32)
    background = np.tile(x_gradient, (height, 1))
    image = cv2.merge(
        [background * 0.75, background * 0.95, background * 1.15]
    ).clip(0, 255).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.circle(mask, (185, 205), 90, 255, -1)
    cv2.circle(mask, (325, 205), 90, 255, -1)
    cv2.rectangle(mask, (430, 95), (570, 185), 255, -1)
    cv2.ellipse(mask, (500, 305), (85, 55), -15, 0, 360, 255, -1)

    colored = image.copy()
    colored[mask > 0] = (185, 220, 245)
    cv2.circle(colored, (185, 205), 90, (170, 205, 245), -1)
    cv2.circle(colored, (325, 205), 90, (175, 235, 215), -1)

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 9, colored.shape).astype(np.int16)
    image = np.clip(colored.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image, mask
