"""Implementations of the segmentation techniques used in the experiment."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def preprocess(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def _smaller_foreground(binary: np.ndarray) -> np.ndarray:
    """Treat the less common side of a binary split as foreground."""
    return cv2.bitwise_not(binary) if np.mean(binary > 0) > 0.5 else binary


def global_threshold(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return _smaller_foreground(binary)


def otsu_threshold(gray: np.ndarray) -> tuple[np.ndarray, float]:
    value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _smaller_foreground(binary), float(value)


def adaptive_threshold(gray: np.ndarray, block_size: int = 301, constant: int = -5) -> np.ndarray:
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd number of at least 3")
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
    )
    return _smaller_foreground(binary)


def watershed_segmentation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = preprocess(image)
    binary, _ = otsu_threshold(gray)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_background = cv2.dilate(opened, kernel, iterations=3)
    distance = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    sure_foreground = np.zeros_like(opened)
    component_count, component_labels = cv2.connectedComponents(opened)
    for label in range(1, component_count):
        component = component_labels == label
        local_maximum = float(distance[component].max())
        sure_foreground[component & (distance >= 0.70 * local_maximum)] = 255
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image.copy(), markers)

    mask = np.where(markers > 1, 255, 0).astype(np.uint8)
    view = image.copy()
    view[markers == -1] = (0, 0, 255)
    return mask, view


def kmeans_segmentation(
    image: np.ndarray, clusters: int = 4, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    if clusters < 2:
        raise ValueError("clusters must be at least 2")
    pixels = image.reshape((-1, 3)).astype(np.float32)
    model = MiniBatchKMeans(
        n_clusters=clusters, random_state=random_state, n_init=5, batch_size=4096
    )
    labels = model.fit_predict(pixels)
    centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
    quantized = centers[labels].reshape(image.shape)

    brightness = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).ravel()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_labels = np.flatnonzero(brightness > otsu_value)
    mask = np.where(
        np.isin(labels.reshape(image.shape[:2]), foreground_labels), 255, 0
    ).astype(np.uint8)
    mask = _smaller_foreground(mask)
    return quantized, mask
