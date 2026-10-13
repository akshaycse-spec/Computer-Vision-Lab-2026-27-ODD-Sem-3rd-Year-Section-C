"""Objective mask and boundary comparison metrics."""

from __future__ import annotations

import cv2
import numpy as np


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)


def boundary_f1(prediction: np.ndarray, truth: np.ndarray, tolerance: int = 2) -> float:
    predicted_edge = _boundary(prediction)
    true_edge = _boundary(truth)
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    predicted_near = cv2.dilate(predicted_edge, kernel)
    true_near = cv2.dilate(true_edge, kernel)

    predicted_count = int(predicted_edge.sum())
    true_count = int(true_edge.sum())
    if predicted_count == 0 or true_count == 0:
        return float(predicted_count == true_count)
    precision = float((predicted_edge & true_near).sum()) / predicted_count
    recall = float((true_edge & predicted_near).sum()) / true_count
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_mask(prediction: np.ndarray, truth: np.ndarray | None = None) -> dict[str, float | str]:
    predicted = prediction > 0
    metrics: dict[str, float | str] = {
        "foreground_ratio": round(float(predicted.mean()), 4),
        "accuracy": "N/A",
        "iou": "N/A",
        "dice": "N/A",
        "boundary_f1": "N/A",
    }
    if truth is None:
        return metrics

    actual = truth > 0
    intersection = int(np.logical_and(predicted, actual).sum())
    union = int(np.logical_or(predicted, actual).sum())
    total = predicted.size
    predicted_count = int(predicted.sum())
    actual_count = int(actual.sum())
    metrics.update(
        accuracy=round(float((predicted == actual).sum()) / total, 4),
        iou=round(intersection / union if union else 1.0, 4),
        dice=round(2 * intersection / (predicted_count + actual_count) if predicted_count + actual_count else 1.0, 4),
        boundary_f1=round(boundary_f1(prediction, truth), 4),
    )
    return metrics

