import numpy as np

from src.evaluation import evaluate_mask
from src.io_utils import create_demo_scene
from src.segmenters import (
    adaptive_threshold,
    global_threshold,
    kmeans_segmentation,
    otsu_threshold,
    preprocess,
    watershed_segmentation,
)


def test_all_methods_preserve_image_dimensions():
    image, _ = create_demo_scene(width=240, height=180)
    gray = preprocess(image)
    masks = [
        global_threshold(gray),
        otsu_threshold(gray)[0],
        adaptive_threshold(gray),
        watershed_segmentation(image)[0],
        kmeans_segmentation(image, clusters=3)[1],
    ]
    for mask in masks:
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})


def test_otsu_and_metrics_are_valid():
    image, truth = create_demo_scene(width=240, height=180)
    mask, threshold = otsu_threshold(preprocess(image))
    scores = evaluate_mask(mask, truth)
    assert 0 <= threshold <= 255
    for name in ("accuracy", "iou", "dice", "boundary_f1"):
        assert 0.0 <= scores[name] <= 1.0


def test_demo_methods_find_foreground():
    image, _ = create_demo_scene(width=240, height=180)
    gray = preprocess(image)
    masks = [global_threshold(gray), otsu_threshold(gray)[0], adaptive_threshold(gray)]
    assert all(0 < np.count_nonzero(mask) < mask.size for mask in masks)

