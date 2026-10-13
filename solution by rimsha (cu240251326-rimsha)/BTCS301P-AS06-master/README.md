# Image Segmentation Techniques - Experiment 6

A compact OpenCV project that implements and compares five image-segmentation methods:

- Global thresholding
- Otsu's thresholding
- Adaptive thresholding
- Watershed segmentation
- K-means color clustering

The program records execution time and, when a ground-truth mask is available, accuracy, IoU, Dice score, and boundary F1 score. With no arguments it creates a reproducible benchmark scene containing uneven illumination and touching objects.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run the built-in benchmark:

```bash
python main.py
```

Run on a real-world image:

```bash
python main.py --image path/to/photo.jpg --output results
```

Optionally provide a binary ground-truth mask for objective scores:

```bash
python main.py --image photo.jpg --ground-truth mask.png --clusters 4
```

The output folder contains each segmented image, `comparison.png`, and `metrics.csv`. For K-means, both the color-cluster image and a foreground mask selected from the brighter clusters are saved.

## Project structure

```text
.
|-- main.py                 # Command-line experiment
|-- requirements.txt       # Python dependencies
|-- src/
|   |-- io_utils.py         # Image loading and benchmark generation
|   |-- segmenters.py       # Five segmentation techniques
|   `-- evaluation.py       # Accuracy and boundary metrics
|-- tests/test_segmenters.py
`-- docs/
    |-- REPORT.md           # Comparative analysis and observations
    `-- ANSWERS.md          # Answers to assignment questions
```

## Expected comparison

Global thresholding is fastest but sensitive to lighting. Otsu automatically selects one global threshold. Adaptive thresholding is stronger under uneven illumination. Watershed is especially useful for touching objects. K-means uses color information and handles multi-region scenes, but costs more computation.

## Notes

- Accuracy scores require a mask whose white pixels represent the foreground.
- For real photographs, inspect all K-means color regions; its automatic binary mask uses brightness as a simple foreground heuristic.
- Re-run with different `--clusters` values to study the effect of K.
