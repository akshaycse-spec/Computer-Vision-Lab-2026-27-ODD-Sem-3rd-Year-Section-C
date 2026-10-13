# BTCS301P-AS02

## Experiment No. 2
**Contrast Enhancement and Histogram-Based Image Processing using Python and OpenCV**

This folder is a complete solution for the assignment. It implements contrast stretching, histogram analysis, global histogram equalization, CLAHE, visual comparison, histogram comparison, and quantitative observations.

## Folder contents

- `main.py` - complete runnable Python program.
- `sample_input.png` - generated low-contrast sample image used by default.
- `ANSWERS.md` - complete answers to all 10 assignment questions.
- `requirements.txt` - required Python packages.
- `run.bat` - Windows one-click setup and run script.
- `run.ps1` - PowerShell setup and run script.
- `output/` - generated enhanced images, plots, metrics, and observations.

## Required C: drive location

Place/extract the folder at:

```text
C:\BTCS301P-AS02
```

Then the main program will be:

```text
C:\BTCS301P-AS02\main.py
```

## Easiest way to run on Windows

1. Install Python 3.10 or newer and enable **Add Python to PATH** during installation.
2. Open `C:\BTCS301P-AS02`.
3. Double-click `run.bat`.
4. The script creates a local virtual environment, installs requirements, runs the experiment, and saves results in `C:\BTCS301P-AS02\output`.

## Manual commands

Open Command Prompt:

```bat
cd /d C:\BTCS301P-AS02
py -m pip install -r requirements.txt
py main.py
```

## Use your own image

```bat
py main.py --input "C:\path\to\your\image.jpg"
```

Optional CLAHE parameters:

```bat
py main.py --input "image.jpg" --clip-limit 2.5 --tile-size 8
```

## Generated output

The program creates:

- `01_original_grayscale.png`
- `02_contrast_stretched.png`
- `03_histogram_equalized.png`
- `04_clahe_enhanced.png`
- `05_image_comparison.png`
- `06_histogram_comparison.png`
- individual histogram plots
- `metrics.csv`
- `observations.txt`

## Experiment flow implemented

1. Import OpenCV, NumPy, and Matplotlib.
2. Load a low-contrast image / convert it to grayscale.
3. Display/save the original image and analyze its histogram.
4. Perform contrast stretching.
5. Apply global Histogram Equalization with OpenCV.
6. Apply CLAHE.
7. Plot and compare histograms.
8. Analyze brightness, contrast, dynamic range, entropy, and feature visibility.
9. Compare the enhancement methods.
10. Save observations for documentation.

## Notes for submission

The sample image is included only to make the assignment runnable immediately. For a laboratory demonstration, you can supply any medical, satellite, low-light, or other low-contrast image using `--input`.
