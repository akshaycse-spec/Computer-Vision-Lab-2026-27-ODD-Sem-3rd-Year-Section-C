# BTCS301P-AS01

**Experiment No. 1:** Implementation of Fundamental Image Processing Operations using Python and OpenCV

This folder contains a complete solution for all implementation steps and all 10 theory questions.

## Folder Contents

- `main.py` - complete executable Python solution.
- `ANSWERS.md` - complete answers to all 10 assignment questions.
- `sample_input.jpg` - sample local color image for immediate execution.
- `requirements.txt` - required Python libraries.
- `run.bat` - Windows one-click setup/run script for `C:\BTCS301P-AS01`.
- `output/` - generated automatically when the program runs.

## Operations Implemented

1. Import OpenCV, NumPy, and Matplotlib.
2. Load a local color image and provide OpenCV/Matplotlib display functionality.
3. Examine dimensions, resolution, channels, total pixels, and data type.
4. Save and compare JPEG and PNG versions.
5. Convert BGR/RGB image data to Grayscale, HSV, and LAB.
6. Resize, rotate, horizontally flip, and vertically flip the image.
7. Generate the image complement/negative.
8. Crop and numerically analyze a Region of Interest (ROI).
9. Save a comparison figure containing the original and processed images.
10. Generate an observations file explaining preprocessing significance.

## Windows Setup - Required Folder Location

Place this entire folder at:

`C:\BTCS301P-AS01`

Then open Command Prompt and run:

```bat
cd /d C:\BTCS301P-AS01
py -m pip install -r requirements.txt
py main.py --show
```

Or simply double-click `run.bat`.

## Run Without GUI Windows

If you only want the generated output files:

```bat
cd /d C:\BTCS301P-AS01
py main.py
```

## Run With Your Own Image

Copy your image into the folder, for example `my_image.jpg`, then run:

```bat
py main.py --image my_image.jpg --show
```

All processed files are created inside `C:\BTCS301P-AS01\output`.

## Expected Output Files

- `01_saved_as_jpeg.jpg`
- `02_saved_as_png.png`
- `03_grayscale.png`
- `04_hsv.png`
- `05_lab.png`
- `06_resized_50_percent.png`
- `07_rotated_45_degrees.png`
- `08_horizontal_flip.png`
- `09_vertical_flip.png`
- `10_negative.png`
- `11_roi.png`
- `12_all_operations_comparison.png`
- `observations.txt`

## Notes

- OpenCV loads color images in **BGR** order by default.
- Matplotlib normally expects **RGB** ordering, so BGR-to-RGB conversion is performed for correct display.
- The script is written to work even on machines without GUI support; all results are saved to disk regardless of GUI availability.
