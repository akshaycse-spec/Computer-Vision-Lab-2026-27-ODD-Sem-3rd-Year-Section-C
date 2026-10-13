# Computer Vision Assignment – Experiment 5

**Experiment:** Feature Extraction and Image Analysis using SIFT and HOG Descriptors

This repository contains the assignment answers and a Google Colab/Jupyter Notebook implementation of SIFT and HOG feature extraction.

## Contents

- `Experiment_5_SIFT_HOG.ipynb` – Complete practical notebook
- `Assignment_Answers.md` – Answers to all 10 questions
- `requirements.txt` – Required Python libraries

## Experiment workflow

1. Import libraries and load a real-world image.
2. Convert the image to grayscale.
3. Detect SIFT keypoints and compute descriptors.
4. Visualize SIFT keypoints.
5. Extract HOG features.
6. Visualize the HOG representation.
7. Compare SIFT and HOG.
8. Perform basic image matching using two related images.
9. Discuss strengths and limitations.
10. Record observations and applications.

## Run in Google Colab

Open `Experiment_5_SIFT_HOG.ipynb` in Google Colab and run the cells from top to bottom.

The notebook uses the sample `astronaut` image from `scikit-image`, so no separate image upload is required.

## Technologies

- Python
- OpenCV
- scikit-image
- NumPy
- Matplotlib

## Expected outputs

The notebook generates:
- Original image
- Grayscale image
- Image with SIFT keypoints
- HOG visualization
- SIFT descriptor statistics
- HOG feature-vector statistics
- Feature matching visualization between two related images
