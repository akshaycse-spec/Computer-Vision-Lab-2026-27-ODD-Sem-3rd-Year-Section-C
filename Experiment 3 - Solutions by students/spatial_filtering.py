import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# EXPERIMENT 3: SPATIAL FILTERING
# -------------------------------------------------

# Find the folder where this Python file is located
folder = os.path.dirname(os.path.abspath(__file__))

# Image path
image_path = os.path.join(folder, "images", "input.jpg")

# Load image
image = cv2.imread(image_path)

# Check whether image was loaded
if image is None:
    print("ERROR: Image could not be loaded!")
    print("Looking for image at:")
    print(image_path)
    exit()

print("Image loaded successfully!")
print("Image path:", image_path)

# -------------------------------------------------
# Convert image to grayscale
# -------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------
# LOW-PASS FILTERS
# -------------------------------------------------

# 1. Gaussian Filter
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Median Filter
median = cv2.medianBlur(gray, 5)

# 3. Average / Mean Filter
average = cv2.blur(gray, (5, 5))

# -------------------------------------------------
# HIGH-PASS FILTERS
# -------------------------------------------------

# 4. Laplacian Filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# 5. Sobel X
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# 6. Sobel Y
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

plt.figure(figsize=(14, 8))

plt.subplot(2, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gaussian, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(median, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(average, cmap="gray")
plt.title("Average Filter")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Filter")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.tight_layout()

# Show all results
plt.show()