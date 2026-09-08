# EXPERIMENT 6
# Implementation and Comparative Analysis of Image Segmentation
# Techniques using Python and OpenCV


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os



# STEP 1: LOAD IMAGE


# Get the folder where this Python file is located
script_folder = os.path.dirname(os.path.abspath(__file__))

# Image path
image_path = os.path.join(script_folder, "image.jpg")

print("========================================")
print("IMAGE SEGMENTATION - EXPERIMENT 6")
print("========================================")

print("\nWorking Folder:")
print(script_folder)

print("\nImage Path:")
print(image_path)


# Load image
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    raise FileNotFoundError(
        "\nERROR: image.jpg not found!\n"
        "Please put image.jpg in the same folder as main.py"
    )

print("\nImage loaded successfully!")



# STEP 2: GRAYSCALE AND GAUSSIAN BLUR


# Convert BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

print("\nGrayscale conversion completed.")
print("Gaussian Blur applied successfully.")



# STEP 3: GLOBAL THRESHOLDING


# Fixed threshold value
threshold_value = 127

_, global_threshold = cv2.threshold(
    blur,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

print("\nGlobal Thresholding completed.")
print("Threshold Value:", threshold_value)



# STEP 4: OTSU'S THRESHOLDING


# Otsu automatically calculates the optimal threshold
otsu_value, otsu_threshold = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("\nOtsu's Thresholding completed.")
print("Optimal Otsu Threshold Value:", otsu_value)



# STEP 5: ADAPTIVE THRESHOLDING


# Adaptive Gaussian Thresholding
adaptive_threshold = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

print("\nAdaptive Thresholding completed.")



# STEP 6: WATERSHED SEGMENTATION


# Convert image into binary image using Otsu
_, binary = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Create morphological kernel
kernel = np.ones((3, 3), np.uint8)


# Remove noise using Morphological Opening
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)


# Find sure background area
sure_background = cv2.dilate(
    opening,
    kernel,
    iterations=3
)


# Distance Transform
distance_transform = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)


# Find sure foreground area
_, sure_foreground = cv2.threshold(
    distance_transform,
    0.5 * distance_transform.max(),
    255,
    0
)

sure_foreground = np.uint8(sure_foreground)


# Find unknown region
unknown = cv2.subtract(
    sure_background,
    sure_foreground
)


# Marker labelling
num_labels, markers = cv2.connectedComponents(
    sure_foreground
)


# Add 1 to all marker labels
markers = markers + 1


# Mark unknown region as 0
markers[unknown == 255] = 0


# Apply Watershed Algorithm
watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)


# Mark boundaries with red
watershed_image[markers == -1] = [0, 0, 255]


# Convert BGR to RGB for Matplotlib
watershed_rgb = cv2.cvtColor(
    watershed_image,
    cv2.COLOR_BGR2RGB
)

print("\nWatershed Segmentation completed.")



# STEP 7: K-MEANS CLUSTERING


# Convert image from BGR to RGB
rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# Reshape image into a list of pixels
pixels = rgb_image.reshape((-1, 3))


# Convert pixel values to float32
pixels = np.float32(pixels)


# Number of clusters
k = 4


# Create K-Means model
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)


# Train K-Means model
kmeans.fit(pixels)


# Get cluster centers
centers = np.uint8(
    kmeans.cluster_centers_
)


# Get cluster labels
labels = kmeans.labels_


# Replace each pixel with its cluster center
segmented_pixels = centers[labels]


# Reshape pixels back to image shape
kmeans_segmented = segmented_pixels.reshape(
    rgb_image.shape
)

print("\nK-Means Clustering completed.")
print("Number of Clusters:", k)



# STEP 8: DISPLAY RESULTS


plt.figure(figsize=(16, 10))


# Original Image
plt.subplot(2, 4, 1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis("off")


# Grayscale Image
plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")


# Gaussian Blur
plt.subplot(2, 4, 3)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")


# Global Thresholding
plt.subplot(2, 4, 4)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")


# Otsu Thresholding
plt.subplot(2, 4, 5)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu Thresholding")
plt.axis("off")


# Adaptive Thresholding
plt.subplot(2, 4, 6)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")


# Watershed Segmentation
plt.subplot(2, 4, 7)
plt.imshow(watershed_rgb)
plt.title("Watershed Segmentation")
plt.axis("off")


# K-Means Segmentation
plt.subplot(2, 4, 8)
plt.imshow(kmeans_segmented)
plt.title("K-Means Segmentation")
plt.axis("off")


plt.tight_layout()
plt.show()



# STEP 9: COMPARISON


print("\n")
print("========================================")
print("COMPARISON OF SEGMENTATION TECHNIQUES")
print("========================================")

print("\n1. Global Thresholding")
print("   - Uses a fixed threshold value.")
print("   - Simple and fast.")
print("   - Works well when illumination is uniform.")


print("\n2. Otsu's Thresholding")
print("   - Automatically finds the optimal threshold.")
print("   - Useful for images with clear foreground and background.")
print("   - No manual threshold selection is required.")


print("\n3. Adaptive Thresholding")
print("   - Calculates threshold for different local regions.")
print("   - Works better with uneven illumination.")
print("   - Useful when lighting conditions vary.")


print("\n4. Watershed Segmentation")
print("   - Separates touching or overlapping objects.")
print("   - Detects object boundaries.")
print("   - Useful for separating individual objects.")


print("\n5. K-Means Clustering")
print("   - Groups pixels into different clusters.")
print("   - Useful for color-based segmentation.")
print("   - Number of clusters can be selected manually.")



# STEP 10: SAVE OUTPUT IMAGES


# Save Global Thresholding result
cv2.imwrite(
    os.path.join(script_folder, "global_threshold.jpg"),
    global_threshold
)


# Save Otsu result
cv2.imwrite(
    os.path.join(script_folder, "otsu_threshold.jpg"),
    otsu_threshold
)


# Save Adaptive Thresholding result
cv2.imwrite(
    os.path.join(script_folder, "adaptive_threshold.jpg"),
    adaptive_threshold
)


# Save Watershed result
cv2.imwrite(
    os.path.join(script_folder, "watershed_segmentation.jpg"),
    watershed_image
)


# Save K-Means result
kmeans_bgr = cv2.cvtColor(
    kmeans_segmented,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    os.path.join(script_folder, "kmeans_segmentation.jpg"),
    kmeans_bgr
)



# EXPERIMENT COMPLETED


print("\n========================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("========================================")

print("\nGenerated Files:")

print("1. global_threshold.jpg")
print("2. otsu_threshold.jpg")
print("3. adaptive_threshold.jpg")
print("4. watershed_segmentation.jpg")
print("5. kmeans_segmentation.jpg")

print("\nAll segmentation techniques were implemented successfully.")