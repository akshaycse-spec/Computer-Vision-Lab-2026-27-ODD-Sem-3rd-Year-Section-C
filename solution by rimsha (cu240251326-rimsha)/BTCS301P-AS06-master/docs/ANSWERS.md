# Assignment Questions and Answers

## 1. What is image segmentation and why is it fundamental?

Image segmentation assigns pixels to meaningful regions or objects. It is fundamental because it converts raw pixels into spatial areas that later tasks can measure, track, classify, or recognize.

## 2. Segmentation vs classification

Classification gives one or more labels to an entire image, such as "car." Segmentation gives a label to each pixel, showing exactly where the car is and describing its shape.

## 3. Global, Otsu, and adaptive thresholding

Global thresholding compares every grayscale pixel with one manually chosen value. Otsu also uses one threshold, but selects it automatically by minimizing within-class variance. Adaptive thresholding computes a local threshold for each neighborhood, making it useful under uneven illumination.

## 4. Watershed algorithm

Watershed treats a grayscale or distance image like terrain. Starting from foreground and background markers, regions expand until they meet at watershed boundaries. Separate markers inside touching objects let the algorithm split them.

## 5. K-means for segmentation

Each pixel is represented by its color values and assigned to the nearest of K cluster centers. Centers and assignments are updated repeatedly. Pixels in the same cluster form a color-based region, producing a quantized segmented image.

## 6. Threshold-based vs clustering-based methods

Thresholding is simple, fast, and mainly separates pixels by intensity, often into two groups. Clustering can use multiple color channels and create several regions, but needs a value for K, is slower, and does not directly understand object meaning.

## 7. Challenges with complex backgrounds or lighting

Common problems are shadows, highlights, low contrast, sensor noise, textured backgrounds, similar foreground/background colors, occlusion, scale variation, and unclear object boundaries. Preprocessing or learned semantic features may be required.

## 8. Five applications

1. Tumor and organ measurement in medical scans.
2. Road, vehicle, and pedestrian understanding for autonomous driving.
3. Land-use and crop mapping in satellite images.
4. Defect detection in industrial inspection.
5. Foreground and background separation in photo editing.

## 9. Benefit to detection and recognition

Segmentation removes irrelevant background, provides exact object boundaries, separates instances, and supplies shape and area information. Detectors and recognizers can therefore focus on cleaner regions and reduce false matches.

## 10. Traditional vs deep learning segmentation

Traditional methods are fast, interpretable, inexpensive, and do not require labelled training data, but are sensitive to hand-chosen parameters and difficult scenes. U-Net learns detailed semantic masks and works well in areas such as medical imaging. Mask R-CNN detects individual objects and predicts a separate mask for each instance. These models handle complex appearance better, but need labelled data, greater computation, longer training, and careful validation.

