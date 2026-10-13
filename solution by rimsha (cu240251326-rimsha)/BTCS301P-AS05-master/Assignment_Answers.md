# Computer Vision Assignment – Experiment 5

## Feature Extraction and Image Analysis using SIFT and HOG Descriptors

### Aim
To implement Scale-Invariant Feature Transform (SIFT) and Histogram of Oriented Gradients (HOG) techniques for feature extraction and analyze their effectiveness in image representation, object recognition, and computer vision applications.

### 1. What is feature extraction, and why is it important in computer vision?
Feature extraction is the process of converting raw image data into meaningful features or descriptors. These features represent important properties such as edges, shapes, textures, and key points. It is important because it reduces complex image data and provides useful information for image matching, classification, tracking, and object recognition.

### 2. Explain the working principle of SIFT.
SIFT (Scale-Invariant Feature Transform) detects distinctive points in an image and creates descriptors for them. Its main stages are:
1. Scale-space construction and keypoint detection.
2. Removal of low-contrast and unstable keypoints.
3. Assignment of orientation to keypoints.
4. Generation of a descriptor based on local gradient information.
The resulting descriptors can be matched between images even when scale or rotation changes.

### 3. What are keypoints and feature descriptors in image analysis?
Keypoints are distinctive locations in an image, such as corners or highly textured regions, that can be detected reliably. A feature descriptor is a numerical representation of the local image region around a keypoint. It allows corresponding features to be compared or matched between images.

### 4. Explain HOG and its significance.
Histogram of Oriented Gradients (HOG) represents an image using the distribution of edge or gradient directions. The image is divided into small cells, gradient orientations are accumulated into histograms, and neighboring cells are normalized in blocks. HOG is significant because it captures shape and edge information effectively, especially for object and pedestrian detection.

### 5. Compare SIFT and HOG.
| Feature | SIFT | HOG |
|---|---|---|
| Representation | Local keypoints and descriptors | Dense gradient histograms |
| Scale robustness | High | Depends on image/object scale |
| Rotation robustness | High | Relatively limited |
| Computation | Generally more computationally demanding | Usually simpler for fixed-size images |
| Best suited for | Image matching, recognition, correspondence | Object/pedestrian detection |
| Output | Keypoints + descriptors | Feature vector |

### 6. Why is SIFT invariant to scale and rotation?
SIFT searches for keypoints across multiple image scales using scale-space. It also assigns a dominant orientation to each keypoint and constructs the descriptor relative to that orientation. Therefore, changes in image size and rotation have much less effect on the resulting descriptor.

### 7. Mention three real-world applications of HOG.
1. Pedestrian detection.
2. Vehicle and object detection.
3. Human activity or shape analysis.

### 8. Why is feature extraction performed before classification or object detection?
Raw images contain a large amount of pixel information. Feature extraction converts this information into a more compact and meaningful representation. This can reduce computational complexity and help a classifier or detector focus on useful visual patterns.

### 9. Advantages and limitations of handcrafted descriptors compared with deep learning features.
**Advantages:** They are interpretable, relatively simple, and can work well with limited training data. They can also be computationally practical for specific tasks.

**Limitations:** Their performance depends strongly on manually designed features and parameters. They may be less adaptable to complex variations than learned deep features. Deep learning methods can learn task-specific representations directly from data, but usually require more data and computational resources.

### 10. How do feature extraction techniques contribute to image matching, face recognition, and object detection?
Feature extraction provides compact visual representations that can be compared or used by recognition systems. SIFT descriptors can be matched between images to find corresponding regions, while gradient-based descriptors such as HOG can represent object shape for detection. In face recognition, extracted features can help distinguish facial patterns and compare faces.
