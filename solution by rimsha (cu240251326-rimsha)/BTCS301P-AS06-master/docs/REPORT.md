# Experiment Report

## Aim

To implement global, Otsu, adaptive, watershed, and K-means segmentation and compare their boundaries, accuracy, and execution time.

## Procedure

1. Load a color image and create a Gaussian-blurred grayscale copy.
2. Apply the three thresholding methods.
3. Use morphology, distance transform, connected components, and watershed markers to separate touching objects.
4. Cluster BGR pixel values with K-means for color segmentation.
5. Save every result and record runtime. When a true mask is supplied, calculate pixel accuracy, IoU, Dice, and boundary F1.

## Comparison

| Method | Main strength | Main limitation | Relative speed |
|---|---|---|---|
| Global threshold | Simple and predictable | Fixed threshold fails under lighting changes | Very fast |
| Otsu | Automatically selects a global threshold | Assumes a roughly bimodal histogram | Very fast |
| Adaptive | Handles local illumination changes | Can amplify texture and noise | Fast |
| Watershed | Separates touching objects and gives boundaries | Marker quality controls the result | Medium |
| K-means | Uses color and supports several regions | K must be selected; iterative | Slowest |

## Observations

- Gaussian blur reduces small noisy regions before thresholding.
- Global thresholding works best when foreground and background intensities are stable.
- Otsu is a stronger default when the histogram has two distinct groups.
- Adaptive thresholding helps with shadows and brightness gradients but may produce fragmented regions.
- Watershed produces meaningful separating boundaries for overlapping or touching objects.
- K-means preserves several color regions and is useful when grayscale intensity alone is insufficient.
- Accuracy can be inflated by a large background, so IoU, Dice, boundary F1, and runtime should be examined together.

## Result

No technique is universally best. Otsu is a practical fast baseline, adaptive thresholding is suitable for uneven lighting, watershed is best for separating touching objects, and K-means is valuable for multi-color scenes. Deep learning is preferable when objects have complex semantic shapes and enough labelled training data is available.

