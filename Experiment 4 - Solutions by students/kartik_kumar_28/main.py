import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load a grayscale image
img = cv2.imread('image.jpg', 0)

# 2. Compute the Discrete Fourier Transform (DFT)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# 3. Shift the zero-frequency component to the center
dft_shift = np.fft.fftshift(dft)

# 4. Compute magnitude spectrum for visualization
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

# Get image dimensions and center point
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask_size = 30 # Size of the filter

# 5. Design and apply a Low-Pass Filter (LPF) mask
mask_lpf = np.zeros((rows, cols, 2), np.uint8)
mask_lpf[crow - mask_size:crow + mask_size, ccol - mask_size:ccol + mask_size] = 1
fshift_lpf = dft_shift * mask_lpf

# 6. Design and apply a High-Pass Filter (HPF) mask
mask_hpf = np.ones((rows, cols, 2), np.uint8)
mask_hpf[crow - mask_size:crow + mask_size, ccol - mask_size:ccol + mask_size] = 0
fshift_hpf = dft_shift * mask_hpf

# 7. Perform Inverse Fourier Transform (IDFT) for both filters
# LPF Reconstruction
f_ishift_lpf = np.fft.ifftshift(fshift_lpf)
img_back_lpf = cv2.idft(f_ishift_lpf)
img_back_lpf = cv2.magnitude(img_back_lpf[:, :, 0], img_back_lpf[:, :, 1])

# HPF Reconstruction
f_ishift_hpf = np.fft.ifftshift(fshift_hpf)
img_back_hpf = cv2.idft(f_ishift_hpf)
img_back_hpf = cv2.magnitude(img_back_hpf[:, :, 0], img_back_hpf[:, :, 1])

# --- NEW ADDITIONS FOR SAVING IMAGES ---

# Normalize the images to 0-255 so they can be saved properly as JPG/PNG
mag_save = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
lpf_save = cv2.normalize(img_back_lpf, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hpf_save = cv2.normalize(img_back_hpf, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save the files to the current directory
cv2.imwrite('magnitude_spectrum.jpg', mag_save)
cv2.imwrite('low_pass_filtered.jpg', lpf_save)
cv2.imwrite('high_pass_filtered.jpg', hpf_save)
print("Images have been saved successfully!")

# 8. Compare the original image, spectrum, and filtered images using Matplotlib
plt.figure(figsize=(12, 10))

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.axis('off')

plt.subplot(223), plt.imshow(img_back_lpf, cmap='gray')
plt.title('Low-Pass Filtered (Smoothed)'), plt.axis('off')

plt.subplot(224), plt.imshow(img_back_hpf, cmap='gray')
plt.title('High-Pass Filtered (Edges)'), plt.axis('off')

plt.tight_layout()
plt.show()