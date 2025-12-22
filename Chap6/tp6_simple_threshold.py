"""
Simple global thresholding for image binarization.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap6/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image as grayscale
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Fixed threshold value
seuil = 127
thresh = np.zeros((H, W), dtype=np.uint8)

# Manual thresholding
for i in range(H):
    for j in range(W):
        if img_array[i, j] > seuil:
            thresh[i, j] = 255
        else:
            thresh[i, j] = 0

# Save result
output_path = os.path.join(output_dir, 'output_tp6_simple_threshold.png')
Image.fromarray(thresh).save(output_path)

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.hist(img_array.flatten(), bins=256, range=(0, 256), color='gray')
plt.axvline(x=seuil, color='red', linestyle='--', label=f'Threshold={seuil}')
plt.title('Histogram')
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title('Binarized Result')
plt.axis('off')

plt.suptitle('TP6: Simple Thresholding')
plt.tight_layout()
plt.show()
