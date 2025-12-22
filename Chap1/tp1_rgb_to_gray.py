"""
Convert RGB image to grayscale using weighted average.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap1/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image
img = Image.open('inputs/img2.png')
img_array = np.array(img)
H, W, _ = img_array.shape

# Manual conversion
gray = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        r, g, b = img_array[i, j]
        # Standard weighted average for grayscale conversion
        gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)

# Save result
output_path = os.path.join(output_dir, 'output_tp1_rgb_to_gray.png')
Image.fromarray(gray).save(output_path)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title('Original (RGB)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Result (Manual)')
plt.axis('off')

plt.suptitle('TP1: RGB to Grayscale Conversion')
plt.tight_layout()
plt.show()
