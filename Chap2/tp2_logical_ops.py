"""
Logical operations: AND, OR on binary images.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap2/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load images as grayscale
img1 = Image.open('inputs/img1.png').convert('L')
img2 = Image.open('inputs/img2.png').convert('L')

# Ensure they have the same size
if img1.size != img2.size:
    img2 = img2.resize(img1.size)

img1_array = np.array(img1)
img2_array = np.array(img2)
H, W = img1_array.shape

# Thresholding to create binary images
bin1 = np.zeros((H, W), dtype=np.uint8)
bin2 = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        bin1[i, j] = 255 if img1_array[i, j] > 127 else 0
        bin2[i, j] = 255 if img2_array[i, j] > 127 else 0

# Manual Logical operations
and_result = np.zeros((H, W), dtype=np.uint8)
or_result = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        and_result[i, j] = 255 if bin1[i, j] == 255 and bin2[i, j] == 255 else 0
        or_result[i, j] = 255 if bin1[i, j] == 255 or bin2[i, j] == 255 else 0

# Save results
Image.fromarray(and_result).save(os.path.join(output_dir, 'output_tp2_logical_and.png'))
Image.fromarray(or_result).save(os.path.join(output_dir, 'output_tp2_logical_or.png'))

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img1_array, cmap='gray')
plt.title('Original 1')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img2_array, cmap='gray')
plt.title('Original 2')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(bin1, cmap='gray')
plt.title('Binary 1')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(and_result, cmap='gray')
plt.title('Logical AND')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(or_result, cmap='gray')
plt.title('Logical OR')
plt.axis('off')

plt.suptitle('TP2: Logical Operations (AND/OR)')
plt.tight_layout()
plt.show()
