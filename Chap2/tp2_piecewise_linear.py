"""
Piecewise linear transformation for custom contrast control.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap2/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image as grayscale
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Define control points for piecewise linear (x, y)
points = [(0, 0), (50, 25), (128, 128), (200, 230), (255, 255)]

# Pre-calculate LUT
LUT = np.zeros(256, dtype=np.uint8)
for k in range(256):
    for p in range(len(points)-1):
        if points[p][0] <= k <= points[p+1][0]:
            x1, y1 = points[p]
            x2, y2 = points[p+1]
            if x2 != x1:
                LUT[k] = int(y1 + (y2 - y1) * (k - x1) / (x2 - x1))
            else:
                LUT[k] = y1
            break

# Manual application
transformed = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        transformed[i, j] = LUT[img_array[i, j]]

# Save result
output_path = os.path.join(output_dir, 'output_tp2_piecewise_linear.png')
Image.fromarray(transformed).save(output_path)

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(transformed, cmap='gray')
plt.title('Piecewise Linear Result')
plt.axis('off')

plt.subplot(2, 2, 3)
px, py = zip(*points)
plt.plot(range(256), LUT, color='red', label='LUT')
plt.scatter(px, py, color='blue', label='Control Points')
plt.title('Transformation Function')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.hist(img_array.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.5, label='Original')
plt.hist(transformed.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.5, label='Processed')
plt.title('Histogram Comparison')
plt.legend()

plt.suptitle('TP2: Piecewise Linear Transformation')
plt.tight_layout()
plt.show()
