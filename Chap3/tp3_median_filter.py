"""
Median filter.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap3/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image (img2 is recommended for median filter in notes)
img = Image.open('inputs/img2.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Parameters
N = 3
r = N // 2
padded = np.pad(img_array, r, mode='reflect')
result = np.zeros((H, W), dtype=np.uint8)

print(f"Applying Median Filter {N}x{N}...")

# Manual median filter
for i in range(H):
    for j in range(W):
        neighbors = []
        for u in range(N):
            for v in range(N):
                neighbors.append(padded[i + u, j + v])
        neighbors.sort()
        result[i, j] = neighbors[len(neighbors) // 2]

# Save result
output_path = os.path.join(output_dir, 'output_tp3_median_filter.png')
Image.fromarray(result).save(output_path)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title(f'Median Filter ({N}x{N})')
plt.axis('off')

plt.suptitle('TP3: Median Filtering')
plt.tight_layout()
plt.show()
