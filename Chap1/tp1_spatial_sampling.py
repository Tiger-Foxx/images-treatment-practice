"""
Spatial sampling by factor N.
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
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Sampling factor
N = 4
new_H, new_W = H // N, W // N
sampled = np.zeros((new_H, new_W), dtype=np.uint8)

# Manual sampling (downsampling without interpolation)
for i in range(new_H):
    for j in range(new_W):
        sampled[i, j] = img_array[i * N, j * N]

# Save result
output_path = os.path.join(output_dir, 'output_tp1_spatial_sampling.png')
Image.fromarray(sampled).save(output_path)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title(f'Original ({H}x{W})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sampled, cmap='gray')
plt.title(f'Sampled x{N} ({new_H}x{new_W})')
plt.axis('off')

plt.suptitle('TP1: Spatial Sampling')
plt.tight_layout()
plt.show()
