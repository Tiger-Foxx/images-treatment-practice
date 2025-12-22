"""
Morphological erosion.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap7/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image and binarize
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Manual binarization
binary = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > 127:
            binary[i, j] = 1
        else:
            binary[i, j] = 0

# Structuring element size
N = 3
r = N // 2
padded = np.pad(binary, r, mode='constant', constant_values=1) # Pad with 1 for erosion
eroded = np.zeros((H, W), dtype=np.uint8)

# Manual erosion
for i in range(H):
    for j in range(W):
        match = True
        for u in range(N):
            for v in range(N):
                if padded[i + u, j + v] == 0:
                    match = False
                    break
            if not match:
                break
        if match:
            eroded[i, j] = 1

# Save result
output_path = os.path.join(output_dir, 'output_tp7_morph_erosion.png')
Image.fromarray((eroded * 255).astype(np.uint8)).save(output_path)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(eroded, cmap='gray')
plt.title(f'Erosion ({N}x{N})')
plt.axis('off')

plt.suptitle('TP7: Morphological Erosion')
plt.tight_layout()
plt.show()
