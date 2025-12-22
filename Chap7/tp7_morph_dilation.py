"""
Morphological dilation for expanding labeled regions.
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

# Structuring element size (NxN)
N = 3
r = N // 2
padded = np.pad(binary, r, mode='constant', constant_values=0)
dilated = np.zeros((H, W), dtype=np.uint8)

print("Applying manual dilation...")

# Manual dilation
for i in range(H):
    for j in range(W):
        any_match = False
        for u in range(N):
            for v in range(N):
                if padded[i + u, j + v] == 1:
                    any_match = True
                    break
            if any_match: break
        
        if any_match:
            dilated[i, j] = 1

# Save result
output_path = os.path.join(output_dir, 'output_tp7_morph_dilation.png')
Image.fromarray((dilated * 255).astype(np.uint8)).save(output_path)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(dilated, cmap='gray')
plt.title(f'Dilation Result ({N}x{N})')
plt.axis('off')

plt.suptitle('TP7: Morphological Dilation')
plt.tight_layout()
plt.show()
