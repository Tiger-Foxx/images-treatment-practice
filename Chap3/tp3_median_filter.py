"""
Median filter.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap3/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img2.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape


N = 3
r = N // 2
padded = np.pad(img_array, r, mode='reflect')
result = np.zeros((H, W), dtype=np.uint8)

print(f"Applying Median Filter {N}x{N}...")


for i in range(H):
    for j in range(W):
        neighbors = []
        for u in range(N):
            for v in range(N):
                neighbors.append(padded[i + u, j + v])
        neighbors.sort()
        result[i, j] = neighbors[len(neighbors) // 2]


output_path = os.path.join(output_dir, 'output_tp3_median_filter.png')
Image.fromarray(result).save(output_path)


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
