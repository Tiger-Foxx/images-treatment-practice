"""
Simplified Watershed segmentation based on image gradient.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap6/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


input_image = 'inputs/img1.png'
if not os.path.exists(input_image):
    input_image = 'inputs/img2.png'

img = Image.open(input_image).convert('L')
img_array = np.array(img).astype(np.float32)
H, W = img_array.shape


gradient = np.zeros((H, W), dtype=np.float32)
for i in range(1, H-1):
    for j in range(1, W-1):
        dx = img_array[i+1, j] - img_array[i-1, j]
        dy = img_array[i, j+1] - img_array[i, j-1]
        gradient[i, j] = np.sqrt(dx**2 + dy**2)


pixels = []
for i in range(H):
    for j in range(W):
        pixels.append((gradient[i, j], i, j))
pixels.sort()

labels = np.zeros((H, W), dtype=int)
next_label = 1

print("Applying simplified Watershed... this may take a moment.")

for _, i, j in pixels:
    
    found_label = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and labels[ni, nj] > 0:
                found_label = labels[ni, nj]
                break
        if found_label > 0: break
    
    if found_label > 0:
        labels[i, j] = found_label
    else:
        labels[i, j] = next_label
        next_label += 1


colored = np.zeros((H, W, 3), dtype=np.uint8)
np.random.seed(42)
colors = np.random.randint(0, 255, size=(next_label + 1, 3), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        colored[i, j] = colors[labels[i, j]]


output_path = os.path.join(output_dir, 'output_tp6_watershed.png')
Image.fromarray(colored).save(output_path)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gradient, cmap='nipy_spectral')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(colored)
plt.title(f'Watershed (Regions: {next_label-1})')
plt.axis('off')

plt.suptitle('TP6: Simplified Watershed Segmentation')
plt.tight_layout()
plt.show()
