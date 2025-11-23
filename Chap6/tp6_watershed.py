"""
Watershed segmentation.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
gradient = np.zeros((H, W))
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
label = 1
for _, i, j in pixels:
    if labels[i, j] == 0:
        labels[i, j] = label
        label += 1
colored = np.zeros((H, W, 3), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        colored[i, j] = [ (labels[i, j] * 37) % 256, (labels[i, j] * 73) % 256, (labels[i, j] * 113) % 256 ]
Image.fromarray(colored).save('Chap6/outputs/output_tp6_watershed.png')