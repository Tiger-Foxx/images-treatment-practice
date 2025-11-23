"""
Nearest neighbor interpolation.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
F = 2
new_H = H * F
new_W = W * F
zoomed = np.zeros((new_H, new_W), dtype=np.uint8)
for i in range(new_H):
    for j in range(new_W):
        orig_i = i // F
        orig_j = j // F
        zoomed[i, j] = img_array[orig_i, orig_j]
Image.fromarray(zoomed).save('Chap2/outputs/output_tp2_interp_nearest.png')