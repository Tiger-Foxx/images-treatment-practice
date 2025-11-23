"""
Bilinear interpolation.
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
        x = i / F
        y = j / F
        x1 = int(x)
        y1 = int(y)
        x2 = min(x1 + 1, H - 1)
        y2 = min(y1 + 1, W - 1)
        dx = x - x1
        dy = y - y1
        I00 = img_array[x1, y1]
        I10 = img_array[x2, y1]
        I01 = img_array[x1, y2]
        I11 = img_array[x2, y2]
        val = (1 - dx) * (1 - dy) * I00 + dx * (1 - dy) * I10 + (1 - dx) * dy * I01 + dx * dy * I11
        zoomed[i, j] = int(val)
Image.fromarray(zoomed).save('Chap2/outputs/output_tp2_interp_bilinear.png')