"""
Linear stretch to full range.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
min_val = 255
max_val = 0
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        if I < min_val:
            min_val = I
        if I > max_val:
            max_val = I
stretched = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        stretched[i, j] = int(255 * (I - min_val) / (max_val - min_val))
Image.fromarray(stretched).save('Chap2/outputs/output_tp2_linear_stretch.png')