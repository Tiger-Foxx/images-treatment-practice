"""
Compute statistics: mean, std, min, max.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
sum_I = 0
count = H * W
min_val = 255
max_val = 0
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        sum_I += I
        if I < min_val:
            min_val = I
        if I > max_val:
            max_val = I
mean = sum_I / count
sum_var = 0
for i in range(H):
    for j in range(W):
        sum_var += (img_array[i, j] - mean) ** 2
std = (sum_var / count) ** 0.5
print(f"Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}")
# Save original as output
img.save('Chap2/outputs/output_tp2_stats.png')