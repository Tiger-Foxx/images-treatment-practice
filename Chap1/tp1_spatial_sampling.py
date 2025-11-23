"""
Spatial sampling by factor N.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
N = 2
new_H = H // N
new_W = W // N
sampled = np.zeros((new_H, new_W), dtype=np.uint8)
for i in range(new_H):
    for j in range(new_W):
        sampled[i, j] = img_array[i * N, j * N]
Image.fromarray(sampled).save('Chap1/outputs/output_tp1_spatial_sampling.png')