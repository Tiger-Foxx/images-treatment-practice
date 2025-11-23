"""
Median filter.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img2.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape
N = 3
r = N // 2
padded = np.pad(img_array, r, mode='constant', constant_values=0)
result = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        neighbors = []
        for u in range(-r, r+1):
            for v in range(-r, r+1):
                neighbors.append(padded[i + r + u, j + r + v])
        neighbors.sort()
        result[i, j] = neighbors[len(neighbors) // 2]
Image.fromarray(result).save('Chap3/outputs/output_tp3_median_filter.png')