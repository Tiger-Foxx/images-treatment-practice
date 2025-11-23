"""
Morphological erosion.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
binary = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > 128:
            binary[i, j] = 1
N = 3
r = N // 2
padded = np.pad(binary, r, mode='constant', constant_values=0)
eroded = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        all_one = True
        for u in range(-r, r+1):
            for v in range(-r, r+1):
                if padded[i + r + u, j + r + v] == 0:
                    all_one = False
                    break
            if not all_one:
                break
        if all_one:
            eroded[i, j] = 1
Image.fromarray((eroded * 255).astype(np.uint8)).save('Chap7/outputs/output_tp7_morph_erosion.png')