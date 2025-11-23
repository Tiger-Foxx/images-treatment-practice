"""
Morphological dilation.
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
dilated = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        any_one = False
        for u in range(-r, r+1):
            for v in range(-r, r+1):
                if padded[i + r + u, j + r + v] == 1:
                    any_one = True
                    break
            if any_one:
                break
        if any_one:
            dilated[i, j] = 1
Image.fromarray((dilated * 255).astype(np.uint8)).save('Chap7/outputs/output_tp7_morph_dilation.png')