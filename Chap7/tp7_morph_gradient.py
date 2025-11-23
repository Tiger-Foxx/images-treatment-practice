"""
Morphological gradient.
"""
import numpy as np
from PIL import Image

def erosion(binary, N=3):
    H, W = binary.shape
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
    return eroded

def dilation(binary, N=3):
    H, W = binary.shape
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
    return dilated

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
binary = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > 128:
            binary[i, j] = 1
dil = dilation(binary)
ero = erosion(binary)
gradient = dil.astype(int) - ero.astype(int)
gradient = np.clip(gradient, 0, 255).astype(np.uint8)
Image.fromarray(gradient).save('Chap7/outputs/output_tp7_morph_gradient.png')