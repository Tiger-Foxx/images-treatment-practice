"""
Simple thresholding.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
seuil = 128
thresh = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > seuil:
            thresh[i, j] = 255
Image.fromarray(thresh).save('Chap6/outputs/output_tp6_simple_threshold.png')