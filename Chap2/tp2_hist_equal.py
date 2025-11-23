"""
Gamma correction.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
gamma = 2.2
corrected = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        corrected[i, j] = int(255 * ((I / 255) ** (1 / gamma)))
Image.fromarray(corrected).save('Chap2/outputs/output_tp2_gamma_correction.png')