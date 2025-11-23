"""
Quantize grayscale image to K bits.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
K = 4  # example, test with 2,4,6
L = 2 ** K
quant = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        I_prime = int((I / 255.0) * (L - 1)) * (255 // (L - 1))
        quant[i, j] = I_prime
Image.fromarray(quant).save('Chap1/outputs/output_tp1_gray_quantization.png')