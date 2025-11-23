"""
Arithmetic operations: add, sub, mult.
"""
import numpy as np
from PIL import Image

img1 = Image.open('inputs/img1.png').convert('L')
img2 = Image.open('inputs/img2.png').convert('L')
img1_array = np.array(img1)
img2_array = np.array(img2)
H, W = img1_array.shape
add_result = np.zeros((H, W), dtype=np.uint8)
sub_result = np.zeros((H, W), dtype=np.uint8)
mult_result = np.zeros((H, W), dtype=np.uint8)
ratio = 0.5
for i in range(H):
    for j in range(W):
        add_result[i, j] = min(img1_array[i, j] + img2_array[i, j], 255)
        sub_result[i, j] = max(img1_array[i, j] - img2_array[i, j], 0)
        mult_result[i, j] = min(int(img1_array[i, j] * ratio), 255)
Image.fromarray(add_result).save('Chap2/outputs/output_tp2_arithmetic_add.png')
Image.fromarray(sub_result).save('Chap2/outputs/output_tp2_arithmetic_sub.png')
Image.fromarray(mult_result).save('Chap2/outputs/output_tp2_arithmetic_mult.png')