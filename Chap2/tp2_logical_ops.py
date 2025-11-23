"""
Logical operations: AND, OR.
"""
import numpy as np
from PIL import Image

img1 = Image.open('inputs/img1.png').convert('L')
img2 = Image.open('inputs/img2.png').convert('L')
img1_array = np.array(img1)
img2_array = np.array(img2)
H, W = img1_array.shape
bin1 = np.zeros((H, W), dtype=np.uint8)
bin2 = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        bin1[i, j] = 255 if img1_array[i, j] > 128 else 0
        bin2[i, j] = 255 if img2_array[i, j] > 128 else 0
and_result = np.zeros((H, W), dtype=np.uint8)
or_result = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        and_result[i, j] = 255 if bin1[i, j] == 255 and bin2[i, j] == 255 else 0
        or_result[i, j] = 255 if bin1[i, j] == 255 or bin2[i, j] == 255 else 0
Image.fromarray(and_result).save('Chap2/outputs/output_tp2_logical_and.png')
Image.fromarray(or_result).save('Chap2/outputs/output_tp2_logical_or.png')