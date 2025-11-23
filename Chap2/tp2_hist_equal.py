"""
Global histogram equalization.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
hist = np.zeros(256, dtype=int)
for i in range(H):
    for j in range(W):
        hist[img_array[i, j]] += 1
total = H * W
hn = hist / total
CDF = np.zeros(256)
CDF[0] = hn[0]
for k in range(1, 256):
    CDF[k] = CDF[k-1] + hn[k]
equalized = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        equalized[i, j] = int(CDF[img_array[i, j]] * 255)
Image.fromarray(equalized).save('Chap2/outputs/output_tp2_hist_equal.png')