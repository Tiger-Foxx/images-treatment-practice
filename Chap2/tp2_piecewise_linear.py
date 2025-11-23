"""
Piecewise linear transformation.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
points = [(0, 0), (50, 25), (128, 128), (200, 230), (255, 255)]
LUT = np.zeros(256, dtype=np.uint8)
for k in range(256):
    for p in range(len(points)-1):
        if points[p][0] <= k <= points[p+1][0]:
            x1, y1 = points[p]
            x2, y2 = points[p+1]
            LUT[k] = int(y1 + (y2 - y1) * (k - x1) / (x2 - x1))
            break
transformed = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        transformed[i, j] = LUT[img_array[i, j]]
Image.fromarray(transformed).save('Chap2/outputs/output_tp2_piecewise_linear.png')