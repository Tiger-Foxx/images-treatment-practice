"""
Otsu thresholding.
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
best_T = 0
best_var = float('inf')
for T in range(1, 255):
    P1 = sum(hist[:T]) / total
    P2 = sum(hist[T:]) / total
    if P1 == 0 or P2 == 0:
        continue
    mu1 = sum(k * hist[k] for k in range(T)) / (P1 * total)
    mu2 = sum(k * hist[k] for k in range(T, 256)) / (P2 * total)
    var = P1 * P2 * (mu1 - mu2)**2
    if var < best_var:
        best_var = var
        best_T = T
thresh = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > best_T:
            thresh[i, j] = 255
Image.fromarray(thresh).save('Chap6/outputs/output_tp6_otsu_threshold.png')