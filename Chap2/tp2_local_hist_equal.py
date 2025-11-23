"""
Local histogram equalization.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
N = 7
r = N // 2
padded = np.pad(img_array, r, mode='constant', constant_values=0)
equalized = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        local_hist = np.zeros(256, dtype=int)
        for u in range(-r, r+1):
            for v in range(-r, r+1):
                local_hist[padded[i+r+u, j+r+v]] += 1
        total_local = N * N
        hn_local = local_hist / total_local
        CDF_local = np.zeros(256)
        CDF_local[0] = hn_local[0]
        for k in range(1, 256):
            CDF_local[k] = CDF_local[k-1] + hn_local[k]
        equalized[i, j] = int(CDF_local[img_array[i, j]] * 255)
Image.fromarray(equalized).save('Chap2/outputs/output_tp2_local_hist_equal.png')