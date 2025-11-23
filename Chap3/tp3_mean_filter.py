"""
Mean filter.
"""
import numpy as np
from PIL import Image

def convolve(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape
    r = kH // 2
    padded = np.pad(img, r, mode='constant', constant_values=0)
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            sum_val = 0
            for u in range(kH):
                for v in range(kW):
                    sum_val += padded[i + u, j + v] * kernel[u, v]
            result[i, j] = sum_val
    return result.astype(np.uint8)

img = Image.open('inputs/img1.png')
img_array = np.array(img)
N = 3
kernel = np.ones((N, N)) / (N * N)
result = convolve(img_array, kernel)
Image.fromarray(result).save('Chap3/outputs/output_tp3_mean_filter.png')