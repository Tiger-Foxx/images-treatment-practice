"""
Laplacian operator.
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
    return result

img = Image.open('inputs/img1.png')
img_array = np.array(img)
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
lap = convolve(img_array, kernel)
lap_abs = np.abs(lap)
lap_norm = (lap_abs / lap_abs.max() * 255).astype(np.uint8)
Image.fromarray(lap_norm).save('Chap5/outputs/output_tp5_laplacian.png')