"""
Sobel operator.
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
kx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
ky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
dx = convolve(img_array, kx)
dy = convolve(img_array, ky)
mag = np.sqrt(dx**2 + dy**2)
mag_norm = (mag / mag.max() * 255).astype(np.uint8)
Image.fromarray(mag_norm).save('Chap5/outputs/output_tp5_sobel.png')