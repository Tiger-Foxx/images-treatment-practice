"""
Non-max suppression.
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
H, W = img_array.shape
kx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
ky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
dx = convolve(img_array, kx)
dy = convolve(img_array, ky)
mag = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(dy, dx)
suppressed = np.zeros((H, W), dtype=np.float32)
for i in range(1, H-1):
    for j in range(1, W-1):
        angle = theta[i, j] * 180 / np.pi
        if (angle < 22.5 and angle >= -22.5) or (angle >= 157.5 or angle < -157.5):
            if mag[i, j] >= mag[i, j-1] and mag[i, j] >= mag[i, j+1]:
                suppressed[i, j] = mag[i, j]
        elif (angle >= 22.5 and angle < 67.5) or (angle < -112.5 and angle >= -157.5):
            if mag[i, j] >= mag[i-1, j+1] and mag[i, j] >= mag[i+1, j-1]:
                suppressed[i, j] = mag[i, j]
        elif (angle >= 67.5 and angle < 112.5) or (angle < -67.5 and angle >= -112.5):
            if mag[i, j] >= mag[i-1, j] and mag[i, j] >= mag[i+1, j]:
                suppressed[i, j] = mag[i, j]
        else:
            if mag[i, j] >= mag[i-1, j-1] and mag[i, j] >= mag[i+1, j+1]:
                suppressed[i, j] = mag[i, j]
suppressed_norm = (suppressed / suppressed.max() * 255).astype(np.uint8)
Image.fromarray(suppressed_norm).save('Chap5/outputs/output_tp5_non_max_suppression.png')