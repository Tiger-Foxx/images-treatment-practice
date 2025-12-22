"""
Gaussian smoothing filter for noise reduction.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap3/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convolve(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape
    r = kH // 2
    padded = np.pad(img, r, mode='reflect')
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            sum_val = 0
            for u in range(kH):
                for v in range(kW):
                    sum_val += padded[i + u, j + v] * kernel[u, v]
            result[i, j] = sum_val
    return result


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img).astype(np.float32)


sigma = 1.0
N = 5
r = N // 2
kernel = np.zeros((N, N))
for u in range(N):
    for v in range(N):
        kernel[u, v] = (1 / (2 * np.pi * sigma**2)) * np.exp(- ((u - r)**2 + (v - r)**2) / (2 * sigma**2))
kernel /= kernel.sum()


result = convolve(img_array, kernel)
result_uint8 = np.clip(result, 0, 255).astype(np.uint8)


output_path = os.path.join(output_dir, 'output_tp3_gaussian_filter.png')
Image.fromarray(result_uint8).save(output_path)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result_uint8, cmap='gray')
plt.title(f'Gaussian Filter (sigma={sigma}, N={N})')
plt.axis('off')

plt.suptitle('TP3: Gaussian Filtering')
plt.tight_layout()
plt.show()
