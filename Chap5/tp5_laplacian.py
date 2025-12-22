"""
Laplacian operator for edge detection.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap5/outputs'
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


kernel = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])

lap = convolve(img_array, kernel)



lap_abs = np.abs(lap)
if lap_abs.max() > 0:
    lap_norm = (lap_abs / lap_abs.max() * 255).astype(np.uint8)
else:
    lap_norm = lap_abs.astype(np.uint8)


output_path = os.path.join(output_dir, 'output_tp5_laplacian.png')
Image.fromarray(lap_norm).save(output_path)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(lap_norm, cmap='gray')
plt.title('Laplacian (Absolute)')
plt.axis('off')

plt.suptitle('TP5: Laplacian Edge Detection')
plt.tight_layout()
plt.show()
