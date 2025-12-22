"""
Sobel operator for edge detection.
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



kx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

ky = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

gx = convolve(img_array, kx)
gy = convolve(img_array, ky)


mag = np.sqrt(gx**2 + gy**2)

if mag.max() > 0:
    mag_norm = (mag / mag.max() * 255).astype(np.uint8)
else:
    mag_norm = mag.astype(np.uint8)


output_path = os.path.join(output_dir, 'output_tp5_sobel.png')
Image.fromarray(mag_norm).save(output_path)


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mag_norm, cmap='gray')
plt.title('Sobel Magnitude')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(np.abs(gx), cmap='gray')
plt.title('Gradient Gx (Horizontal)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.abs(gy), cmap='gray')
plt.title('Gradient Gy (Vertical)')
plt.axis('off')

plt.suptitle('TP5: Sobel Edge Detection')
plt.tight_layout()
plt.show()
