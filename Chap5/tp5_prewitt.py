"""
Prewitt operator for edge detection.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
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

# Load image as grayscale
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img).astype(np.float32)

# Prewitt Kernels
kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) # Horizontal gradient
ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) # Vertical gradient

# Apply convolution
dx = convolve(img_array, kx)
dy = convolve(img_array, ky)
mag = np.sqrt(dx**2 + dy**2)

# Normalization
mag_max = mag.max()
if mag_max > 0:
    mag_norm = (mag / mag_max * 255).astype(np.uint8)
else:
    mag_norm = mag.astype(np.uint8)

# Save result
output_path = os.path.join(output_dir, 'output_tp5_prewitt.png')
Image.fromarray(mag_norm).save(output_path)

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mag_norm, cmap='gray')
plt.title('Prewitt Magnitude')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(dx, cmap='PRGn')
plt.title('H-Gradient (dx)')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(dy, cmap='PRGn')
plt.title('V-Gradient (dy)')
plt.axis('off')
plt.colorbar()

plt.suptitle('TP5: Prewitt Operator')
plt.tight_layout()
plt.show()
