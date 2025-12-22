"""
Roberts cross operator for edge detection.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap5/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convolve_2x2(img, kernel):
    # Specialized convolution for 2x2 Roberts kernels
    H, W = img.shape
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(H - 1):
        for j in range(W - 1):
            # Apply 2x2 kernel
            val = img[i, j] * kernel[0, 0] + img[i, j+1] * kernel[0, 1] + \
                  img[i+1, j] * kernel[1, 0] + img[i+1, j+1] * kernel[1, 1]
            result[i, j] = val
    return result

# Load image as grayscale
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img).astype(np.float32)
H, W = img_array.shape

# Roberts Kernels
dx_kernel = np.array([[1, 0], [0, -1]])
dy_kernel = np.array([[0, 1], [-1, 0]])

# Apply cross differences
dx = convolve_2x2(img_array, dx_kernel)
dy = convolve_2x2(img_array, dy_kernel)
mag = np.sqrt(dx**2 + dy**2)

# Normalization
mag_max = mag.max()
if mag_max > 0:
    mag_norm = (mag / mag_max * 255).astype(np.uint8)
else:
    mag_norm = mag.astype(np.uint8)

# Save result
output_path = os.path.join(output_dir, 'output_tp5_roberts.png')
Image.fromarray(mag_norm).save(output_path)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mag_norm, cmap='gray')
plt.title('Roberts Cross Magnitude')
plt.axis('off')

plt.suptitle('TP5: Roberts Operator')
plt.tight_layout()
plt.show()
