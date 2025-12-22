"""
2D convolution.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap3/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convolve(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape
    r = kH // 2
    # Padding with zeros
    padded = np.pad(img, r, mode='constant', constant_values=0)
    result = np.zeros((H, W), dtype=np.float32)
    
    # Manual convolution
    for i in range(H):
        for j in range(W):
            sum_val = 0
            for u in range(kH):
                for v in range(kW):
                    # Kernel is often flipped in convolution, but for symmetric or academic cases correlation is used.
                    # Here we use standard correlation-style convolution as taught in basic courses.
                    sum_val += padded[i + u, j + v] * kernel[u, v]
            result[i, j] = sum_val
    
    # Clip values to 0-255 range
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

# Load image
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)

# Define a kernel (e.g., sharpening kernel)
kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

result = convolve(img_array, kernel)

# Save result
output_path = os.path.join(output_dir, 'output_tp3_convolution_2d.png')
Image.fromarray(result).save(output_path)

# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Convolved Image (Sharpening)')
plt.axis('off')

plt.suptitle('TP3: General 2D Convolution')
plt.tight_layout()
plt.show()
