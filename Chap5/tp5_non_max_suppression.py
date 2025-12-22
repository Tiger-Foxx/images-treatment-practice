"""
Non-maximum suppression for edge thinning.
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
H, W = img_array.shape


kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


dx = convolve(img_array, kx)
dy = convolve(img_array, ky)
mag = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(ky, kx) 

suppressed = np.zeros((H, W), dtype=np.float32)
angle = np.arctan2(dy, dx) * 180.0 / np.pi
angle[angle < 0] += 180


for i in range(1, H-1):
    for j in range(1, W-1):
        q = 255
        r = 255
        
        
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
            q = mag[i, j+1]
            r = mag[i, j-1]
        
        elif (22.5 <= angle[i,j] < 67.5):
            q = mag[i+1, j-1]
            r = mag[i-1, j+1]
        
        elif (67.5 <= angle[i,j] < 112.5):
            q = mag[i+1, j]
            r = mag[i-1, j]
        
        elif (112.5 <= angle[i,j] < 157.5):
            q = mag[i-1, j-1]
            r = mag[i+1, j+1]

        if (mag[i,j] >= q) and (mag[i,j] >= r):
            suppressed[i,j] = mag[i,j]
        else:
            suppressed[i,j] = 0


if suppressed.max() > 0:
    suppressed_norm = (suppressed / suppressed.max() * 255).astype(np.uint8)
else:
    suppressed_norm = suppressed.astype(np.uint8)


output_path = os.path.join(output_dir, 'output_tp5_non_max_suppression.png')
Image.fromarray(suppressed_norm).save(output_path)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mag, cmap='magma')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(suppressed_norm, cmap='gray')
plt.title('Non-Max Suppression (Thinned)')
plt.axis('off')

plt.suptitle('TP5: Non-Maximum Suppression')
plt.tight_layout()
plt.show()
