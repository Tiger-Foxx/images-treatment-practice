"""
Gradient computation using central finite differences.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap5/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img).astype(np.float32)
H, W = img_array.shape

dx = np.zeros((H, W), dtype=np.float32)
dy = np.zeros((H, W), dtype=np.float32)


for i in range(1, H-1):
    for j in range(1, W-1):
        
        dx[i, j] = img_array[i+1, j] - img_array[i-1, j]
        
        dy[i, j] = img_array[i, j+1] - img_array[i, j-1]

mag = np.sqrt(dx**2 + dy**2)


mag_max = mag.max()
if mag_max > 0:
    mag_norm = (mag / mag_max * 255).astype(np.uint8)
else:
    mag_norm = mag.astype(np.uint8)


output_path = os.path.join(output_dir, 'output_tp5_gradient_finite_diff.png')
Image.fromarray(mag_norm).save(output_path)


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mag_norm, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(dx, cmap='RdBu')
plt.title('V-Gradient (dx)')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(dy, cmap='RdBu')
plt.title('H-Gradient (dy)')
plt.axis('off')
plt.colorbar()

plt.suptitle('TP5: Gradient by Finite Differences')
plt.tight_layout()
plt.show()
