"""
Nearest neighbor interpolation for image zooming.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap2/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape


F = 2.0
new_H = int(H * F)
new_W = int(W * F)
zoomed = np.zeros((new_H, new_W), dtype=np.uint8)


for i in range(new_H):
    for j in range(new_W):
        
        orig_i = int(i / F)
        orig_j = int(j / F)
        
        
        orig_i = min(orig_i, H - 1)
        orig_j = min(orig_j, W - 1)
        
        zoomed[i, j] = img_array[orig_i, orig_j]


output_path = os.path.join(output_dir, 'output_tp2_interp_nearest.png')
Image.fromarray(zoomed).save(output_path)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title(f'Original ({H}x{W})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(zoomed, cmap='gray')
plt.title(f'Nearest Zoom x{F} ({new_H}x{new_W})')
plt.axis('off')

plt.suptitle('TP2: Nearest Neighbor Interpolation')
plt.tight_layout()
plt.show()
