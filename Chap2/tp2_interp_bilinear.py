"""
Bilinear interpolation.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap2/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Zoom factor
F = 2.5
new_H = int(H * F)
new_W = int(W * F)
zoomed = np.zeros((new_H, new_W), dtype=np.uint8)

# Manual bilinear interpolation
for i in range(new_H):
    for j in range(new_W):
        # Map back to original image space
        orig_y = i / F
        orig_x = j / F
        
        y1 = int(np.floor(orig_y))
        x1 = int(np.floor(orig_x))
        y2 = min(y1 + 1, H - 1)
        x2 = min(x1 + 1, W - 1)
        
        dy = orig_y - y1
        dx = orig_x - x1
        
        # 4 neighbor pixels
        I11 = img_array[y1, x1]
        I21 = img_array[y2, x1]
        I12 = img_array[y1, x2]
        I22 = img_array[y2, x2]
        
        # Interpolation formula
        weighted_val = (1 - dx) * (1 - dy) * I11 + \
                       (1 - dx) * dy * I21 + \
                       dx * (1 - dy) * I12 + \
                       dx * dy * I22
                       
        zoomed[i, j] = int(weighted_val)

# Save result
output_path = os.path.join(output_dir, 'output_tp2_interp_bilinear.png')
Image.fromarray(zoomed).save(output_path)

# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title(f'Original ({H}x{W})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(zoomed, cmap='gray')
plt.title(f'Zoomed Bilinear x{F} ({new_H}x{new_W})')
plt.axis('off')

plt.suptitle('TP2: Bilinear Interpolation')
plt.tight_layout()
plt.show()
