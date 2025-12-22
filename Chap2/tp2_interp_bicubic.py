"""
Manual bicubic interpolation.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap2/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def cubic_kernel(x):
    # Catmull-Rom cubic spline kernel
    x = abs(x)
    if x <= 1:
        return 1.5*x**3 - 2.5*x**2 + 1
    elif x < 2:
        return -0.5*x**3 + 2.5*x**2 - 4*x + 2
    return 0

# Load image as grayscale
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img).astype(np.float32)
H, W = img_array.shape

# Zoom factor
F = 2.0
new_H = int(H * F)
new_W = int(W * F)
zoomed = np.zeros((new_H, new_W), dtype=np.float32)

print(f"Applying Bicubic Interpolation x{F}. This might take a few seconds...")

# Manual bicubic interpolation
# For each pixel in output, find 16 neighbors in input
for i in range(new_H):
    for j in range(new_W):
        y = i / F
        x = j / F
        
        y_int = int(y)
        x_int = int(x)
        
        val = 0
        for m in range(-1, 3):
            for n in range(-1, 3):
                yy = y_int + m
                xx = x_int + n
                
                # Boundary check (clamping)
                yy_clamp = max(0, min(H - 1, yy))
                xx_clamp = max(0, min(W - 1, xx))
                
                weight = cubic_kernel(y - yy) * cubic_kernel(x - xx)
                val += img_array[yy_clamp, xx_clamp] * weight
        
        zoomed[i, j] = val

zoomed = np.clip(zoomed, 0, 255).astype(np.uint8)

# Save result
output_path = os.path.join(output_dir, 'output_tp2_interp_bicubic.png')
Image.fromarray(zoomed).save(output_path)

# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title(f'Original ({H}x{W})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(zoomed, cmap='gray')
plt.title(f'Bicubic Zoom x{F} ({new_H}x{new_W})')
plt.axis('off')

plt.suptitle('TP2: Bicubic Interpolation')
plt.tight_layout()
plt.show()
