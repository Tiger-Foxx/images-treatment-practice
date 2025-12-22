"""
Linear stretch to full range.
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

# Find min and max manually
min_val = 255
max_val = 0
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        if I < min_val: min_val = I
        if I > max_val: max_val = I

# Apply linear stretch
stretched = np.zeros((H, W), dtype=np.uint8)
diff = max_val - min_val if max_val > min_val else 1
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        stretched[i, j] = int(255 * (I - min_val) / diff)

# Save result
output_path = os.path.join(output_dir, 'output_tp2_linear_stretch.png')
Image.fromarray(stretched).save(output_path)

# Visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title(f'Original (min={min_val}, max={max_val})')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(stretched, cmap='gray')
plt.title('Stretched to [0, 255]')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(img_array.flatten(), bins=256, range=(0, 256), color='gray')
plt.title('Original Histogram')

plt.subplot(2, 2, 4)
plt.hist(stretched.flatten(), bins=256, range=(0, 256), color='blue')
plt.title('Stretched Histogram')

plt.suptitle('TP2: Linear Contrast Stretching')
plt.tight_layout()
plt.show()
