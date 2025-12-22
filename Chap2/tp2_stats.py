"""
Compute statistics: mean, std, min, max.
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
img_array = np.array(img).astype(np.float32)
H, W = img_array.shape

# Manual calculations
sum_I = 0
count = H * W
min_val = 255
max_val = 0
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        sum_I += I
        if I < min_val:
            min_val = int(I)
        if I > max_val:
            max_val = int(I)

mean_val = sum_I / count

sum_var = 0
for i in range(H):
    for j in range(W):
        sum_var += (img_array[i, j] - mean_val) ** 2
std_val = (sum_var / count) ** 0.5

# Visualization
plt.figure(figsize=(8, 6))
plt.imshow(img_array, cmap='gray')
stats_text = f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val}\nMax: {max_val}"
plt.text(10, 25, stats_text, color='red', fontsize=12, fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.7))
plt.title('Image Statistics')
plt.axis('off')

# Save result
output_path = os.path.join(output_dir, 'output_tp2_stats.png')
plt.savefig(output_path)
plt.show()

print(stats_text)
