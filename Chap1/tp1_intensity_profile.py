"""
Extract intensity profile along horizontal line at middle.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap1/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Line to extract (middle row)
y = H // 2
profile = []
for x in range(W):
    profile.append(img_array[y, x])

# Save result plot
output_path = os.path.join(output_dir, 'output_tp1_intensity_profile.png')

# Visualization
plt.figure(figsize=(12, 6))

# Show image with a line indicating the profile location
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.axhline(y=y, color='r', linestyle='--')
plt.title(f'Grayscale Image (Line at y={y})')
plt.axis('off')

# Show the intensity profile
plt.subplot(1, 2, 2)
plt.plot(profile, color='blue')
plt.title('Intensity Profile')
plt.xlabel('Pixel Column')
plt.ylabel('Intensity Value')
plt.grid(True)

plt.suptitle('TP1: Intensity Profile Extraction')
plt.tight_layout()
plt.savefig(output_path)
plt.show()
