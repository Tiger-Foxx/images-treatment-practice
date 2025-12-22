"""
Extract intensity profile along horizontal line at middle.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap1/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape


y = H // 2
profile = []
for x in range(W):
    profile.append(img_array[y, x])


output_path = os.path.join(output_dir, 'output_tp1_intensity_profile.png')


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.axhline(y=y, color='r', linestyle='--')
plt.title(f'Grayscale Image (Line at y={y})')
plt.axis('off')


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
