"""
Convert RGB image to grayscale using weighted average.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap1/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img2.png')
img_array = np.array(img)
H, W, _ = img_array.shape


gray = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        r, g, b = img_array[i, j]
        
        gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)


output_path = os.path.join(output_dir, 'output_tp1_rgb_to_gray.png')
Image.fromarray(gray).save(output_path)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title('Original (RGB)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Result (Manual)')
plt.axis('off')

plt.suptitle('TP1: RGB to Grayscale Conversion')
plt.tight_layout()
plt.show()
