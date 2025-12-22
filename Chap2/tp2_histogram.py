"""
Compute histogram of grayscale image.
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


hist = np.zeros(256, dtype=int)
for i in range(H):
    for j in range(W):
        hist[img_array[i, j]] += 1


output_path = os.path.join(output_dir, 'output_tp2_histogram.png')


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(256), hist, color='gray', width=1.0)
plt.title('Grayscale Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.suptitle('TP2: Histogram Calculation')
plt.tight_layout()
plt.savefig(output_path)
plt.show()
