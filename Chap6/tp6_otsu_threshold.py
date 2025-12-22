"""
Otsu's method for automatic thresholding.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap6/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape


hist = np.zeros(256, dtype=int)
for i in range(H):
    for j in range(W):
        hist[img_array[i, j]] += 1


total = H * W
best_T = 0
max_variance = -1


for T in range(256):
    
    P1 = np.sum(hist[:T]) / total
    P2 = 1.0 - P1
    
    if P1 == 0 or P2 == 0:
        continue
    
    
    mu1 = np.sum(np.arange(T) * hist[:T]) / (P1 * total)
    mu2 = np.sum(np.arange(T, 256) * hist[T:]) / (P2 * total)
    
    
    variance = P1 * P2 * (mu1 - mu2)**2
    
    if variance > max_variance:
        max_variance = variance
        best_T = T

print(f"Optimal Otsu Threshold: {best_T}")


thresh = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > best_T:
            thresh[i, j] = 255
        else:
            thresh[i, j] = 0


output_path = os.path.join(output_dir, 'output_tp6_otsu_threshold.png')
Image.fromarray(thresh).save(output_path)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.bar(range(256), hist, color='gray', width=1.0)
plt.axvline(x=best_T, color='red', linestyle='--', label=f'Otsu T={best_T}')
plt.title('Histogram & Threshold')
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title('Otsu Segmentation')
plt.axis('off')

plt.suptitle('TP6: Otsu Automatic Thresholding')
plt.tight_layout()
plt.show()
