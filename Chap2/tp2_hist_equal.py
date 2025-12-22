"""
Global histogram equalization.
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


total = H * W
hn = hist / total
CDF = np.zeros(256)
CDF[0] = hn[0]
for k in range(1, 256):
    CDF[k] = CDF[k-1] + hn[k]


equalized = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        equalized[i, j] = int(CDF[img_array[i, j]] * 255)


output_path = os.path.join(output_dir, 'output_tp2_hist_equal.png')
Image.fromarray(equalized).save(output_path)


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.bar(range(256), hist, color='gray', width=1.0)
plt.title('Original Histogram')

plt.subplot(2, 2, 4)
plt.plot(range(256), CDF, color='red', label='CDF')
plt.bar(range(256), np.histogram(equalized, bins=256, range=(0, 256))[0] / total, color='blue', alpha=0.5, label='Equalized Hist')
plt.title('Equalized Histogram & CDF')
plt.legend()

plt.suptitle('TP2: Global Histogram Equalization')
plt.tight_layout()
plt.show()
