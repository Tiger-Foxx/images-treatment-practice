"""
Linear contrast stretching with saturation (clipping).
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


Smin = 50
Smax = 200
stretched = np.zeros((H, W), dtype=np.uint8)


lut = np.zeros(256, dtype=np.uint8)
for k in range(256):
    if k < Smin:
        lut[k] = 0
    elif k > Smax:
        lut[k] = 255
    else:
        
        lut[k] = int(255 * (k - Smin) / (Smax - Smin))


for i in range(H):
    for j in range(W):
        stretched[i, j] = lut[img_array[i, j]]


output_path = os.path.join(output_dir, 'output_tp2_linear_saturation.png')
Image.fromarray(stretched).save(output_path)


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(stretched, cmap='gray')
plt.title(f'Saturated Stretch [{Smin}, {Smax}]')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(range(256), lut, color='red')
plt.title('Transformation Function')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.hist(img_array.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.5, label='Original')
plt.hist(stretched.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.5, label='Processed')
plt.title('Histogram Comparison')
plt.legend()

plt.suptitle('TP2: Linear Stretch with Saturation')
plt.tight_layout()
plt.show()
