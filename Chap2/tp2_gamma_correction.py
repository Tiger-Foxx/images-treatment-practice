"""
Gamma correction for image luminance control.
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


gamma = 2.2 
corrected = np.zeros((H, W), dtype=np.uint8)


lut = np.zeros(256, dtype=np.uint8)
for k in range(256):
    lut[k] = int(255 * ((k / 255.0) ** (1.0 / gamma)))


for i in range(H):
    for j in range(W):
        corrected[i, j] = lut[img_array[i, j]]


output_path = os.path.join(output_dir, 'output_tp2_gamma_correction.png')
Image.fromarray(corrected).save(output_path)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(corrected, cmap='gray')
plt.title(f'Gamma Corrected (gamma={gamma})')
plt.axis('off')

plt.suptitle('TP2: Gamma Correction')
plt.tight_layout()
plt.show()
