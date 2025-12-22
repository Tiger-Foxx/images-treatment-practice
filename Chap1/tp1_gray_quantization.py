"""
Quantize grayscale image to K bits.
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

def quantize(image, k_bits):
    L = 2 ** k_bits
    quantized = np.zeros((H, W), dtype=np.uint8)
    
    
    factor = 255 // (L - 1)
    for i in range(H):
        for j in range(W):
            val = image[i, j]
            
            level = int((val / 255.0) * (L - 1))
            quantized[i, j] = level * factor
    return quantized


k_values = [2, 4, 8]
results = []
for k in k_values:
    res = quantize(img_array, k)
    results.append(res)
    output_path = os.path.join(output_dir, f'output_tp1_gray_quant_k{k}.png')
    Image.fromarray(res).save(output_path)


plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original (8-bit)')
plt.axis('off')

for i, k in enumerate(k_values):
    plt.subplot(1, 4, i + 2)
    plt.imshow(results[i], cmap='gray')
    plt.title(f'Quantized (K={k})')
    plt.axis('off')

plt.suptitle('TP1: Grayscale Quantization')
plt.tight_layout()
plt.show()
