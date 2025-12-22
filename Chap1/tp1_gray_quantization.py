"""
Quantize grayscale image to K bits.
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

def quantize(image, k_bits):
    L = 2 ** k_bits
    quantized = np.zeros((H, W), dtype=np.uint8)
    # Number of intervals is L. Factor to map 0-255 to 0-(L-1) is (L-1)/255
    # Then we map back to 0-255 by multiplying by 255/(L-1)
    factor = 255 // (L - 1)
    for i in range(H):
        for j in range(W):
            val = image[i, j]
            # Map to discrete level and back to 0-255 range
            level = int((val / 255.0) * (L - 1))
            quantized[i, j] = level * factor
    return quantized

# Test with different K
k_values = [2, 4, 8]
results = []
for k in k_values:
    res = quantize(img_array, k)
    results.append(res)
    output_path = os.path.join(output_dir, f'output_tp1_gray_quant_k{k}.png')
    Image.fromarray(res).save(output_path)

# Visualization
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
