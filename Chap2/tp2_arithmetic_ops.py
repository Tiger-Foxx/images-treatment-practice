"""
Arithmetic operations: addition, subtraction, and multiplication.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap2/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img1 = Image.open('inputs/img1.png').convert('L')
img2 = Image.open('inputs/img2.png').convert('L')


if img1.size != img2.size:
    img2 = img2.resize(img1.size)

img1_array = np.array(img1, dtype=np.int16) 
img2_array = np.array(img2, dtype=np.int16)
H, W = img1_array.shape

add_result = np.zeros((H, W), dtype=np.uint8)
sub_result = np.zeros((H, W), dtype=np.uint8)
mult_result = np.zeros((H, W), dtype=np.uint8)
ratio = 0.5


for i in range(H):
    for j in range(W):
        
        add_val = img1_array[i, j] + img2_array[i, j]
        add_result[i, j] = min(add_val, 255)
        
        
        sub_val = img1_array[i, j] - img2_array[i, j]
        sub_result[i, j] = max(sub_val, 0)
        
        
        mult_result[i, j] = int(img1_array[i, j] * ratio)


Image.fromarray(add_result).save(os.path.join(output_dir, 'output_tp2_arithmetic_add.png'))
Image.fromarray(sub_result).save(os.path.join(output_dir, 'output_tp2_arithmetic_sub.png'))
Image.fromarray(mult_result).save(os.path.join(output_dir, 'output_tp2_arithmetic_mult.png'))


plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img1_array, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img2_array, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(add_result, cmap='gray')
plt.title('Addition (img1 + img2)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(sub_result, cmap='gray')
plt.title('Subtraction (img1 - img2)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(mult_result, cmap='gray')
plt.title(f'Multiplication (img1 * {ratio})')
plt.axis('off')

plt.suptitle('TP2: Image Arithmetic Operations')
plt.tight_layout()
plt.show()
