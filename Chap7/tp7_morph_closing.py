"""
Morphological closing (Dilation then Erosion) for filling holes.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap7/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def erosion(binary, N=3):
    H, W = binary.shape
    r = N // 2
    padded = np.pad(binary, r, mode='constant', constant_values=1) 
    eroded = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            match = True
            for u in range(N):
                for v in range(N):
                    if padded[i + u, j + v] == 0:
                        match = False
                        break
                if not match: break
            if match:
                eroded[i, j] = 1
    return eroded

def dilation(binary, N=3):
    H, W = binary.shape
    r = N // 2
    padded = np.pad(binary, r, mode='constant', constant_values=0) 
    dilated = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            any_match = False
            for u in range(N):
                for v in range(N):
                    if padded[i + u, j + v] == 1:
                        any_match = True
                        break
                if any_match: break
            if any_match:
                dilated[i, j] = 1
    return dilated


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape
binary = (img_array > 127).astype(np.uint8)


closed = erosion(dilation(binary, N=3), N=3)


output_path = os.path.join(output_dir, 'output_tp7_morph_closing.png')
Image.fromarray((closed * 255).astype(np.uint8)).save(output_path)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(closed, cmap='gray')
plt.title('Closing Result (Dilation -> Erosion)')
plt.axis('off')

plt.suptitle('TP7: Morphological Closing')
plt.tight_layout()
plt.show()
