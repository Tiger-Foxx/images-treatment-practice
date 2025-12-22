"""
Hysteresis thresholding for edge linking (Canny-style).
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from collections import deque


output_dir = 'Chap5/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convolve(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape
    r = kH // 2
    padded = np.pad(img, r, mode='reflect')
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            sum_val = 0
            for u in range(kH):
                for v in range(kW):
                    sum_val += padded[i + u, j + v] * kernel[u, v]
            result[i, j] = sum_val
    return result


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img).astype(np.float32)
H, W = img_array.shape


kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


dx = convolve(img_array, kx)
dy = convolve(img_array, ky)
mag = np.sqrt(dx**2 + dy**2)


high_thresh = 100
low_thresh = 40

edges = np.zeros((H, W), dtype=np.uint8)
strong_i, strong_j = np.where(mag >= high_thresh)
weak_i, weak_j = np.where((mag >= low_thresh) & (mag < high_thresh))


queue = deque()
for i, j in zip(strong_i, strong_j):
    edges[i, j] = 255
    queue.append((i, j))


while queue:
    x, y = queue.popleft()
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0: continue
            ni, nj = x + di, y + dj
            if 0 <= ni < H and 0 <= nj < W:
                if edges[ni, nj] == 0 and mag[ni, nj] >= low_thresh:
                    edges[ni, nj] = 255
                    queue.append((ni, nj))


output_path = os.path.join(output_dir, 'output_tp5_hysteresis.png')
Image.fromarray(edges).save(output_path)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mag, cmap='inferno')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title(f'Hysteresis (Low={low_thresh}, High={high_thresh})')
plt.axis('off')

plt.suptitle('TP5: Hysteresis Edge Linking')
plt.tight_layout()
plt.show()
