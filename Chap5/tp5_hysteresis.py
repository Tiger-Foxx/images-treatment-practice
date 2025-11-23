"""
Hysteresis.
"""
import numpy as np
from PIL import Image

def convolve(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape
    r = kH // 2
    padded = np.pad(img, r, mode='constant', constant_values=0)
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            sum_val = 0
            for u in range(kH):
                for v in range(kW):
                    sum_val += padded[i + u, j + v] * kernel[u, v]
            result[i, j] = sum_val
    return result

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
kx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
ky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
dx = convolve(img_array, kx)
dy = convolve(img_array, ky)
mag = np.sqrt(dx**2 + dy**2)
high = 200
low = 100
edges = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if mag[i, j] > high:
            edges[i, j] = 255
from collections import deque
queue = deque()
for i in range(H):
    for j in range(W):
        if edges[i, j] == 255:
            queue.append((i, j))
while queue:
    x, y = queue.popleft()
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = x + di, y + dj
            if 0 <= ni < H and 0 <= nj < W and edges[ni, nj] == 0 and mag[ni, nj] > low:
                edges[ni, nj] = 255
                queue.append((ni, nj))
Image.fromarray(edges).save('Chap5/outputs/output_tp5_hysteresis.png')