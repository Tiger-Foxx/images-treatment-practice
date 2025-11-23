"""
Region growing.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img2.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape
seed = (H//2, W//2)
tol = 20
visited = np.zeros((H, W), dtype=bool)
region = np.zeros((H, W), dtype=np.uint8)
from collections import deque
queue = deque()
queue.append(seed)
visited[seed] = True
region[seed] = 255
mean = img_array[seed]
count = 1
while queue:
    x, y = queue.popleft()
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = x + di, y + dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                if abs(img_array[ni, nj] - mean) < tol:
                    visited[ni, nj] = True
                    region[ni, nj] = 255
                    queue.append((ni, nj))
                    mean = (mean * count + img_array[ni, nj]) / (count + 1)
                    count += 1
Image.fromarray(region).save('Chap6/outputs/output_tp6_region_growing.png')