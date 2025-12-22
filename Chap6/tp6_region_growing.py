"""
Region growing segmentation starting from a seed point.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from collections import deque

# Create outputs directory if it doesn't exist
output_dir = 'Chap6/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image
img = Image.open('inputs/img2.png').convert('L')
img_array = np.array(img).astype(np.int16)
H, W = img_array.shape

# Seed point (center by default)
seed = (H // 2, W // 2)
tol = 20

# Tracking
visited = np.zeros((H, W), dtype=bool)
region = np.zeros((H, W), dtype=np.uint8)
queue = deque([seed])
visited[seed] = True
region[seed] = 255

initial_val = img_array[seed]
count = 1
running_mean = float(initial_val)

print(f"Applying Region Growing from seed {seed} with tolerance {tol}...")

# Manual growth
while queue:
    curr_x, curr_y = queue.popleft()
    
    # 8-connectivity
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0: continue
            ni, nj = curr_x + di, curr_y + dj
            
            if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                # Similarity criterion: difference with current region mean
                if abs(img_array[ni, nj] - running_mean) < tol:
                    visited[ni, nj] = True
                    region[ni, nj] = 255
                    queue.append((ni, nj))
                    # Update mean
                    running_mean = (running_mean * count + img_array[ni, nj]) / (count + 1)
                    count += 1

# Save result
output_path = os.path.join(output_dir, 'output_tp6_region_growing.png')
Image.fromarray(region).save(output_path)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.scatter(seed[1], seed[0], color='red', s=50, label='Seed')
plt.title('Original Image')
plt.legend()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(region, cmap='gray')
plt.title('Grown Region')
plt.axis('off')

plt.suptitle('TP6: Region Growing Segmentation')
plt.tight_layout()
plt.show()
