"""
Connected components labeling (Two-pass algorithm).
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap6/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape


binary = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > 127:
            binary[i, j] = 1


labels = np.zeros((H, W), dtype=int)
next_label = 1
parent = {}

for i in range(H):
    for j in range(W):
        if binary[i, j] == 1:
            
            neighbors = []
            if i > 0 and labels[i-1, j] > 0:
                neighbors.append(labels[i-1, j])
            if j > 0 and labels[i, j-1] > 0:
                neighbors.append(labels[i, j-1])
            
            if not neighbors:
                labels[i, j] = next_label
                parent[next_label] = next_label
                next_label += 1
            else:
                min_label = min(neighbors)
                labels[i, j] = min_label
                for n in neighbors:
                    
                    root_n = n
                    while parent[root_n] != root_n:
                        root_n = parent[root_n]
                    root_min = min_label
                    while parent[root_min] != root_min:
                        root_min = parent[root_min]
                    if root_n != root_min:
                        parent[root_n] = root_min


for i in range(H):
    for j in range(W):
        if labels[i, j] > 0:
            root = labels[i, j]
            while parent[root] != root:
                root = parent[root]
            labels[i, j] = root


max_label = 0 if not parent else max(labels.flatten())
colored = np.zeros((H, W, 3), dtype=np.uint8)
if max_label > 0:
    
    colors = np.random.RandomState(42).randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if labels[i, j] > 0:
                colored[i, j] = colors[labels[i, j]]


output_path = os.path.join(output_dir, 'output_tp6_connected_components_labeling.png')
Image.fromarray(colored).save(output_path)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(colored)
plt.title(f'Labels Found: {len(np.unique(labels)) - 1}')
plt.axis('off')

plt.suptitle('TP6: Connected Components Labeling')
plt.tight_layout()
plt.show()
