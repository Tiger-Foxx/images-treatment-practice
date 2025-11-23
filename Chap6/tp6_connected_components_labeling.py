"""
Connected components labeling.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
binary = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > 128:
            binary[i, j] = 1
labels = np.zeros((H, W), dtype=int)
label = 1
equiv = {}
for i in range(H):
    for j in range(W):
        if binary[i, j] == 1:
            neighbors = []
            if i > 0 and labels[i-1, j] > 0:
                neighbors.append(labels[i-1, j])
            if j > 0 and labels[i, j-1] > 0:
                neighbors.append(labels[i, j-1])
            if neighbors:
                min_label = min(neighbors)
                labels[i, j] = min_label
                for n in neighbors:
                    if n != min_label:
                        equiv[n] = min_label
            else:
                labels[i, j] = label
                label += 1
for i in range(H):
    for j in range(W):
        if labels[i, j] > 0:
            root = labels[i, j]
            while root in equiv:
                root = equiv[root]
            labels[i, j] = root
max_label = labels.max()
colored = np.zeros((H, W, 3), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if labels[i, j] > 0:
            colored[i, j] = [ (labels[i, j] * 37) % 256, (labels[i, j] * 73) % 256, (labels[i, j] * 113) % 256 ]
Image.fromarray(colored).save('Chap6/outputs/output_tp6_connected_components_labeling.png')