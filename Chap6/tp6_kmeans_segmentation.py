"""
K-means segmentation.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img2.png')
img_array = np.array(img)
H, W, _ = img_array.shape
K = 3
pixels = []
for i in range(H):
    for j in range(W):
        pixels.append(img_array[i, j])
pixels = np.array(pixels)
centers = pixels[np.random.choice(len(pixels), K, replace=False)]
for _ in range(10):
    labels = np.zeros(len(pixels), dtype=int)
    for idx, p in enumerate(pixels):
        dists = [np.linalg.norm(p - c) for c in centers]
        labels[idx] = np.argmin(dists)
    new_centers = np.zeros((K, 3))
    counts = np.zeros(K)
    for idx, p in enumerate(pixels):
        new_centers[labels[idx]] += p
        counts[labels[idx]] += 1
    for k in range(K):
        if counts[k] > 0:
            centers[k] = new_centers[k] / counts[k]
segmented = np.zeros((H, W, 3), dtype=np.uint8)
idx = 0
for i in range(H):
    for j in range(W):
        segmented[i, j] = centers[labels[idx]]
        idx += 1
Image.fromarray(segmented).save('Chap6/outputs/output_tp6_kmeans_segmentation.png')