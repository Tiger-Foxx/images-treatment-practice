"""
K-means clustering for color-based image segmentation.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap6/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img2.png').convert('RGB')
img_array = np.array(img)
H, W, C = img_array.shape


K = 3
max_iter = 10
np.random.seed(42)


pixels = img_array.reshape((-1, 3)).astype(np.float32)


random_indices = np.random.choice(len(pixels), K, replace=False)
centers = pixels[random_indices]

print(f"Applying K-means segmentation (K={K})...")

for i in range(max_iter):
    
    
    labels = np.zeros(len(pixels), dtype=int)
    for idx, p in enumerate(pixels):
        
        dists = np.sqrt(np.sum((p - centers)**2, axis=1))
        labels[idx] = np.argmin(dists)
    
    
    new_centers = np.zeros((K, 3), dtype=np.float32)
    counts = np.zeros(K)
    for idx, p in enumerate(pixels):
        new_centers[labels[idx]] += p
        counts[labels[idx]] += 1
    
    for k in range(K):
        if counts[k] > 0:
            centers[k] = new_centers[k] / counts[k]
    
    print(f" Iteration {i+1}/{max_iter} complete.")


segmented_pixels = centers[labels].astype(np.uint8)
segmented = segmented_pixels.reshape((H, W, 3))


output_path = os.path.join(output_dir, 'output_tp6_kmeans_segmentation.png')
Image.fromarray(segmented).save(output_path)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title(f'K-means Segmentation (K={K})')
plt.axis('off')

plt.suptitle('TP6: K-means Clustering')
plt.tight_layout()
plt.show()
