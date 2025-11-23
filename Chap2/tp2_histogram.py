"""
Compute histogram of grayscale image.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
hist = np.zeros(256, dtype=int)
for i in range(H):
    for j in range(W):
        hist[img_array[i, j]] += 1
plt.bar(range(256), hist)
plt.title('Histogram')
plt.savefig('Chap2/outputs/output_tp2_histogram.png')
plt.show()