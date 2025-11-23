"""
Convert RGB image to grayscale using weighted average.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img2.png')
img_array = np.array(img)
H, W, _ = img_array.shape
gray = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        r, g, b = img_array[i, j]
        gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
Image.fromarray(gray).save('Chap1/outputs/output_tp1_rgb_to_gray.png')