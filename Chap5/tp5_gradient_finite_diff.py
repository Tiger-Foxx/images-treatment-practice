"""
Gradient by finite differences.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
dx = np.zeros((H, W), dtype=np.float32)
dy = np.zeros((H, W), dtype=np.float32)
for i in range(1, H-1):
    for j in range(1, W-1):
        dx[i, j] = img_array[i+1, j] - img_array[i-1, j]
        dy[i, j] = img_array[i, j+1] - img_array[i, j-1]
mag = np.sqrt(dx**2 + dy**2)
mag_norm = (mag / mag.max() * 255).astype(np.uint8)
Image.fromarray(mag_norm).save('Chap5/outputs/output_tp5_gradient_finite_diff.png')