"""
Linear stretch with saturation.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
Smin = 50
Smax = 200
stretched = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        I = img_array[i, j]
        if I < Smin:
            stretched[i, j] = 0
        elif I > Smax:
            stretched[i, j] = 255
        else:
            stretched[i, j] = int(255 * (I - Smin) / (Smax - Smin))
Image.fromarray(stretched).save('Chap2/outputs/output_tp2_linear_saturation.png')