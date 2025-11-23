"""
Extract intensity profile along horizontal line at middle.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('inputs/img1.png')
img_array = np.array(img)
H, W = img_array.shape
y = H // 2
profile = []
for x in range(W):
    profile.append(img_array[y, x])
plt.plot(profile)
plt.title('Intensity Profile')
plt.savefig('Chap1/outputs/output_tp1_intensity_profile.png')
plt.show()