"""
2D DFT.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def dft1d(signal):
    N = len(signal)
    F = np.zeros(N, dtype=complex)
    for u in range(N):
        for k in range(N):
            F[u] += signal[k] * np.exp(-2j * np.pi * u * k / N)
    return F

img = Image.open('inputs/img1.png')
img_array = np.array(img)
patch = img_array[:64, :64]
H, W = patch.shape
dft_rows = np.zeros((H, W), dtype=complex)
for i in range(H):
    dft_rows[i, :] = dft1d(patch[i, :])
dft_2d = np.zeros((H, W), dtype=complex)
for j in range(W):
    dft_2d[:, j] = dft1d(dft_rows[:, j])
mag = np.log(1 + np.abs(dft_2d))
plt.imshow(mag, cmap='gray')
plt.title('2D DFT Magnitude')
plt.savefig('Chap4/outputs/output_tp4_dft_2d.png')
plt.show()