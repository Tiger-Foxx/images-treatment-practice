"""
Notch filter.
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
shifted = np.zeros((H, W), dtype=complex)
shifted[:H//2, :W//2] = dft_2d[H//2:, W//2:]
shifted[:H//2, W//2:] = dft_2d[H//2:, :W//2]
shifted[H//2:, :W//2] = dft_2d[:H//2, W//2:]
shifted[H//2:, W//2:] = dft_2d[:H//2, :W//2]
notch_u = 10
notch_v = 10
shifted[notch_u, notch_v] = 0
shifted[H - notch_u, W - notch_v] = 0
mag = np.log(1 + np.abs(shifted))
plt.imshow(mag, cmap='gray')
plt.title('Notch Filtered DFT')
plt.savefig('Chap4/outputs/output_tp4_notch_filter.png')
plt.show()