"""
Highpass enhancement.
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
rayon = 10
highpass = shifted.copy()
for u in range(H):
    for v in range(W):
        if np.sqrt((u - H//2)**2 + (v - W//2)**2) < rayon:
            highpass[u, v] = 0
mag = np.log(1 + np.abs(highpass))
plt.imshow(mag, cmap='gray')
plt.title('Highpass Enhancement')
plt.savefig('Chap4/outputs/output_tp4_enhance_highpass.png')
plt.show()