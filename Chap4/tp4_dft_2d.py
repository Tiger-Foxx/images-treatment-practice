"""
2D DFT.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap4/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def dft1d(signal):
    N = len(signal)
    F = np.zeros(N, dtype=complex)
    
    for u in range(N):
        for k in range(N):
            F[u] += signal[k] * np.exp(-2j * np.pi * u * k / N)
    return F


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)


patch = img_array[:64, :64]
H, W = patch.shape

print(f"Computing 2D DFT on {H}x{W} patch. This will take a moment...")


dft_rows = np.zeros((H, W), dtype=complex)
for i in range(H):
    dft_rows[i, :] = dft1d(patch[i, :])


dft_2d = np.zeros((H, W), dtype=complex)
for j in range(W):
    dft_2d[:, j] = dft1d(dft_rows[:, j])


mag = np.log(1 + np.abs(dft_2d))


output_path = os.path.join(output_dir, 'output_tp4_dft_2d.png')
plt.imsave(output_path, mag, cmap='gray')


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(patch, cmap='gray')
plt.title(f'Original Patch ({H}x{W})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mag, cmap='gray')
plt.title('2D DFT Log-Magnitude Spectrum')
plt.axis('off')

plt.suptitle('TP4: 2D Discrete Fourier Transform')
plt.tight_layout()
plt.show()
