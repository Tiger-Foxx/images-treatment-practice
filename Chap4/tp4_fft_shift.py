"""
FFT shift to center low frequencies.
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

print("Computing 2D DFT and shifting... please wait.")


dft_rows = np.zeros((H, W), dtype=complex)
for i in range(H):
    dft_rows[i, :] = dft1d(patch[i, :])


dft_2d = np.zeros((H, W), dtype=complex)
for j in range(W):
    dft_2d[:, j] = dft1d(dft_rows[:, j])


shifted = np.zeros((H, W), dtype=complex)
h_mid, w_mid = H // 2, W // 2


shifted[:h_mid, :w_mid] = dft_2d[h_mid:, w_mid:]
shifted[:h_mid, w_mid:] = dft_2d[h_mid:, :w_mid]
shifted[h_mid:, :w_mid] = dft_2d[:h_mid, w_mid:]
shifted[h_mid:, w_mid:] = dft_2d[:h_mid, :w_mid]


mag_orig = np.log(1 + np.abs(dft_2d))
mag_shifted = np.log(1 + np.abs(shifted))


output_path = os.path.join(output_dir, 'output_tp4_fft_shift.png')
plt.imsave(output_path, mag_shifted, cmap='gray')


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(patch, cmap='gray')
plt.title('Original Patch')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mag_orig, cmap='gray')
plt.title('DFT Spectrum (Unshifted)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mag_shifted, cmap='gray')
plt.title('DFT Spectrum (Shifted)')
plt.axis('off')

plt.suptitle('TP4: FFT Quadrant Shifting')
plt.tight_layout()
plt.show()
