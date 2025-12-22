"""
Ideal lowpass frequency filtering for image smoothing.
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

def idft1d(F):
    N = len(F)
    f = np.zeros(N, dtype=complex)
    for k in range(N):
        for u in range(N):
            f[k] += F[u] * np.exp(2j * np.pi * u * k / N)
    return f / N


img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img).astype(np.float32)


patch = img_array[:64, :64]
H, W = patch.shape

print("Applying Frequency Lowpass Filter... please wait.")


temp = np.zeros((H, W), dtype=complex)
for i in range(H): temp[i, :] = dft1d(patch[i, :])
F_2d = np.zeros((H, W), dtype=complex)
for j in range(W): F_2d[:, j] = dft1d(temp[:, j])


h_mid, w_mid = H // 2, W // 2
shifted = np.zeros((H, W), dtype=complex)
shifted[:h_mid, :w_mid] = F_2d[h_mid:, w_mid:]
shifted[:h_mid, w_mid:] = F_2d[h_mid:, :w_mid]
shifted[h_mid:, :w_mid] = F_2d[:h_mid, w_mid:]
shifted[h_mid:, w_mid:] = F_2d[:h_mid, :w_mid]

rayon = 15
low_shifted = shifted.copy()
for u in range(H):
    for v in range(W):
        if np.sqrt((u - h_mid)**2 + (v - w_mid)**2) > rayon:
            low_shifted[u, v] = 0 


inv_shifted = np.zeros((H, W), dtype=complex)
inv_shifted[h_mid:, w_mid:] = low_shifted[:h_mid, :w_mid]
inv_shifted[h_mid:, :w_mid] = low_shifted[:h_mid, w_mid:]
inv_shifted[:h_mid, w_mid:] = low_shifted[h_mid:, :w_mid]
inv_shifted[:h_mid, :w_mid] = low_shifted[h_mid:, w_mid:]

temp_inv = np.zeros((H, W), dtype=complex)
for i in range(H): temp_inv[i, :] = idft1d(inv_shifted[i, :])
result_img = np.zeros((H, W), dtype=complex)
for j in range(W): result_img[:, j] = idft1d(temp_inv[:, j])

result_abs = np.abs(result_img)
result_uint8 = np.clip(result_abs, 0, 255).astype(np.uint8)


output_path = os.path.join(output_dir, 'output_tp4_lowpass_freq.png')
Image.fromarray(result_uint8).save(output_path)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(patch, cmap='gray')
plt.title('Original Patch')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(low_shifted)), cmap='gray')
plt.title('Lowpass Spectrum (Log Mag)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result_uint8, cmap='gray')
plt.title('Lowpass Result (Blurred)')
plt.axis('off')

plt.suptitle('TP4: Frequency Domain Lowpass Filter')
plt.tight_layout()
plt.show()
