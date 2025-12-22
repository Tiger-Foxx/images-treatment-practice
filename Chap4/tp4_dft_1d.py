"""
1D Discrete Fourier Transform (DFT).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap4/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate a synthetic signal: sum of sines + noise
N = 128
n = np.arange(N)
f = 0.5 * np.sin(2 * np.pi * n / 20) + 0.2 * np.sin(2 * np.pi * n / 5) + 0.1 * np.random.randn(N)

# Manual 1D DFT
F = np.zeros(N, dtype=complex)
for u in range(N):
    for k in range(N):
        F[u] += f[k] * np.exp(-2j * np.pi * u * k / N)

magnitude = np.abs(F)

# Save result plot
output_path = os.path.join(output_dir, 'output_tp4_dft_1d.png')

# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(n, f, color='blue')
plt.title('Original Signal (Noisy Sines)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(n[:N//2], magnitude[:N//2], color='red') # Only show positive frequencies
plt.title('DFT Magnitude Spectrum')
plt.grid(True)

plt.suptitle('TP4: 1D Discrete Fourier Transform')
plt.tight_layout()
plt.savefig(output_path)
plt.show()
