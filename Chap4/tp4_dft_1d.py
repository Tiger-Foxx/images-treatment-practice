"""
1D DFT.
"""
import numpy as np
import matplotlib.pyplot as plt

N = 64
f = np.zeros(N)
for k in range(N):
    f[k] = np.sin(2 * np.pi * k / 10) + 0.5 * np.random.randn()
F = np.zeros(N, dtype=complex)
for u in range(N):
    for k in range(N):
        F[u] += f[k] * np.exp(-2j * np.pi * u * k / N)
magnitude = np.abs(F)
plt.plot(magnitude)
plt.title('1D DFT Magnitude')
plt.savefig('Chap4/outputs/output_tp4_dft_1d.png')
plt.show()