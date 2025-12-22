"""
Local histogram equalization.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
output_dir = 'Chap2/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image
img = Image.open('inputs/img1.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape

# Parameters
N = 15  # Window size (NxN)
r = N // 2
padded = np.pad(img_array, r, mode='reflect') # reflect is better for edges
equalized = np.zeros((H, W), dtype=np.uint8)

print(f"Applying Local Hist Equalization with window {N}x{N}. This may take a few seconds...")

# Manual local equalization
for i in range(H):
    for j in range(W):
        # 1. Extract local window
        local_window = padded[i:i+N, j:j+N]
        # 2. Compute local histogram
        local_hist = np.zeros(256, dtype=int)
        for u in range(N):
            for v in range(N):
                local_hist[local_window[u, v]] += 1
        
        # 3. Compute local CDF
        total_local = N * N
        cum_sum = 0
        target_val = img_array[i, j]
        # Optimization: only compute CDF up to target_val
        for k in range(target_val + 1):
            cum_sum += local_hist[k]
        
        # 4. Map pixel
        equalized[i, j] = int((cum_sum / total_local) * 255)

# Save result
output_path = os.path.join(output_dir, 'output_tp2_local_hist_equal.png')
Image.fromarray(equalized).save(output_path)

# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title(f'Local Equalization ({N}x{N})')
plt.axis('off')

plt.suptitle('TP2: Local Histogram Equalization')
plt.tight_layout()
plt.show()
