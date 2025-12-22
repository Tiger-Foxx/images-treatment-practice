"""
Freeman chain code (8-connectivity contour following).
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


output_dir = 'Chap7/outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img = Image.open('inputs/img2.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape


binary = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > 127:
            binary[i, j] = 1


start_node = None
for i in range(H):
    for j in range(W):
        if binary[i, j] == 1:
            start_node = (i, j)
            break
    if start_node:
        break

if start_node:
    chain = []
    contour_coords = [start_node]
    current = start_node
    
    
    
    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    visited = set()
    visited.add(start_node)
    
    while True:
        found = False
        for d in range(8):
            di, dj = dirs[d]
            ni, nj = current[0] + di, current[1] + dj
            
            if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] == 1 and (ni, nj) not in visited:
                
                chain.append(d)
                visited.add((ni, nj))
                contour_coords.append((ni, nj))
                current = (ni, nj)
                found = True
                break
        if not found:
            break
            
    print(f"Chain code length: {len(chain)}")
    print("Chain code:", chain[:20], "..." if len(chain) > 20 else "")
    
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Binarized Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(binary, cmap='gray', alpha=0.3)
    if len(contour_coords) > 0:
        y_coords, x_coords = zip(*contour_coords)
        plt.plot(x_coords, y_coords, 'r-', linewidth=2)
        plt.scatter(x_coords[0], y_coords[0], color='green', s=50, label='Start')
    plt.title('Contour Tracing (Freeman)')
    plt.legend()
    plt.axis('off')
    
    plt.suptitle('TP7: Freeman Chain Code')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'output_tp7_freeman_chain_code.png')
    plt.savefig(output_path)
    plt.show()
else:
    print("No object found in image.")
