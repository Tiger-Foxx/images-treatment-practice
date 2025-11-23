"""
Freeman chain code.
"""
import numpy as np
from PIL import Image

img = Image.open('inputs/img2.png').convert('L')
img_array = np.array(img)
H, W = img_array.shape
binary = np.zeros((H, W), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        if img_array[i, j] > 128:
            binary[i, j] = 1
start = None
for i in range(H):
    for j in range(W):
        if binary[i, j] == 1:
            start = (i, j)
            break
    if start:
        break
if start:
    chain = []
    current = start
    direction = 0
    visited = set()
    visited.add(start)
    while True:
        found = False
        dirs = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        for d in range(8):
            di, dj = dirs[d]
            ni, nj = current[0] + di, current[1] + dj
            if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] == 1 and (ni, nj) not in visited:
                chain.append(d)
                visited.add((ni, nj))
                current = (ni, nj)
                found = True
                break
        if not found:
            break
    print("Chain code:", chain)
else:
    print("No object found")
Image.fromarray((binary * 255).astype(np.uint8)).save('Chap7/outputs/output_tp7_freeman_chain_code.png')