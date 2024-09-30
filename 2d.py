# Question 2, Task D
# Shaik Sahil Chanda, 23b0943
# P. Gnana Koushik Reddy, 23b1000
# D. Kishan Teja, 23b1061

import numpy as np
import matplotlib.pyplot as plt

def simulate_galton_board(N, depth):
    
    steps = np.random.randint(0, 2, size=(N, depth))
    
    # final positions
    final_positions = (np.sum(steps, axis=1))
    
    counts = np.bincount(final_positions, minlength=depth + 1)
    
    return counts


N = 10**5 
h = 100  # 10 50

counts = simulate_galton_board(N, h)
x_values = np.arange(-h // 2, h // 2 + 1)

plt.figure(figsize=(6, 4))
plt.bar(x_values, counts / N, color='b', alpha=0.7)
plt.title(f'Galton Board Simulation with Depth = {h}')
plt.xlabel('Pocket')
plt.ylabel('Normalised count')
plt.show()
