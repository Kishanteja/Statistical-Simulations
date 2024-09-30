# Question 2, Task C
# Shaik Sahil Chanda, 23b0943
# P. Gnana Koushik Reddy, 23b1000
# D. Kishan Teja, 23b1061

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def sample(loc, scale, size=1):
    # uniform random numbers in [0, 1]
    uniform_samples = np.random.uniform(0, 1, size)
    # Using the inverse CDF from norm to transform them to Gaussian samples
    gaussian_samples = norm.ppf(uniform_samples, loc=loc, scale=scale)
    return gaussian_samples

N = 10**5
params = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]
samples = {param: sample(param[0], np.sqrt(param[1]), size=N) for param in params}

plt.figure(figsize=(12, 8))

colors = ['b', 'g', 'r', 'y']

for (param, sample_data), color in zip(samples.items(), colors):
    mu, sigmasq = param
    plt.hist(sample_data, bins=100, density=True, alpha=0.6, color=color, label=f'μ={mu}, σ²={sigmasq}')

plt.title('Histograms of obtained Gaussian Distributions')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.show()