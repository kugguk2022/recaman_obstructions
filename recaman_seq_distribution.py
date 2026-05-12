import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def recaman_sequence(max_n):
    a = [0]
    visited = {0}
    for n in range(1, max_n + 1):
        candidate = a[-1] - n
        if candidate > 0 and candidate not in visited:
            a.append(candidate)
        else:
            a.append(a[-1] + n)
        visited.add(a[-1])
    return np.array(a)

print("Computing Recamán sequence up to n=1,000,000 ...")
seq = recaman_sequence(1_000_000)

print(f"Max a_n: {seq.max()}")
print(f"Mean a_n: {seq.mean():.2f}")
print(f"Median a_n: {np.median(seq):.2f}")
print(f"Std a_n: {seq.std():.2f}")

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(seq, bins=200, density=True, alpha=0.7, color='steelblue', edgecolor='black')
plt.title('Recamán Sequence Distribution (n=1,000,000)')
plt.xlabel('a_n')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.savefig('recaman_distribution_1e6.png', dpi=150, bbox_inches='tight')
print("Histogram saved to recaman_distribution_1e6.png")

# Top 10 largest values and their positions
sorted_idx = np.argsort(seq)[-10:][::-1]
print("\nTop 10 largest a_n and their n:")
for idx in sorted_idx:
    print(f"n={idx+1:7d} -> a_n={seq[idx]:8d}")