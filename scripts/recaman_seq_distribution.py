from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def recaman_sequence(max_n: int) -> np.ndarray:
    values = [0]
    visited = {0}
    for n in range(1, max_n + 1):
        candidate = values[-1] - n
        if candidate > 0 and candidate not in visited:
            values.append(candidate)
        else:
            values.append(values[-1] + n)
        visited.add(values[-1])
    return np.array(values)


print("Computing Recaman sequence up to n=1,000,000 ...")
seq = recaman_sequence(1_000_000)

print(f"Max a_n: {seq.max()}")
print(f"Mean a_n: {seq.mean():.2f}")
print(f"Median a_n: {np.median(seq):.2f}")
print(f"Std a_n: {seq.std():.2f}")

output_dir = Path(__file__).resolve().parent.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "recaman_distribution_1e6.png"

plt.figure(figsize=(10, 6))
plt.hist(seq, bins=200, density=True, alpha=0.7, color="steelblue", edgecolor="black")
plt.title("Recaman Sequence Distribution (n=1,000,000)")
plt.xlabel("a_n")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Histogram saved to {output_path}")

sorted_idx = np.argsort(seq)[-10:][::-1]
print("\nTop 10 largest a_n and their n:")
for idx in sorted_idx:
    print(f"n={idx + 1:7d} -> a_n={seq[idx]:8d}")
