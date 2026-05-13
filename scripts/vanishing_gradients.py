"""
vanishing_gradients.py
======================
Visualises the "soft memory potential" of the Recamán sequence and detects
where its gradient vanishes — indicating unexplored gaps that are hard to
reach by gradient-based search.

Potential:   U(x) = Σ_m  exp(-(x-m)² / σ²)
Gradient:   dU/dx = Σ_m  -2(x-m)/σ²  · exp(-(x-m)² / σ²)   (analytical)

Large-value behaviour
---------------------
For x >> max(centers), every term exp(-(x-m)²/σ²) → 0 so both U and dU/dx
vanish *analytically* — no computation needed.  The gradient-horizon formula
below gives the exact x beyond which |dU/dx| is guaranteed below any ε.

For large *n_steps* use `recaman_bitmap()` which replaces the Python set with
a compact bytearray (1 byte per candidate value), cutting memory ~56× and
allowing n up to ~10^7 in seconds.  Beyond ~10^8 steps the upper bound on
the sequence (~n²/2) exceeds available RAM and a streaming/approximate
approach is needed.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# All generated files are written here regardless of the working directory.
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Recamán sequence
# ---------------------------------------------------------------------------

def classical_recaman(n_steps: int = 20) -> tuple[list[int], set[int]]:
    """Return (sequence, visited_set) for the first *n_steps* Recamán terms.

    Uses a Python ``set`` — fast for n ≲ 10^5, memory-heavy beyond that.
    For larger n use :func:`recaman_bitmap`.
    """
    seq: list[int] = [0]
    visited: set[int] = {0}
    for n in range(1, n_steps + 1):
        cand = seq[-1] - n
        if cand > 0 and cand not in visited:
            seq.append(cand)
        else:
            seq.append(seq[-1] + n)
        visited.add(seq[-1])
    return seq, visited


def recaman_bitmap(n_steps: int) -> tuple[list[int], np.ndarray]:
    """Memory-efficient Recamán for large *n_steps* using a bytearray bitmap.

    The worst-case maximum value after n steps is the n-th triangular number
    T(n) = n*(n+1)//2 (all steps go up).  A bytearray of that length uses
    ~1 byte per slot — about 56× less than a Python set of integers.

    Practical limits on a typical machine
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * n = 10^5  →  bitmap ≈ 5 GB  ... set ≈ 100 MB   → use set
    * n = 10^6  →  bitmap ≈ 500 MB → feasible
    * n = 10^7  →  bitmap ≈ 50 GB  → needs a 64-GB machine
    * n = 10^17 →  impossible in RAM; use analytic bound instead

    Returns
    -------
    seq     : list of sequence values (length n_steps+1)
    centers : sorted numpy array of visited values
    """
    # Upper bound: every step goes upward → a(n) ≤ T(n)
    max_val = (n_steps * (n_steps + 1)) // 2 + 1
    _RAM_LIMIT = 2 * 10**9  # 2 GB guard
    if max_val > _RAM_LIMIT:
        raise MemoryError(
            f"n_steps={n_steps:,} would need a bitmap of {max_val/1e9:.1f} GB. "
            "Use analytic_gradient_horizon() instead."
        )

    bitmap = bytearray(max_val)  # O(T(n)) bytes
    seq: list[int] = [0]
    bitmap[0] = 1
    for n in range(1, n_steps + 1):
        prev = seq[-1]
        cand = prev - n
        if cand > 0 and not bitmap[cand]:
            seq.append(cand)
        else:
            seq.append(prev + n)
        bitmap[seq[-1]] = 1

    centers = np.flatnonzero(np.frombuffer(bitmap, dtype=np.uint8)).astype(np.float64)
    return seq, centers


# ---------------------------------------------------------------------------
# Potential and gradient — fully vectorised over x
# ---------------------------------------------------------------------------

def soft_memory_potential(
    x: np.ndarray | float,
    centers: np.ndarray,
    sigma: float = 2.5,
) -> np.ndarray:
    """Gaussian mixture potential at scalar or array *x*.

    Parameters
    ----------
    x:       query point(s)
    centers: 1-D array of visited Recamán values
    sigma:   bandwidth of each Gaussian
    """
    x = np.asarray(x, dtype=float)
    diff = x[..., np.newaxis] - centers          # shape (..., M)
    return np.exp(-(diff ** 2) / sigma ** 2).sum(axis=-1)


def analytical_gradient_U(
    x: np.ndarray | float,
    centers: np.ndarray,
    sigma: float = 2.5,
) -> np.ndarray:
    """Exact derivative dU/dx of the Gaussian mixture potential.

    dU/dx = Σ_m  -2(x-m)/σ²  · exp(-(x-m)²/σ²)
    """
    x = np.asarray(x, dtype=float)
    diff = x[..., np.newaxis] - centers          # shape (..., M)
    weights = np.exp(-(diff ** 2) / sigma ** 2)
    return ((-2.0 / sigma ** 2) * diff * weights).sum(axis=-1)


# ---------------------------------------------------------------------------
# Analytical large-x bound
# ---------------------------------------------------------------------------

def gradient_horizon(centers: np.ndarray, sigma: float = 2.5,
                     threshold: float = 0.1) -> float:
    """Return the x beyond which |dU/dx| is *guaranteed* below *threshold*.

    For x > max(centers)+d the upper bound on |dU/dx| is

        B(d) = M * (2d/σ²) * exp(-(d/σ)²)

    This bound rises from 0, peaks at d* = σ/√2, then decays to 0.
    We find the d > d* where B(d) first falls below *threshold*.
    """
    from scipy.optimize import brentq  # local import — optional dependency

    M = len(centers)
    x_max = float(centers.max())

    # Peak of the bound B(d): occurs at d = σ/√2
    d_peak = sigma / np.sqrt(2.0)
    B_peak = M * (2.0 * d_peak / sigma ** 2) * np.exp(-(d_peak / sigma) ** 2)

    if B_peak <= threshold:
        # Bound never exceeds threshold — horizon is at x_max itself
        return x_max

    def B(d: float) -> float:
        return M * (2.0 * d / sigma ** 2) * np.exp(-(d / sigma) ** 2) - threshold

    # B(d_peak) > 0 (confirmed above).  Find upper bracket where B < 0.
    d_hi = d_peak * 2.0
    while B(d_hi) > 0:
        d_hi *= 2.0

    d_star = brentq(B, d_peak, d_hi)
    return x_max + d_star


def check_large_x(x: float, centers: np.ndarray, sigma: float = 2.5,
                  threshold: float = 0.1) -> None:
    """Print an analytical and numerical verdict for a single large query x.

    For x ~ 10^17 direct float64 arithmetic is fine — numpy handles it — but
    the result is always analytically 0 when x >> max(centers) + few·σ.
    """
    horizon = gradient_horizon(centers, sigma, threshold)
    U = float(soft_memory_potential(x, centers, sigma))
    G = float(analytical_gradient_U(x, centers, sigma))
    beyond = x > horizon
    print(f"\n--- Large-x check  x = {x:.3e} ---")
    print(f"  max(centers)       = {centers.max():.1f}")
    print(f"  gradient horizon   = {horizon:.3f}")
    print(f"  x beyond horizon?  {beyond}")
    print(f"  U(x)  (float64)    = {U:.6e}")
    print(f"  dU/dx (float64)    = {G:.6e}")
    if beyond:
        print("  -> gradient is ANALYTICALLY VANISHING at this x.")
    else:
        print("  -> x is within the active region.")


# ---------------------------------------------------------------------------
# Spot-check table
# ---------------------------------------------------------------------------

SPOT_POINTS = [0, 1, 2, 3, 6, 7, 13, 17, 20, 25, 42, 50]
GRADIENT_THRESHOLD = 0.1


def print_gradient_table(
    centers: np.ndarray,
    sigma: float = 2.5,
    threshold: float = GRADIENT_THRESHOLD,
) -> None:
    xs = np.array(SPOT_POINTS, dtype=float)
    U  = soft_memory_potential(xs, centers, sigma)
    G  = analytical_gradient_U(xs, centers, sigma)

    print(f"\n{'x':>6} | {'U(x)':>8} | {'dU/dx':>10} | status")
    print("-" * 52)
    for xi, ui, gi in zip(xs, U, G):
        status = "strong gradient" if abs(gi) > threshold else "VANISHING / GAP"
        print(f"{xi:6.1f} | {ui:8.4f} | {gi:10.5f} | {status}")


# ---------------------------------------------------------------------------
# Continuous-range sweep and plot
# ---------------------------------------------------------------------------

def plot_landscape(
    seq: list[int],
    centers: np.ndarray,
    sigma: float = 2.5,
    threshold: float = GRADIENT_THRESHOLD,
) -> None:
    x_max = max(centers) + 10
    xs = np.linspace(0, x_max, 2000)
    U  = soft_memory_potential(xs, centers, sigma)
    G  = analytical_gradient_U(xs, centers, sigma)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Recamán soft-memory potential  (σ={sigma}, n_steps={len(seq)-1})",
        fontsize=13,
    )

    # --- panel 1: potential ---
    ax = axes[0]
    ax.plot(xs, U, color="steelblue", lw=1.5, label="U(x)")
    ax.vlines(centers, 0, U.max(), colors="salmon", lw=0.6, alpha=0.5, label="visited")
    ax.set_ylabel("U(x)")
    ax.legend(fontsize=8)

    # --- panel 2: gradient ---
    ax = axes[1]
    ax.plot(xs, G, color="darkorange", lw=1.5, label="|dU/dx|")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("dU/dx")
    ax.legend(fontsize=8)

    # --- panel 3: vanishing-gradient mask ---
    ax = axes[2]
    vanishing = np.abs(G) < threshold
    ax.fill_between(xs, 0, 1, where=vanishing,
                    color="crimson", alpha=0.4, label="vanishing (|grad|<threshold)")
    ax.fill_between(xs, 0, 1, where=~vanishing,
                    color="mediumseagreen", alpha=0.3, label="strong gradient")
    ax.set_ylabel("gradient regime")
    ax.set_xlabel("x")
    ax.set_yticks([])
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUTPUTS_DIR / "vanishing_gradients_landscape.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved -> {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(n_steps: int = 20, sigma: float = 2.5,
         large_x_demo: float = 1e17) -> None:
    # --- small n: use set-based generator ---
    seq, visited = classical_recaman(n_steps)
    centers = np.array(sorted(visited), dtype=float)

    print(f"Recamán sequence (n_steps={n_steps}):")
    print(" ", seq)
    print(f"\nVisited set ({len(centers)} values): {sorted(visited)}")

    coverage = len(centers) / (centers.max() + 1)
    print(f"Coverage of [0, {int(centers.max())}]: {coverage:.1%}")

    # --- gradient table and landscape ---
    print_gradient_table(centers, sigma=sigma)
    plot_landscape(seq, centers, sigma=sigma)

    # --- analytical large-x demo (e.g. x = 10^17) ---
    check_large_x(large_x_demo, centers, sigma=sigma)

    # --- larger n with bitmap: show gradient horizon grows with coverage ---
    print("\n=== Gradient horizon vs n_steps ===")
    print(f"{'n_steps':>10} | {'max visited':>12} | {'|visited|':>10} | "
          f"{'coverage':>9} | {'horizon':>10}")
    print("-" * 62)
    for ns in [20, 100, 500, 2000]:
        try:
            s2, c2 = recaman_bitmap(ns)
        except MemoryError as e:
            print(f"{ns:10,} | {str(e)}")
            continue
        cov2 = len(c2) / (c2.max() + 1)
        h2 = gradient_horizon(c2, sigma=sigma)
        print(f"{ns:10,} | {int(c2.max()):12,} | {len(c2):10,} | "
              f"{cov2:9.1%} | {h2:10.2f}")


if __name__ == "__main__":
    main(n_steps=20, sigma=2.5, large_x_demo=1e17)
