"""
recaman_modm_scan.py
====================

Scan: which choice of state separates the obstruction bit b_n best?

We measure conditional mutual information between b_n and various candidate
states, computed on the real Recaman run.

Candidates:
  - S_{n-1}     : the 2-state wheel  (baseline; expected near 0)
  - a_{n-1} mod m  for m in a grid
  - n mod m        for the same grid
  - (a_{n-1} mod m, n mod m)  joint
  - a_{n-1} mod m for m = n   (trivially perfect; sanity check, not a model)

For each candidate state X, compute:
    I(b ; X) = H(b) - H(b | X)
where everything is empirical Shannon entropy.

I(b;X) = 0 means X tells you nothing about b beyond the marginal.
I(b;X) close to H(b) ~ 1 bit means X almost determines b.

This is the right metric: it is invariant under relabelling of states and
directly answers "does conditioning on X help predict b?".
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import List, Tuple


# ---------------------------------------------------------------- Recaman -----

def recaman(N: int) -> Tuple[List[int], List[int], List[int]]:
    a = [0] * (N + 1)
    b = [0] * (N + 1)
    S = [0] * (N + 1)
    visited = {0}
    state = 0
    for n in range(1, N + 1):
        cand = a[n - 1] - n
        if cand > 0 and cand not in visited:
            a[n] = cand
            b[n] = 0
        else:
            a[n] = a[n - 1] + n
            b[n] = 1
        visited.add(a[n])
        if b[n] == 1:
            state = 1 - state
        S[n] = state
    return a, b, S


# ---------------------------------------------------------- Entropy helpers ---

def entropy(counts) -> float:
    """Shannon entropy in bits of a Counter / dict of counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def mutual_information(b: List[int], x: List[int]) -> Tuple[float, float, float]:
    """
    Return (H(b), H(b|x), I(b;x)) in bits.
    b and x are aligned lists of equal length, no leading zero.
    """
    Hb = entropy(Counter(b))
    # H(b | x) = sum_x P(x) H(b | X=x)
    by_x = defaultdict(Counter)
    for bi, xi in zip(b, x):
        by_x[xi][bi] += 1
    total = len(b)
    H_b_given_x = 0.0
    for xi, sub in by_x.items():
        w = sum(sub.values()) / total
        H_b_given_x += w * entropy(sub)
    return Hb, H_b_given_x, Hb - H_b_given_x


# ---------------------------------------------------- Per-state diagnostics ---

def per_state_block_rate(b: List[int], x: List[int], top_k: int = 6) -> List[Tuple[int, int, int, float]]:
    """
    Return list of (state, total_visits, blocked_count, block_rate),
    sorted by total_visits descending, truncated to top_k.
    """
    by_x = defaultdict(lambda: [0, 0])  # [total, blocked]
    for bi, xi in zip(b, x):
        by_x[xi][0] += 1
        if bi == 1:
            by_x[xi][1] += 1
    rows = [(xi, tot, blk, blk / tot if tot else 0.0) for xi, (tot, blk) in by_x.items()]
    rows.sort(key=lambda r: -r[1])
    return rows[:top_k]


# ----------------------------------------------------------- Main scan --------

def scan(N: int, m_grid: List[int]):
    print(f"\n{'=' * 70}")
    print(f"  Conditional information scan, N = {N:,}")
    print(f"{'=' * 70}")

    a, b, S = recaman(N)
    # Use bits 1..N
    b_arr = b[1:]
    Hb = entropy(Counter(b_arr))
    print(f"\n  H(b) marginal entropy = {Hb:.4f} bits  (max 1.0)")
    print(f"  P(b=1) = {sum(b_arr)/N:.4f}")
    print()

    # --- Baseline: 2-state wheel using S_{n-1} ---
    x_wheel = [S[n - 1] for n in range(1, N + 1)]
    Hb_, H_cond, I = mutual_information(b_arr, x_wheel)
    print(f"  Candidate: S_{{n-1}} (2-state wheel)")
    print(f"    states used = 2,  I(b ; X) = {I:.6f} bits   ({100*I/Hb:.3f}% of H(b))")

    # --- a_{n-1} mod m ---
    print(f"\n  Candidate: a_{{n-1}} mod m")
    print(f"    {'m':>5}  {'states':>7}  {'I(b;X)':>10}  {'%H(b)':>7}")
    results_a = []
    for m in m_grid:
        x = [a[n - 1] % m for n in range(1, N + 1)]
        _, _, I = mutual_information(b_arr, x)
        used = len(set(x))
        results_a.append((m, used, I))
        print(f"    {m:>5}  {used:>7}  {I:>10.4f}  {100*I/Hb:>6.2f}%")

    # --- n mod m ---
    print(f"\n  Candidate: n mod m")
    print(f"    {'m':>5}  {'states':>7}  {'I(b;X)':>10}  {'%H(b)':>7}")
    for m in m_grid:
        x = [n % m for n in range(1, N + 1)]
        _, _, I = mutual_information(b_arr, x)
        used = len(set(x))
        print(f"    {m:>5}  {used:>7}  {I:>10.4f}  {100*I/Hb:>6.2f}%")

    # --- joint (a_{n-1} mod m, n mod m) ---
    print(f"\n  Candidate: (a_{{n-1}} mod m, n mod m) joint")
    print(f"    {'m':>5}  {'states':>7}  {'I(b;X)':>10}  {'%H(b)':>7}")
    for m in m_grid:
        x = [(a[n - 1] % m) * m + (n % m) for n in range(1, N + 1)]
        _, _, I = mutual_information(b_arr, x)
        used = len(set(x))
        print(f"    {m:>5}  {used:>7}  {I:>10.4f}  {100*I/Hb:>6.2f}%")

    # --- best a_{n-1} mod m: show per-state block rates ---
    best_m = max(results_a, key=lambda r: r[2])[0]
    print(f"\n  Top-6 most-visited residues for best m = {best_m} (a_{{n-1}} mod m):")
    rows = per_state_block_rate(b_arr, [a[n - 1] % best_m for n in range(1, N + 1)], top_k=8)
    print(f"    {'r':>5}  {'visits':>8}  {'blocked':>8}  {'rate':>7}")
    for r, tot, blk, rate in rows:
        print(f"    {r:>5}  {tot:>8}  {blk:>8}  {rate:>7.4f}")


if __name__ == "__main__":
    m_grid = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]
    scan(50_000, m_grid)
    scan(200_000, m_grid)
