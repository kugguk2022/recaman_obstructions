"""
recaman_heldout.py
==================

The mod-m scan suggested high I(b;X) for large m, but most of that is overfit:
when state cardinality approaches sample size, each state is visited 1-2 times
and H(b|X=x) collapses to zero artificially.

This script does the right test: split the run into a train half and a test
half. Fit q(x) = Pr[b=1 | X=x] on train. Score log-likelihood on test.
Compare against the marginal-only baseline q = P(b=1).

If the conditional model beats the marginal on held-out test data, the
state X carries real predictive information about b. Otherwise it was overfit.

Metric: per-bit log-likelihood improvement over marginal, in bits.
    delta = (LL_cond - LL_marginal) / N_test
A positive delta means real predictive value; zero means no help; negative
means the state actively hurts on held-out data.
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Tuple, Callable


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


def fit_q(b_train: List[int], x_train: List[int], smoothing: float = 1.0):
    """
    Fit q(x) = Pr[b=1 | X=x] with add-smoothing.
    Returns a dict and a fallback (marginal).
    """
    counts = defaultdict(lambda: [0.0, 0.0])  # [n_zero, n_one]
    for bi, xi in zip(b_train, x_train):
        counts[xi][bi] += 1.0
    q = {}
    for xi, (n0, n1) in counts.items():
        q[xi] = (n1 + smoothing) / (n0 + n1 + 2 * smoothing)
    marginal = sum(b_train) / len(b_train)
    return q, marginal


def loglik_per_bit(b: List[int], x: List[int], q: dict, fallback: float) -> float:
    """Average per-bit log-likelihood (in bits) under model q."""
    total = 0.0
    for bi, xi in zip(b, x):
        p1 = q.get(xi, fallback)
        # Avoid log(0) with tiny clip
        p1 = min(max(p1, 1e-9), 1 - 1e-9)
        if bi == 1:
            total += math.log2(p1)
        else:
            total += math.log2(1 - p1)
    return total / len(b)


def loglik_marginal(b: List[int], p1: float) -> float:
    p1 = min(max(p1, 1e-9), 1 - 1e-9)
    return (sum(b) * math.log2(p1) + (len(b) - sum(b)) * math.log2(1 - p1)) / len(b)


def evaluate(name: str, b_train: List[int], x_train: List[int],
             b_test: List[int], x_test: List[int]):
    q, fallback = fit_q(b_train, x_train)
    ll_cond = loglik_per_bit(b_test, x_test, q, fallback)
    ll_marg = loglik_marginal(b_test, fallback)
    delta = ll_cond - ll_marg
    n_states_train = len(set(x_train))
    n_states_test = len(set(x_test))
    overlap = len(set(x_train) & set(x_test))
    print(f"  {name:<42}  states_train={n_states_train:>6}  "
          f"test_seen={overlap:>6}  delta = {delta:+.6f} bits/bit")


def run(N: int):
    print(f"\n{'=' * 76}")
    print(f"  Held-out validation, N = {N:,}  (train: 1..{N//2:,}  test: {N//2+1:,}..{N:,})")
    print(f"{'=' * 76}\n")
    a, b, S = recaman(N)
    half = N // 2

    # Train and test halves
    b_train = b[1:half + 1]
    b_test  = b[half + 1:N + 1]

    def x_for(fn: Callable[[int], int], lo: int, hi: int):
        return [fn(n) for n in range(lo, hi + 1)]

    print(f"  Marginal: P(b=1) on train = {sum(b_train)/len(b_train):.4f},  "
          f"on test = {sum(b_test)/len(b_test):.4f}\n")

    # Baselines
    evaluate("wheel S_{n-1}",
             b_train, [S[n-1] for n in range(1, half+1)],
             b_test,  [S[n-1] for n in range(half+1, N+1)])

    print()
    print("  a_{n-1} mod m:")
    for m in [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 4096]:
        evaluate(f"  m = {m}",
                 b_train, x_for(lambda n: a[n-1] % m, 1, half),
                 b_test,  x_for(lambda n: a[n-1] % m, half+1, N))

    print()
    print("  n mod m:")
    for m in [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128]:
        evaluate(f"  m = {m}",
                 b_train, x_for(lambda n: n % m, 1, half),
                 b_test,  x_for(lambda n: n % m, half+1, N))

    print()
    print("  joint (a_{n-1} mod m, n mod m):")
    for m in [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256]:
        evaluate(f"  m = {m}",
                 b_train, x_for(lambda n: (a[n-1] % m) * m + (n % m), 1, half),
                 b_test,  x_for(lambda n: (a[n-1] % m) * m + (n % m), half+1, N))

    print()
    print("  Other candidates:")
    # last-k bits of b history as state
    for k in [1, 2, 3, 4, 8, 16]:
        def hist_state(n, k=k):
            if n - k < 1:
                return -1
            v = 0
            for j in range(k):
                v = (v << 1) | b[n - 1 - j]
            return v
        evaluate(f"  last-{k} bits of b",
                 b_train, x_for(hist_state, 1, half),
                 b_test,  x_for(hist_state, half+1, N))

    # n mod 2 AND last-1 bit
    evaluate("  (n mod 2, b_{n-1})",
             b_train, x_for(lambda n: (n % 2) * 2 + (b[n-1] if n > 1 else 0), 1, half),
             b_test,  x_for(lambda n: (n % 2) * 2 + (b[n-1] if n > 1 else 0), half+1, N))

    # |V_{n-1}| / n  bucketed -- proxy for density
    # We track the visited-set fraction in low region [0, a_{n-1}]
    # Build it incrementally
    print()
    print("  Density proxies:")

    visited_in_low_region_train_by_bucket = []
    visited_in_low_region_test_by_bucket = []
    visited = set()
    for n in range(1, N + 1):
        visited.add(a[n])

    # Bucket the visited-set density rho_n = |V_{n-1} cap [0, a_{n-1}]| / max(a_{n-1}, 1)
    # Approximate via |V_{n-1}| / max a so far -- since V_{n-1} = first n-1 values
    # |V_{n-1}| = n-1 (visited grows by 1 each step in Recaman: no repeats)
    # So rho_n = (n-1) / max_{k < n} a_k -- this is the "fill density"
    
    # Recompute on the fly
    max_so_far = [0] * (N + 1)
    a_so_far = 0
    for n in range(1, N + 1):
        if a[n] > a_so_far:
            a_so_far = a[n]
        max_so_far[n] = a_so_far

    # rho_n = (n-1)/max_so_far[n-1] is the fill ratio just before step n
    def rho_bucket(n, n_buckets=10):
        if n <= 1 or max_so_far[n-1] == 0:
            return 0
        rho = (n - 1) / max_so_far[n - 1]
        # Bucket into 0..n_buckets-1
        return min(int(rho * n_buckets), n_buckets - 1)

    for n_buckets in [4, 10, 20]:
        evaluate(f"  rho_n bucketed into {n_buckets}",
                 b_train, [rho_bucket(n, n_buckets) for n in range(1, half+1)],
                 b_test,  [rho_bucket(n, n_buckets) for n in range(half+1, N+1)])


if __name__ == "__main__":
    run(50_000)
    run(200_000)
