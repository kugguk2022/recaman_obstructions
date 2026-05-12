#!/usr/bin/env python3
"""
markov_2x2.py
=============

Fit an honest first-order 2x2 Markov surrogate to the true Recaman
obstruction stream.

State is the previous obstruction bit:
  0 = free/down step
  1 = blocked/up step

This is a descriptive lag-1 model, not a claim that Recaman is exactly
Markovian on two states. The script reports where the fit is strong
(transition probabilities, stationary mass, phase-slip rate) and where
residual structure remains (k-block KL gap against a fitted null).
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarkovFit:
    transition: np.ndarray
    counts: np.ndarray

    @property
    def q_prev0(self) -> float:
        return float(self.transition[0, 1])

    @property
    def q_prev1(self) -> float:
        return float(self.transition[1, 1])


def generate_recaman(n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate Recaman values a_n and obstruction bits b_n up to n_terms."""
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1")

    values = np.zeros(n_terms + 1, dtype=np.int64)
    bits = np.zeros(n_terms + 1, dtype=np.int8)
    visited = {0}

    for n in range(1, n_terms + 1):
        candidate = int(values[n - 1]) - n
        if candidate > 0 and candidate not in visited:
            values[n] = candidate
        else:
            values[n] = values[n - 1] + n
            bits[n] = 1
        visited.add(int(values[n]))

    return values, bits


def fit_markov_2x2(bits: np.ndarray) -> MarkovFit:
    """
    Fit T[i, j] = Pr[b_n = j | b_{n-1} = i] for i, j in {0, 1}.

    The comparisons start at n=2 because b_0 is only a sentinel.
    """
    if len(bits) < 3:
        raise ValueError("Need at least two real obstruction bits to fit transitions")

    counts = np.zeros((2, 2), dtype=np.int64)
    for prev_bit, curr_bit in zip(bits[1:-1], bits[2:]):
        counts[int(prev_bit), int(curr_bit)] += 1

    row_totals = counts.sum(axis=1, keepdims=True)
    if np.any(row_totals == 0):
        raise ValueError("Both bit states must appear at least once to fit a 2x2 chain")

    transition = counts / row_totals
    return MarkovFit(transition=transition, counts=counts)


def empirical_distribution(bits: np.ndarray) -> np.ndarray:
    counts = np.bincount(bits[1:].astype(np.int64), minlength=2)
    return counts / counts.sum()


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """
    Closed-form stationary law for a 2-state chain.

      T[0,1] = p01
      T[1,0] = p10

    Then pi = (p10, p01) / (p01 + p10) when the denominator is nonzero.
    """
    p01 = float(transition[0, 1])
    p10 = float(transition[1, 0])
    denom = p01 + p10
    if denom == 0.0:
        raise ValueError("Stationary distribution is not unique for this degenerate chain")
    return np.array([p10 / denom, p01 / denom], dtype=np.float64)


def phase_slip_rate(bits: np.ndarray) -> tuple[float, int, int]:
    """
    Phase slip = b_n == b_{n-1} for n >= 2.

    The old version incorrectly included the sentinel comparison (b_1, b_0).
    """
    n_pairs = len(bits) - 2
    if n_pairs <= 0:
        raise ValueError("Need at least two real obstruction bits to measure slips")
    n_slips = int(np.sum(bits[2:] == bits[1:-1]))
    return n_slips / n_pairs, n_slips, n_pairs


def run_length_histogram(bits: np.ndarray) -> Counter[int]:
    """Histogram of maximal runs of identical real obstruction bits."""
    runs: Counter[int] = Counter()
    if len(bits) <= 2:
        return runs

    current = int(bits[1])
    length = 1
    for raw_value in bits[2:]:
        value = int(raw_value)
        if value == current:
            length += 1
        else:
            runs[length] += 1
            current = value
            length = 1
    runs[length] += 1
    return runs


def k_block_distribution(bits: np.ndarray, k: int) -> dict[tuple[int, ...], float]:
    """Empirical distribution of length-k bit blocks over the real stream b_1..b_N."""
    if k < 1:
        raise ValueError("k must be positive")
    if len(bits) - 1 < k:
        return {}

    counts: Counter[tuple[int, ...]] = Counter()
    for i in range(1, len(bits) - k + 1):
        block = tuple(int(x) for x in bits[i : i + k])
        counts[block] += 1

    total = sum(counts.values())
    return {block: count / total for block, count in counts.items()}


def kl_divergence(p: dict[tuple[int, ...], float], q: dict[tuple[int, ...], float], eps: float = 1e-12) -> float:
    """KL(p || q) over the union of keys, with epsilon smoothing on missing q-mass."""
    total = 0.0
    for key in set(p) | set(q):
        pk = p.get(key, 0.0)
        qk = max(q.get(key, 0.0), eps)
        if pk > 0.0:
            total += pk * math.log(pk / qk)
    return total


def simulate_markov_bits(
    n_terms: int,
    transition: np.ndarray,
    seed: int,
    initial_bit: int,
) -> np.ndarray:
    """Simulate the fitted 2x2 chain without mutating global RNG state."""
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1")

    rng = np.random.default_rng(seed)
    bits = np.zeros(n_terms + 1, dtype=np.int8)
    bits[1] = np.int8(initial_bit)
    for n in range(2, n_terms + 1):
        prev_bit = int(bits[n - 1])
        bits[n] = np.int8(rng.random() < transition[prev_bit, 1])
    return bits


def mean_scaled_value(values: np.ndarray) -> float:
    """Return mean(a_n / n) over n = 1..N."""
    n = np.arange(1, len(values), dtype=np.float64)
    return float(np.mean(values[1:] / n))


def print_run_table(real_runs: Counter[int], null_runs: Counter[int], max_len: int = 6) -> None:
    print("\nRun-length comparison (identical-bit runs)")
    print(f"{'len':>5}  {'real':>10}  {'null':>10}  {'ratio':>10}")
    for run_len in range(1, max_len + 1):
        real_count = real_runs.get(run_len, 0)
        null_count = null_runs.get(run_len, 0)
        ratio = real_count / null_count if null_count else float("inf")
        print(f"{run_len:>5}  {real_count:>10}  {null_count:>10}  {ratio:>10.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a 2x2 Markov surrogate to the true Recaman obstruction stream.")
    parser.add_argument("-N", "--terms", type=int, default=200_000, help="Number of Recaman terms to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the fitted Markov null simulation.")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Block sizes used for KL comparisons against the fitted null.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Recaman Obstruction Stream: Honest 2x2 Markov Fit")
    print(f"N = {args.terms:,}   seed = {args.seed}")
    print("=" * 72)

    values, real_bits = generate_recaman(args.terms)
    fit = fit_markov_2x2(real_bits)
    stationary = stationary_distribution(fit.transition)
    empirical = empirical_distribution(real_bits)
    slip_rate, n_slips, n_pairs = phase_slip_rate(real_bits)

    null_bits = simulate_markov_bits(
        n_terms=args.terms,
        transition=fit.transition,
        seed=args.seed,
        initial_bit=int(real_bits[1]),
    )
    null_bits_2 = simulate_markov_bits(
        n_terms=args.terms,
        transition=fit.transition,
        seed=args.seed + 1,
        initial_bit=int(real_bits[1]),
    )
    null_empirical = empirical_distribution(null_bits)
    null_slip_rate, null_slips, _ = phase_slip_rate(null_bits)

    print("\n[1] True Recaman run")
    print(f"max a_n                = {int(values.max()):,}")
    print(f"mean(a_n / n)          = {mean_scaled_value(values):.6f}")
    print(f"empirical p(b=0), p(b=1) = ({empirical[0]:.6f}, {empirical[1]:.6f})")

    print("\n[2] Fitted 2x2 transition matrix")
    print("Rows = previous bit, columns = next bit")
    print(f"counts[0->0, 0->1]     = ({fit.counts[0, 0]:,}, {fit.counts[0, 1]:,})")
    print(f"counts[1->0, 1->1]     = ({fit.counts[1, 0]:,}, {fit.counts[1, 1]:,})")
    print(f"P(next=1 | prev=0)     = {fit.q_prev0:.6f}")
    print(f"P(next=1 | prev=1)     = {fit.q_prev1:.6f}")
    print(f"|q_prev0 - q_prev1|    = {abs(fit.q_prev0 - fit.q_prev1):.6f}")

    print("\n[3] Stationary mass and phase slips")
    print(f"stationary pi(0), pi(1) = ({stationary[0]:.6f}, {stationary[1]:.6f})")
    print(f"null p(b=0), p(b=1)     = ({null_empirical[0]:.6f}, {null_empirical[1]:.6f})")
    print(f"real slip rate          = {slip_rate:.6f}  ({n_slips:,}/{n_pairs:,})")
    print(f"null slip rate          = {null_slip_rate:.6f}  ({null_slips:,}/{n_pairs:,})")

    real_runs = run_length_histogram(real_bits)
    null_runs = run_length_histogram(null_bits)
    print_run_table(real_runs, null_runs)

    print("\n[4] Residual structure beyond the 2x2 fit")
    for k in args.block_sizes:
        real_dist = k_block_distribution(real_bits, k)
        null_dist = k_block_distribution(null_bits, k)
        null_dist_2 = k_block_distribution(null_bits_2, k)
        kl_real_vs_null = kl_divergence(real_dist, null_dist)
        kl_null_vs_null = kl_divergence(null_dist, null_dist_2)
        print(
            f"k={k}: KL(real || null) = {kl_real_vs_null:.6f}   "
            f"KL(null || null) = {kl_null_vs_null:.6f}   "
            f"excess = {kl_real_vs_null - kl_null_vs_null:+.6f}"
        )

    print("\nSummary")
    print("The lag-1 2x2 model captures the near-alternating transition bias well.")
    print("Any positive KL excess above the null-vs-null baseline is residual memory")
    print("that the two-state first-order surrogate does not capture.")


if __name__ == "__main__":
    main()
