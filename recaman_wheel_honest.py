"""
recaman_wheel_honest.py
=======================

Replacement for the earlier racamman_markov.py, which had three bugs:

  (B1) Generated b_n as a Bernoulli draw from a hand-crafted state-conditional
       probability. This makes the obstruction process Markovian by construction.
       Real Recaman b_n = 1[a_{n-1}-n <= 0  or  a_{n-1}-n in V_{n-1}] is
       DETERMINISTIC given the full visited set V_{n-1}, which grows without
       bound. No finite-state Markov chain can encode it.

  (B2) Trained empirical block-probabilities on data the model itself produced,
       so the training was circular and proved nothing.

  (B3) Pretended the resulting Markov chain could PREDICT hit-probabilities
       for specific target values. Predicting "does Recaman visit m at step
       n0?" needs the actual V_{n-1}, which a finite-state shadow throws away.

This script does it honestly:

  Step 1. Run the TRUE Recaman generator with the full visited set, to N terms.
          Extract the real obstruction stream {b_n} and the real wheel state
          stream {S_n}.

  Step 2. WHEEL AS DESCRIPTOR. Measure q(w) = Pr[b_n = 1 | S_n = w] from the
          real run. This is empirical and not circular: the bits come from
          real Recaman, the conditioning is on the wheel state.

  Step 3. WHEEL AS GENERATIVE NULL. Simulate a Markov wheel using the q(w)
          measured in Step 2. Compare the synthetic obstruction statistics
          against the real ones. Any gap is exactly the memory effect the
          wheel cannot capture.

  Step 4. Report the KL divergence between the real obstruction-block
          structure and the wheel-Markov approximation. A nonzero
          divergence is the formal statement "Recaman is not Markovian on
          this state space."

Run:  python3 recaman_wheel_honest.py
"""

from __future__ import annotations
import random
from collections import Counter, defaultdict
from typing import List, Tuple


# -----------------------------------------------------------------------------
# Wheel: reverse-complement involution Theta_3 on length-6 words over {0,1,2,3}
# -----------------------------------------------------------------------------

def precompute_theta3() -> List[int]:
    """All 4096 length-6 words over {0,1,2,3}, with their Theta_3 image."""
    theta3 = [0] * 4096
    for idx in range(4096):
        digits = [(idx >> (2 * (5 - i))) & 3 for i in range(6)]
        new_digits = [3 - d for d in reversed(digits)]
        new_idx = 0
        for d in new_digits:
            new_idx = (new_idx << 2) | d
        theta3[idx] = new_idx
    return theta3


# Two-state restriction: 210 and 321 are the canonical attractors.
# In the 4096-encoding, 210 is the digit-tuple (2,1,0) repeated/padded
# We encode the minimal 2-state wheel as state in {0,1} (0 = "210", 1 = "321"),
# and the involution flips state.
def two_state_wheel_flip(s: int) -> int:
    return 1 - s


# -----------------------------------------------------------------------------
# Step 1. True Recaman generator with full memory
# -----------------------------------------------------------------------------

def recaman(N: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Generate Recaman to N terms with the full visited-set memory.
    Returns: (a, b, S)
      a[n] = the Recaman value at step n (a[0] = 0)
      b[n] = obstruction bit at step n (for n >= 1):
             1 if the rule went UP (blocked), 0 if it went DOWN (free)
      S[n] = two-state wheel state AFTER processing step n (S[0] = 0)
             Updated by: if b[n]=1 flip, else stay.
    """
    a = [0] * (N + 1)
    b = [0] * (N + 1)
    S = [0] * (N + 1)
    visited = {0}
    state = 0
    for n in range(1, N + 1):
        cand = a[n - 1] - n
        if cand > 0 and cand not in visited:
            a[n] = cand
            b[n] = 0  # free, went down
        else:
            a[n] = a[n - 1] + n
            b[n] = 1  # blocked, went up
        visited.add(a[n])
        if b[n] == 1:
            state = two_state_wheel_flip(state)
        S[n] = state
    return a, b, S


# -----------------------------------------------------------------------------
# Step 2. WHEEL AS DESCRIPTOR
#   Measure q(w) = Pr[b_n = 1 | S_{n-1} = w] from the real run.
#   We condition on the PRE-step state S_{n-1}, since b_n is what we predict
#   from the state at the moment of decision.
# -----------------------------------------------------------------------------

def measure_q_from_real(b: List[int], S: List[int]) -> Tuple[List[float], List[int], List[int]]:
    """
    Returns q[w] for w in {0,1} = Pr[b_n=1 | S_{n-1}=w], plus counts.
    """
    N = len(b) - 1
    total = [0, 0]
    blocked = [0, 0]
    for n in range(1, N + 1):
        w = S[n - 1]
        total[w] += 1
        if b[n] == 1:
            blocked[w] += 1
    q = [blocked[w] / total[w] if total[w] else 0.5 for w in (0, 1)]
    return q, blocked, total


# -----------------------------------------------------------------------------
# Step 3. WHEEL AS GENERATIVE NULL
#   Simulate the 2-state Markov wheel using q(w) measured above. The simulation
#   ignores V_{n-1} entirely; b_n is drawn as Bernoulli(q[S_{n-1}]).
# -----------------------------------------------------------------------------

def simulate_wheel_null(N: int, q: List[float], seed: int = 0) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    b = [0] * (N + 1)
    S = [0] * (N + 1)
    state = 0
    for n in range(1, N + 1):
        bn = 1 if rng.random() < q[state] else 0
        b[n] = bn
        if bn == 1:
            state = two_state_wheel_flip(state)
        S[n] = state
    return b, S


# -----------------------------------------------------------------------------
# Step 4. Compare the real obstruction structure to the Markov null
#   We score along three axes:
#     (a) bit marginal P(b=1)
#     (b) run-length distribution of consecutive bits
#     (c) k-block empirical distribution for small k (k=4)
#   For (c) we compute the KL divergence between empirical block distributions.
#   A large divergence is the formal failure of the Markov approximation.
# -----------------------------------------------------------------------------

import math

def bit_marginal(b: List[int]) -> float:
    return sum(b[1:]) / (len(b) - 1)

def run_lengths(b: List[int]) -> Counter:
    """Lengths of maximal runs of identical bits."""
    runs = Counter()
    if len(b) <= 1:
        return runs
    cur = b[1]
    cnt = 1
    for x in b[2:]:
        if x == cur:
            cnt += 1
        else:
            runs[cnt] += 1
            cur = x
            cnt = 1
    runs[cnt] += 1
    return runs

def kblock_dist(b: List[int], k: int) -> dict:
    """Empirical distribution of length-k bit blocks."""
    counts = Counter()
    for i in range(1, len(b) - k + 1):
        tup = tuple(b[i : i + k])
        counts[tup] += 1
    total = sum(counts.values())
    return {key: c / total for key, c in counts.items()} if total else {}

def kl_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    """KL(p || q) over the union of keys; missing q-mass gets eps."""
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, eps)
        if pk > 0:
            s += pk * math.log(pk / max(qk, eps))
    return s


# -----------------------------------------------------------------------------
# Main: run the honest comparison at N = 50,000 and N = 200,000
# -----------------------------------------------------------------------------

def report(N: int):
    print(f"\n{'=' * 64}")
    print(f"  N = {N:,}")
    print(f"{'=' * 64}")

    # Step 1: real Recaman
    a, b_real, S_real = recaman(N)

    print(f"\n[Step 1] True Recaman run")
    print(f"  max a_n        : {max(a):,}")
    print(f"  avg a_n        : {sum(a)/N:.1f}")
    print(f"  Pr[b=1] (real) : {bit_marginal(b_real):.4f}")

    # Step 2: measure q(w) from the real run
    q, blocked, total = measure_q_from_real(b_real, S_real)
    print(f"\n[Step 2] Wheel-as-descriptor: q(w) measured from real Recaman")
    print(f"  q(S=0)         : {q[0]:.4f}   ({blocked[0]:,} blocked / {total[0]:,} visits)")
    print(f"  q(S=1)         : {q[1]:.4f}   ({blocked[1]:,} blocked / {total[1]:,} visits)")
    print(f"  |q(0) - q(1)|  : {abs(q[0]-q[1]):.4f}   <- predictive content of wheel state")

    # Step 3: simulate the Markov null
    b_null, S_null = simulate_wheel_null(N, q, seed=42)
    print(f"\n[Step 3] Wheel-as-null: synthetic obstruction stream using measured q")
    print(f"  Pr[b=1] (null) : {bit_marginal(b_null):.4f}")

    # Step 4: compare run lengths and k-block structure
    runs_real = run_lengths(b_real)
    runs_null = run_lengths(b_null)
    max_len = max(max(runs_real, default=0), max(runs_null, default=0))
    print(f"\n[Step 4] Comparison: real vs. Markov-null")
    print(f"  Run-length distribution (top 8):")
    print(f"    {'len':>4}  {'real':>8}  {'null':>8}  {'ratio':>7}")
    for L in sorted(set(list(runs_real) + list(runs_null)))[:8]:
        r = runs_real.get(L, 0)
        n = runs_null.get(L, 0)
        ratio = r / n if n else float('inf')
        print(f"    {L:>4}  {r:>8}  {n:>8}  {ratio:>7.3f}")
    if max_len > 8:
        tail_real = sum(c for L, c in runs_real.items() if L > 8)
        tail_null = sum(c for L, c in runs_null.items() if L > 8)
        print(f"    >8   {tail_real:>8}  {tail_null:>8}  "
              f"{tail_real/tail_null if tail_null else float('inf'):>7.3f}")

    # KL divergences on k-block distributions
    print(f"  KL divergence on k-block distributions:")
    for k in (2, 4, 6, 8):
        pk = kblock_dist(b_real, k)
        qk = kblock_dist(b_null, k)
        kl = kl_divergence(pk, qk)
        # null model self-KL as a sanity baseline
        b_null2, _ = simulate_wheel_null(N, q, seed=123)
        qk2 = kblock_dist(b_null2, k)
        kl_self = kl_divergence(qk, qk2)
        print(f"    k={k}: KL(real || null) = {kl:.4f}    "
              f"KL(null1 || null2) = {kl_self:.4f}   "
              f"excess = {kl - kl_self:+.4f}")

    print(f"\n  --> Excess KL is the memory effect the wheel cannot capture.")
    print(f"      If excess KL > 0 by more than null1-null2 baseline noise,")
    print(f"      the obstruction process is provably not Markovian on the 2-state wheel.")


if __name__ == "__main__":
    report(50_000)
    report(200_000)
