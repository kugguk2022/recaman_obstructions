#!/usr/bin/env python3
"""
recaman_obstruction_scorer.py
=============================

Focused on VALID OBSTRUCTIONS (Chaffin holes / missing numbers).

This script:
- Implements the 42 features used in the repo
- Loads the top 5 strongest linear formulas from the random search
- Scores any integer and returns how "obstruction-like" it is
- Includes a small validation demo on real holes vs controls

Run: python3 recaman_obstruction_scorer.py
"""

import numpy as np
from collections import Counter
import math

# ---------------------------------------------------------------------------
# 42 Features (exact same as in the repo)
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def encode_number(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(42)

    s = str(n)
    digits = [int(d) for d in s]
    length = len(digits)
    digit_sum = sum(digits)
    alt_sum = sum(d * (1 if i % 2 == 0 else -1) for i, d in enumerate(digits))
    first_digit = digits[0]
    last1 = digits[-1]
    last2 = int(s[-2:]) if length >= 2 else last1
    last3 = int(s[-3:]) if length >= 3 else last2

    digit_counts = [digits.count(d) for d in range(10)]
    even_count = sum(digit_counts[d] for d in [0,2,4,6,8])
    odd_count = sum(digit_counts[d] for d in [1,3,5,7,9])
    c321 = digit_counts[1] + digit_counts[2] + digit_counts[3]
    c210 = digit_counts[0] + digit_counts[1] + digit_counts[2]

    basis_lift_3 = 3 * c321 + 2 * c210
    basis_lift_2 = 2 * c321 + 1 * c210
    basis_lift_1 = 1 * c321 + 0 * c210

    mods = [n % p for p in PRIMES]

    eq_3_0 = 1 if digit_counts[3] == digit_counts[0] else 0
    eq_2_0 = 1 if digit_counts[2] == digit_counts[0] else 0
    eq_1_0 = 1 if digit_counts[1] == digit_counts[0] else 0
    eq_321_210 = 1 if c321 == c210 else 0
    digit_sum_mod3_zero = 1 if digit_sum % 3 == 0 else 0
    alt_sum_mod3_zero = 1 if alt_sum % 3 == 0 else 0

    features = [
        length, digit_sum, alt_sum, first_digit, last1, last2, last3,
        len(set(digits)), even_count, odd_count,
        c321, c210, basis_lift_3, basis_lift_2, basis_lift_1
    ] + digit_counts + mods + [
        eq_3_0, eq_2_0, eq_1_0, eq_321_210,
        digit_sum_mod3_zero, alt_sum_mod3_zero
    ]
    return np.array(features, dtype=float)

# ---------------------------------------------------------------------------
# Top 5 Linear Formulas (from your JSON - strongest ones)
# ---------------------------------------------------------------------------

FORMULAS = {
    "F1 (best)": lambda f: (
        -3*f[35] -3*f[36] +3*f[23] +3*f[21] +3*f[20] +3*f[19] -3*f[18] +3*f[26] +
        3*f[29] +3*f[1] +3*f[9] +3*f[3] -3*f[0] -3*f[13] -3*f[14] +2*f[37] -2*f[34] -2*f[2] -2*f[16] +
        2*f[11] -2*f[15] -2*f[27] -2*f[17] -1*f[35] +1*f[5] -1*f[12] +1*f[30] +1*f[28] +1*f[24] +
        1*f[8] -1*f[4] +1*f[22] +1*f[12]
    ),
    "F2": lambda f: (
        3*f[28] +3*f[37] +3*f[17] -3*f[24] +3*f[19] +3*f[16] +3*f[25] +3*f[21] -
        2*f[29] -2*f[22] +2*f[5] +2*f[7] +2*f[4] -2*f[0] +2*f[8] +2*f[9] -2*f[12] -2*f[15] -2*f[20] -2*f[21] -
        1*f[6] -1*f[30] +1*f[34] -1*f[31] -1*f[35] +1*f[36] -1*f[2] -1*f[11] -1*f[13] +1*f[18] -1*f[27] +
        1*f[26] +1*f[24] -1*f[23] -1*f[13]
    ),
    "F3": lambda f: (
        3*f[30] +3*f[35] -3*f[22] +3*f[21] +3*f[20] +3*f[18] -3*f[26] -3*f[16] -3*f[25] -3*f[27] -
        3*f[11] -3*f[9] -3*f[2] -3*f[3] -3*f[14] +3*f[12] +2*f[36] -2*f[37] -2*f[0] -2*f[4] +2*f[31] +
        2*f[30] -2*f[37] -2*f[34] -2*f[23] -2*f[6] +2*f[19] -2*f[18] +2*f[29] -2*f[13] -1*f[35] +1*f[8] -
        1*f[24] +1*f[15] +1*f[24] -1*f[12] -1*f[7]
    ),
    "F4": lambda f: (
        -3*f[35] -3*f[34] -3*f[31] -3*f[28] -3*f[30] -3*f[11] +3*f[22] -3*f[16] +3*f[26] -3*f[21] +
        3*f[5] +3*f[3] -3*f[14] +3*f[15] -3*f[12] -2*f[37] -2*f[12] -2*f[8] +2*f[4] -2*f[18] -2*f[24] +
        2*f[0] +2*f[23] +1*f[30] +1*f[37] +1*f[36] +1*f[1] -1*f[29] -1*f[19] +1*f[16] -1*f[20] +1*f[25] +
        1*f[9] -1*f[24] +1*f[21] +1*f[13] -1*f[7] +1*f[12]
    ),
    "F5": lambda f: (
        3*f[35] +3*f[6] -3*f[1] -3*f[0] -3*f[9] +3*f[11] +3*f[17] -3*f[24] +3*f[26] +3*f[8] -
        3*f[4] +3*f[2] +3*f[3] +3*f[12] +2*f[25] -2*f[31] +2*f[37] -2*f[36] -2*f[29] +2*f[16] -2*f[19] +
        2*f[20] +2*f[28] +2*f[22] +1*f[34] -1*f[5] -1*f[30] -1*f[37] +1*f[31] -1*f[21] -1*f[24] -1*f[18] +
        1*f[21] -1*f[12] +1*f[7] -1*f[15] +1*f[0]
    )
}

def score_number(n: int, formula_name: str = "F1 (best)") -> float:
    """Score a number using one of the top formulas. Higher = more obstruction-like."""
    feats = encode_number(n)
    return FORMULAS[formula_name](feats)

# ---------------------------------------------------------------------------
# Demo: Score some real holes vs random controls
# ---------------------------------------------------------------------------

def demo():
    print("=" * 70)
    print("Recamán Obstruction Scorer — Valid Holes vs Controls")
    print("=" * 70)

    # Small set of known holes (from Chaffin list, 6-10 digits)
    holes = [852655, 1000003, 1234567, 2000000, 3141592, 5000000, 9876543, 10000000]
    controls = [852656, 1000004, 1234568, 2000001, 3141593, 5000001, 9876544, 10000001]

    print("\nScoring with best formula (F1):")
    print(f"{'Number':>12}  {'Type':<10}  {'Score':>10}")
    print("-" * 40)

    hole_scores = []
    control_scores = []

    for h in holes:
        s = score_number(h)
        hole_scores.append(s)
        print(f"{h:>12}  {'HOLE':<10}  {s:>10.3f}")

    for c in controls:
        s = score_number(c)
        control_scores.append(s)
        print(f"{c:>12}  {'CONTROL':<10}  {s:>10.3f}")

    print("\n" + "-" * 40)
    print(f"Mean HOLE score   : {np.mean(hole_scores):.3f}")
    print(f"Mean CONTROL score: {np.mean(control_scores):.3f}")
    print(f"Separation        : {np.mean(hole_scores) - np.mean(control_scores):.3f}")

    # Simple AUC estimate (very small sample)
    from sklearn.metrics import roc_auc_score
    y_true = [1]*len(holes) + [0]*len(controls)
    y_score = hole_scores + control_scores
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"Simple AUC (8+8)  : {auc:.3f}")
    except:
        print("AUC not computed (sklearn not available)")

if __name__ == "__main__":
    demo()