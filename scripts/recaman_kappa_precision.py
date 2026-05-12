"""
recaman_kappa_precision.py
==========================

High-precision measurement of:
  (a) phase-slip rate ρ_slip = Pr[b_n ≠ b_{n-1}] at large N
  (b) median q/p ratio on hole and control sets

Then PSLQ against a constants battery, with attention to coefficient sizes
(small integer coefficients = real relation; huge coefficients = noise fit).
"""

from __future__ import annotations
import math, random
from typing import List, Tuple
import mpmath as mp

mp.mp.dps = 30


def recaman(N: int):
    a = [0] * (N + 1)
    b = [0] * (N + 1)
    visited = {0}
    for n in range(1, N + 1):
        cand = a[n - 1] - n
        if cand > 0 and cand not in visited:
            a[n] = cand; b[n] = 0
        else:
            a[n] = a[n - 1] + n; b[n] = 1
        visited.add(a[n])
    return a, b, visited


def slip_count(b):
    s = 0
    for n in range(2, len(b)):
        if b[n] != b[n - 1]:
            s += 1
    return s


def pslq_small(target_mp, constants, max_coeff=2000, tol=1e-25):
    """
    Run PSLQ but only accept results with small coefficients.
    A real algebraic/transcendental relation has integer coeffs <~ 1000.
    Larger coefficients with the same residual = numerical artifact.
    """
    print(f"  target = {target_mp}")
    found = []
    for name, val in constants.items():
        if name == "1":
            continue
        try:
            rel = mp.pslq([target_mp, val, mp.mpf(1)], maxcoeff=max_coeff, tol=tol)
        except Exception:
            rel = None
        if rel is not None and rel[0] != 0:
            a, b, c = rel
            max_abs = max(abs(int(a)), abs(int(b)), abs(int(c)))
            if max_abs <= max_coeff:
                est = -(b * val + c) / a
                err = abs(target_mp - est)
                found.append((name, (int(a), int(b), int(c)), float(err), max_abs))
    # also try 3-constant relations
    triples = [
        ("pi", "log2"), ("pi", "log3"), ("zeta2", "log2"),
        ("pi", "zeta3"), ("sqrt2", "log2"),
    ]
    for n1, n2 in triples:
        v1, v2 = constants[n1], constants[n2]
        try:
            rel = mp.pslq([target_mp, v1, v2, mp.mpf(1)], maxcoeff=max_coeff, tol=tol)
        except Exception:
            rel = None
        if rel is not None and rel[0] != 0:
            a, b, c, d = rel
            max_abs = max(abs(int(a)), abs(int(b)), abs(int(c)), abs(int(d)))
            if max_abs <= max_coeff:
                est = -(b * v1 + c * v2 + d) / a
                err = abs(target_mp - est)
                found.append((f"{n1}+{n2}", (int(a), int(b), int(c), int(d)),
                              float(err), max_abs))
    found.sort(key=lambda x: x[3])  # sort by smallest max-coefficient
    return found


def main():
    print("[step 1] high-precision phase-slip measurement at large N")
    for N in [200_000, 1_000_000, 5_000_000]:
        a, b, V = recaman(N)
        s = slip_count(b)
        rho = mp.mpf(s) / mp.mpf(N - 1)
        print(f"  N = {N:>9,}  slips = {s:>7,}  ρ_slip = {mp.nstr(rho, 20)}")
        if N == 5_000_000:
            rho_best = rho
            a_big, b_big, V_big = a, b, V

    print(f"\n[step 2] best ρ_slip estimate: {mp.nstr(rho_best, 25)}")
    # 1-sigma estimate from Bernoulli sampling
    n_trials = N - 1
    p = float(rho_best)
    se = math.sqrt(p * (1 - p) / n_trials)
    print(f"   standard error (Bernoulli) ≈ {se:.2e}")
    print(f"   95% CI: [{p - 2*se:.6f}, {p + 2*se:.6f}]")

    constants = {
        "1": mp.mpf(1),
        "pi": mp.pi,
        "e": mp.e,
        "phi": (1 + mp.sqrt(5)) / 2,
        "sqrt2": mp.sqrt(2),
        "sqrt3": mp.sqrt(3),
        "sqrt5": mp.sqrt(5),
        "log2": mp.log(2),
        "log3": mp.log(3),
        "ln_phi": mp.log((1 + mp.sqrt(5)) / 2),
        "zeta2": mp.zeta(2),
        "zeta3": mp.zeta(3),
        "catalan": mp.catalan,
        "euler_gamma": mp.euler,
    }

    print(f"\n[step 3] PSLQ on ρ_slip with small-coefficient filter (max=2000)")
    found = pslq_small(rho_best, constants, max_coeff=2000)
    if not found:
        print("  no small-coefficient relations found → ρ_slip is unlikely to be a")
        print("  simple algebraic/transcendental combination of these constants.")
    else:
        print(f"  {len(found)} candidate relations with max-coeff ≤ 2000:")
        for name, coeffs, err, mxc in found[:10]:
            print(f"    vs {name:18s}  coeffs={coeffs}  max_abs={mxc}  err={err:.2e}")

    # also check the trivial 1/ρ
    inv = 1 / rho_best
    print(f"\n[step 4] is 1/ρ_slip ≈ {mp.nstr(inv, 15)} a recognisable constant?")
    found_inv = pslq_small(inv, constants, max_coeff=2000)
    if found_inv:
        for name, coeffs, err, mxc in found_inv[:5]:
            print(f"    vs {name:18s}  coeffs={coeffs}  max_abs={mxc}  err={err:.2e}")

    # κ_defect measurement using the median q/p ratio (robust)
    print(f"\n[step 5] median q/p ratio on Recaman holes vs controls at N=5M")
    max_val = max(a_big)
    visited = V_big
    candidates_unvis = [h for h in range(1, max_val + 1) if h not in visited]
    visited_list = sorted(visited)
    rng = random.Random(7)
    holes_sample = rng.sample(candidates_unvis, min(100_000, len(candidates_unvis)))
    nonholes_sample = rng.sample(visited_list, min(100_000, len(visited_list)))

    def u_p(h, p):
        r = h % p
        return min(r, p - r)
    def X_(h): return 1 + u_p(h, 2)
    def Y_(h): return 1 + u_p(h, 3)
    def Z_(h):
        d = [int(c) for c in str(h)]
        s = 1.0
        for i in range(len(d)):
            for j in range(i+1, len(d)):
                if d[i] == d[j]:
                    s += 2.0 ** -(j - i)
        return s
    def ratio(h):
        X, Y, Z = X_(h), Y_(h), Z_(h)
        return (X*X + Y*Y + Z*Z) / (X*Y*Z)

    rh = sorted(ratio(h) for h in holes_sample)
    rc = sorted(ratio(h) for h in nonholes_sample)
    med_h = rh[len(rh)//2]
    med_c = rc[len(rc)//2]
    print(f"  median q/p on holes   = {med_h:.10f}")
    print(f"  median q/p on control = {med_c:.10f}")
    print(f"  difference            = {med_h - med_c:+.6f}")

    print(f"\n[step 6] PSLQ on median κ for holes")
    found_h = pslq_small(mp.mpf(med_h), constants, max_coeff=200)
    if not found_h:
        print("  no small-coefficient relations (max_coeff ≤ 200) for hole κ")
    else:
        for name, coeffs, err, mxc in found_h[:5]:
            print(f"    vs {name:18s}  coeffs={coeffs}  max_abs={mxc}  err={err:.2e}")


if __name__ == "__main__":
    main()
