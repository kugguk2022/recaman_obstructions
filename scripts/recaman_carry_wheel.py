"""
recaman_carry_wheel.py
======================
Self-contained analysis of the Recamán *carry wheel*:

  - y_n  = a_n - a_{n-1}  (signed step: +n if blocked/up, -n if free/down)
  - gap  = |y_n| = n      (step size)
  - carry_n(n)  = v_10(n+1)   (10-adic valuation: how many base-10 carries
                                occur when incrementing n)
  - demand ∈ {0, 1, 2}:
      0 = reset   (gap == 1, i.e. n=1)
      1 = expansion  (y > 0, step went up)
      2 = contraction (y < 0, step went down)

  wheel_state = (demand, carry_n)

Four plots are saved to outputs/:
  1. Demand distribution by carry_n depth
  2. P(contraction) vs carry_n depth   (monotone rise is the key finding)
  3. Wheel-state transition heatmap    (top states, row-normalised)
  4. Carry-depth frequency distribution for n vs a_n
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 1_000_000
BASE = 10


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def build_events(n_steps: int) -> pd.DataFrame:
    """Run the Recamán sequence and return one row per step."""
    a = 0
    visited: set[int] = {0}
    rows: list[dict] = []
    for n in range(1, n_steps + 1):
        cand = a - n
        if cand > 0 and cand not in visited:
            a_new = cand
        else:
            a_new = a + n
        visited.add(a_new)
        rows.append({"n": n, "a_n": a_new, "y": a_new - a})
        a = a_new
    df = pd.DataFrame(rows)
    df["gap"] = df["y"].abs()          # == n by definition
    return df


def v_base(x: int, b: int = BASE) -> int:
    """b-adic valuation of x: largest k s.t. b^k | x. Returns 0 for x=0."""
    x = abs(int(x))
    if x == 0:
        return 0
    k = 0
    while x % b == 0:
        k += 1
        x //= b
    return k


def carry_depth(x: int, b: int = BASE) -> int:
    """Number of carries when computing x + 1 in base b = v_b(x + 1)."""
    return v_base(int(x) + 1, b)


def classify_demand(y: int, gap: int) -> int:
    if gap == 1:
        return 0   # reset (step 1)
    if y > 0:
        return 1   # expansion
    return 2       # contraction


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------

print(f"Generating Recamán sequence to n={N_STEPS:,} ...")
df = build_events(N_STEPS)

df["demand"]     = df.apply(lambda r: classify_demand(r["y"], r["gap"]), axis=1)
df["carry_n"]    = df["n"].apply(carry_depth)
df["carry_a"]    = df["a_n"].apply(carry_depth)
df["wheel_state"] = list(zip(df["demand"], df["carry_n"]))

# Transition table
df["next_state"] = df["wheel_state"].shift(-1)
transitions = (
    df.dropna(subset=["next_state"])
      .groupby(["wheel_state", "next_state"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

print("\nTop-20 wheel-state transitions:")
print(transitions.head(20).to_string(index=False))

print("\nP(demand type) by carry_n depth:")
print(f"{'carry_n':>8} | {'count':>8} | {'P(reset)':>9} | {'P(expand)':>10} | {'P(contract)':>12}")
print("-" * 58)
for c in range(6):
    sub = df[df["carry_n"] == c]
    if len(sub) == 0:
        continue
    print(f"{c:8d} | {len(sub):8,} | "
          f"{(sub['demand']==0).mean():9.4f} | "
          f"{(sub['demand']==1).mean():10.4f} | "
          f"{(sub['demand']==2).mean():12.4f}")


# ---------------------------------------------------------------------------
# Plot 1 — Demand distribution by carry_n
# ---------------------------------------------------------------------------

max_carry = int(df["carry_n"].max())
carry_vals = range(max_carry + 1)
p_reset    = [df[df["carry_n"]==c]["demand"].eq(0).mean() for c in carry_vals]
p_expand   = [df[df["carry_n"]==c]["demand"].eq(1).mean() for c in carry_vals]
p_contract = [df[df["carry_n"]==c]["demand"].eq(2).mean() for c in carry_vals]

x = np.arange(max_carry + 1)
width = 0.28

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, p_reset,    width, label="reset (d=0)",      color="steelblue")
ax.bar(x,         p_expand,   width, label="expansion (d=1)",  color="mediumseagreen")
ax.bar(x + width, p_contract, width, label="contraction (d=2)", color="tomato")
ax.set_xlabel("carry_n depth  v₁₀(n+1)")
ax.set_ylabel("proportion")
ax.set_title(f"Demand distribution by carry depth  (n={N_STEPS:,})")
ax.set_xticks(x)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out1 = OUTPUTS_DIR / "carry_wheel_demand_by_depth.png"
plt.savefig(out1, dpi=150)
print(f"\nPlot 1 -> {out1}")
plt.show()
plt.close()


# ---------------------------------------------------------------------------
# Plot 2 — P(contraction) vs carry_n  (the key monotone signal)
# ---------------------------------------------------------------------------

depths       = [c for c in carry_vals if len(df[df["carry_n"]==c]) >= 10]
p_cont_exact = [df[df["carry_n"]==c]["demand"].eq(2).mean() for c in depths]
counts       = [len(df[df["carry_n"]==c]) for c in depths]
# 95% Wilson CI half-width
ci = [1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0
      for p, n in zip(p_cont_exact, counts)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(depths, p_cont_exact, yerr=ci, fmt="o-", color="tomato",
            capsize=4, lw=2, markersize=7, label="P(contraction) ± 95% CI")
ax.set_xlabel("carry_n depth  v₁₀(n+1)")
ax.set_ylabel("P(contraction)")
ax.set_title(f"P(contraction | carry depth)  (n={N_STEPS:,})")
ax.set_xticks(depths)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
out2 = OUTPUTS_DIR / "carry_wheel_pcontraction_vs_depth.png"
plt.savefig(out2, dpi=150)
print(f"Plot 2 -> {out2}")
plt.show()
plt.close()


# ---------------------------------------------------------------------------
# Plot 3 — Wheel-state transition heatmap  (top states, row-normalised)
# ---------------------------------------------------------------------------

# Keep top-k states by frequency
K = 12
top_states = (
    df["wheel_state"].value_counts().head(K).index.tolist()
)
mask = (df["wheel_state"].isin(top_states) &
        df["next_state"].isin(top_states))
sub_tr = df[mask].copy()

state_labels = [str(s) for s in top_states]
state_idx    = {s: i for i, s in enumerate(top_states)}

mat = np.zeros((K, K))
for _, row in sub_tr.iterrows():
    i = state_idx[row["wheel_state"]]
    j = state_idx[row["next_state"]]
    mat[i, j] += 1

row_sums = mat.sum(axis=1, keepdims=True)
mat_norm = np.divide(mat, row_sums, where=row_sums > 0)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mat_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="transition probability")
ax.set_xticks(range(K)); ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(K)); ax.set_yticklabels(state_labels, fontsize=8)
ax.set_xlabel("next state  (demand, carry_n)")
ax.set_ylabel("current state  (demand, carry_n)")
ax.set_title(f"Carry-wheel transition heatmap (top-{K} states, row-normalised)")
for i in range(K):
    for j in range(K):
        if mat_norm[i, j] > 0.05:
            ax.text(j, i, f"{mat_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
plt.tight_layout()
out3 = OUTPUTS_DIR / "carry_wheel_transition_heatmap.png"
plt.savefig(out3, dpi=150)
print(f"Plot 3 -> {out3}")
plt.show()
plt.close()


# ---------------------------------------------------------------------------
# Plot 4 — Carry-depth frequency: n vs a_n
# ---------------------------------------------------------------------------

carry_counts_n  = df["carry_n"].value_counts().sort_index()
carry_counts_a  = df["carry_a"].value_counts().sort_index()
all_depths = sorted(set(carry_counts_n.index) | set(carry_counts_a.index))

cn = np.array([carry_counts_n.get(d, 0) for d in all_depths], dtype=float)
ca = np.array([carry_counts_a.get(d, 0) for d in all_depths], dtype=float)
cn /= cn.sum()
ca /= ca.sum()

x2 = np.arange(len(all_depths))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x2 - 0.2, cn, 0.4, label="carry_n  v₁₀(n+1)", color="steelblue", alpha=0.8)
ax.bar(x2 + 0.2, ca, 0.4, label="carry_a  v₁₀(a_n+1)", color="darkorange", alpha=0.8)
ax.set_xlabel("carry depth")
ax.set_ylabel("relative frequency")
ax.set_title(f"Carry-depth frequency: step index n vs sequence value a_n  (n={N_STEPS:,})")
ax.set_xticks(x2); ax.set_xticklabels(all_depths)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out4 = OUTPUTS_DIR / "carry_wheel_depth_frequency.png"
plt.savefig(out4, dpi=150)
print(f"Plot 4 -> {out4}")
plt.show()
plt.close()

print("\nDone. All plots written to outputs/.")
