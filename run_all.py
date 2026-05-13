#!/usr/bin/env python
"""
run_all.py — one-click reproducibility script for claude_racaman.

Runs every analysis in dependency order and writes all outputs to outputs/.

Usage:
    python run_all.py              # full run
    python run_all.py --dry-run    # print commands without executing
    python run_all.py --skip wheel # skip a named step (can repeat)

Steps (in order):
  1. version_c     — event-structured obstruction modeling (AUC dataset D)
  2. randmat        — random matched-control feature search (42-dim)
  3. wheel          — wheel / phase-slip validation
  4. real_vs_fake   — real-vs-fake sequence sweep + AUC summary
  5. seq_dist       — large-sample sequence distribution histogram
  6. phase_3d       — 3D phase-space arc-lift render
  7. carry_wheel    — carry-wheel analysis and plots (n=1,000,000)
  8. vanishing      — soft-memory potential + gradient-horizon analysis
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"
OBSTRUCTIONS = ROOT / "obstructions.txt"

STEPS: list[tuple[str, list[str]]] = [
    (
        "version_c",
        [
            sys.executable, str(SCRIPTS / "321_210_version_c.py"),
            "--input-file", str(OBSTRUCTIONS),
            "--datasets", "ABCD",
            "--save-json", str(OUTPUTS / "version_c_obstructions_results.json"),
        ],
    ),
    (
        "randmat",
        [
            sys.executable, str(SCRIPTS / "321_210_randmat.py"),
            "--input-file", str(OBSTRUCTIONS),
            "--controls-per-positive", "1",
            "--save-best-file", str(OUTPUTS / "best_obstructions_random.json"),
        ],
    ),
    (
        "wheel",
        [
            sys.executable, str(SCRIPTS / "recaman_wheel_validator.py"),
        ],
    ),
    (
        "real_vs_fake",
        [
            sys.executable, str(SCRIPTS / "recaman_real_vs_fake_sweep.py"),
            "--save-json", str(OUTPUTS / "recaman_real_vs_fake_sweep.json"),
        ],
    ),
    (
        "auc_summary",
        [
            sys.executable, str(SCRIPTS / "recaman_real_vs_fake_auc_summary.py"),
            "--output", str(OUTPUTS / "recaman_real_vs_fake_auc_summary.csv"),
        ],
    ),
    (
        "seq_dist",
        [
            sys.executable, str(SCRIPTS / "recaman_seq_distribution.py"),
        ],
    ),
    (
        "phase_3d",
        [
            sys.executable, str(SCRIPTS / "recaman_phase_space_3d.py"),
            "--steps", "2800",
            "--mode", "arc-lift",
            "--twist", "1.8",
            "--save", str(OUTPUTS / "recaman_phase_arc_readme.png"),
        ],
    ),
    (
        "carry_wheel",
        [
            sys.executable, str(SCRIPTS / "recaman_carry_wheel.py"),
        ],
    ),
    (
        "vanishing",
        [
            sys.executable, str(SCRIPTS / "vanishing_gradients.py"),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running them.")
    p.add_argument("--skip", metavar="STEP", action="append", default=[],
                   help="Skip a named step. Can be repeated. "
                        "Valid names: " + ", ".join(n for n, _ in STEPS))
    p.add_argument("--only", metavar="STEP", action="append", default=[],
                   help="Run only these named steps (overrides --skip).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    valid_names = {n for n, _ in STEPS}
    for name in args.skip + args.only:
        if name not in valid_names:
            print(f"ERROR: unknown step '{name}'. Valid: {', '.join(sorted(valid_names))}")
            sys.exit(1)

    steps_to_run = [
        (name, cmd) for name, cmd in STEPS
        if (not args.only or name in args.only)
        and name not in args.skip
    ]

    print(f"=== run_all.py — {len(steps_to_run)} step(s) ===\n")
    total_start = time.perf_counter()
    failures: list[str] = []

    for name, cmd in steps_to_run:
        label = f"[{name}]"
        display_cmd = " ".join(
            f'"{c}"' if " " in c else c for c in cmd
        )
        print(f"{label} {display_cmd}")

        if args.dry_run:
            print(f"{label} (dry-run — skipped)\n")
            continue

        t0 = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.perf_counter() - t0

        if result.returncode == 0:
            print(f"{label} OK  ({elapsed:.1f}s)\n")
        else:
            print(f"{label} FAILED (exit {result.returncode}, {elapsed:.1f}s)\n")
            failures.append(name)

    if not args.dry_run:
        total = time.perf_counter() - total_start
        print(f"=== Finished in {total:.1f}s ===")
        if failures:
            print(f"Failed steps: {', '.join(failures)}")
            sys.exit(1)
        else:
            print("All steps completed successfully.")
            print(f"Outputs written to: {OUTPUTS}")


if __name__ == "__main__":
    main()
