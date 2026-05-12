#!/usr/bin/env python3
"""
recaman_real_vs_fake_auc_summary.py
===================================

Extract a compact AUC-only summary from the finite-horizon real-vs-fake
Recaman sweep output.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "recaman_real_vs_fake_sweep.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "recaman_real_vs_fake_auc_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract an AUC-only summary from a Recaman sweep JSON file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Sweep JSON input path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    return parser.parse_args()


def load_runs(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Expected 'runs' list in {path}")
    return runs


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    args = parse_args()
    runs = load_runs(args.input)

    rows: list[dict[str, str]] = []
    for run in runs:
        horizons = run["horizons"]
        summary = run["summary"]
        rows.append(
            {
                "n1": str(horizons["n1"]),
                "n2": str(horizons["n2"]),
                "value42_auc": fmt(summary["value42_auc"]),
                "value42_minus_scale_auc": fmt(summary["value42_minus_scale_auc"]),
                "local_gap_auc": fmt(summary["local_gap_auc"]),
                "local_gap_pure_auc": fmt(summary["local_gap_pure_auc"]),
                "value42_plus_local_auc": fmt(summary["value42_plus_local_auc"]),
                "value42_plus_local_deconf_auc": fmt(summary["value42_plus_local_deconf_auc"]),
                "local_gap_pure_lift": fmt(summary["local_gap_pure_lift_over_value42"]),
                "deconf_combined_lift": fmt(summary["deconf_combined_lift_over_value42"]),
                "full_combined_lift": fmt(summary["full_combined_lift_over_value42"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "n1",
        "n2",
        "value42_auc",
        "value42_minus_scale_auc",
        "local_gap_auc",
        "local_gap_pure_auc",
        "value42_plus_local_auc",
        "value42_plus_local_deconf_auc",
        "local_gap_pure_lift",
        "deconf_combined_lift",
        "full_combined_lift",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("AUC-only summary")
    print("=" * 72)
    for row in rows:
        print(
            f"N1={int(row['n1']):,} N2={int(row['n2']):,}  "
            f"value42={row['value42_auc']}  "
            f"local_gap_pure={row['local_gap_pure_auc']}  "
            f"deconf_combined={row['value42_plus_local_deconf_auc']}"
        )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
