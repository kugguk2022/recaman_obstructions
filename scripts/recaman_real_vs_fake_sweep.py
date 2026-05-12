#!/usr/bin/env python3
"""
recaman_real_vs_fake_sweep.py
=============================

Run the finite-horizon Recaman real-vs-fake ablation over multiple
``(N1, N2)`` horizon pairs and save a compact comparison report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_PATH = Path(__file__).resolve().parent / "recaman_real_vs_fake.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "recaman_real_vs_fake_sweep.json"
DEFAULT_SWEEP = "50000:250000,100000:500000,150000:750000,200000:1000000"


def load_experiment_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recaman_real_vs_fake_module", EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment module from {EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_sweep(spec_text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chunk in spec_text.split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid sweep pair {item!r}; expected N1:N2.")
        n1 = int(parts[0].replace("_", ""))
        n2 = int(parts[1].replace("_", ""))
        if n2 <= n1:
            raise ValueError(f"Invalid sweep pair {item!r}; require N2 > N1.")
        pairs.append((n1, n2))
    if not pairs:
        raise ValueError("No horizon pairs parsed from sweep specification.")
    return pairs


def run_one_horizon(
    exp: ModuleType,
    randmat: ModuleType,
    n1: int,
    n2: int,
    max_per_class: int | None,
    folds: int,
    seed: int,
    n_estimators: int,
    max_depth: int,
) -> dict[str, object]:
    snapshot, late_values, still_values, visited_prefix_sorted, _arrival_step = exp.build_finite_horizon_labels(
        n1=n1,
        n2=n2,
    )
    sample = exp.sample_balanced_by_digit_length(
        late_values=late_values,
        still_values=still_values,
        seed=seed,
        max_per_class=max_per_class,
    )

    sampled_numbers = sample.late_values + sample.still_values
    y = np.array([1] * len(sample.late_values) + [0] * len(sample.still_values), dtype=np.int8)

    X_value = exp.build_value42_matrix(randmat, sampled_numbers)
    X_local, local_names = exp.build_local_gap_features(
        numbers=sampled_numbers,
        visited_prefix_sorted=visited_prefix_sorted,
        h1=snapshot.h1,
        n1=n1,
    )
    X_combined = np.hstack([X_value, X_local])

    value_names = list(randmat.FEATURE_NAMES)
    combined_names = value_names + local_names
    local_drop_frontier = {"relative_value_h_over_H1", "residual_h_minus_N1"}
    value_drop_scale = {"length", "first_digit"}

    X_local_pure, local_pure_names = exp.drop_named_features(X_local, local_names, local_drop_frontier)
    X_value_deconf, value_deconf_names = exp.drop_named_features(X_value, value_names, value_drop_scale)
    X_combined_deconf = np.hstack([X_value_deconf, X_local_pure])
    combined_deconf_names = value_deconf_names + local_pure_names

    models = [
        exp.run_model("value42_rf", X_value, y, folds, seed, n_estimators, max_depth),
        exp.run_model("value42_minus_scale_rf", X_value_deconf, y, folds, seed, n_estimators, max_depth),
        exp.run_model("local_gap_rf", X_local, y, folds, seed, n_estimators, max_depth),
        exp.run_model("local_gap_pure_rf", X_local_pure, y, folds, seed, n_estimators, max_depth),
        exp.run_model("value42_plus_local_rf", X_combined, y, folds, seed, n_estimators, max_depth),
        exp.run_model("value42_plus_local_deconf_rf", X_combined_deconf, y, folds, seed, n_estimators, max_depth),
    ]
    models_by_name = {model.name: model for model in models}

    top_features = exp.fit_feature_importance(
        X=X_combined_deconf,
        y=y,
        feature_names=combined_deconf_names,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    return {
        "horizons": {
            "n1": snapshot.n1,
            "n2": snapshot.n2,
            "h1": snapshot.h1,
            "terminal_value_n1": snapshot.terminal_value_n1,
            "terminal_value_n2": snapshot.terminal_value_n2,
            "visited_n1": snapshot.visited_n1,
            "visited_n2": snapshot.visited_n2,
            "unvisited_at_n1": snapshot.unvisited_at_n1,
            "late_arrivals": snapshot.late_arrivals,
            "still_unvisited": snapshot.still_unvisited,
            "late_fraction": snapshot.late_fraction,
            "delay_mean": snapshot.delay_mean,
            "delay_median": snapshot.delay_median,
            "delay_p90": snapshot.delay_p90,
        },
        "sample": {
            "per_class": sample.per_class,
            "total": sample.total,
            "late_digit_lengths": exp.format_digit_lengths(sample.late_values),
            "still_digit_lengths": exp.format_digit_lengths(sample.still_values),
        },
        "models": {
            model.name: exp.model_result_dict(model)
            for model in models
        },
        "summary": {
            "value42_auc": models_by_name["value42_rf"].mean_auc,
            "value42_minus_scale_auc": models_by_name["value42_minus_scale_rf"].mean_auc,
            "local_gap_auc": models_by_name["local_gap_rf"].mean_auc,
            "local_gap_pure_auc": models_by_name["local_gap_pure_rf"].mean_auc,
            "value42_plus_local_auc": models_by_name["value42_plus_local_rf"].mean_auc,
            "value42_plus_local_deconf_auc": models_by_name["value42_plus_local_deconf_rf"].mean_auc,
            "local_gap_pure_lift_over_value42": (
                models_by_name["local_gap_pure_rf"].mean_auc - models_by_name["value42_rf"].mean_auc
            ),
            "deconf_combined_lift_over_value42": (
                models_by_name["value42_plus_local_deconf_rf"].mean_auc - models_by_name["value42_rf"].mean_auc
            ),
            "full_combined_lift_over_value42": (
                models_by_name["value42_plus_local_rf"].mean_auc - models_by_name["value42_rf"].mean_auc
            ),
        },
        "top_features_value42_plus_local_deconf": [
            {"name": name, "importance": score}
            for name, score in top_features
        ],
        "feature_sets": {
            "value42_names": value_names,
            "local_gap_names": local_names,
            "local_gap_pure_names": local_pure_names,
            "value42_plus_local_names": combined_names,
            "value42_plus_local_deconf_names": combined_deconf_names,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Recaman real-vs-fake ablations over multiple horizons.")
    parser.add_argument(
        "--sweep",
        type=str,
        default=DEFAULT_SWEEP,
        help="Comma-separated N1:N2 horizon pairs.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50_000,
        help="Maximum sampled examples per class for each horizon.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=300, help="RandomForest tree count.")
    parser.add_argument("--max-depth", type=int, default=8, help="RandomForest max depth.")
    parser.add_argument("--save-json", type=Path, default=DEFAULT_OUTPUT, help="Write sweep results to JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = parse_sweep(args.sweep)
    exp = load_experiment_module()
    randmat = exp.load_randmat_module()

    print("=" * 72)
    print("Recaman Real-vs-Fake Horizon Sweep")
    print(f"pairs = {', '.join(f'{n1:,}:{n2:,}' for n1, n2 in pairs)}")
    print("=" * 72)

    runs: list[dict[str, object]] = []
    for index, (n1, n2) in enumerate(pairs, start=1):
        print(f"\n[{index}/{len(pairs)}] N1={n1:,} N2={n2:,}")
        run = run_one_horizon(
            exp=exp,
            randmat=randmat,
            n1=n1,
            n2=n2,
            max_per_class=args.max_per_class,
            folds=args.folds,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        summary = run["summary"]
        assert isinstance(summary, dict)
        print(
            "  "
            f"value42={summary['value42_auc']:.4f}  "
            f"local_gap_pure={summary['local_gap_pure_auc']:.4f}  "
            f"deconf_combined={summary['value42_plus_local_deconf_auc']:.4f}"
        )
        print(
            "  "
            f"lift(local_gap_pure)={summary['local_gap_pure_lift_over_value42']:+.4f}  "
            f"lift(deconf_combined)={summary['deconf_combined_lift_over_value42']:+.4f}"
        )
        runs.append(run)

    payload = {
        "settings": {
            "sweep": [{"n1": n1, "n2": n2} for n1, n2 in pairs],
            "max_per_class": args.max_per_class,
            "folds": args.folds,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
        "runs": runs,
    }

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
