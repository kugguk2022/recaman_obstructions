#!/usr/bin/env python3
"""
recaman_real_vs_fake.py
=======================

Finite-horizon "real vs fake hole" experiment for Recaman.

Definitions at horizons N1 < N2:
  - U1 = values h in [1, H1] with H1 = max(a_0, ..., a_N1) that are unvisited at N1
  - late_arrival: h in U1 that is first visited during steps N1+1 .. N2
  - still_unvisited: h in U1 that remains unvisited at N2

This script builds those exact finite labels, then evaluates:
  1. a plain 42-feature value-only baseline reused from 321_210_randmat.py
  2. the same baseline plus cheap local gap features from the visited set at N1

The goal is to answer the concrete question first:
can finite-horizon late arrivals be separated from still-unvisited values
without committing to the geometric slip-budget story yet?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent.parent
RANDMAT_PATH = Path(__file__).resolve().parent / "321_210_randmat.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "recaman_real_vs_fake_results.json"


@dataclass(frozen=True)
class HorizonSnapshot:
    n1: int
    n2: int
    h1: int
    terminal_value_n1: int
    terminal_value_n2: int
    visited_n1: int
    visited_n2: int
    unvisited_at_n1: int
    late_arrivals: int
    still_unvisited: int
    late_fraction: float
    delay_mean: float
    delay_median: float
    delay_p90: float


@dataclass(frozen=True)
class DatasetSample:
    late_values: list[int]
    still_values: list[int]

    @property
    def per_class(self) -> int:
        return len(self.late_values)

    @property
    def total(self) -> int:
        return len(self.late_values) + len(self.still_values)


@dataclass(frozen=True)
class ModelResult:
    name: str
    feature_dim: int
    aucs: list[float]

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.aucs))


def load_randmat_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recaman_randmat_helpers", RANDMAT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {RANDMAT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def format_digit_lengths(values: list[int]) -> str:
    counts = Counter(len(str(value)) for value in values)
    return ", ".join(f"{length}d:{count}" for length, count in sorted(counts.items()))


def run_to_n1(n1: int) -> tuple[int, set[int], int]:
    value = 0
    visited = {0}
    max_seen = 0
    for step in range(1, n1 + 1):
        candidate = value - step
        if candidate > 0 and candidate not in visited:
            value = candidate
        else:
            value = value + step
        visited.add(value)
        if value > max_seen:
            max_seen = value
    return value, visited, max_seen


def continue_to_n2(
    n1: int,
    n2: int,
    value: int,
    visited: set[int],
    horizon_max: int,
    unvisited_mask: bytearray,
) -> tuple[int, set[int], np.ndarray, list[int]]:
    arrival_step = np.zeros(horizon_max + 1, dtype=np.uint32)
    late_values: list[int] = []

    for step in range(n1 + 1, n2 + 1):
        candidate = value - step
        if candidate > 0 and candidate not in visited:
            value = candidate
        else:
            value = value + step
        visited.add(value)
        if 0 < value <= horizon_max and unvisited_mask[value]:
            arrival_step[value] = step
            late_values.append(value)
            unvisited_mask[value] = 0

    return value, visited, arrival_step, late_values


def build_finite_horizon_labels(n1: int, n2: int) -> tuple[HorizonSnapshot, list[int], list[int], list[int], np.ndarray]:
    if n2 <= n1:
        raise ValueError("N2 must be larger than N1")

    value_n1, visited_n1, h1 = run_to_n1(n1)
    visited_prefix_sorted = sorted(visited_n1)

    unvisited_mask = bytearray(b"\x01") * (h1 + 1)
    unvisited_mask[0] = 0
    for value in visited_prefix_sorted:
        unvisited_mask[value] = 0
    unvisited_at_n1 = int(sum(unvisited_mask))

    value_n2, visited_n2, arrival_step, late_values = continue_to_n2(
        n1=n1,
        n2=n2,
        value=value_n1,
        visited=set(visited_n1),
        horizon_max=h1,
        unvisited_mask=unvisited_mask,
    )
    still_values = [value for value in range(1, h1 + 1) if unvisited_mask[value]]

    delays = np.asarray([int(arrival_step[value]) - n1 for value in late_values], dtype=np.int64)
    snapshot = HorizonSnapshot(
        n1=n1,
        n2=n2,
        h1=h1,
        terminal_value_n1=value_n1,
        terminal_value_n2=value_n2,
        visited_n1=len(visited_n1),
        visited_n2=len(visited_n2),
        unvisited_at_n1=unvisited_at_n1,
        late_arrivals=len(late_values),
        still_unvisited=len(still_values),
        late_fraction=(len(late_values) / unvisited_at_n1) if unvisited_at_n1 else float("nan"),
        delay_mean=float(np.mean(delays)) if len(delays) else float("nan"),
        delay_median=float(np.median(delays)) if len(delays) else float("nan"),
        delay_p90=float(np.quantile(delays, 0.90)) if len(delays) else float("nan"),
    )
    return snapshot, late_values, still_values, visited_prefix_sorted, arrival_step


def proportional_allocation(capacity: dict[int, int], total_target: int) -> dict[int, int]:
    if total_target <= 0:
        return {key: 0 for key in capacity}

    total_capacity = sum(capacity.values())
    if total_target >= total_capacity:
        return dict(capacity)

    raw = {key: (total_target * value / total_capacity) for key, value in capacity.items()}
    alloc = {key: min(capacity[key], int(math.floor(raw[key]))) for key in capacity}
    assigned = sum(alloc.values())
    remainders = sorted(
        ((raw[key] - alloc[key], key) for key in capacity if alloc[key] < capacity[key]),
        reverse=True,
    )
    idx = 0
    while assigned < total_target and idx < len(remainders):
        _, key = remainders[idx]
        alloc[key] += 1
        assigned += 1
        idx += 1
    return alloc


def sample_balanced_by_digit_length(
    late_values: list[int],
    still_values: list[int],
    seed: int,
    max_per_class: int | None,
) -> DatasetSample:
    rng = random.Random(seed)
    late_by_len: dict[int, list[int]] = defaultdict(list)
    still_by_len: dict[int, list[int]] = defaultdict(list)
    for value in late_values:
        late_by_len[len(str(value))].append(value)
    for value in still_values:
        still_by_len[len(str(value))].append(value)

    capacity = {
        length: min(len(late_by_len[length]), len(still_by_len[length]))
        for length in sorted(set(late_by_len) & set(still_by_len))
        if min(len(late_by_len[length]), len(still_by_len[length])) > 0
    }
    if not capacity:
        raise ValueError("No overlapping digit-length buckets between late and still classes.")

    total_available = sum(capacity.values())
    target = total_available if max_per_class is None else min(total_available, max_per_class)
    allocation = proportional_allocation(capacity, target)

    sampled_late: list[int] = []
    sampled_still: list[int] = []
    for length, count in allocation.items():
        if count <= 0:
            continue
        sampled_late.extend(rng.sample(late_by_len[length], count))
        sampled_still.extend(rng.sample(still_by_len[length], count))

    sampled_late.sort()
    sampled_still.sort()
    if len(sampled_late) != len(sampled_still):
        raise AssertionError("Balanced sampling failed: class sizes differ.")
    return DatasetSample(late_values=sampled_late, still_values=sampled_still)


def build_value42_matrix(randmat: ModuleType, numbers: list[int]) -> np.ndarray:
    return randmat.build_feature_matrix(numbers)


def build_local_gap_features(
    numbers: list[int],
    visited_prefix_sorted: list[int],
    h1: int,
    n1: int,
) -> tuple[np.ndarray, list[str]]:
    visited_ext = visited_prefix_sorted
    rows = []
    names = [
        "relative_value_h_over_H1",
        "residual_h_minus_N1",
        "left_gap",
        "right_gap",
        "gap_size",
        "position_in_gap",
        "distance_to_nearest_boundary",
        "local_fill_65",
        "local_fill_257",
    ]

    for h in numbers:
        idx = bisect_right(visited_ext, h)
        prev_visited = visited_ext[idx - 1]
        next_visited = visited_ext[idx]
        left_gap = h - prev_visited
        right_gap = next_visited - h
        gap_size = next_visited - prev_visited - 1
        position_in_gap = left_gap / (gap_size + 1) if gap_size >= 0 else 0.0
        nearest_boundary = min(left_gap, right_gap)

        lo_small = h - 32
        hi_small = h + 32
        lo_large = h - 128
        hi_large = h + 128
        fill_small = bisect_right(visited_ext, hi_small) - bisect_left(visited_ext, lo_small)
        fill_large = bisect_right(visited_ext, hi_large) - bisect_left(visited_ext, lo_large)

        rows.append(
            [
                h / h1,
                float(h - n1),
                float(left_gap),
                float(right_gap),
                float(gap_size),
                float(position_in_gap),
                float(nearest_boundary),
                fill_small / 65.0,
                fill_large / 257.0,
            ]
        )

    return np.asarray(rows, dtype=float), names


def evaluate_rf_auc(
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
    n_estimators: int,
    max_depth: int,
) -> list[float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed + fold_id,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], scores)))
    return aucs


def fit_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_estimators: int,
    max_depth: int,
    top_k: int = 12,
) -> list[tuple[str, float]]:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)
    importances = list(zip(feature_names, clf.feature_importances_))
    importances.sort(key=lambda item: item[1], reverse=True)
    return [(name, float(score)) for name, score in importances[:top_k]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-horizon real-vs-fake Recaman experiment.")
    parser.add_argument("--n1", type=int, default=200_000, help="First horizon N1.")
    parser.add_argument("--n2", type=int, default=1_000_000, help="Second horizon N2, with N2 > N1.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=100_000,
        help="Maximum sampled examples per class after digit-length matching.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=300, help="RandomForest tree count.")
    parser.add_argument("--max-depth", type=int, default=8, help="RandomForest max depth.")
    parser.add_argument("--save-json", type=Path, default=DEFAULT_OUTPUT, help="Write results to JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    randmat = load_randmat_module()

    print("=" * 72)
    print("Recaman Finite-Horizon Real-vs-Fake Experiment")
    print(f"N1 = {args.n1:,}   N2 = {args.n2:,}")
    print("=" * 72)

    print("\n[1] Generating exact finite-horizon labels")
    snapshot, late_values, still_values, visited_prefix_sorted, arrival_step = build_finite_horizon_labels(
        n1=args.n1,
        n2=args.n2,
    )
    print(f"H1 = max(a_0..a_N1)      = {snapshot.h1:,}")
    print(f"visited by N1            = {snapshot.visited_n1:,}")
    print(f"unvisited at N1          = {snapshot.unvisited_at_n1:,}")
    print(f"late arrivals by N2      = {snapshot.late_arrivals:,}")
    print(f"still unvisited at N2    = {snapshot.still_unvisited:,}")
    print(f"late fraction in U1      = {snapshot.late_fraction:.4f}")
    print(f"late delay mean / median = {snapshot.delay_mean:.1f} / {snapshot.delay_median:.1f}")
    print(f"late delay p90           = {snapshot.delay_p90:.1f}")

    print("\n[2] Building balanced evaluation sample")
    sample = sample_balanced_by_digit_length(
        late_values=late_values,
        still_values=still_values,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    print(f"sampled per class        = {sample.per_class:,}")
    print(f"total sampled            = {sample.total:,}")
    print(f"late digit lengths       = {format_digit_lengths(sample.late_values)}")
    print(f"still digit lengths      = {format_digit_lengths(sample.still_values)}")

    sampled_numbers = sample.late_values + sample.still_values
    y = np.array([1] * len(sample.late_values) + [0] * len(sample.still_values), dtype=np.int8)

    print("\n[3] Building feature matrices")
    X_value = build_value42_matrix(randmat, sampled_numbers)
    X_local, local_names = build_local_gap_features(
        numbers=sampled_numbers,
        visited_prefix_sorted=visited_prefix_sorted,
        h1=snapshot.h1,
        n1=args.n1,
    )
    X_combined = np.hstack([X_value, X_local])
    value_names = list(randmat.FEATURE_NAMES)
    combined_names = value_names + local_names
    print(f"value-only dim           = {X_value.shape[1]}")
    print(f"value+local dim          = {X_combined.shape[1]}")

    print("\n[4] Evaluating 42-feature baseline")
    value_aucs = evaluate_rf_auc(
        X=X_value,
        y=y,
        folds=args.folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    print(f"fold AUCs                = {[round(x, 4) for x in value_aucs]}")
    print(f"mean AUC                 = {np.mean(value_aucs):.4f}")

    print("\n[5] Evaluating 42 + local-gap features")
    combined_aucs = evaluate_rf_auc(
        X=X_combined,
        y=y,
        folds=args.folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    print(f"fold AUCs                = {[round(x, 4) for x in combined_aucs]}")
    print(f"mean AUC                 = {np.mean(combined_aucs):.4f}")
    print(f"lift over value-only     = {np.mean(combined_aucs) - np.mean(value_aucs):+.4f}")

    print("\n[6] Top feature importances for value+local model")
    top_features = fit_feature_importance(
        X=X_combined,
        y=y,
        feature_names=combined_names,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    for name, score in top_features:
        print(f"{name:28s} {score:.4f}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
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
                "late_digit_lengths": format_digit_lengths(sample.late_values),
                "still_digit_lengths": format_digit_lengths(sample.still_values),
            },
            "models": {
                "value42_rf": {
                    "feature_dim": int(X_value.shape[1]),
                    "fold_aucs": value_aucs,
                    "mean_auc": float(np.mean(value_aucs)),
                },
                "value42_plus_local_rf": {
                    "feature_dim": int(X_combined.shape[1]),
                    "fold_aucs": combined_aucs,
                    "mean_auc": float(np.mean(combined_aucs)),
                    "lift_over_value42": float(np.mean(combined_aucs) - np.mean(value_aucs)),
                },
            },
            "top_features_value42_plus_local": [
                {"name": name, "importance": score} for name, score in top_features
            ],
        }
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.save_json}")

    _ = arrival_step  # retained for future delay-stratified analysis
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
