from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

RAW = """
PASTE YOUR DATASET HERE
"""

PLACEHOLDER = "PASTE YOUR DATASET HERE"
RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
NUMBER_RE = re.compile(r"\d+")
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=np.int64)
BASIS = np.array([[3.0, 2.0], [2.0, 1.0], [1.0, 0.0]], dtype=float)
NUMBER_FEATURE_NAMES = (
    "length",
    "digit_sum",
    "alt_sum",
    "first_digit",
    "last1",
    "last2",
    "last3",
    "distinct_digits",
    "even_digit_count",
    "odd_digit_count",
    "c321",
    "c210",
    "basis_lift_3",
    "basis_lift_2",
    "basis_lift_1",
    *(f"digit_count_{digit}" for digit in range(10)),
    *(f"mod_{prime}" for prime in PRIMES.tolist()),
    "eq_count_3_0",
    "eq_count_2_0",
    "eq_count_1_0",
    "eq_count_321_210",
    "digit_sum_mod3_zero",
    "alt_sum_mod3_zero",
)
ANCHOR_CONTEXT_NAMES = (
    "event_index",
    "candidate_anchor",
    "prev_start",
    "current_gap",
    "prev_gap",
    "gap_ratio",
    "gap_delta",
    "log_gap",
    "v2_gap",
    "v3_gap",
    "recaman_residual",
    "recaman_side",
    "recaman_collision",
    "recaman_forward_needed",
    "gap_compression",
    "gap_explosion",
)
GAP_CONTEXT_NAMES = (
    "event_index",
    "candidate_gap",
    "candidate_start",
    "prev_start",
    "prev_gap",
    "gap_ratio",
    "gap_delta",
    "log_gap",
    "v2_gap",
    "v3_gap",
    "recaman_gap_residual",
    "recaman_gap_side",
    "recaman_start_collision",
    "recaman_forward_needed",
    "gap_compression",
    "gap_explosion",
)


@dataclass(frozen=True)
class Event:
    index: int
    start: int
    end: int
    length: int
    event_type: str
    gap_from_prev: int | None
    gap_regime: str


@dataclass(frozen=True)
class DatasetResult:
    name: str
    feature_dim: int
    contexts: int
    examples: int
    aucs: list[float]

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.aucs))


def prefixed_names(prefix: str, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in names)


ANCHOR_FEATURE_NAMES = (
    *ANCHOR_CONTEXT_NAMES,
    *prefixed_names("anchor_", NUMBER_FEATURE_NAMES),
    *prefixed_names("gap_", NUMBER_FEATURE_NAMES),
    *prefixed_names("prev_gap_", NUMBER_FEATURE_NAMES),
)
GAP_FEATURE_NAMES = (
    *GAP_CONTEXT_NAMES,
    *prefixed_names("gap_", NUMBER_FEATURE_NAMES),
    *prefixed_names("candidate_start_", NUMBER_FEATURE_NAMES),
    *prefixed_names("prev_gap_", NUMBER_FEATURE_NAMES),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version C: range-compressed obstruction modeling with blocked CV and gap dynamics."
    )
    parser.add_argument("--input-file", type=Path, help="Read the obstruction list from a text file.")
    parser.add_argument("--stdin", action="store_true", help="Read the obstruction list from stdin.")
    parser.add_argument("--raw", help="Read the obstruction list from a raw string argument.")
    parser.add_argument("--controls-per-positive", type=int, default=1, help="Matched controls per positive context.")
    parser.add_argument("--blocked-folds", type=int, default=5, help="Blocked folds over event order.")
    parser.add_argument(
        "--cv-scheme",
        choices=("forward", "blocked"),
        default="forward",
        help="Validation scheme. 'forward' avoids train-on-future leakage; 'blocked' keeps the legacy blocked K-fold.",
    )
    parser.add_argument(
        "--purge-contexts",
        type=int,
        default=1,
        help="Contexts to purge immediately before each forward test block.",
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="RandomForest trees.")
    parser.add_argument("--max-depth", type=int, default=5, help="RandomForest max depth.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--datasets",
        default="ABCD",
        help="Subset of datasets to evaluate. Choices are A, B, C, D.",
    )
    parser.add_argument("--save-json", type=Path, help="Write the full version-c results to JSON.")
    return parser.parse_args(argv)


def load_raw_dataset(args: argparse.Namespace) -> str:
    if args.raw is not None:
        return args.raw
    if args.stdin:
        return sys.stdin.read()
    if args.input_file is not None:
        return args.input_file.read_text(encoding="utf-8")
    return RAW


def classify_event_type(length: int) -> str:
    if length == 1:
        return "singleton"
    if length == 2:
        return "pair"
    if length <= 50:
        return "short"
    if length <= 1000:
        return "medium"
    return "huge"


def classify_gap_regime(gap: int | None, prev_gap: int | None) -> str:
    if gap is None or prev_gap is None or prev_gap <= 0:
        return "bootstrap"
    if gap <= prev_gap / 2:
        return "compression"
    if gap >= prev_gap * 2:
        return "explosion"
    return "stable"


def parse_events(raw: str) -> list[Event]:
    rows: list[tuple[int, int]] = []
    for raw_line in raw.splitlines():
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue

        match = RANGE_RE.fullmatch(line)
        if match:
            start, end = map(int, match.groups())
            if start > end:
                start, end = end, start
            rows.append((start, end))
        else:
            values = [int(token) for token in NUMBER_RE.findall(line)]
            for value in values:
                rows.append((value, value))

    rows.sort()
    events: list[Event] = []
    prev_start: int | None = None
    prev_gap: int | None = None
    for index, (start, end) in enumerate(rows, start=1):
        length = end - start + 1
        gap = None if prev_start is None else start - prev_start
        regime = classify_gap_regime(gap, prev_gap)
        events.append(
            Event(
                index=index,
                start=start,
                end=end,
                length=length,
                event_type=classify_event_type(length),
                gap_from_prev=gap,
                gap_regime=regime,
            )
        )
        prev_start = start
        prev_gap = gap
    return events


def digit_length(n: int) -> int:
    return 1 if n == 0 else len(str(n))


def band_bounds(length: int) -> tuple[int, int]:
    if length <= 1:
        return 0, 9
    return 10 ** (length - 1), (10 ** length) - 1


def encode_number(n: int) -> np.ndarray:
    digits = np.fromiter((int(char) for char in str(max(0, n))), dtype=np.int16)
    counts = np.bincount(digits, minlength=10).astype(float)
    c321 = counts[1] + counts[2] + counts[3]
    c210 = counts[0] + counts[1] + counts[2]
    basis_lift = BASIS @ np.array([c321, c210], dtype=float)
    digit_sum = float(digits.sum())
    alt_sum = float(digits[::2].sum() - digits[1::2].sum())

    return np.array(
        [
            float(len(digits)),
            digit_sum,
            alt_sum,
            float(digits[0]),
            float(n % 10),
            float(n % 100),
            float(n % 1000),
            float(np.count_nonzero(counts)),
            float(counts[0::2].sum()),
            float(counts[1::2].sum()),
            c321,
            c210,
            *basis_lift.tolist(),
            *counts.tolist(),
            *[float(n % prime) for prime in PRIMES.tolist()],
            float(counts[3] == counts[0]),
            float(counts[2] == counts[0]),
            float(counts[1] == counts[0]),
            float(c321 == c210),
            float(digit_sum % 3 == 0),
            float(alt_sum % 3 == 0),
        ],
        dtype=float,
    )


ZERO_NUMBER_FEATURES = np.zeros(len(NUMBER_FEATURE_NAMES), dtype=float)


def v_adic(n: int, prime: int) -> int:
    if n <= 0:
        return 0
    count = 0
    value = n
    while value % prime == 0:
        value //= prime
        count += 1
    return count


def sign(n: int) -> int:
    if n > 0:
        return 1
    if n < 0:
        return -1
    return 0


def build_anchor_features(
    candidate_anchor: int,
    event_index: int,
    prev_start: int | None,
    prev_gap: int | None,
    seen_starts: set[int],
) -> np.ndarray:
    prev_start_value = 0 if prev_start is None else prev_start
    current_gap = candidate_anchor if prev_start is None else candidate_anchor - prev_start
    previous_gap = 0 if prev_gap is None else prev_gap
    gap_ratio = current_gap / previous_gap if previous_gap > 0 else 0.0
    gap_delta = current_gap - previous_gap
    recaman_residual = current_gap - event_index
    backward_candidate = candidate_anchor - event_index
    collision = int(backward_candidate in seen_starts)
    forward_needed = int(backward_candidate <= 0 or collision)
    gap_compression = int(previous_gap > 0 and current_gap <= previous_gap / 2)
    gap_explosion = int(previous_gap > 0 and current_gap >= previous_gap * 2)

    context = np.array(
        [
            float(event_index),
            float(candidate_anchor),
            float(prev_start_value),
            float(current_gap),
            float(previous_gap),
            float(gap_ratio),
            float(gap_delta),
            float(math.log1p(max(current_gap, 0))),
            float(v_adic(current_gap, 2)),
            float(v_adic(current_gap, 3)),
            float(recaman_residual),
            float(sign(recaman_residual)),
            float(collision),
            float(forward_needed),
            float(gap_compression),
            float(gap_explosion),
        ],
        dtype=float,
    )
    gap_features = ZERO_NUMBER_FEATURES if current_gap <= 0 else encode_number(current_gap)
    prev_gap_features = ZERO_NUMBER_FEATURES if previous_gap <= 0 else encode_number(previous_gap)
    return np.concatenate([context, encode_number(candidate_anchor), gap_features, prev_gap_features])


def build_gap_features(
    candidate_gap: int,
    event_index: int,
    prev_start: int,
    prev_gap: int | None,
    seen_starts: set[int],
) -> np.ndarray:
    candidate_start = prev_start + candidate_gap
    previous_gap = 0 if prev_gap is None else prev_gap
    gap_ratio = candidate_gap / previous_gap if previous_gap > 0 else 0.0
    gap_delta = candidate_gap - previous_gap
    recaman_gap_residual = candidate_gap - event_index
    backward_candidate = candidate_start - event_index
    collision = int(backward_candidate in seen_starts)
    forward_needed = int(backward_candidate <= 0 or collision)
    gap_compression = int(previous_gap > 0 and candidate_gap <= previous_gap / 2)
    gap_explosion = int(previous_gap > 0 and candidate_gap >= previous_gap * 2)

    context = np.array(
        [
            float(event_index),
            float(candidate_gap),
            float(candidate_start),
            float(prev_start),
            float(previous_gap),
            float(gap_ratio),
            float(gap_delta),
            float(math.log1p(candidate_gap)),
            float(v_adic(candidate_gap, 2)),
            float(v_adic(candidate_gap, 3)),
            float(recaman_gap_residual),
            float(sign(recaman_gap_residual)),
            float(collision),
            float(forward_needed),
            float(gap_compression),
            float(gap_explosion),
        ],
        dtype=float,
    )
    prev_gap_features = ZERO_NUMBER_FEATURES if previous_gap <= 0 else encode_number(previous_gap)
    return np.concatenate([context, encode_number(candidate_gap), encode_number(candidate_start), prev_gap_features])


def sample_anchor_controls(
    positive_anchor: int,
    prev_start: int | None,
    blocked_points: set[int],
    rng: random.Random,
    k: int,
) -> list[int]:
    length = digit_length(positive_anchor)
    lo, hi = band_bounds(length)
    lower = lo if prev_start is None else max(lo, prev_start + 1)
    if lower > hi:
        raise ValueError(f"No control anchors available for {positive_anchor}.")

    chosen: set[int] = set()
    attempts = 0
    max_attempts = max(2000, 100 * k)
    while len(chosen) < k and attempts < max_attempts:
        candidate = rng.randint(lower, hi)
        attempts += 1
        if candidate == positive_anchor or candidate in blocked_points or candidate in chosen:
            continue
        chosen.add(candidate)

    if len(chosen) < k:
        raise ValueError(f"Could not sample enough control anchors for {positive_anchor}.")
    return sorted(chosen)


def sample_gap_controls(
    positive_gap: int,
    prev_start: int,
    blocked_points: set[int],
    rng: random.Random,
    k: int,
) -> list[int]:
    length = digit_length(positive_gap)
    lo, hi = band_bounds(length)
    lower = max(1, lo)

    chosen: set[int] = set()
    attempts = 0
    max_attempts = max(4000, 200 * k)
    while len(chosen) < k and attempts < max_attempts:
        candidate_gap = rng.randint(lower, hi)
        candidate_start = prev_start + candidate_gap
        attempts += 1
        if candidate_gap == positive_gap or candidate_gap in chosen:
            continue
        if candidate_start in blocked_points:
            continue
        chosen.add(candidate_gap)

    if len(chosen) < k:
        raise ValueError(f"Could not sample enough control gaps for {positive_gap}.")
    return sorted(chosen)


def build_dataset(
    name: str,
    events: list[Event],
    controls_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = random.Random(seed)
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    group_rows: list[int] = []
    feature_names = ANCHOR_FEATURE_NAMES if name in {"A", "B", "C"} else GAP_FEATURE_NAMES

    visible_blocked_points: set[int] = set()
    seen_starts: set[int] = set()
    prev_start: int | None = None
    prev_gap: int | None = None
    context_index = 0

    for event in events:
        visible_blocked_points.update(range(event.start, event.end + 1))
        current_gap = None if prev_start is None else event.start - prev_start

        if name == "A" and event.length == 1:
            context_index += 1
            X_rows.append(build_anchor_features(event.start, event.index, prev_start, prev_gap, seen_starts))
            y_rows.append(1)
            group_rows.append(context_index)
            for control in sample_anchor_controls(
                event.start,
                prev_start,
                visible_blocked_points,
                rng,
                controls_per_positive,
            ):
                X_rows.append(build_anchor_features(control, event.index, prev_start, prev_gap, seen_starts))
                y_rows.append(0)
                group_rows.append(context_index)

        elif name == "B" and event.length > 1:
            context_index += 1
            X_rows.append(build_anchor_features(event.start, event.index, prev_start, prev_gap, seen_starts))
            y_rows.append(1)
            group_rows.append(context_index)
            for control in sample_anchor_controls(
                event.start,
                prev_start,
                visible_blocked_points,
                rng,
                controls_per_positive,
            ):
                X_rows.append(build_anchor_features(control, event.index, prev_start, prev_gap, seen_starts))
                y_rows.append(0)
                group_rows.append(context_index)

        elif name == "C" and event.length > 1:
            context_index += 1
            X_rows.append(build_anchor_features(event.end, event.index, prev_start, prev_gap, seen_starts))
            y_rows.append(1)
            group_rows.append(context_index)
            for control in sample_anchor_controls(
                event.end,
                prev_start,
                visible_blocked_points,
                rng,
                controls_per_positive,
            ):
                X_rows.append(build_anchor_features(control, event.index, prev_start, prev_gap, seen_starts))
                y_rows.append(0)
                group_rows.append(context_index)

        elif name == "D" and prev_start is not None and current_gap is not None:
            context_index += 1
            X_rows.append(build_gap_features(current_gap, event.index, prev_start, prev_gap, seen_starts))
            y_rows.append(1)
            group_rows.append(context_index)
            for control in sample_gap_controls(
                current_gap,
                prev_start,
                visible_blocked_points,
                rng,
                controls_per_positive,
            ):
                X_rows.append(build_gap_features(control, event.index, prev_start, prev_gap, seen_starts))
                y_rows.append(0)
                group_rows.append(context_index)

        seen_starts.add(event.start)
        prev_start = event.start
        prev_gap = current_gap

    if not X_rows:
        raise ValueError(f"Dataset {name} produced no examples.")

    return (
        np.vstack(X_rows),
        np.array(y_rows, dtype=np.int8),
        np.array(group_rows, dtype=np.int32),
        feature_names,
    )


def blocked_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
    n_estimators: int,
    max_depth: int,
) -> list[float]:
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two blocked folds.")

    split_groups = [np.array(chunk, dtype=np.int32) for chunk in np.array_split(unique_groups, n_splits)]
    aucs: list[float] = []
    for fold_id, test_groups in enumerate(split_groups):
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed + fold_id,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X[train_mask], y[train_mask])
        scores = clf.predict_proba(X[test_mask])[:, 1]
        aucs.append(float(roc_auc_score(y[test_mask], scores)))
    return aucs


def forward_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
    n_estimators: int,
    max_depth: int,
    purge_contexts: int,
) -> list[float]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("Need at least three contexts for forward CV.")

    n_chunks = min(folds + 1, len(unique_groups))
    split_groups = [np.array(chunk, dtype=np.int32) for chunk in np.array_split(unique_groups, n_chunks) if len(chunk)]
    if len(split_groups) < 2:
        raise ValueError("Need at least one train block and one test block for forward CV.")

    aucs: list[float] = []
    for fold_id in range(1, len(split_groups)):
        test_groups = split_groups[fold_id]
        train_candidates = np.concatenate(split_groups[:fold_id])
        if purge_contexts > 0:
            cutoff = test_groups[0] - purge_contexts
            train_groups = train_candidates[train_candidates < cutoff]
        else:
            train_groups = train_candidates
        if len(train_groups) == 0:
            continue

        test_mask = np.isin(groups, test_groups)
        train_mask = np.isin(groups, train_groups)
        if np.unique(y[train_mask]).size < 2 or np.unique(y[test_mask]).size < 2:
            continue

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed + fold_id,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X[train_mask], y[train_mask])
        scores = clf.predict_proba(X[test_mask])[:, 1]
        aucs.append(float(roc_auc_score(y[test_mask], scores)))

    if len(aucs) < 1:
        raise ValueError("Forward CV produced no valid folds.")
    return aucs


def summarize_events(events: list[Event]) -> dict[str, object]:
    event_type_counts = Counter(event.event_type for event in events)
    gap_regime_counts = Counter(event.gap_regime for event in events)
    lengths = [event.length for event in events]
    starts = [event.start for event in events]
    gaps = [event.gap_from_prev for event in events if event.gap_from_prev is not None]
    return {
        "events": len(events),
        "singletons": event_type_counts["singleton"],
        "ranges": len(events) - event_type_counts["singleton"],
        "event_type_counts": dict(event_type_counts),
        "gap_regime_counts": dict(gap_regime_counts),
        "range_length_min": min(lengths),
        "range_length_max": max(lengths),
        "start_min": min(starts),
        "start_max": max(starts),
        "gap_min": min(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "gap_median": float(np.median(gaps)) if gaps else None,
    }


def print_dataset_result(result: DatasetResult) -> None:
    rounded = [round(value, 4) for value in result.aucs]
    print(f"=== dataset {result.name} ===")
    print(f"contexts: {result.contexts}")
    print(f"examples: {result.examples}")
    print(f"validation_auc: {rounded}")
    print(f"mean_validation_auc: {result.mean_auc:.4f}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load_raw_dataset(args)
    if raw.strip() == PLACEHOLDER:
        raise ValueError("RAW still contains the placeholder dataset.")

    events = parse_events(raw)
    if not events:
        raise ValueError("No events were parsed from the obstruction list.")

    selected = sorted(set(args.datasets.upper()))
    invalid = [name for name in selected if name not in {"A", "B", "C", "D"}]
    if invalid:
        raise ValueError(f"Unsupported dataset codes: {invalid}")

    summary = summarize_events(events)
    print(f"events: {summary['events']}")
    print(f"singletons: {summary['singletons']}")
    print(f"ranges: {summary['ranges']}")
    print(f"event_type_counts: {summary['event_type_counts']}")
    print(f"gap_regime_counts: {summary['gap_regime_counts']}")
    print()

    results_payload: dict[str, object] = {
        "summary": summary,
        "cv_scheme": args.cv_scheme,
        "purge_contexts": args.purge_contexts,
        "datasets": {},
    }

    for offset, name in enumerate(selected):
        X, y, groups, feature_names = build_dataset(
            name=name,
            events=events,
            controls_per_positive=args.controls_per_positive,
            seed=args.seed + offset,
        )
        if args.cv_scheme == "forward":
            aucs = forward_cv_auc(
                X=X,
                y=y,
                groups=groups,
                folds=args.blocked_folds,
                seed=args.seed + 100 * (offset + 1),
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                purge_contexts=args.purge_contexts,
            )
        else:
            aucs = blocked_cv_auc(
                X=X,
                y=y,
                groups=groups,
                folds=args.blocked_folds,
                seed=args.seed + 100 * (offset + 1),
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            )
        result = DatasetResult(
            name=name,
            feature_dim=int(X.shape[1]),
            contexts=int(groups.max()),
            examples=int(X.shape[0]),
            aucs=aucs,
        )
        print_dataset_result(result)
        results_payload["datasets"][name] = {
            "feature_dim": result.feature_dim,
            "contexts": result.contexts,
            "examples": result.examples,
            "auc": aucs,
            "mean_auc": result.mean_auc,
            "feature_names": list(feature_names),
        }

    if args.save_json is not None:
        args.save_json.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
        print(f"saved results: {args.save_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
