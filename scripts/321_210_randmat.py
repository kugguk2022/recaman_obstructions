from __future__ import annotations

import argparse
import heapq
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

RAW = """
PASTE YOUR DATASET HERE
"""

EPS = 1e-9
PLACEHOLDER = "PASTE YOUR DATASET HERE"
RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
NUMBER_RE = re.compile(r"\d+")
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=np.int64)
BASIS = np.array([[3.0, 2.0], [2.0, 1.0], [1.0, 0.0]], dtype=float)
FEATURE_NAMES = (
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


@dataclass(frozen=True)
class SearchConfig:
    trials: int
    out_dim: int
    coeff_low: int
    coeff_high: int
    keep_top: int
    seed: int
    search_sample_per_class: int
    controls_per_positive: int
    cv_folds: int
    search_mode: str
    ramanujan_order: int | None = None


@dataclass(frozen=True)
class Candidate:
    score: float
    separation: float
    auc: float
    diversity: float
    matrix: np.ndarray
    positive_mean_signature: np.ndarray
    negative_mean_signature: np.ndarray
    cv_auc_scores: np.ndarray | None = None

    @property
    def cv_auc_mean(self) -> float | None:
        if self.cv_auc_scores is None:
            return None
        return float(np.mean(self.cv_auc_scores))


class BinaryProjectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BinaryProjectionTransformer":
        X = np.asarray(X, dtype=float)
        self.mean_, self.scale_ = fit_normalizer(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xn = normalize_features(X, self.mean_, self.scale_)
        return heaviside(Xn @ self.matrix.T)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search 321/210-inspired generator matrices and validate them with CV AUC."
    )
    parser.add_argument("--input-file", type=Path, help="Read the positive set from a text file.")
    parser.add_argument("--stdin", action="store_true", help="Read the positive set from stdin.")
    parser.add_argument("--raw", help="Read the positive set from a raw string argument.")
    parser.add_argument("--no-expand-ranges", action="store_true", help="Do not expand a-b ranges.")
    parser.add_argument("--max-range-expand", type=int, default=5000, help="Expand ranges only when b-a <= this.")
    parser.add_argument("--controls-per-positive", type=int, default=3, help="Matched random controls per positive.")
    parser.add_argument(
        "--search-mode",
        choices=("random", "ramanujan"),
        default="random",
        help="Projection family to search.",
    )
    parser.add_argument("--trials", type=int, default=10000, help="Random matrices to test.")
    parser.add_argument("--out-dim", type=int, default=10, help="Projection signature width.")
    parser.add_argument("--coeff-low", type=int, default=-3, help="Minimum random matrix coefficient.")
    parser.add_argument("--coeff-high", type=int, default=3, help="Maximum random matrix coefficient.")
    parser.add_argument("--keep-top", type=int, default=20, help="How many heuristic candidates to keep.")
    parser.add_argument("--search-sample-per-class", type=int, default=4000, help="Per-class cap used during the matrix search.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds for generator testing.")
    parser.add_argument("--save-best-file", type=Path, help="Write the best candidate, matrix, and row formulas to a JSON file.")
    parser.add_argument(
        "--ramanujan-order",
        type=int,
        help="Optional Paley graph order q. Must be prime, q %% 4 == 1, and at least feature_dim + out_dim.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args(argv)


def load_raw_dataset(args: argparse.Namespace) -> str:
    if args.raw is not None:
        return args.raw
    if args.stdin:
        return sys.stdin.read()
    if args.input_file is not None:
        return args.input_file.read_text(encoding="utf-8")
    return RAW


def parse_numbers(raw: str, expand_ranges: bool = True, max_range_expand: int = 5000) -> list[int]:
    numbers: set[int] = set()
    for raw_line in raw.splitlines():
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue

        match = RANGE_RE.fullmatch(line)
        if match:
            start, end = map(int, match.groups())
            if start > end:
                start, end = end, start
            if expand_ranges and (end - start) <= max_range_expand:
                values = range(start, end + 1)
            else:
                values = (start, end)
        else:
            found = NUMBER_RE.findall(line)
            if not found:
                continue
            values = (int(token) for token in found)

        for value in values:
            if value < 0:
                raise ValueError(f"Negative values are not supported: {value}")
            numbers.add(value)

    return sorted(numbers)


def digit_length(n: int) -> int:
    return 1 if n == 0 else len(str(n))


def band_bounds(length: int) -> tuple[int, int]:
    if length <= 1:
        return 0, 9
    return 10 ** (length - 1), (10 ** length) - 1


def sample_controls_for_length(
    length: int,
    positives: list[int],
    k: int,
    rng: random.Random,
) -> list[int]:
    lo, hi = band_bounds(length)
    needed = len(positives) * k
    blocked = set(positives)
    span = hi - lo + 1
    available = span - len(blocked)
    if available <= 0:
        raise ValueError(f"No controls are available for {length}-digit positives.")
    if needed > available:
        raise ValueError(
            f"Need {needed} unique controls for {length}-digit positives, but only {available} are available."
        )

    if span <= 1_000_000:
        pool = [value for value in range(lo, hi + 1) if value not in blocked]
        return rng.sample(pool, needed)

    chosen: set[int] = set()
    attempts = 0
    max_attempts = max(needed * 50, 1000)
    while len(chosen) < needed and attempts < max_attempts:
        candidate = rng.randint(lo, hi)
        attempts += 1
        if candidate in blocked or candidate in chosen:
            continue
        chosen.add(candidate)

    if len(chosen) < needed:
        raise ValueError(
            f"Could not sample enough unique {length}-digit controls. Reduce controls-per-positive."
        )
    return sorted(chosen)


def make_controls(positives: list[int], k: int = 3, seed: int = 1) -> list[int]:
    rng = random.Random(seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    for value in positives:
        buckets[digit_length(value)].append(value)

    controls: list[int] = []
    for length, bucket in sorted(buckets.items()):
        controls.extend(sample_controls_for_length(length, bucket, k, rng))
    return controls


def encode_number(n: int) -> np.ndarray:
    digits = np.fromiter((int(char) for char in str(n)), dtype=np.int16)
    counts = np.bincount(digits, minlength=10).astype(float)
    c321 = counts[1] + counts[2] + counts[3]
    c210 = counts[0] + counts[1] + counts[2]
    basis_lift = BASIS @ np.array([c321, c210], dtype=float)
    digit_sum = float(digits.sum())
    alt_sum = float(digits[::2].sum() - digits[1::2].sum())

    features = np.array(
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
    if features.shape[0] != len(FEATURE_NAMES):
        raise AssertionError("Feature vector and FEATURE_NAMES are out of sync.")
    return features


def build_feature_matrix(numbers: list[int]) -> np.ndarray:
    if not numbers:
        raise ValueError("Cannot build a feature matrix from an empty number set.")
    return np.vstack([encode_number(number) for number in numbers])


def fit_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < EPS] = 1.0
    return mean, scale


def normalize_features(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def heaviside(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.uint8)


def symmetric_auc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        return 0.5
    return max(auc, 1.0 - auc)


def candidate_score(separation: float, auc: float, diversity: float) -> float:
    auc_gain = max(0.0, 2.0 * (auc - 0.5))
    return 0.6 * separation + 0.3 * auc_gain + 0.1 * diversity


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = int(n ** 0.5)
    for factor in range(3, limit + 1, 2):
        if n % factor == 0:
            return False
    return True


def next_prime_one_mod_four(lower_bound: int) -> int:
    candidate = max(5, lower_bound)
    if candidate % 2 == 0:
        candidate += 1
    while True:
        if candidate % 4 == 1 and is_prime(candidate):
            return candidate
        candidate += 2


def resolve_ramanujan_order(feature_dim: int, out_dim: int, requested_order: int | None) -> int:
    minimum_order = feature_dim + out_dim
    if requested_order is None:
        return next_prime_one_mod_four(minimum_order)
    if requested_order < minimum_order:
        raise ValueError(
            f"ramanujan-order must be at least feature_dim + out_dim = {minimum_order}."
        )
    if requested_order % 4 != 1 or not is_prime(requested_order):
        raise ValueError("ramanujan-order must be prime and congruent to 1 mod 4.")
    return requested_order


def paley_sign_table(order: int) -> np.ndarray:
    signs = np.full(order, -1, dtype=np.int8)
    signs[0] = 0
    for value in range(1, order):
        signs[(value * value) % order] = 1
    return signs


def sanitize_matrix(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    zero_rows = np.where(~M.any(axis=1))[0]
    if zero_rows.size == 0:
        return M
    repaired = M.copy()
    cols = rng.integers(0, repaired.shape[1], size=zero_rows.size)
    signs = rng.choice(np.array([-1, 1], dtype=int), size=zero_rows.size)
    repaired[zero_rows, cols] = signs
    return repaired


def evaluate_matrix(Xn: np.ndarray, y: np.ndarray, M: np.ndarray) -> Candidate:
    Z = heaviside(Xn @ M.T)
    positive_mask = y == 1
    negative_mask = ~positive_mask
    positive_mean = Z[positive_mask].mean(axis=0)
    negative_mean = Z[negative_mask].mean(axis=0)
    separation = float(np.abs(positive_mean - negative_mean).mean())
    weights = (1 << np.arange(Z.shape[1], dtype=np.uint64))
    codes = Z @ weights
    auc = symmetric_auc(y, codes)
    diversity = float(np.unique(codes).size / len(codes))
    score = candidate_score(separation, auc, diversity)
    return Candidate(score, separation, auc, diversity, M, positive_mean, negative_mean)


def subsample_for_search(
    X: np.ndarray,
    y: np.ndarray,
    per_class_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if per_class_limit <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    for label in (0, 1):
        label_idx = np.flatnonzero(y == label)
        if len(label_idx) <= per_class_limit:
            indices.append(label_idx)
            continue
        chosen = rng.choice(label_idx, size=per_class_limit, replace=False)
        indices.append(np.sort(chosen))
    keep = np.sort(np.concatenate(indices))
    return X[keep], y[keep]


def search_generated_matrices(
    X: np.ndarray,
    y: np.ndarray,
    config: SearchConfig,
    matrix_factory,
) -> list[Candidate]:
    if config.out_dim > 62:
        raise ValueError("out-dim must be <= 62 so signature codes fit in uint64.")

    mean, scale = fit_normalizer(X)
    Xn = normalize_features(X, mean, scale)
    heap: list[tuple[tuple[float, float, float], int, Candidate]] = []
    Xn_search, y_search = subsample_for_search(Xn, y, config.search_sample_per_class, config.seed)

    for trial_index in range(config.trials):
        M = matrix_factory(trial_index)
        candidate = evaluate_matrix(Xn_search, y_search, M)
        key = (candidate.score, candidate.auc, candidate.diversity)
        if len(heap) < config.keep_top:
            heapq.heappush(heap, (key, trial_index, candidate))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, trial_index, candidate))

    return [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]


def random_matrix_search(X: np.ndarray, y: np.ndarray, config: SearchConfig) -> list[Candidate]:
    if config.coeff_low > config.coeff_high:
        raise ValueError("coeff-low cannot exceed coeff-high.")

    rng = np.random.default_rng(config.seed)

    def matrix_factory(_: int) -> np.ndarray:
        M = rng.integers(
            config.coeff_low,
            config.coeff_high + 1,
            size=(config.out_dim, X.shape[1]),
        )
        return sanitize_matrix(M, rng)

    return search_generated_matrices(X, y, config, matrix_factory)


def build_ramanujan_matrix(
    feature_dim: int,
    out_dim: int,
    order: int,
    signs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    permutation = rng.permutation(order)
    feature_vertices = permutation[:feature_dim]
    row_vertices = permutation[feature_dim : feature_dim + out_dim]
    row_flips = rng.choice(np.array([-1, 1], dtype=np.int8), size=out_dim)
    col_flips = rng.choice(np.array([-1, 1], dtype=np.int8), size=feature_dim)
    row_offsets = rng.integers(0, order, size=out_dim)

    matrix = np.empty((out_dim, feature_dim), dtype=np.int8)
    for row_index, row_vertex in enumerate(row_vertices):
        residues = signs[(row_vertex + row_offsets[row_index] - feature_vertices) % order]
        matrix[row_index] = row_flips[row_index] * residues * col_flips
    return matrix.astype(int, copy=False)


def ramanujan_matrix_search(X: np.ndarray, y: np.ndarray, config: SearchConfig) -> list[Candidate]:
    if config.ramanujan_order is None:
        raise ValueError("ramanujan-order was not resolved.")

    rng = np.random.default_rng(config.seed)
    signs = paley_sign_table(config.ramanujan_order)

    def matrix_factory(_: int) -> np.ndarray:
        return build_ramanujan_matrix(
            feature_dim=X.shape[1],
            out_dim=config.out_dim,
            order=config.ramanujan_order,
            signs=signs,
            rng=rng,
        )

    return search_generated_matrices(X, y, config, matrix_factory)


def search_candidates(X: np.ndarray, y: np.ndarray, config: SearchConfig) -> list[Candidate]:
    if config.search_mode == "random":
        return random_matrix_search(X, y, config)
    if config.search_mode == "ramanujan":
        return ramanujan_matrix_search(X, y, config)
    raise ValueError(f"Unsupported search mode: {config.search_mode}")


def cross_validate_signature(
    X: np.ndarray,
    y: np.ndarray,
    matrix: np.ndarray,
    seed: int,
    folds: int,
) -> np.ndarray:
    class_counts = np.bincount(y)
    n_splits = min(folds, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least two examples per class for cross-validation.")

    pipeline = Pipeline(
        steps=[
            ("projection", BinaryProjectionTransformer(matrix)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)


def rerank_candidates_by_cv(
    candidates: list[Candidate],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
) -> list[Candidate]:
    rescored: list[Candidate] = []
    for index, candidate in enumerate(candidates):
        cv_scores = cross_validate_signature(X, y, candidate.matrix, seed + index, folds)
        rescored.append(
            Candidate(
                score=candidate.score,
                separation=candidate.separation,
                auc=candidate.auc,
                diversity=candidate.diversity,
                matrix=candidate.matrix,
                positive_mean_signature=candidate.positive_mean_signature,
                negative_mean_signature=candidate.negative_mean_signature,
                cv_auc_scores=cv_scores,
            )
        )
    return sorted(
        rescored,
        key=lambda candidate: (candidate.cv_auc_mean or -1.0, candidate.score, candidate.auc),
        reverse=True,
    )


def format_digit_lengths(values: list[int]) -> str:
    counts = Counter(digit_length(value) for value in values)
    return ", ".join(f"{length}d:{count}" for length, count in sorted(counts.items()))


def describe_row(weights: np.ndarray, max_terms: int | None = 8) -> str:
    indices = np.argsort(np.abs(weights))[::-1]
    parts: list[str] = []
    for index in indices:
        coeff = int(weights[index])
        if coeff == 0:
            continue
        label = FEATURE_NAMES[index]
        parts.append(f"{coeff:+d}*{label}")
        if max_terms is not None and len(parts) == max_terms:
            break
    return " ".join(parts) if parts else "0"


def save_best_candidate(
    output_path: Path,
    positives: list[int],
    controls: list[int],
    X: np.ndarray,
    config: SearchConfig,
    candidate: Candidate,
) -> None:
    payload = {
        "dataset": {
            "positives": len(positives),
            "controls": len(controls),
            "feature_dim": int(X.shape[1]),
            "positive_digit_lengths": format_digit_lengths(positives),
            "control_digit_lengths": format_digit_lengths(controls),
        },
        "search": {
            "mode": config.search_mode,
            "trials": config.trials,
            "out_dim": config.out_dim,
            "coeff_low": config.coeff_low,
            "coeff_high": config.coeff_high,
            "keep_top": config.keep_top,
            "seed": config.seed,
            "search_sample_per_class": config.search_sample_per_class,
            "controls_per_positive": config.controls_per_positive,
            "cv_folds": config.cv_folds,
            "ramanujan_order": config.ramanujan_order,
        },
        "candidate": {
            "heuristic_score": candidate.score,
            "separation": candidate.separation,
            "code_auc": candidate.auc,
            "diversity": candidate.diversity,
            "rf_cv_auc_scores": candidate.cv_auc_scores.tolist() if candidate.cv_auc_scores is not None else None,
            "rf_cv_auc_mean": candidate.cv_auc_mean,
            "positive_mean_signature": candidate.positive_mean_signature.tolist(),
            "negative_mean_signature": candidate.negative_mean_signature.tolist(),
            "matrix": candidate.matrix.tolist(),
            "row_formulas": [describe_row(row, max_terms=None) for row in candidate.matrix],
            "row_formulas_concise": [describe_row(row, max_terms=8) for row in candidate.matrix],
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_report(
    positives: list[int],
    controls: list[int],
    X: np.ndarray,
    config: SearchConfig,
    candidates: list[Candidate],
) -> None:
    print(f"positives: {len(positives)}")
    print(f"controls: {len(controls)}")
    print(f"feature_dim: {X.shape[1]}")
    print(f"search mode: {config.search_mode}")
    if config.ramanujan_order is not None:
        print(f"ramanujan order: {config.ramanujan_order}")
    print(f"positive digit lengths: {format_digit_lengths(positives)}")
    print(f"control digit lengths:  {format_digit_lengths(controls)}")
    print(f"search sample per class: {config.search_sample_per_class}")
    print()

    for rank, candidate in enumerate(candidates[:5], start=1):
        print(f"=== candidate {rank} ===")
        print(
            "heuristic:",
            round(candidate.score, 4),
            "sep:",
            round(candidate.separation, 4),
            "code_auc:",
            round(candidate.auc, 4),
            "diversity:",
            round(candidate.diversity, 4),
        )
        if candidate.cv_auc_scores is not None:
            rounded = np.round(candidate.cv_auc_scores, 4)
            print("rf_cv_auc:", rounded.tolist())
            print("mean rf_cv_auc:", round(float(np.mean(candidate.cv_auc_scores)), 4))
        print("positive mean signature:", np.round(candidate.positive_mean_signature, 3))
        print("negative mean signature:", np.round(candidate.negative_mean_signature, 3))
        print("rows:")
        for row_index, row in enumerate(candidate.matrix, start=1):
            print(f"  r{row_index:02d}: {describe_row(row)}")
        print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load_raw_dataset(args)
    expand_ranges = not args.no_expand_ranges
    positives = parse_numbers(raw, expand_ranges=expand_ranges, max_range_expand=args.max_range_expand)
    if not positives:
        raise ValueError(
            "No positives were parsed. Provide data via --stdin, --raw, --input-file, or edit RAW."
        )
    if raw.strip() == PLACEHOLDER:
        raise ValueError("RAW still contains the placeholder dataset.")

    controls = make_controls(positives, k=args.controls_per_positive, seed=args.seed)
    X_pos = build_feature_matrix(positives)
    X_neg = build_feature_matrix(controls)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int8)
    ramanujan_order = None
    if args.search_mode == "ramanujan":
        ramanujan_order = resolve_ramanujan_order(X.shape[1], args.out_dim, args.ramanujan_order)

    config = SearchConfig(
        trials=args.trials,
        out_dim=args.out_dim,
        coeff_low=args.coeff_low,
        coeff_high=args.coeff_high,
        keep_top=args.keep_top,
        seed=args.seed,
        search_sample_per_class=args.search_sample_per_class,
        controls_per_positive=args.controls_per_positive,
        cv_folds=args.cv_folds,
        search_mode=args.search_mode,
        ramanujan_order=ramanujan_order,
    )

    heuristic_top = search_candidates(X, y, config)
    ranked = rerank_candidates_by_cv(heuristic_top, X, y, args.seed, args.cv_folds)
    print_report(positives, controls, X, config, ranked)
    if args.save_best_file is not None and ranked:
        save_best_candidate(args.save_best_file, positives, controls, X, config, ranked[0])
        print(f"saved best candidate: {args.save_best_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
