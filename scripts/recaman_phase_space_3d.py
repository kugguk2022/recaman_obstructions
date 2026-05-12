from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D phase-space embeddings of the Recaman sequence."
    )
    parser.add_argument("--steps", type=int, default=2800, help="Number of Recaman steps to generate.")
    parser.add_argument(
        "--mode",
        choices=("delay", "spatiotemporal", "arc-lift"),
        default="delay",
        help="3D embedding mode.",
    )
    parser.add_argument("--tau", type=int, default=1, help="Delay for delay embedding.")
    parser.add_argument(
        "--samples-per-arc",
        type=int,
        default=40,
        help="Samples per step arc in arc-lift mode.",
    )
    parser.add_argument(
        "--twist",
        type=float,
        default=0.0,
        help="Global twist strength for arc-lift mode. Values around 1-3 create a spiral sheet.",
    )
    parser.add_argument(
        "--elevation-scale",
        type=float,
        default=1.0,
        help="Vertical lift scale for arc-lift mode.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.0,
        help="Line width for trajectory segments.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional image path. If provided, the figure is saved there.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI when saving.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    return parser.parse_args()


def recaman(steps: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(steps + 1, dtype=np.int64)
    blocked = np.zeros(steps + 1, dtype=np.int8)
    visited = {0}
    for n in range(1, steps + 1):
        candidate = int(values[n - 1] - n)
        if candidate > 0 and candidate not in visited:
            values[n] = candidate
            blocked[n] = 0
        else:
            values[n] = values[n - 1] + n
            blocked[n] = 1
        visited.add(int(values[n]))
    return values, blocked


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def build_delay_embedding(values: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray]:
    if tau < 1:
        raise ValueError("tau must be >= 1")
    if len(values) <= 2 * tau:
        raise ValueError("Need more steps for the requested delay.")
    points = np.column_stack(
        [
            values[:-2 * tau],
            values[tau:-tau],
            values[2 * tau :],
        ]
    ).astype(float)
    colors = normalize(np.arange(len(points)))
    return points, colors


def build_spatiotemporal_embedding(values: np.ndarray, blocked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(1, len(values), dtype=float)
    sequence = values[1:].astype(float)
    delta = np.diff(values).astype(float)
    signed_step = np.where(blocked[1:] == 1, delta, -delta)
    points = np.column_stack([indices, sequence, signed_step])
    colors = normalize(sequence)
    return points, colors


def rotate_xy(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def build_arc_lift_embedding(
    values: np.ndarray,
    blocked: np.ndarray,
    samples_per_arc: int,
    twist: float,
    elevation_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if samples_per_arc < 4:
        raise ValueError("samples-per-arc must be >= 4")

    arc_points: list[np.ndarray] = []
    arc_colors: list[np.ndarray] = []
    total_steps = len(values) - 1
    max_gap = max(1.0, float(np.max(np.abs(np.diff(values)))))
    height_scale = elevation_scale / max_gap

    for n in range(1, len(values)):
        start = float(values[n - 1])
        end = float(values[n])
        radius = abs(end - start) / 2.0
        center = (start + end) / 2.0
        theta = np.linspace(np.pi, 0.0, samples_per_arc) if end >= start else np.linspace(0.0, np.pi, samples_per_arc)
        x = center + radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.full_like(theta, (n - 1) * elevation_scale)
        z += np.linspace(0.0, radius * height_scale, samples_per_arc)

        if twist != 0.0:
            angle = twist * (n / total_steps) * 2.0 * np.pi
            x, y = rotate_xy(x, y, np.full_like(theta, angle))

        arc_points.append(np.column_stack([x, y, z]))
        arc_colors.append(np.full(samples_per_arc, blocked[n], dtype=float))

    points = np.vstack(arc_points)
    colors = np.concatenate(arc_colors)
    return points, colors


def make_segments(points: np.ndarray) -> np.ndarray:
    return np.stack([points[:-1], points[1:]], axis=1)


def style_axes(ax: plt.Axes, title: str) -> None:
    bg = "#0f1117"
    grid = (0.22, 0.26, 0.36, 0.7)
    ax.set_facecolor(bg)
    ax.figure.set_facecolor(bg)
    ax.set_title(title, color="white", pad=14)
    ax.set_xlabel("X", color="white", labelpad=10)
    ax.set_ylabel("Y", color="white", labelpad=10)
    ax.set_zlabel("Z", color="white", labelpad=10)
    ax.tick_params(colors="#d5d9e0")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.08, 0.09, 0.12, 1.0))
        axis._axinfo["grid"]["color"] = grid
        axis._axinfo["grid"]["linewidth"] = 0.8


def plot_embedding(
    points: np.ndarray,
    colors: np.ndarray,
    blocked: np.ndarray | None,
    args: argparse.Namespace,
    title: str,
) -> None:
    fig = plt.figure(figsize=(11, 8), facecolor="#0f1117")
    ax = fig.add_subplot(111, projection="3d")
    style_axes(ax, title)

    segments = make_segments(points)
    segment_colors = plt.cm.Greys_r(normalize(np.arange(len(segments))))
    line_collection = Line3DCollection(
        segments,
        colors=segment_colors,
        linewidths=args.line_width,
        alpha=0.85,
    )
    ax.add_collection3d(line_collection)

    if blocked is None:
        low = np.array([0.33, 0.64, 1.0, 1.0])
        high = np.array([0.72, 1.0, 0.72, 1.0])
        scatter_colors = low + colors.reshape(-1, 1) * (high - low)
    else:
        scatter_colors = np.where(
            blocked.reshape(-1, 1) > 0,
            np.array([[1.0, 0.55, 0.58, 1.0]]),
            np.array([[0.68, 1.0, 0.68, 1.0]]),
        )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=scatter_colors, s=args.point_size, depthshade=False)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    centers = (maxs + mins) / 2.0
    radius = spans.max() / 2.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.view_init(elev=25, azim=-58)
    plt.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    values, blocked = recaman(args.steps)

    if args.mode == "delay":
        points, colors = build_delay_embedding(values, args.tau)
        title = f"Recaman 3D Delay Embedding (N={args.steps}, tau={args.tau})"
        plot_embedding(points, colors, blocked[2 * args.tau :], args, title)
    elif args.mode == "spatiotemporal":
        points, colors = build_spatiotemporal_embedding(values, blocked)
        title = f"Recaman 3D Spatiotemporal Embedding (N={args.steps})"
        plot_embedding(points, colors, blocked[1:], args, title)
    else:
        points, colors = build_arc_lift_embedding(
            values,
            blocked,
            samples_per_arc=args.samples_per_arc,
            twist=args.twist,
            elevation_scale=args.elevation_scale,
        )
        title = f"Recaman 3D Arc Lift (N={args.steps}, twist={args.twist})"
        plot_embedding(points, colors, None, args, title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
