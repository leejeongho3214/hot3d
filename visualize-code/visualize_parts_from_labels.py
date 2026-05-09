#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize object point clouds with part labels from label_merged.json."
    )
    parser.add_argument(
        "--obj-pkl",
        type=Path,
        default=Path("/Users/jeongho/Desktop/hot3d_vis/obj.pkl"),
        help="Path to obj.pkl",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/Users/jeongho/Library/CloudStorage/SynologyDrive-home/Vscode/hot3d/label_merged.json"),
        help="Path to label_merged.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/jeongho/Desktop/hot3d_vis/part_vis"),
        help="Directory to save rendered images",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Scatter point size",
    )
    parser.add_argument(
        "--overview-cols",
        type=int,
        default=4,
        help="Number of columns in the overview mosaic",
    )
    parser.add_argument(
        "--overview-scale",
        type=float,
        default=1.0,
        help="Scale factor for overview figure size",
    )
    return parser.parse_args()


def load_inputs(obj_pkl_path: Path, labels_path: Path) -> tuple[dict, dict]:
    with obj_pkl_path.open("rb") as f:
        obj_data = pickle.load(f)
    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return obj_data, labels


def build_part_colors(part_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10" if len(part_names) <= 10 else "tab20")
    return {part: cmap(i % cmap.N) for i, part in enumerate(part_names)}


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_single_object(
    object_name: str,
    points: np.ndarray,
    part_to_indices: dict[str, list[int]],
    output_path: Path,
    point_size: float,
) -> None:
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    part_names = list(part_to_indices.keys())
    part_colors = build_part_colors(part_names)

    for part_name, indices in part_to_indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            points[idx, 2],
            s=point_size,
            color=part_colors[part_name],
            alpha=0.95,
            linewidths=0,
        )

    set_equal_axes(ax, points)
    ax.set_title(object_name, pad=18)
    ax.view_init(elev=20, azim=40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{part_name} ({len(indices)})",
            markerfacecolor=part_colors[part_name],
            markersize=8,
        )
        for part_name, indices in part_to_indices.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.18, 1.02))

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_overview(
    labels: dict[str, dict[str, list[int]]],
    obj_pcs: dict[str, np.ndarray],
    output_path: Path,
    point_size: float,
    cols: int,
    scale: float,
) -> None:
    object_names = list(labels.keys())
    cols = max(1, int(cols))
    rows = math.ceil(len(object_names) / cols)
    fig = plt.figure(figsize=(cols * 4.3 * scale, rows * 3.6 * scale))

    for plot_index, object_name in enumerate(object_names, start=1):
        ax = fig.add_subplot(rows, cols, plot_index, projection="3d")
        points = np.asarray(obj_pcs[object_name])
        part_to_indices = labels[object_name]
        part_names = list(part_to_indices.keys())
        part_colors = build_part_colors(part_names)

        for part_name, indices in part_to_indices.items():
            idx = np.asarray(indices, dtype=np.int64)
            ax.scatter(
                points[idx, 0],
                points[idx, 1],
                points[idx, 2],
                s=point_size * 0.7,
                color=part_colors[part_name],
                alpha=0.95,
                linewidths=0,
            )

        set_equal_axes(ax, points)
        ax.set_title(object_name, fontsize=max(8, 9 * scale), pad=8)
        ax.view_init(elev=18, azim=40)
        ax.set_axis_off()

    fig.suptitle(
        "HOT3D object part point-cloud overview",
        fontsize=max(12, 16 * scale),
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def validate(labels: dict, obj_pcs: dict) -> list[str]:
    validated_names = []
    for object_name, part_to_indices in labels.items():
        if object_name not in obj_pcs:
            raise KeyError(f"{object_name} is missing from obj.pkl")
        points = np.asarray(obj_pcs[object_name])
        point_count = len(points)

        covered = []
        for part_name, indices in part_to_indices.items():
            if not indices:
                raise ValueError(f"{object_name}:{part_name} has no indices")
            idx = np.asarray(indices, dtype=np.int64)
            if idx.min() < 0 or idx.max() >= point_count:
                raise IndexError(
                    f"{object_name}:{part_name} index out of range for {point_count} points"
                )
            covered.extend(indices)

        if len(set(covered)) != point_count:
            missing = sorted(set(range(point_count)) - set(covered))
            overlap_count = len(covered) - len(set(covered))
            raise ValueError(
                f"{object_name} labels do not form a clean partition; "
                f"missing={len(missing)}, overlaps={overlap_count}"
            )
        validated_names.append(object_name)
    return validated_names


def main() -> None:
    args = parse_args()
    obj_data, labels = load_inputs(args.obj_pkl, args.labels)
    obj_pcs = obj_data["obj_pcs"]

    validated_names = validate(labels, obj_pcs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_object_dir = args.output_dir / "objects"
    per_object_dir.mkdir(parents=True, exist_ok=True)

    for object_name in validated_names:
        render_single_object(
            object_name=object_name,
            points=np.asarray(obj_pcs[object_name]),
            part_to_indices=labels[object_name],
            output_path=per_object_dir / f"{object_name}.png",
            point_size=args.point_size,
        )

    render_overview(
        labels=labels,
        obj_pcs=obj_pcs,
        output_path=args.output_dir / "overview.png",
        point_size=args.point_size,
        cols=args.overview_cols,
        scale=args.overview_scale,
    )

    print(f"Rendered {len(validated_names)} labeled objects")
    print(f"Overview: {args.output_dir / 'overview.png'}")
    print(f"Per-object images: {per_object_dir}")


if __name__ == "__main__":
    main()
