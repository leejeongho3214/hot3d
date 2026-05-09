#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("/Users/jeongho/Desktop/hot3d_vis")
DEFAULT_INPUT_FILES = [
    "s_bps_1stage.pkl",
    "s_cov_map.pkl",
    "s_gaze_cov.pkl",
    "s_gaze_cov_point++.pkl",
    "diffh2o.pkl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw actual wrist trajectories used by eval_metric SD/OD."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-files", nargs="+", default=DEFAULT_INPUT_FILES)
    parser.add_argument("--output-dir", type=Path, default=Path("wrist_trajectory_vis"))
    parser.add_argument("--max-overall", type=int, default=100)
    parser.add_argument("--max-prompt", type=int, default=80)
    parser.add_argument("--ranked-prompts", type=int, default=3)
    parser.add_argument(
        "--samples-per-ranked-prompt",
        type=int,
        default=5,
        help="Maximum number of samples drawn for each ranked high/low-SD prompt.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=26,
        help="Maximum number of trajectories to label A, B, C, ... per plot.",
    )
    return parser.parse_args()


def _to_numpy(data) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _selected_hands(text: str) -> tuple[bool, bool]:
    text = str(text).lower()
    use_right = "right" in text
    use_left = "left" in text
    if "both" in text:
        use_left = True
        use_right = True
    return use_left, use_right


def _sequence_length(data) -> int:
    arr = _to_numpy(data)
    if arr.ndim >= 2:
        return int(arr.shape[0])
    return 0


def _looks_like_text_field(value) -> bool:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return isinstance(value, str) and len(value.strip()) > 0


def _looks_like_object_meta_list(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and (
        "object_name" in first or "obj_pc_org" in first or "data_id" in first
    )


def _looks_like_eval_meta_list(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and (
        "contact" in first
        or "penetration_ok" in first
        or "success" in first
        or "pen_max_mm" in first
    )


def _batch_size(data) -> int:
    if data is None:
        return 0
    if isinstance(data, (list, tuple)):
        return len(data)
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        if data.ndim >= 3:
            return int(data.shape[0])
        if data.ndim >= 2:
            return 1
        return 0
    return 1


def _batch_item(data, idx: int):
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return data[idx] if len(data) > idx else None
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        if data.ndim >= 3:
            return data[idx] if data.shape[0] > idx else None
        if data.ndim >= 2:
            return data if idx == 0 else None
        return data if idx == 0 else None
    return data


def iter_samples(record):
    if not isinstance(record, (list, tuple)):
        return
    object_meta_list = None
    if len(record) >= 10 and _looks_like_eval_meta_list(record[4]):
        x_obj, lhand, rhand, text = record[:4]
        if len(record) > 10 and _looks_like_object_meta_list(record[10]):
            object_meta_list = record[10]
    elif len(record) == 11:
        if _looks_like_object_meta_list(record[4]) or _looks_like_object_meta_list(record[10]):
            x_obj, lhand, rhand, text = record[:4]
            object_meta_list = record[10] if _looks_like_object_meta_list(record[10]) else record[4]
        elif _looks_like_text_field(record[6]):
            x_obj, lhand, rhand, text = record[3], record[4], record[5], record[6]
        else:
            _fine_lhand, _fine_rhand, x_obj, text, lhand, rhand = record[:6]
    elif len(record) == 10:
        x_obj, lhand, rhand, text = record[:4]
    elif len(record) == 8:
        x_obj, lhand, rhand, text = record[:4]
    elif len(record) == 7:
        x_obj, lhand, rhand, text = record[:4]
    else:
        return

    batch_size = max(_batch_size(x_obj), _batch_size(lhand), _batch_size(rhand), _batch_size(text))
    for i in range(batch_size):
        l_i = _batch_item(lhand, i)
        r_i = _batch_item(rhand, i)
        if l_i is None or r_i is None:
            continue
        text_i = _batch_item(text, i)
        if text_i is None:
            text_i = text
        yield {
            "text": str(text_i),
            "obj_params": _batch_item(x_obj, i),
            "lhand_params": l_i,
            "rhand_params": r_i,
            "object_meta": _batch_item(object_meta_list, i),
            "sample_idx": i,
        }


def rot6d_to_rotmat(rot6d) -> np.ndarray:
    x = torch.as_tensor(rot6d, dtype=torch.float32).reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = torch.nn.functional.normalize(a1)
    b2 = torch.nn.functional.normalize(
        a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1).detach().cpu().numpy()


def canonicalize_wrist_trajectory(traj: np.ndarray, obj_params) -> np.ndarray | None:
    obj = _to_numpy(obj_params)
    if obj.ndim == 1:
        obj = obj[None, :]
    if obj.ndim != 2 or obj.shape[1] < 9 or obj.shape[0] == 0:
        return None
    t = min(traj.shape[0], obj.shape[0])
    out = np.full_like(traj, np.nan, dtype=np.float32)
    rot = rot6d_to_rotmat(obj[:t, 3:9])
    trans = obj[:t, :3].astype(np.float64)
    for frame_idx in range(t):
        frame = traj[frame_idx]
        valid = np.all(np.isfinite(frame), axis=1)
        if valid.any():
            out[frame_idx, valid] = np.einsum(
                "ni,ij->nj",
                frame[valid].astype(np.float64) - trans[frame_idx],
                rot[frame_idx],
            ).astype(np.float32)
    return out


def wrist_trajectory(sample: dict) -> np.ndarray | None:
    use_left, use_right = _selected_hands(sample["text"])
    nframes = max(_sequence_length(sample["lhand_params"]), _sequence_length(sample["rhand_params"]))
    if nframes <= 0 or (not use_left and not use_right):
        return None
    traj = np.full((nframes, 2, 3), np.nan, dtype=np.float32)
    if use_left:
        l = _to_numpy(sample["lhand_params"])
        if l.ndim == 2 and l.shape[1] >= 3:
            t = min(nframes, l.shape[0])
            traj[:t, 0, :] = l[:t, :3]
    if use_right:
        r = _to_numpy(sample["rhand_params"])
        if r.ndim == 2 and r.shape[1] >= 3:
            t = min(nframes, r.shape[0])
            traj[:t, 1, :] = r[:t, :3]
    return canonicalize_wrist_trajectory(traj, sample["obj_params"])


def active_wrist_centroid(traj: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(traj), axis=2)
    out = np.full((traj.shape[0], 3), np.nan, dtype=np.float32)
    for frame_idx in range(traj.shape[0]):
        if np.any(valid[frame_idx]):
            out[frame_idx] = np.mean(traj[frame_idx, valid[frame_idx]], axis=0)
    return out


def wrist_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    t = min(a.shape[0], b.shape[0])
    vals = []
    for hand_idx in (0, 1):
        aa = a[:t, hand_idx, :]
        bb = b[:t, hand_idx, :]
        valid = np.all(np.isfinite(aa), axis=1) & np.all(np.isfinite(bb), axis=1)
        if valid.any():
            vals.extend(np.linalg.norm(aa[valid] - bb[valid], axis=1).tolist())
    if vals:
        return float(np.mean(vals))

    a_center = active_wrist_centroid(a[:t])
    b_center = active_wrist_centroid(b[:t])
    valid = np.all(np.isfinite(a_center), axis=1) & np.all(np.isfinite(b_center), axis=1)
    if not valid.any():
        return None
    return float(np.linalg.norm(a_center[valid] - b_center[valid], axis=1).mean())


def mean_pairwise(trajs: list[np.ndarray]) -> float:
    if len(trajs) < 2:
        return 0.0
    vals = []
    for i in range(len(trajs)):
        for j in range(i + 1, len(trajs)):
            dist = wrist_distance(trajs[i], trajs[j])
            if dist is not None:
                vals.append(dist)
    return float(np.mean(vals)) if vals else 0.0


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def trajectory_label(index: int) -> str:
    label = ""
    index = int(index)
    while True:
        label = chr(ord("A") + (index % 26)) + label
        index = index // 26 - 1
        if index < 0:
            return label


def plot_trajectories(
    rows: list[dict],
    title: str,
    output_path: Path,
    max_rows: int,
    max_labels: int,
) -> None:
    rows = [r for r in rows if r.get("wrist_traj") is not None]
    if len(rows) > max_rows:
        idx = np.linspace(0, len(rows) - 1, max_rows, dtype=np.int64)
        rows = [rows[int(i)] for i in idx]

    fig = plt.figure(figsize=(12, 5.7))
    ax3d = fig.add_subplot(121, projection="3d")
    axxy = fig.add_subplot(122)
    all_points = []
    for row_idx, row in enumerate(rows):
        traj = row["wrist_traj"]
        use_left, use_right = _selected_hands(row["text"])
        active_hand_count = int(use_left) + int(use_right)
        row_label = trajectory_label(row_idx)
        for hand_idx, color, label in [(0, "#c0392b", "left"), (1, "#2980b9", "right")]:
            if (hand_idx == 0 and not use_left) or (hand_idx == 1 and not use_right):
                continue
            pts = traj[:, hand_idx, :]
            valid = np.all(np.isfinite(pts), axis=1)
            pts = pts[valid]
            if len(pts) == 0:
                continue
            all_points.append(pts)
            ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.32, linewidth=1.2)
            ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=color, s=12, alpha=0.8)
            axxy.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.32, linewidth=1.2)
            axxy.scatter(pts[0, 0], pts[0, 1], color=color, s=12, alpha=0.8)
            if row_idx < max_labels:
                marker_label = row_label if active_hand_count <= 1 else f"{row_label}-{label[0].upper()}"
                ax3d.text(
                    pts[0, 0],
                    pts[0, 1],
                    pts[0, 2],
                    marker_label,
                    color=color,
                    fontsize=9,
                    weight="bold",
                )
                axxy.text(
                    pts[0, 0],
                    pts[0, 1],
                    marker_label,
                    color=color,
                    fontsize=9,
                    weight="bold",
                )

    if all_points:
        stacked = np.concatenate(all_points, axis=0)
        set_equal_3d_axes(ax3d, stacked)
        axxy.set_aspect("equal", adjustable="box")
    ax3d.set_title("3D wrist trajectories")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    axxy.set_title("x-y projection")
    axxy.set_xlabel("x (m)")
    axxy.set_ylabel("y (m)")
    axxy.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def select_rows_for_prompt_plot(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    idx = np.linspace(0, len(rows) - 1, max_samples, dtype=np.int64)
    return [rows[int(i)] for i in idx]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in args.input_files:
        path = Path(file_name)
        if not path.is_absolute():
            path = args.input_dir.expanduser() / path
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue
        with path.open("rb") as f:
            records = pickle.load(f)

        rows = []
        for record in records:
            for sample in iter_samples(record) or []:
                traj = wrist_trajectory(sample)
                if traj is None:
                    continue
                sample["wrist_traj"] = traj
                rows.append(sample)
        if not rows:
            print(f"[WARN] no trajectories: {path.name}")
            continue

        stem = path.stem
        plot_trajectories(
            rows,
            f"{path.name}: all wrist trajectories ({len(rows)} samples)",
            output_dir / f"{stem}_overall_wrist_trajectories.png",
            args.max_overall,
            args.max_labels,
        )

        by_text: dict[str, list[dict]] = {}
        for row in rows:
            by_text.setdefault(row["text"], []).append(row)
        ranked = sorted(
            ((text, mean_pairwise([r["wrist_traj"] for r in group]), group) for text, group in by_text.items() if len(group) >= 2),
            key=lambda x: x[1],
            reverse=True,
        )
        if ranked:
            top_text, top_sd, top_rows = ranked[0]
            plot_trajectories(
                top_rows,
                f"{path.name}: highest-SD prompt (SD={top_sd:.3f}m)\n{top_text}",
                output_dir / f"{stem}_highest_sd_prompt_wrist_trajectories.png",
                args.max_prompt,
                args.max_labels,
            )

            k = max(0, int(args.ranked_prompts))
            for rank, (text, sd, group) in enumerate(ranked[:k], start=1):
                group_for_plot = select_rows_for_prompt_plot(
                    group, args.samples_per_ranked_prompt
                )
                plot_trajectories(
                    group_for_plot,
                    f"{path.name}: high SD #{rank} (SD={sd:.3f}m, showing {len(group_for_plot)}/{len(group)} samples)\n{text}",
                    output_dir / f"{stem}_high_sd_{rank:02d}_wrist_trajectories.png",
                    args.max_prompt,
                    args.max_labels,
                )
            for rank, (text, sd, group) in enumerate(reversed(ranked[-k:]), start=1):
                group_for_plot = select_rows_for_prompt_plot(
                    group, args.samples_per_ranked_prompt
                )
                plot_trajectories(
                    group_for_plot,
                    f"{path.name}: low SD #{rank} (SD={sd:.3f}m, showing {len(group_for_plot)}/{len(group)} samples)\n{text}",
                    output_dir / f"{stem}_low_sd_{rank:02d}_wrist_trajectories.png",
                    args.max_prompt,
                    args.max_labels,
                )
        print(f"{path.name}: wrote wrist trajectory plots for {len(rows)} samples")


if __name__ == "__main__":
    main()
