import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import itertools
import json
import inspect
import re
from typing import Optional

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
import trimesh

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

from data_loaders.mano_layer import MANOHandModel
from hot3d.mano import build_mano_aa

if not hasattr(inspect, "getargspec"):
    # Compatibility for chumpy on Python 3.11+
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
import numpy as np

# NumPy 1.24+ compatibility for legacy packages (e.g., chumpy)
for _name, _type in [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("object", object),
    ("str", str),
]:
    # Use __dict__ to avoid triggering NumPy deprecation warnings during attribute access
    if _name not in np.__dict__:
        setattr(np, _name, _type)

import trimesh

from collections import defaultdict

import os
import torch
import numpy as np


from rot import *

import rerun as rr
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)


def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)

def _sanitize_entity_path(text: str) -> str:
    sanitized = re.sub(r"\s+", "_", text.strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.\\-]", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "entry"

class ObjectModel:
    def __init__(self, pkl_file):
        self.pkl_file = pkl_file
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            self.object_name = data["object_name"]
            self.obj_pcs = data["obj_pcs"]
            self.obj_pc_normals = data["obj_pc_normals"]
            self.point_sets = data["point_sets"]
            self.obj_path = data["obj_path"]
            if "obj_pc_top" in data:
                self.obj_pc_top = data["obj_pc_top"]
            else:
                self.obj_pc_top = None

    def __call__(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_name[object_name]
        point_set = self.point_sets[object_name].copy()
        obj_pc = self.obj_pcs[object_name].copy()
        obj_pc_normal = self.obj_pc_normals[object_name].copy()
        obj_path = self.obj_path[object_name]
        if self.obj_pc_top is not None:
            obj_pc_top = self.obj_pc_top[object_name].copy()
            return point_set, obj_pc, obj_pc_normal, obj_path, obj_pc_top
        else:
            return point_set, obj_pc, obj_pc_normal, obj_path


home = os.path.expanduser("~")
mano_hand_model_path = os.path.join(home, "Desktop/hot3d_vis/mano_v1_2/models")
mano_hand_model = None
if mano_hand_model_path is not None:
    mano_hand_model = MANOHandModel(mano_hand_model_path)


def process_hand_result(hand_layer, hand_params):
    hand_pose = hand_params[:, 3:]
    hand_pose = rot6d_to_axis_angle(hand_pose).reshape(-1, 48)
    hand_trans = hand_params[:, :3]
    duration = hand_trans.shape[0]
    out = hand_layer(
        global_orient=hand_pose[:, :3],
        hand_pose=hand_pose[:, 3:48],
        betas=torch.zeros((duration, 10)),
    )
    hand_trans = hand_trans.unsqueeze(1)
    hand_vertices = out.vertices + hand_trans
    hand_faces = hand_layer.faces.copy().astype(np.int16)
    hand_faces = torch.LongTensor(hand_faces)
    return hand_vertices, hand_faces


def process_obj_result(obj_verts, obj_params):
    obj_trans = obj_params[:, :3]
    obj_rot6d = obj_params[:, 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
    obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(1)
    return obj_verts_transformed


def rigid_transform(A, B):
    """두 포인트 클라우드 A, B (N,3) numpy array에 대해, B로 정렬하는 R, t 반환"""
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


with open(os.path.join(home, "Desktop/hot3d_vis/instance.json"), "r") as f:
    instance_ = json.load(f)

object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
obj_pc = dict()

for obj_name in object_model.obj_pcs.keys():
    _, pc, _, _ = object_model(obj_name)
    obj_pc[obj_name] = torch.tensor(pc)


def _extract_of_with_target(text: str) -> Optional[str]:
    match = re.search(r"\bof\s+(.+?)\s+with\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    target = re.sub(r"\s+", " ", match.group(1)).strip()
    return target.lower() if target else None


def _build_grouping_keys(text: str) -> tuple[str, str]:
    object_key = _extract_of_with_target(text) or text
    action_key = f"{object_key}::{text}"
    return object_key, action_key

def _reduce_cov_counts(cov_frame: np.ndarray, num_points: int) -> Optional[np.ndarray]:
    cov_counts = None
    if cov_frame.ndim >= 2:
        if cov_frame.shape[-1] == num_points:
            axes = tuple(range(cov_frame.ndim - 1))
            cov_counts = cov_frame.sum(axis=axes)
        elif cov_frame.shape[0] == num_points:
            axes = tuple(range(1, cov_frame.ndim))
            cov_counts = cov_frame.sum(axis=axes)
    elif cov_frame.ndim == 1 and cov_frame.shape[0] == num_points:
        cov_counts = cov_frame
    return cov_counts


def _compute_point_colors(cov_values: np.ndarray) -> np.ndarray:
    max_count = float(np.max(cov_values)) if np.max(cov_values) > 0 else 1.0
    intensity = np.clip(cov_values / max_count, 0.0, 1.0).astype(np.float32)
    low_color = np.array([0, 0, 255], dtype=np.float32)
    high_color = np.array([255, 255, 0], dtype=np.float32)
    return (low_color + (high_color - low_color) * intensity[:, None]).astype(np.uint8)


def _render_histogram_image(counts: np.ndarray, height: int = 160, bar_width: int = 4) -> np.ndarray:
    if counts.size == 0:
        return np.full((height, 1, 3), 255, dtype=np.uint8)
    max_count = int(np.max(counts))
    if max_count <= 0:
        return np.full((height, 1, 3), 255, dtype=np.uint8)
    width = counts.shape[0] * bar_width
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    max_value = float(np.max(counts))
    for idx, value in enumerate(counts):
        if value <= 0:
            continue
        bar_height = int((value / max_value) * (height - 1))
        x0 = idx * bar_width
        x1 = x0 + bar_width
        y0 = height - bar_height
        img[y0:height, x0:x1, :] = 0
    return img


# The saved pickle references "__main__.Hot3DActionDataset". Make sure this
# name exists so pickle can resolve the dataset class.

def _to_1d_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _load_item_list(obj):
    if isinstance(obj, dict) and "save_list" in obj:
        obj = obj["save_list"]
    if not isinstance(obj, (list, tuple)):
        raise ValueError("Input does not contain a list of [text, duration] pairs.")
    return obj


def _aggregate_by_text(item_list):
    by_text = defaultdict(list)
    for item in item_list:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text, duration = item[0], item[1]
        durations = _to_1d_numpy(duration)
        if durations.size == 0:
            continue
        by_text[str(text)].append(durations)
    flat_by_text = {}
    for text, chunks in by_text.items():
        flat_by_text[text] = np.concatenate(chunks, axis=0)
    return flat_by_text


def _compute_stats(flat_by_text):
    rows = []
    for text, values in flat_by_text.items():
        if values.size == 0:
            continue
        rows.append(
            {
                "text": text,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "var": float(np.var(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    rows.sort(key=lambda r: (-r["count"], r["text"]))
    return rows


def _save_stats_csv(rows, path):
    if not rows:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("text,count,mean,std,var,min,max\n")
        for row in rows:
            safe_text = row["text"].replace('"', '""')
            f.write(
                f"\"{safe_text}\",{row['count']},{row['mean']:.2f},"
                f"{row['std']:.2f},{row['var']:.2f},{row['min']:.2f},{row['max']:.2f}\n"
            )


def _plot_mean_std(rows, output_path, top_k, close=True):
    if not rows:
        return
    rows = rows[:top_k] if top_k > 0 else rows
    labels = [r["text"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    fig_w = max(8, 0.45 * len(labels))
    plt.figure(figsize=(fig_w, 5))
    x = np.arange(len(labels))
    bars = plt.bar(x, means, yerr=stds, capsize=3)
    ax = plt.gca()
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.02,
            f"{means[idx]:.2f}\n{stds[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            transform=ax.get_xaxis_transform(),
        )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Duration (mean ± std)")
    plt.title("Per-text duration mean/std")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if close:
        plt.close()


def _plot_violin(flat_by_text, rows, output_path, top_k, close=True):
    if not rows:
        return
    rows = rows[:top_k] if top_k > 0 else rows
    labels = [r["text"] for r in rows]
    data = [flat_by_text[label] for label in labels]
    fig_w = max(8, 0.5 * len(labels))
    plt.figure(figsize=(fig_w, 5))
    parts = plt.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Duration")
    plt.title("Per-text duration distribution (violin)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if close:
        plt.close()


def _plot_overall_hist(flat_by_text, output_path, bins=40, close=True):
    all_values = np.concatenate(list(flat_by_text.values()), axis=0) if flat_by_text else np.array([])
    if all_values.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(all_values, bins=bins, color="#2a6f97", alpha=0.85)
    plt.xlabel("Duration")
    plt.ylabel("Count")
    plt.title("Overall duration distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if close:
        plt.close()


# The saved pickle references "__main__.Hot3DActionDataset". Make sure this
# name exists so pickle can resolve the dataset class.

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot motion length statistics.")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.expanduser("~"), "Desktop/hot3d_vis/grab_ori.pkl"),
        help="Path to pickle containing save_list of [text, duration] pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots; defaults to input file directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K texts (by count) to visualize; use 0 for all.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    return parser.parse_args()


def _log_point_cloud(base_path: str, points: np.ndarray, cov_values: np.ndarray) -> None:
    colors = _compute_point_colors(cov_values)
    rr.log(
        f"{base_path}/point_cloud",
        rr.Points3D(
            positions=points,
            radii=0.005,
            colors=colors,
        ),
        static=True,
    )


def main() -> None:
    args = _parse_args()

    with open(args.input, "rb") as f:
        item_list = pickle.load(f)
    item_list = _load_item_list(item_list)

    flat_by_text = _aggregate_by_text(item_list)
    stats_rows = _compute_stats(flat_by_text)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    stats_path = os.path.join(output_dir, f"{base_name}_stats.csv")
    _save_stats_csv(stats_rows, stats_path)

    mean_std_path = os.path.join(output_dir, f"{base_name}_mean_std.png")
    violin_path = os.path.join(output_dir, f"{base_name}_violin.png")
    overall_hist_path = os.path.join(output_dir, f"{base_name}_overall_hist.png")

    _plot_mean_std(stats_rows, mean_std_path, args.top_k, close=not args.show)
    _plot_violin(flat_by_text, stats_rows, violin_path, args.top_k, close=not args.show)
    _plot_overall_hist(flat_by_text, overall_hist_path, close=not args.show)

    if args.show:
        if stats_rows:
            plt.show()



if __name__ == "__main__":
    main()
