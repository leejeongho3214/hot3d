import argparse
import csv
import json
import inspect
import re
from typing import Optional

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

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
import trimesh

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)


def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)


import os
import torch
import numpy as np

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rot import *

import rerun as rr
import pickle


import os
from mano import build_mano_aa



_rng = np.random.default_rng()


def random_rgb_color() -> list[int]:
    """Return a random RGB color as a list of ints in [0, 255]."""
    return _rng.integers(0, 256, size=3, dtype=np.uint8).tolist()


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
    hand_joints = out.joints + hand_trans
    hand_faces = hand_layer.faces.copy().astype(np.int16)
    hand_faces = torch.LongTensor(hand_faces)
    return hand_vertices, hand_faces, hand_joints


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


def apply_rigid_transform(points, R, t):
    """Apply a rigid transform to (N,3) points."""
    return (points @ R.T) + t


def _to_torch(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


def _set_frame_time(frame_idx: int) -> None:
    # Rerun API differs by version.
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_idx)
    else:
        rr.set_time("frame", sequence=frame_idx)


def _compute_cov_map_from_course(
    course_lhand,
    course_rhand,
    x_obj,
    obj_points,
    l_hand_layer,
    r_hand_layer,
    threshold: float = 0.02,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return None


def _compute_contact_from_vertices(
    obj_vertices_np: np.ndarray,
    l_hand_vertices_np: np.ndarray,
    r_hand_vertices_np: np.ndarray,
    threshold: float = 0.02,
) -> np.ndarray:
    nframes = obj_vertices_np.shape[0]
    contact_per_frame = np.zeros((nframes, obj_vertices_np.shape[1]), dtype=bool)

    for frame_idx in range(nframes):
        obj_frame = obj_vertices_np[frame_idx]
        frame_contact = np.zeros(obj_frame.shape[0], dtype=bool)
        if l_hand_vertices_np is not None:
            l_frame = l_hand_vertices_np[
                min(frame_idx, l_hand_vertices_np.shape[0] - 1)
            ]
            l_dist = np.linalg.norm(obj_frame[:, None, :] - l_frame[None, :, :], axis=2)
            frame_contact |= l_dist.min(axis=1) < threshold
        if r_hand_vertices_np is not None:
            r_frame = r_hand_vertices_np[
                min(frame_idx, r_hand_vertices_np.shape[0] - 1)
            ]
            r_dist = np.linalg.norm(obj_frame[:, None, :] - r_frame[None, :, :], axis=2)
            frame_contact |= r_dist.min(axis=1) < threshold
        contact_per_frame[frame_idx] = frame_contact
    return contact_per_frame


def _load_part_map(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_part_map_path() -> Optional[str]:
    candidates = [
        os.path.join(os.path.expanduser("~"), "label_merged.json"),
        os.path.join(os.path.expanduser("~"), "labels_merged.json"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "label_merged.json")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "labels_merged.json")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _part_mask_from_map(
    part_map: dict, object_key: str, part_key: str, num_points: int
) -> Optional[np.ndarray]:
    if object_key not in part_map:
        return None
    parts = part_map.get(object_key, {})
    if part_key not in parts:
        return None
    indices = np.asarray(parts[part_key], dtype=np.int64)
    if indices.size == 0:
        return None
    indices = indices[(indices >= 0) & (indices < num_points)]
    if indices.size == 0:
        return None
    mask = np.zeros(num_points, dtype=bool)
    mask[indices] = True
    return mask


def _precision_recall(
    contact_mask: np.ndarray, part_mask: np.ndarray
) -> tuple[float, float, float]:
    tp = np.logical_and(contact_mask, part_mask).sum()
    contact_count = contact_mask.sum()
    part_count = part_mask.sum()
    precision = (tp / contact_count) if contact_count > 0 else 0.0
    recall = (tp / part_count) if part_count > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def _framewise_binary_from_cov_map(
    cov_map, nframes: int, num_points: int
) -> np.ndarray:
    """Best-effort conversion of a cov/contact map into frame-wise binary contact."""
    if cov_map is None or nframes <= 0:
        return np.zeros((max(nframes, 0),), dtype=bool)
    arr = np.asarray(cov_map)
    if arr.size == 0:
        return np.zeros((nframes,), dtype=bool)
    arr = np.squeeze(arr)

    # If any axis matches nframes, treat it as the frame axis.
    frame_axes = [ax for ax, size in enumerate(arr.shape) if size == nframes]
    if frame_axes:
        frame_axis = frame_axes[0]
        arr_f = np.moveaxis(arr, frame_axis, 0)
        if arr_f.ndim == 1:
            return arr_f > 0
        arr_f = arr_f.reshape(nframes, -1)
        return (arr_f > 0).any(axis=1)

    # No clear frame axis: collapse to a scalar and broadcast.
    scalar_contact = bool(np.any(arr > 0))
    return np.full((nframes,), scalar_contact, dtype=bool)


def _reduce_cov_to_point_mask(cov_map, num_points: int) -> Optional[np.ndarray]:
    """Reduce a cov/contact map to a per-point boolean mask when possible."""
    if cov_map is None:
        return None
    arr = np.asarray(cov_map)
    if arr.ndim == 1 and arr.shape[0] == num_points:
        return arr > 0
    if arr.size == 0:
        return None
    point_axes = [ax for ax, size in enumerate(arr.shape) if size == num_points]
    if not point_axes:
        return None
    point_axis = point_axes[0]
    arr_p = np.moveaxis(arr, point_axis, -1)
    reduce_axes = tuple(range(arr_p.ndim - 1))
    return (arr_p > 0).any(axis=reduce_axes)


def _framewise_point_mask_from_cov_map(cov_map, num_points: int) -> Optional[np.ndarray]:
    """Best-effort conversion to per-frame per-point mask of shape [T, num_points]."""
    if cov_map is None or num_points <= 0:
        return None
    arr = np.asarray(cov_map)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.full((1, num_points), bool(arr > 0), dtype=bool)

    point_axes = [ax for ax, size in enumerate(arr.shape) if size == num_points]
    if not point_axes:
        return None

    point_axis = point_axes[0]
    arr_p = np.moveaxis(arr, point_axis, -1)  # [..., num_points]
    if arr_p.ndim == 1:
        return (arr_p > 0).reshape(1, num_points)

    # Use the largest non-point axis as frame axis, collapse the rest.
    non_point_shape = arr_p.shape[:-1]
    frame_axis = int(np.argmax(non_point_shape))
    arr_fp = np.moveaxis(arr_p, frame_axis, 0)  # [T, ..., num_points]
    if arr_fp.ndim > 2:
        reduce_axes = tuple(range(1, arr_fp.ndim - 1))
        arr_fp = (arr_fp > 0).any(axis=reduce_axes)
    else:
        arr_fp = arr_fp > 0

    if arr_fp.ndim != 2 or arr_fp.shape[1] != num_points or arr_fp.shape[0] <= 0:
        return None
    return arr_fp.astype(bool)


def _binary_contact_metrics(input_contact: np.ndarray, gen_contact: np.ndarray) -> dict:
    """Compute frame-wise accuracy, precision, and recall with confusion counts."""
    if input_contact.shape != gen_contact.shape:
        raise ValueError("input_contact and gen_contact must have the same shape.")
    input_contact = input_contact.astype(bool)
    gen_contact = gen_contact.astype(bool)
    tp = int(np.logical_and(gen_contact, input_contact).sum())
    fp = int(np.logical_and(gen_contact, ~input_contact).sum())
    fn = int(np.logical_and(~gen_contact, input_contact).sum())
    tn = int(np.logical_and(~gen_contact, ~input_contact).sum())
    total = max(tp + fp + fn + tn, 1)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": int(total),
    }


def _topk_mass(counts: np.ndarray, k_ratio: float = 0.05) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    n = counts.shape[0]
    k = max(1, int(np.ceil(n * k_ratio)))
    topk = np.partition(counts, -k)[-k:]
    return float(topk.sum() / total)


def _gini_index(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = sorted_counts.size
    cum = np.sum((np.arange(1, n + 1) * sorted_counts))
    return float((2.0 * cum) / (n * total) - (n + 1) / n)


def _simpson_diversity(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _coverage_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def _jaccard_diversity_samples(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 0.0
    values = []
    for i in range(len(masks) - 1):
        for j in range(i + 1, len(masks)):
            a = masks[i]
            b = masks[j]
            union = np.logical_or(a, b).sum()
            if union == 0:
                continue
            inter = np.logical_and(a, b).sum()
            values.append(1.0 - (inter / union))
    return float(np.mean(values)) if values else 0.0


def _object_diameter(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    return float(np.linalg.norm(max_xyz - min_xyz))


def _spatial_spread(
    mask: np.ndarray,
    points: np.ndarray,
    max_points: int = 200,
) -> float:
    if mask.sum() < 2:
        return 0.0
    pts = points[mask]
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    mean_pairwise = dists[np.triu_indices(dists.shape[0], k=1)].mean()
    diameter = _object_diameter(points)
    if diameter <= 0:
        return 0.0
    return float(mean_pairwise / diameter)


def _knn_coverage(
    mask: np.ndarray,
    points: np.ndarray,
    k: int = 5,
    max_points: int = 200,
) -> float:
    if mask.sum() < 2:
        return 0.0
    pts = points[mask]
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    kk = min(k, pts.shape[0] - 1)
    nearest = np.partition(dists, kk, axis=1)[:, :kk]
    mean_knn = nearest.mean()
    diameter = _object_diameter(points)
    if diameter <= 0:
        return 0.0
    return float(mean_knn / diameter)


_PART_KEYWORDS = [
    "handle",
    "rim",
    "body",
    "top",
    "bottom",
    "side",
    "edge",
    "surface",
    "base",
    "cover",
    "cap",
    "lid",
    "lip",
    "tip",
    "front",
    "back",
    "middle",
    "center",
]


def _extract_part_keyword(text: str) -> Optional[str]:
    for keyword in _PART_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return keyword
    return None


def _sanitize_entity_path(text: str) -> str:
    """Make a rerun-friendly path segment while keeping the text semantics."""
    sanitized = re.sub(r"\s+", "_", text.strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.\\-]", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "entry"


def _normalize_action_key(text: str) -> str:
    """Return a grouping key that ignores hand laterality and keeps object part names."""
    lowered = re.sub(r"\s+", " ", text.lower()).strip(" .")
    if not lowered:
        return ""
    base, sep, _ = lowered.partition(" with ")
    if not base:
        base = lowered
    base = re.sub(r"\b(right|left|hands?|hand)\b", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        base = lowered
    if " of " in base:
        before_of, obj = base.split(" of ", 1)
        part_keyword = _extract_part_keyword(before_of)
        if not part_keyword:
            part_keyword = _extract_part_keyword(lowered)
        if part_keyword and part_keyword not in before_of:
            verb = before_of.split()[0] if before_of.split() else before_of
            base = f"{verb} {part_keyword} of {obj}"
    if base:
        return base
    cleaned = re.sub(r"\b(right|left|hands?|hand)\b", "", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else lowered


def _extract_of_with_target(text: str) -> Optional[str]:
    """Return the target between 'of' and 'with' for grouping."""
    match = re.search(r"\bof\s+(.+?)\s+with\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    target = re.sub(r"\s+", " ", match.group(1)).strip()
    return target.lower() if target else None


def _build_grouping_keys(text: str) -> tuple[str, str]:
    """Group by object first, then by full text within the object."""
    object_key = _extract_of_with_target(text) or text
    action_key = f"{object_key}::{text}"
    return object_key, action_key


with open(os.path.join(home, "Desktop/hot3d_vis/instance.json"), "r") as f:
    instance_ = json.load(f)

object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
obj_pc = dict()

for obj_name in object_model.obj_pcs.keys():
    _, pc, _, _ = object_model(obj_name)
    obj_pc[obj_name] = torch.tensor(pc)

l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

_CONTACT_METRIC_HEADER = [
    "file_name",
    "method",
    "text",
    "acc",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "tn",
    "total",
]

_TARGET_TEXTS = [
    "Grab handle of mug_patterned with right hand.",
    "Grab handle of mug_white with right hand.",
]


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    sep = "-+-".join("-" * width for width in widths)
    out = []
    out.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    out.append(sep)
    for row in rows:
        out.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def _aggregate_metrics(metrics_list: list[dict], totals: dict) -> dict:
    if not metrics_list:
        return {}
    accs = [m["acc"] for m in metrics_list]
    precs = [m["precision"] for m in metrics_list]
    recs = [m["recall"] for m in metrics_list]
    f1s = [m["f1"] for m in metrics_list]
    sample_count = (
        int(sum(m.get("samples", 1) for m in metrics_list))
        if metrics_list and isinstance(metrics_list[0], dict)
        else len(metrics_list)
    )
    return {
        "acc": float(np.mean(accs)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
        "tp": int(totals["tp"]),
        "fp": int(totals["fp"]),
        "fn": int(totals["fn"]),
        "tn": int(totals["tn"]),
        "total": int(totals["total"]),
        "samples": sample_count,
    }


def _print_file_text_metrics(file_name: str, per_text_rows: list[dict]) -> None:
    if not per_text_rows:
        print(f"[{file_name}] no valid metrics")
        return
    headers = ["text", "samples", "acc", "precision", "recall", "f1"]
    rows = [
        [
            row["text"],
            str(row["samples"]),
            f"{row['acc']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
        ]
        for row in sorted(per_text_rows, key=lambda x: x["f1"], reverse=True)
    ]
    print(f"\n[{file_name}] per-text metrics")
    print(_render_table(headers, rows))


def _write_metrics_csv(out_csv: str, all_results: list[dict]) -> None:
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CONTACT_METRIC_HEADER)
        for result in all_results:
            file_name = result["file_name"]
            for row in result["per_text_rows"]:
                writer.writerow(
                    [
                        file_name,
                        "gt_cov_map vs actual_contact",
                        row["text"],
                        f"{row['acc']:.4f}",
                        f"{row['precision']:.4f}",
                        f"{row['recall']:.4f}",
                        f"{row['f1']:.4f}",
                        row["tp"],
                        row["fp"],
                        row["fn"],
                        row["tn"],
                        row["total"],
                    ]
                )


def _print_cross_file_summary(all_results: list[dict]) -> None:
    headers = ["file", "texts", "samples", "acc", "precision", "recall", "f1"]
    rows = []
    for result in sorted(all_results, key=lambda x: x["overall"]["f1"], reverse=True):
        overall = result["overall"]
        rows.append(
            [
                result["file_name"],
                str(result["num_texts"]),
                str(overall["samples"]),
                f"{overall['acc']:.3f}",
                f"{overall['precision']:.3f}",
                f"{overall['recall']:.3f}",
                f"{overall['f1']:.3f}",
            ]
        )
    if rows:
        print("\n=== Cross-file summary (sorted by F1) ===")
        print(_render_table(headers, rows))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    acc = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        float((2.0 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total": int(total),
    }


def _print_target_text_metric_rankings(
    all_results: list[dict], target_texts: list[str]
) -> None:
    target_lookup = {text.lower(): text for text in target_texts}
    merged = {
        text: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "samples": 0}
        for text in target_texts
    }

    for result in all_results:
        for row in result.get("per_text_rows", []):
            text_key = str(row.get("text", "")).lower()
            if text_key not in target_lookup:
                continue
            canonical = target_lookup[text_key]
            merged[canonical]["tp"] += int(row.get("tp", 0))
            merged[canonical]["fp"] += int(row.get("fp", 0))
            merged[canonical]["fn"] += int(row.get("fn", 0))
            merged[canonical]["tn"] += int(row.get("tn", 0))
            merged[canonical]["samples"] += int(row.get("samples", 0))

    available = []
    for text in target_texts:
        counts = merged[text]
        conf = _metrics_from_confusion(
            counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        )
        conf["text"] = text
        conf["samples"] = counts["samples"]
        if conf["total"] > 0:
            available.append(conf)

    if not available:
        print("\n=== Target text ranking ===")
        print("No matching samples found for target texts.")
        return

    print("\n=== Target text ranking (metric-wise) ===")
    for metric in ["acc", "precision", "recall", "f1"]:
        ranked = sorted(
            available, key=lambda x: (-x[metric], x["text"].lower())
        )
        print(f"\n[{metric}]")
        for idx, row in enumerate(ranked, start=1):
            print(
                f"{idx}. {row['text']} | {metric}={row[metric]:.4f} | "
                f"samples={row['samples']}"
            )


def visualize_rr(recoding_name, idx, visualize=True):
    file_name = os.path.basename(recoding_name)

    with open(recoding_name, "rb") as f:
        item = pickle.load(f)

        action_to_sample_counts = defaultdict(int)
        contact_metric_stats = defaultdict(list)
        contact_metric_totals = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}
        )

        for (
            fine_lhand,
            fine_rhand,
            x_obj,
            text,
            course_lhand,
            course_rhand,
            gt_obj,
            cond_enc,
            gt_cov_map,
            est_cov_map,
            _,
        ) in item:
            for batch_idx in range(len(course_lhand)):
                text_entry = str(text[batch_idx])
                object_key, action_key = _build_grouping_keys(text_entry)
                base_path = _sanitize_entity_path(object_key)
                text_path = _sanitize_entity_path(text_entry)
                hand_path = "r_hand" if "right" in text_entry.lower() else "l_hand"
                sample_idx = action_to_sample_counts[action_key]
                action_to_sample_counts[action_key] += 1
                sample_gt_path = (
                    f"runs/{_sanitize_entity_path(idx)}/original/course/"
                    f"{base_path}/{text_path}/sample_{sample_idx:03d}"
                )

                obj_name = text_entry.split("of ")[-1].split(" with")[0].lower()

                if obj_name not in obj_pc:
                    print(f"[WARN] unresolved object key: '{obj_name}' from text '{text_entry}'")
                    continue
                obj_vertices = process_obj_result(obj_pc[obj_name], x_obj[batch_idx])
                obj_vertices_np = obj_vertices.detach().cpu().numpy()
                gt_cov_map_raw = gt_cov_map[batch_idx] if gt_cov_map is not None else None
                est_cov_map_raw = (
                    est_cov_map[batch_idx] if est_cov_map is not None else None
                )

                if obj_vertices_np is None:
                    continue

                num_points = obj_vertices_np.shape[1]
                obj_nframes = obj_vertices_np.shape[0]
                gt_frame_mask = _framewise_point_mask_from_cov_map(gt_cov_map_raw, num_points)
                est_frame_mask = _framewise_point_mask_from_cov_map(est_cov_map_raw, num_points)
                gt_point_mask = (
                    gt_frame_mask.any(axis=0)
                    if gt_frame_mask is not None
                    else _reduce_cov_to_point_mask(gt_cov_map_raw, num_points)
                )
                est_point_mask = (
                    est_frame_mask.any(axis=0)
                    if est_frame_mask is not None
                    else _reduce_cov_to_point_mask(est_cov_map_raw, num_points)
                )

                r_hand_vertices_gt, r_hand_faces, _ = process_hand_result(
                    r_hand_layer, _to_torch(course_rhand[batch_idx])
                )
                l_hand_vertices_gt, l_hand_faces, _ = process_hand_result(
                    l_hand_layer, _to_torch(course_lhand[batch_idx])
                )
                r_hand_np = r_hand_vertices_gt.detach().cpu().numpy()
                l_hand_np = l_hand_vertices_gt.detach().cpu().numpy()
                use_right = "right" in text_entry.lower()
                use_left = "left" in text_entry.lower()
                if not use_right and not use_left:
                    use_right = True
                    use_left = True
                actual_frame_mask = _compute_contact_from_vertices(
                    obj_vertices_np,
                    l_hand_np if use_left else None,
                    r_hand_np if use_right else None,
                )
                actual_point_mask = actual_frame_mask.any(axis=0)
                # For gt-vs-actual comparison, use cumulative actual contact over time.
                actual_cumulative_frame_mask = np.logical_or.accumulate(
                    actual_frame_mask, axis=0
                )

                if gt_point_mask is not None and gt_point_mask.shape == actual_point_mask.shape:
                    point_metrics = _binary_contact_metrics(gt_point_mask, actual_point_mask)
                    contact_metric_stats[text_entry].append(point_metrics)
                    totals = contact_metric_totals[text_entry]
                    totals["tp"] += point_metrics["tp"]
                    totals["fp"] += point_metrics["fp"]
                    totals["fn"] += point_metrics["fn"]
                    totals["tn"] += point_metrics["tn"]
                    totals["total"] += point_metrics["total"]

                if visualize:
                    r_mesh = trimesh.Trimesh(
                        vertices=r_hand_vertices_gt[0], faces=r_hand_faces, process=False
                    )
                    l_mesh = trimesh.Trimesh(
                        vertices=l_hand_vertices_gt[0], faces=l_hand_faces, process=False
                    )

                render_nframes = obj_nframes
                if visualize:
                    if "right" in text_entry.lower():
                        render_nframes = min(render_nframes, r_hand_vertices_gt.shape[0])
                    elif "left" in text_entry.lower():
                        render_nframes = min(render_nframes, l_hand_vertices_gt.shape[0])
                    else:
                        render_nframes = min(
                            render_nframes,
                            r_hand_vertices_gt.shape[0],
                            l_hand_vertices_gt.shape[0],
                        )
                if actual_frame_mask is not None and actual_frame_mask.shape[0] > 0:
                    render_nframes = min(render_nframes, actual_frame_mask.shape[0])
                render_nframes = max(render_nframes, 1)

                for frame_idx in range(render_nframes):
                    if visualize:
                        _set_frame_time(frame_idx)
                        if "right" in text_entry.lower():
                            rr.log(
                                f"{sample_gt_path}/{hand_path}",
                                rr.Mesh3D(
                                    vertex_positions=r_hand_vertices_gt[frame_idx],
                                    triangle_indices=r_hand_faces,
                                    vertex_normals=r_mesh.vertex_normals,
                                ),
                            )
                        elif "left" in text_entry.lower():
                            rr.log(
                                f"{sample_gt_path}/{hand_path}",
                                rr.Mesh3D(
                                    vertex_positions=l_hand_vertices_gt[frame_idx],
                                    triangle_indices=l_hand_faces,
                                    vertex_normals=l_mesh.vertex_normals,
                                ),
                            )
                        else:
                            rr.log(
                                f"{sample_gt_path}/r_hand",
                                rr.Mesh3D(
                                    vertex_positions=r_hand_vertices_gt[frame_idx],
                                    triangle_indices=r_hand_faces,
                                    vertex_normals=r_mesh.vertex_normals,
                                ),
                            )
                            rr.log(
                                f"{sample_gt_path}/l_hand",
                                rr.Mesh3D(
                                    vertex_positions=l_hand_vertices_gt[frame_idx],
                                    triangle_indices=l_hand_faces,
                                    vertex_normals=l_mesh.vertex_normals,
                                ),
                            )
                    obj_frame_np = obj_vertices_np[frame_idx]
                    colors_cov = np.zeros((obj_frame_np.shape[0], 3), dtype=np.uint8)
                    colors_cov[:] = [0, 0, 255]

                    gt_mask_now = None
                    est_mask_now = None
                    actual_mask_cumulative_now = None
                    if gt_frame_mask is not None:
                        gt_idx = min(frame_idx, gt_frame_mask.shape[0] - 1)
                        gt_mask_now = gt_frame_mask[gt_idx]
                    elif gt_point_mask is not None:
                        gt_mask_now = gt_point_mask

                    if est_frame_mask is not None:
                        est_idx = min(frame_idx, est_frame_mask.shape[0] - 1)
                        est_mask_now = est_frame_mask[est_idx]
                    elif est_point_mask is not None:
                        est_mask_now = est_point_mask

                    if actual_frame_mask is not None:
                        actual_mask_cumulative_now = actual_cumulative_frame_mask[frame_idx]
                        actual_mask_compare_now = actual_cumulative_frame_mask[frame_idx]
                    else:
                        actual_mask_compare_now = None

                    if visualize:
                        colors_actual = np.zeros((obj_frame_np.shape[0], 3), dtype=np.uint8)
                        colors_actual[:] = [0, 0, 255]
                        if actual_mask_cumulative_now is not None:
                            colors_actual[actual_mask_cumulative_now] = [0, 255, 0]
                        rr.log(
                            f"{sample_gt_path}/contact_map_actual",
                            rr.Points3D(
                                positions=obj_frame_np,
                                radii=0.005,
                                colors=colors_actual,
                            ),
                        )

                        # Keep a comparison view with actual contact only.
                        if actual_mask_compare_now is not None:
                            colors_cov[actual_mask_compare_now] = [0, 255, 0]
                        rr.log(
                            f"{sample_gt_path}/contact_map_compare",
                            rr.Points3D(
                                positions=obj_frame_np,
                                radii=0.005,
                                colors=colors_cov,
                            ),
                        )

        per_text_rows = []
        for text_key, metrics_list in contact_metric_stats.items():
            if not metrics_list:
                continue
            totals = contact_metric_totals.get(
                text_key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}
            )
            agg = _aggregate_metrics(metrics_list, totals)
            if not agg:
                continue
            agg["text"] = text_key
            per_text_rows.append(agg)

        if per_text_rows:
            _print_file_text_metrics(file_name, per_text_rows)
            overall = _aggregate_metrics(
                [row for row in per_text_rows],
                {
                    "tp": sum(row["tp"] for row in per_text_rows),
                    "fp": sum(row["fp"] for row in per_text_rows),
                    "fn": sum(row["fn"] for row in per_text_rows),
                    "tn": sum(row["tn"] for row in per_text_rows),
                    "total": sum(row["total"] for row in per_text_rows),
                },
            )
        else:
            overall = {
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "total": 0,
                "samples": 0,
            }

        return {
            "file_name": file_name,
            "per_text_rows": per_text_rows,
            "overall": overall,
            "num_texts": len(per_text_rows),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action="store_true", help="enable rerun visualization"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(home, "Desktop/hot3d_vis"),
        help="directory containing pkl files",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[
        # "grab_exc_rot_aug.pkl",
        # "grab_exc_rot_aug_gaze_emb_afford_mix_init_5_trans.pkl",
        # "grab_exc_rot_aug_afford.pkl",
        # "grab_exc_rot_aug_afford_mix.pkl",
        # "grab_exc_rot_aug_gaze_emb_toekn.pkl",
        "grab_exc_rot_aug_gaze_emb_token_vec.pkl",
        "grab_exc_rot_aug_gaze_emb_token_vec_ro.pkl",
        # "grab_exc_rot_aug_gaze_emb_token_vec_ro_afford_mix.pkl",
                 ],
        help="one or more pickle files to compare",
    )
    args = parser.parse_args()

    if args.visualize:
        # Default to spawning the viewer so --visualize shows output immediately.
        # Set HOT3D_RERUN_SPAWN=0 to disable auto-spawn.
        spawn_viewer = os.environ.get("HOT3D_RERUN_SPAWN", "1") == "1"
        rr.init("Input Data", spawn=spawn_viewer)
        if not spawn_viewer:
            print(
                "[INFO] Rerun viewer auto-spawn disabled (set HOT3D_RERUN_SPAWN=1 to enable)."
            )

    all_results = []
    for file_name in args.input_files:
        recoding_name = (
            file_name
            if os.path.isabs(file_name)
            else os.path.join(args.input_dir, file_name)
        )
        if not os.path.exists(recoding_name):
            print(f"[WARN] missing file: {recoding_name}")
            continue
        all_results.append(
            visualize_rr(recoding_name, file_name, visualize=args.visualize)
        )

    if not all_results:
        print("[WARN] no valid input files were processed.")
        return

    _print_cross_file_summary(all_results)
    out_csv = os.path.join(os.path.expanduser("~"), "contact_metrics.csv")
    _write_metrics_csv(out_csv, all_results)
    print(f"\nSaved metrics CSV: {out_csv}")
    _print_target_text_metric_rankings(all_results, _TARGET_TEXTS)


if __name__ == "__main__":
    main()
