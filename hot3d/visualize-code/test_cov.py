import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import itertools
import json
import ast
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


def rotate_pc_y(points, degrees):
    radians = np.deg2rad(degrees)
    cos_t = np.cos(radians)
    sin_t = np.sin(radians)
    rot = torch.tensor(
        [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]],
        dtype=points.dtype,
        device=points.device,
    )
    return points @ rot.T


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
    cov_values = np.asarray(cov_values, dtype=np.float32)
    if cov_values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vmin = float(np.min(cov_values))
    vmax = float(np.max(cov_values))
    if vmax <= vmin:
        intensity = np.zeros_like(cov_values, dtype=np.float32)
    else:
        # Normalize to [0, 1] and use gamma to enhance contrast.
        intensity = (cov_values - vmin) / (vmax - vmin)
        intensity = np.clip(intensity, 0.0, 1.0) ** 0.6
    low_color = np.array([0, 0, 255], dtype=np.float32)
    high_color = np.array([255, 255, 0], dtype=np.float32)
    return (low_color + (high_color - low_color) * intensity[:, None]).astype(np.uint8)


def _render_histogram_image(
    counts: np.ndarray, height: int = 160, bar_width: int = 4
) -> np.ndarray:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize contact coverage.")
    parser.add_argument(
        "--mode",
        choices=("avg", "individual", "both"),
        default="avg",
        help="avg: aggregated average per action, individual: per item, both: show both.",
    )
    parser.add_argument(
        "--hist-overlap",
        action="store_true",
        help="log histogram of contact index overlap across samples.",
    )
    return parser.parse_args()


def _log_point_cloud(
    base_path: str, points: np.ndarray, cov_values: np.ndarray, offset: np.ndarray
) -> None:
    colors = _compute_point_colors(cov_values)
    points = np.asarray(points)
    if points.ndim == 2 and points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    elif points.ndim != 2 or points.shape[1] != 3:
        points = points.reshape(-1, 3)
    points = points + np.asarray(offset)
    if points.size:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extent = float((maxs - mins).max())
        point_radius = max(0.003, 0.02 * extent)
    else:
        point_radius = 0.02
    rr.log(
        f"{base_path}/point_cloud",
        rr.Points3D(
            positions=points,
            radii=point_radius,
            colors=colors,
        ),
        static=True,
    )


def _log_title(base_path: str, title: str, offset: np.ndarray) -> None:
    rr.log(
        f"{base_path}/title",
        rr.Points3D(
            positions=[offset + np.array([0.0, 0.25, 0.0], dtype=np.float32)],
            radii=0.01,
            colors=[255, 255, 255],
            labels=[title],
        ),
        static=True,
    )


rr.init("Input Data", spawn=True)


def main(file_name: str, offset_x: float = 0.0) -> None:
    args = _parse_args()
    offset = np.array([offset_x, 0.0, 0.0], dtype=np.float32)

    home = os.path.expanduser("~")
    with open(os.path.join(home, file_name), "rb") as f:
        item_list = pickle.load(f)

    file_tag = os.path.basename(file_name)
    _log_title(file_tag, file_tag, offset)

    aggregates = {}
    aggregate_counts = {}
    aggregate_points = {}
    overlap_counts = {}
    global_overlap_counts = None

    sample_counter = 0
    for idx, (rotated_pc, cov, text) in enumerate(item_list):

        def _maybe_parse_list_string(value):
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    try:
                        parsed = ast.literal_eval(stripped)
                        if isinstance(parsed, (list, tuple)):
                            return parsed
                    except (ValueError, SyntaxError):
                        pass
            return value

        def _flatten_texts(value):
            value = _maybe_parse_list_string(value)
            if isinstance(value, (list, tuple)):
                out = []
                for v in value:
                    out.extend(_flatten_texts(v))
                return out
            return [value]

        raw_texts = _flatten_texts(text)
        text_keys = [str(t) for t in raw_texts]

        cov_arr = None
        per_text_cov = None
        if cov is not None:
            cov_arr = np.asarray(cov)
            per_text_cov = (
                cov_arr
                if cov_arr.ndim >= 2 and cov_arr.shape[0] == len(text_keys)
                else None
            )

        for text_idx, text_key in enumerate(text_keys):
            obj_points_source = rotated_pc
            if isinstance(obj_points_source, torch.Tensor):
                obj_points_source = obj_points_source.detach().cpu().numpy()
            else:
                obj_points_source = np.asarray(obj_points_source)
            if obj_points_source.ndim == 3 and obj_points_source.shape[0] == len(
                text_keys
            ):
                # rotated_pc is per-text point cloud: (num_text, num_points, 3)
                obj_points_source = obj_points_source[text_idx]
            elif obj_points_source.ndim >= 4 and obj_points_source.shape[0] == len(
                text_keys
            ):
                obj_points_source = obj_points_source[text_idx]
            if obj_points_source.ndim == 2:
                obj_points_source = obj_points_source[None, ...]
            if obj_points_source.ndim != 3:
                continue
            num_points = obj_points_source.shape[1]

            if args.mode in ("individual", "both"):
                base_path = f"{file_tag}/{_sanitize_entity_path(text_key)}"
                for sample_idx, obj_points in enumerate(obj_points_source):
                    if obj_points.ndim != 2 or obj_points.shape[1] != 3:
                        continue
                    cov_values = np.zeros(num_points, dtype=np.int64)
                    if cov_arr is not None:
                        cov_arr_text = (
                            np.asarray(per_text_cov[text_idx])
                            if per_text_cov is not None
                            else cov_arr
                        )
                        cov_counts = _reduce_cov_counts(cov_arr_text, num_points)
                        if cov_counts is not None:
                            cov_values = cov_counts
                    item_path = f"{base_path}/items/{sample_counter:06d}"
                    _log_point_cloud(
                        item_path,
                        obj_points,
                        cov_values,
                        offset,
                    )
                    sample_counter += 1
            if args.mode in ("avg", "both"):
                cov_values = np.zeros(num_points, dtype=np.int64)
                if cov_arr is not None:
                    cov_arr_text = (
                        np.asarray(per_text_cov[text_idx])
                        if per_text_cov is not None
                        else cov_arr
                    )
                    cov_counts = _reduce_cov_counts(cov_arr_text, num_points)
                    if cov_counts is not None:
                        cov_values = cov_counts
                aggregate_counts[text_key] = aggregate_counts.get(text_key, 0) + 1
                if text_key not in aggregates:
                    aggregates[text_key] = cov_values.astype(np.float64)
                else:
                    aggregates[text_key] += cov_values
                if text_key not in aggregate_points:
                    aggregate_points[text_key] = obj_points_source[0]

    if args.mode in ("avg", "both"):
        for text_key, cov_sum in aggregates.items():
            count = max(aggregate_counts.get(text_key, 1), 1)
            cov_avg = cov_sum / float(count)
            base_path = f"{file_tag}/{_sanitize_entity_path(text_key)}/avg"
            points = aggregate_points.get(text_key)
            if points is None:
                continue
            _log_point_cloud(
                base_path,
                points,
                cov_avg,
                offset,
            )

        # _log_point_cloud("ori_pc", ori_pc[0], np.zeros(1024, dtype=np.int64), offset)


if __name__ == "__main__":
    main("Desktop/hot3d_vis/contact_grab_exc_3_obj_rot_aug_afford.pkl", offset_x=0.24)
    # main("Desktop/hot3d_vis/contact_grab_exc_rot_aug.pkl", offset_x=0.0)
    # main("Desktop/hot3d_vis/contact_grab_exc_two_obj_rot.pkl", offset_x=-3.0)
