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


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)


def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)


def _log_text(label: str, text: str, static: bool = False) -> None:
    if hasattr(rr, "TextLog"):
        rr.log(label, rr.TextLog(text), static=static)
    elif hasattr(rr, "TextDocument"):
        rr.log(label, rr.TextDocument(text), static=static)
    elif hasattr(rr, "AnyValues"):
        rr.log(label, rr.AnyValues(text=text), static=static)
    else:
        print(f"{label}: {text}")


def _get_batch_value(value, batch_idx: int):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
        if len(value) > batch_idx:
            return value[batch_idx]
        return value
    return value


def _sanitize_path_token(value) -> str:
    token = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    return token or "unknown"


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
    if cov_frame.ndim == 2:
        if cov_frame.shape[0] == num_points:
            cov_counts = cov_frame.sum(axis=1)
        elif cov_frame.shape[1] == num_points:
            cov_counts = cov_frame.sum(axis=0)
    elif cov_frame.ndim == 1 and cov_frame.shape[0] == num_points:
        cov_counts = cov_frame
    return cov_counts


def _compute_point_colors(cov_values: np.ndarray) -> np.ndarray:
    max_count = float(np.max(cov_values)) if np.max(cov_values) > 0 else 1.0
    intensity = np.clip(cov_values / max_count, 0.0, 1.0).astype(np.float32)
    low_color = np.array([0, 0, 255], dtype=np.float32)
    high_color = np.array([255, 255, 0], dtype=np.float32)
    return (low_color + (high_color - low_color) * intensity[:, None]).astype(np.uint8)


# The saved pickle references "__main__.Hot3DActionDataset". Make sure this
# name exists so pickle can resolve the dataset class.

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize contact coverage.")
    parser.add_argument(
        "--mode",
        choices=("avg", "individual", "both"),
        default="individual",
        help="avg: aggregated average per action, individual: per item, both: show both.",
    )
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

    rr.init("Input Data", spawn=True)

    home = os.path.expanduser("~")
    with open(os.path.join(home, "Desktop/hot3d_vis/contact_results_e12000.pkl"), "rb") as f:
        item_list = pickle.load(f)

    aggregates = {}

    for idx, (_, cov, text) in enumerate(item_list):
        object_key, action_key = _build_grouping_keys(str(text))
        if object_key not in obj_pc:
            continue
        obj_points = obj_pc[object_key].detach().cpu().numpy()
        num_points = obj_points.shape[0]

        if cov is None:
            continue
        cov_arr = np.asarray(cov)
        cov_mean = cov_arr.mean(axis=0) if cov_arr.ndim >= 2 else cov_arr
        cov_counts = _reduce_cov_counts(np.asarray(cov_mean), num_points)
        if cov_counts is None:
            continue

        if args.mode in ("individual", "both"):
            base_path = _sanitize_entity_path(object_key)
            text_path = _sanitize_entity_path(str(text))
            item_path = f"{base_path}/{text_path}/items/{idx:04d}"
            _log_point_cloud(item_path, obj_points, cov_counts)

        if args.mode in ("avg", "both"):
            entry = aggregates.get(action_key)
            if entry is None:
                aggregates[action_key] = {
                    "object_key": object_key,
                    "text": str(text),
                    "sum": cov_counts.astype(np.float32),
                    "count": 1,
                }
            else:
                entry["sum"] += cov_counts
                entry["count"] += 1

    if args.mode in ("avg", "both"):
        for action_key, entry in aggregates.items():
            object_key = entry["object_key"]
            text_entry = entry["text"]
            base_path = _sanitize_entity_path(object_key)
            text_path = _sanitize_entity_path(text_entry)
            base_path = f"{base_path}/{text_path}"

            points = obj_pc[object_key].detach().cpu().numpy()
            cov_avg = entry["sum"] / max(entry["count"], 1)
            _log_point_cloud(base_path, points, cov_avg)


if __name__ == "__main__":
    main()
