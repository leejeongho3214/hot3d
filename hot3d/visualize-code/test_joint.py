import sys
import os

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


def _normalize_for_lookup(token: str, keep_underscores: bool = True) -> str:
    token = token.lower().strip()
    token = token.replace("-", "_")
    token = token.replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]", "", token)
    if not keep_underscores:
        token = token.replace("_", "")
    return token


def _extract_object_key(text_entry: str) -> Optional[str]:
    lowered = text_entry.lower()
    candidate = lowered.split("of ", 1)[-1] if "of " in lowered else lowered
    for sep in [" with", " using", " by", " to", ".", ",", ";"]:
        if sep in candidate:
            candidate = candidate.split(sep, 1)[0]
    candidate = candidate.strip()
    if not candidate:
        return None

    norm_cand = _normalize_for_lookup(candidate)
    norm_cand_flat = _normalize_for_lookup(candidate, keep_underscores=False)

    for key in obj_pc.keys():
        norm_key = _normalize_for_lookup(key)
        norm_key_flat = _normalize_for_lookup(key, keep_underscores=False)
        if norm_cand == norm_key or norm_cand_flat == norm_key_flat:
            return key
        if norm_key and (norm_key in norm_cand or norm_cand in norm_key):
            return key
        if norm_key_flat and (
            norm_key_flat in norm_cand_flat or norm_cand_flat in norm_key_flat
        ):
            return key
    return None


# The saved pickle references "__main__.Hot3DActionDataset". Make sure this
# name exists so pickle can resolve the dataset class.

rr.init("Input Data", spawn=True)

home = os.path.expanduser("~")
default_results_path = os.path.join(home, "Desktop/hot3d_vis/wrong.pkl")
fallback_results_path = os.path.join(home, "Desktop/hot3d_vis/contact_results.pkl")
results_path = (
    default_results_path
    if os.path.exists(default_results_path)
    else fallback_results_path
)

with open(results_path, "rb") as f:
    item_list = pickle.load(f)

for item_idx, item in enumerate(item_list):
    if not isinstance(item, dict):
        continue

    text_entry = item.get("text", "")
    text_path = _sanitize_path_token(text_entry)
    sample_idx = item.get("index", item_idx)
    hand_label = _sanitize_path_token(item.get("hand", "hand"))
    base_path = f"item_{item_idx:03d}/{text_path}/sample_{sample_idx:06d}/{hand_label}"

    obj_arr = np.asarray(item[""])
    if obj_arr is None:
        obj_name = item.get("object")
        if obj_name in obj_pc:
            obj_arr = obj_pc[obj_name].detach().cpu().numpy()
    motion_arr = None if item.get("motion") is None else np.asarray(item["motion"])

    rr.log(
        f"{base_path}/object_points",
        rr.Points3D(
            positions=obj_arr.reshape(-1, 3),
            radii=0.004,
            colors=[0, 120, 255],
        ),
        static=True,
    )

    if motion_arr is None:
        continue

    if motion_arr.ndim == 2:
        motion_arr = motion_arr[None, ...]

    num_frames = motion_arr.shape[0]
    for frame_idx in range(num_frames):
        rr.set_time_sequence("frame", frame_idx)
        joints = motion_arr[frame_idx]
        rr.log(
            f"{base_path}/hand_joints",
            rr.Points3D(
                positions=joints,
                radii=0.01,
                colors=[255, 0, 0],
            ),
        )

        if obj_arr is not None and obj_arr.ndim == 3 and frame_idx < obj_arr.shape[0]:
            rr.log(
                f"{base_path}/object_points",
                rr.Points3D(
                    positions=obj_arr[frame_idx],
                    radii=0.004,
                    colors=[0, 120, 255],
                ),
            )
