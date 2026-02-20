
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import itertools
import json
import inspect
import re
import argparse
from typing import Optional

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
import trimesh

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

from data_loaders.mano_layer import MANOHandModel
if not hasattr(inspect, "getargspec"):
    # Compatibility for chumpy on Python 3.11+
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
import numpy as np
# NumPy 1.24+ compatibility for legacy packages (e.g., chumpy)
for _name, _type in [("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str)]:
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

from hot3d_action_dataset import Hot3DActionDataset as _Hot3DActionDataset


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)

def log_image(
    image: np.array,
    label: str,
    static=False
) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(
    pose: SE3,
    label: str,
    static=False
) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)
    
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
        betas=torch.zeros((duration, 10))
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
        Vt[2,:] *= -1
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

_sample_counter = itertools.count()  # global counter so train/eval don't collide

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
        if norm_key_flat and (norm_key_flat in norm_cand_flat or norm_cand_flat in norm_key_flat):
            return key
    return None


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


# The saved pickle references "__main__.Hot3DActionDataset". Make sure this
# name exists so pickle can resolve the dataset class.
globals()["Hot3DActionDataset"] = _Hot3DActionDataset

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataset cov maps.")
    parser.add_argument(
        "--mode",
        choices=("avg", "individual", "both"),
        default="avg",
        help="avg: aggregated average per text, individual: per item, both: show both.",
    )
    return parser.parse_args()


rr.init("Input Data", spawn=True)
args = _parse_args()

home = os.path.expanduser("~")
with open(os.path.join(home, "Desktop/hot3d_vis/dataset/grab_ori_bs.pkl"), "rb") as f:
    item_list = pickle.load(f)

aggregates = {}

for item in item_list:
    obj_param = item['x_obj']

    cov_map = item['cov_map']
    text = item['text']
    act_id = item['act_id']

    entry_counts = defaultdict(int)

    for batch_idx in range(len(text)):
        text_entry = text[batch_idx]

        object_key = _extract_object_key(text_entry)

        if object_key != "mug_white":
            continue

        entry_counts[(object_key, text_entry)] += 1
        sanitized_entry = re.sub(r"\s+", "_", text_entry.strip())
        log_prefix = f"{object_key}/{sanitized_entry}/{act_id[batch_idx]}"
        base_path = log_prefix

        num_points = obj_pc[object_key].shape[0]
        cov_arr = np.asarray(cov_map[batch_idx])
        style_covs = []
        if cov_arr.ndim >= 3 and cov_arr.shape[-1] == num_points:
            for style_idx in range(cov_arr.shape[0]):
                style_arr = cov_arr[style_idx]
                if style_arr.ndim >= 2:
                    style_arr = style_arr.mean(axis=0)
                cov_counts = _reduce_cov_counts(style_arr, num_points)
                if cov_counts is None:
                    continue
                style_covs.append((style_idx, cov_counts))
        else:
            if cov_arr.ndim >= 2:
                cov_arr = cov_arr.mean(axis=0)
            cov_counts = _reduce_cov_counts(cov_arr, num_points)
            if cov_counts is not None:
                style_covs.append((0, cov_counts))

        if args.mode in ("avg", "both"):
            for style_idx, cov_counts in style_covs:
                key = (object_key, text_entry, style_idx)
                entry = aggregates.get(key)
                if entry is None:
                    aggregates[key] = {
                        "object_key": object_key,
                        "text": text_entry,
                        "style_idx": style_idx,
                        "sum": cov_counts.astype(np.float32),
                        "count": 1,
                    }
                else:
                    entry["sum"] += cov_counts
                    entry["count"] += 1
        
        if args.mode in ("individual", "both"):
            obj_vertices = process_obj_result(
                obj_pc[object_key], torch.as_tensor(obj_param[batch_idx]).detach().clone()
            )
            for frame_idx in range(obj_vertices.shape[0]):
                rr.set_time_sequence("frame", frame_idx)
                
                # rr.log(
                #     f"{base_path}/object_pc",
                #     rr.Points3D(
                #         positions=obj_vertices[frame_idx],
                #         radii=0.005,
                #         colors=[0, 255, 0],
                #         labels=[act_id[batch_idx]]
                #     )
                # )

                for style_idx, cov_counts in style_covs:
                    colors_cov = _compute_point_colors(cov_counts)
                    rr.log(
                        f"{base_path}/style_{style_idx:02d}/cov_map",
                        rr.Points3D(
                            positions=obj_vertices[frame_idx],
                            radii=0.005,
                            colors=colors_cov,
                        ),
                    )

if args.mode in ("avg", "both"):
    for entry in aggregates.values():
        object_key = entry["object_key"]
        text_entry = entry["text"]
        style_idx = entry["style_idx"]
        sanitized_entry = re.sub(r"\s+", "_", text_entry.strip())
        base_path = f"{object_key}/{sanitized_entry}"
        points = obj_pc[object_key].detach().cpu().numpy()
        cov_avg = entry["sum"] / max(entry["count"], 1)
        colors_cov = _compute_point_colors(cov_avg)
        rr.log(
            f"{base_path}/style_{style_idx:02d}/cov_map_avg",
            rr.Points3D(
                positions=points,
                radii=0.005,
                colors=colors_cov,
            ),
            static=True,
        )
