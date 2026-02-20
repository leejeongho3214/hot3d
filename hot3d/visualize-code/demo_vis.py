import argparse
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


from data_loaders.mano_layer import MANOHandModel
import os

from mano import build_mano_aa


_rng = np.random.default_rng()


def random_rgb_color() -> list[int]:
    """Return a random RGB color as a list of ints in [0, 255]."""
    return _rng.integers(0, 256, size=3, dtype=np.uint8).tolist()


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def apply_rigid_transform(points, R, t):
    """Apply a rigid transform to (N,3) points."""
    return (points @ R.T) + t


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
        
def visualize_rr(recoding_name, file_name, color, align: bool = True):

    with open(f"{home}/Desktop/hot3d_vis/qq2.pkl", "rb") as f:
        item2 = pickle.load(f)

    with open(recoding_name, "rb") as f:
        item = pickle.load(f)

        for item_idx, (hand_verts, hand_faces, obj_verts, est_contact_map, text) in enumerate(item):
            align_tag = "aligned" if align else "raw"
            base_path = f"{align_tag}/item"
            sample_path = f"{base_path}/sample_{item_idx:03d}"

            hand_verts_np = _to_numpy(hand_verts)
            hand_faces_np = _to_numpy(hand_faces)
            if hand_faces_np is None:
                hand_faces_np = r_hand_layer.faces.copy()
            hand_faces_np = np.asarray(hand_faces_np, dtype=np.int32)

            # rhand_verts_np = _to_numpy(rhand_verts)
            # rhand_faces_np = _to_numpy(rhand_faces)
            # if rhand_faces_np is None:
            #     rhand_faces_np = r_hand_layer.faces.copy()
            # rhand_faces_np = np.asarray(rhand_faces_np, dtype=np.int32)

            obj_verts_np = _to_numpy(obj_verts)
            contact_np = _to_numpy(est_contact_map)

            ref_obj = obj_verts_np[0] if obj_verts_np is not None and obj_verts_np.shape[0] > 0 else None
            num_frames = max(hand_verts_np.shape[0], obj_verts_np.shape[0])
            for frame_idx in range(num_frames):
                rr.set_time("frame", sequence=frame_idx)

                hand_frame = hand_verts_np[min(frame_idx, hand_verts_np.shape[0] - 1)]
                # rhand_frame = rhand_verts_np[min(frame_idx, rhand_verts_np.shape[0] - 1)]   
                obj_frame = obj_verts_np[min(frame_idx, obj_verts_np.shape[0] - 1)]
                contact_frame = None
                if contact_np is not None:
                    contact_frame = contact_np[min(frame_idx, contact_np.shape[0] - 1)]

                if align and ref_obj is not None and obj_frame is not None and obj_frame.shape == ref_obj.shape:
                    R, t = rigid_transform(obj_frame, ref_obj)
                    hand_frame = apply_rigid_transform(hand_frame, R, t)
                    obj_frame = ref_obj
                    if contact_frame is not None and contact_frame.shape[0] != obj_frame.shape[0]:
                        contact_frame = None

                rr.log(
                    f"{sample_path}/hand",
                    rr.Mesh3D(
                        vertex_positions=hand_frame,
                        triangle_indices=hand_faces_np,
                        vertex_normals=trimesh.Trimesh(
                            vertices=hand_frame,
                            faces=hand_faces_np,
                            process=False,
                        ).vertex_normals,
                    ),
                )

                # obj_frame = item2[0][-1].squeeze()
                colors = np.zeros((obj_frame.shape[0], 3), dtype=np.uint8)
                colors[:] = [0, 0, 255]
                if contact_frame is not None and contact_frame.shape[0] == obj_frame.shape[0]:
                    contact_mask = contact_frame > 0
                    colors[contact_mask] = [255, 255, 0]

                rr.log(
                    f"{sample_path}/object",
                    rr.Points3D(
                        positions=obj_frame,
                        radii=0.005,
                        colors=colors,
                    ),
                )



def main():
    rr.init("Input Data", spawn=True)
    
    file_name = "qq"
    recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    # visualize_rr(recoding_name, file_name, random_rgb_color(), align=True)
    visualize_rr(recoding_name, file_name, random_rgb_color(), align=False)
    

if __name__ == "__main__":
    main()
