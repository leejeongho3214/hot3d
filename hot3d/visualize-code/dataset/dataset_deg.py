import argparse
import json
import inspect
import re
from typing import Optional
import math
if not hasattr(inspect, "getargspec"):
    # Compatibility for chumpy on Python 3.11+
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
import numpy as np
# NumPy 1.24+ compatibility for legacy packages (e.g., chumpy)
for _name, _type in [("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str)]:
    # Use __dict__ to avoid triggering NumPy deprecation warnings during attribute access
    if _name not in np.__dict__:
        setattr(np, _name, _type)

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
import trimesh

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

import os
import torch
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rot import *

import rerun as rr
import pickle

from data_loaders.mano_layer import MANOHandModel
import os

from mano import build_mano_aa

CONTACT_DISTANCE_THRESHOLD = 0.01

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

def compute_hand_global_rotmat(hand_params: torch.Tensor) -> torch.Tensor:
    """Return per-frame wrist rotation matrices."""
    hand_pose = rot6d_to_axis_angle(hand_params[:, 3:]).reshape(-1, 48)
    global_orient = hand_pose[:, :3]
    return axis_angle_to_rotmat(global_orient)

def compute_relative_rotations(hand_rotmats: torch.Tensor, obj_rotmats: torch.Tensor) -> torch.Tensor:
    """Rotation from object frame to hand frame."""
    return torch.matmul(hand_rotmats, obj_rotmats.transpose(1, 2))
    
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

def infer_object_name(text_entry: str, act_identifier: Optional[str] = None) -> Optional[str]:
    if act_identifier is not None:
        inst = instance_.get(str(act_identifier))
        if inst:
            instance_name = inst.get("instance_name", "")
            if instance_name in obj_pc:
                return instance_name
            lowered = instance_name.lower()
            if lowered in obj_pc:
                return lowered
    text_lower = text_entry.lower()
    candidates = []
    if "of " in text_lower:
        candidates.append(text_lower.split("of ")[-1])
    candidates.append(text_lower)
    separators = [" with", " using", " by", " to", ".", ",", ";"]

    for cand in candidates:
        trimmed = cand
        for sep in separators:
            if sep in trimmed:
                trimmed = trimmed.split(sep)[0]
        normalized = _normalize_for_lookup(trimmed)
        if normalized in obj_pc:
            return normalized

    normalized_text = _normalize_for_lookup(text_lower, keep_underscores=False)
    for obj_name in obj_pc.keys():
        obj_norm = _normalize_for_lookup(obj_name.replace("_", " "), keep_underscores=False)
        if obj_norm and obj_norm in normalized_text:
            return obj_name
        parts = obj_name.split("_")
        if len(parts) > 1:
            reversed_norm = _normalize_for_lookup(" ".join(reversed(parts)), keep_underscores=False)
            if reversed_norm and reversed_norm in normalized_text:
                return obj_name
    return None


def main():
    rr.init("Input Data", spawn=True)
    
    with open(os.path.join(home, "Desktop/hot3d_vis/dataset/acc_ori_train.pkl"), "rb") as f:
        item_list = pickle.load(f)

    global_entry_idx = 1

    for item_name in item_list.keys():
        item = item_list[item_name]
        l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
        r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

        obj_param = item['x_obj']
        l_hand, r_hand = item['x_lhand'], item['x_rhand']
        
        text = item['action']
        act_id = item['act_id']

        entry_counts = defaultdict(int)

        for batch_idx in range(len(text)):
            text_entry = text[batch_idx]
            
            if text_entry == "  ":
                text_entry = "Grasp rim of Vase with left hand."
            
            entry_counts[text_entry] += 1
            sanitized_entry = re.sub(r"\s+", "_", text_entry.strip())
            log_prefix = f"{sanitized_entry}/{entry_counts[text_entry]}"
            root_prefix = f"{global_entry_idx}"

            r_hand_params_tensor = torch.tensor(r_hand[batch_idx], dtype=torch.float32)
            l_hand_params_tensor = torch.tensor(l_hand[batch_idx], dtype=torch.float32)

            r_hand_vertices, r_hand_faces = process_hand_result(r_hand_layer, r_hand_params_tensor)
            l_hand_vertices, l_hand_faces = process_hand_result(l_hand_layer, l_hand_params_tensor)

            object_name = infer_object_name(text_entry, act_id[batch_idx])
            if object_name is None:
                print(f"[WARN] object name not resolved for: {text_entry}")
                continue
            obj_params_tensor = torch.tensor(obj_param[batch_idx], dtype=torch.float32)
            obj_vertices = process_obj_result(obj_pc[object_name], obj_params_tensor)
            obj_trans = obj_params_tensor[:, :3]
            obj_rotmats = rot6d_to_rotmat(obj_params_tensor[:, 3:9]).reshape(-1, 3, 3)
            
            l_hand_rotmats = compute_hand_global_rotmat(l_hand_params_tensor)
            r_hand_rotmats = compute_hand_global_rotmat(r_hand_params_tensor)
            l_hand_trans = l_hand_params_tensor[:, :3]
            r_hand_trans = r_hand_params_tensor[:, :3]
            rel_rot_l = compute_relative_rotations(l_hand_rotmats, obj_rotmats)
            rel_rot_r = compute_relative_rotations(r_hand_rotmats, obj_rotmats)
            rel_axis_angle_l = rotation_matrix_to_angle_axis(rel_rot_l)
            rel_axis_angle_r = rotation_matrix_to_angle_axis(rel_rot_r)

            template_pc = obj_pc[object_name]
            pc_min = torch.min(template_pc, dim=0).values
            pc_max = torch.max(template_pc, dim=0).values
            axis_length = torch.norm(pc_max - pc_min).item() * 0.2
            axis_length = max(axis_length, 0.05)
            axis_radius = axis_length * 0.1
            axis_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

            r_mesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
            l_mesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)

            text_entry_lower = text_entry.lower()
            if "both" in text_entry_lower:
                hand_entries = [
                    ("r_hand", r_hand_vertices, r_hand_faces, r_mesh.vertex_normals, r_hand_trans, rel_rot_r, rel_axis_angle_r),
                    ("l_hand", l_hand_vertices, l_hand_faces, l_mesh.vertex_normals, l_hand_trans, rel_rot_l, rel_axis_angle_l),
                ]
            elif "right" in text_entry_lower:
                hand_entries = [
                    ("r_hand", r_hand_vertices, r_hand_faces, r_mesh.vertex_normals, r_hand_trans, rel_rot_r, rel_axis_angle_r),
                ]
            else:
                hand_entries = [
                    ("l_hand", l_hand_vertices, l_hand_faces, l_mesh.vertex_normals, l_hand_trans, rel_rot_l, rel_axis_angle_l),
                ]
            hand_labels = [entry[0] for entry in hand_entries]
            hand_contact_info = {label: None for label in hand_labels}
            num_frames = obj_vertices.shape[0]

            for frame_idx in range(num_frames):
                for hand_label, vertices, _, _, _, rel_rot, rel_axis_angle in hand_entries:
                    axis_angle = rel_axis_angle[frame_idx].detach().cpu().numpy()
                    angle_rad = np.linalg.norm(axis_angle)
                    angle_deg = math.degrees(angle_rad)
                    if hand_contact_info[hand_label] is None:
                        hand_points = vertices[frame_idx]
                        obj_points = obj_vertices[frame_idx]
                        min_dist = torch.cdist(hand_points, obj_points).min().item()
                        if min_dist <= CONTACT_DISTANCE_THRESHOLD:
                            hand_contact_info[hand_label] = {
                                "frame": frame_idx,
                                "angle_deg": angle_deg,
                                "min_dist": min_dist,
                                "hand_label": hand_label,
                            }

            ordered_contacts = [
                info for info in hand_contact_info.values() if info is not None
            ]
            ordered_contacts.sort(key=lambda info: info["angle_deg"])
            order_map = {}
            for info in ordered_contacts:
                order_map[info["hand_label"]] = f"{info['angle_deg']:.2f}deg"
            for label in hand_labels:
                if label not in order_map:
                    order_map[label] = "deg"

            for frame_idx in range(num_frames):
                rr.set_time_sequence("frame", frame_idx)
                origin = obj_trans[frame_idx].detach().cpu().numpy()
                rotation = obj_rotmats[frame_idx].detach().cpu().numpy()
                axis_segments = []
                for axis_idx in range(3):
                    endpoint = origin + axis_length * rotation[:, axis_idx]
                    axis_segments.append([origin.tolist(), endpoint.tolist()])

                for hand_label, vertices, faces, normals, hand_translations, rel_rot, rel_axis_angle in hand_entries:
                    order_folder = order_map[hand_label]
                    base_path = f"{root_prefix}/{order_folder}"

                    rr.log(
                        f"{base_path}/{hand_label}",
                        rr.Mesh3D(
                            vertex_positions=vertices[frame_idx],
                            triangle_indices=faces,
                            vertex_normals=normals,
                        ),
                    )
                    
                    hand_origin = hand_translations[frame_idx].detach().cpu().numpy()
                    rel_segments = []
                    rel_rot_np = rel_rot[frame_idx].detach().cpu().numpy()
                    for axis_idx in range(3):
                        endpoint = hand_origin + axis_length * rel_rot_np[:, axis_idx]
                        rel_segments.append([hand_origin.tolist(), endpoint.tolist()])

                    rr.log(
                        f"{base_path}/{hand_label}_relative_axes",
                        rr.LineStrips3D(
                            rel_segments,
                            colors=axis_colors,
                            radii=axis_radius,
                        ),
                    )
                    
                    axis_angle = rel_axis_angle[frame_idx].detach().cpu().numpy()
                    angle_rad = np.linalg.norm(axis_angle)
                    angle_deg = math.degrees(angle_rad)
                    rr.log(f"{base_path}/{hand_label}_relative_angle_deg", rr.Scalar(angle_deg))

                    rr.log(
                        f"{base_path}/object_pc",
                        rr.Points3D(
                            positions=obj_vertices[frame_idx],
                            radii=0.005,
                            colors=[0, 255, 0],
                        )
                    )

                    rr.log(
                        f"{base_path}/object_axes",
                        rr.LineStrips3D(
                            axis_segments,
                            colors=axis_colors,
                            radii=axis_radius,
                        ),
                    )

            global_entry_idx += 1

if __name__ == "__main__":
    main()
    
