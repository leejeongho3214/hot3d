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
    return obj_verts_transformed, obj_rotmat, obj_trans
    
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
  "body",
  "bottom",
  "bridge",
  "center",
  "edge",
  "frame",
  "handle",
  "long_edge",
  "rim",
  "roof",
  "short_edge",
  "tail",
]

_PART_REMAP = {
    # Canonicalize mislabeled parts for specific objects.
    "can_soup": {"top": "rim"},
}

def _compute_layout_offset(sample_index: int, spacing: float = 0.4, per_row: int = 5) -> torch.Tensor:
    """
    Generate a simple grid offset so that each visualized sample is spaced apart.
    """
    row = sample_index // per_row
    col = sample_index % per_row
    return torch.tensor([col * spacing, row * spacing, 0.0], dtype=torch.float32)

def _extract_part_keyword(text: str) -> Optional[str]:
    lowered = text.lower()
    for keyword in _PART_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return keyword
    return None


def _slugify_component(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug if slug else fallback


def _find_object_from_text(text: str, object_keys) -> Optional[str]:
    """
    Find an object name mentioned in text by matching known object keys.
    Ignores 'right hand', 'left hand', 'both hands', etc.
    """
    lowered = text.lower()

    # Prefer "of <object>" pattern.
    match = re.search(r"\bof\s+([a-z0-9\s]+?)(?:\s+with|\s+using|\.|$)", lowered)
    candidate = match.group(1).strip() if match else None

    # If we found something like "of vase", use that as the first candidate.
    candidates = []
    if candidate and "hand" not in candidate:
        candidates.append(candidate)

    # Add any object key that appears verbatim in the text.
    for key in object_keys:
        if re.search(rf"\b{re.escape(key.lower())}\b", lowered):
            candidates.append(key)

    for cand in candidates:
        for key in object_keys:
            if cand.lower() == key.lower():
                return key
    return None


with open(os.path.join(home, "Desktop/hot3d_vis/instance.json"), "r") as f:
    instance_ = json.load(f)
    
object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
obj_pc = dict()

for obj_name in object_model.obj_pcs.keys():
    _, pc, _, _ = object_model(obj_name)
    obj_pc[obj_name] = torch.tensor(pc)


def main():
    rr.init("Input Data", spawn=True)
    
    with open(os.path.join(home, "Desktop/hot3d_vis/dataset/acc_ori_train.pkl"), "rb") as f:
        item_list = pickle.load(f)

    entry_counts = defaultdict(int)
    object_offsets = {}
    object_counter = 0

    for item_name in item_list.keys():
        item = item_list[item_name]
        l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
        r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

        obj_param = item['x_obj']
        l_hand, r_hand = item['x_lhand'], item['x_rhand']
        
        text = item['action']
        act_id = item['act_id']

        for batch_idx in range(len(text)):
            text_entry = text[batch_idx]
            
            if text_entry == "  ":
                text_entry = "Grasp rim of Vase with left hand."

            obj_key = _find_object_from_text(text_entry, obj_pc.keys())
            if obj_key is None:
                print(f"Unable to resolve object for text '{text_entry}', skipping.")
                continue

            part_keyword = _extract_part_keyword(text_entry) or "part"
            if obj_key in _PART_REMAP and part_keyword in _PART_REMAP[obj_key]:
                part_keyword = _PART_REMAP[obj_key][part_keyword]

            obj_log = _slugify_component(obj_key, "object")
            part_log = _slugify_component(part_keyword, "part")

            log_prefix = f"{part_log}/{obj_log}"
            entry_counts[log_prefix] += 1
            sample_prefix = f"{log_prefix}/sample_{entry_counts[log_prefix]:03d}"
            if obj_log not in object_offsets:
                object_offsets[obj_log] = _compute_layout_offset(object_counter)
                object_counter += 1
            sample_offset = object_offsets[obj_log]

            r_hand_vertices, r_hand_faces = process_hand_result(r_hand_layer, torch.tensor(r_hand[batch_idx]))
            l_hand_vertices, l_hand_faces = process_hand_result(l_hand_layer, torch.tensor(l_hand[batch_idx]))
            obj_vertices, obj_rotmat, obj_trans = process_obj_result(obj_pc[obj_key], torch.tensor(obj_param[batch_idx]))
            
            r_mesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
            l_mesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)

            # Log object once in its canonical frame so only hand motion is visible.
            rr.log(
                f"{sample_prefix}/object_pc",
                rr.Points3D(
                    positions=obj_pc[obj_key] + sample_offset,
                    radii=0.005,
                    colors=[0, 255, 0],
                    labels=[act_id[batch_idx]]
                ),
                static=True,
            )
            
            for frame_idx in range(obj_vertices.shape[0]):
                rr.set_time_sequence("frame", frame_idx)

                # Transform hands into object-local frame to keep object fixed.
                R = obj_rotmat[frame_idx]
                t = obj_trans[frame_idx]

                def to_obj_frame(verts):
                    return torch.einsum("ij,kj->ki", R.T, verts - t)

                r_hand_obj = to_obj_frame(r_hand_vertices[frame_idx]) + sample_offset
                l_hand_obj = to_obj_frame(l_hand_vertices[frame_idx]) + sample_offset
                
                if "right" in text_entry.lower():
                    rr.log(
                        f"{sample_prefix}/r_hand",
                        rr.Mesh3D(
                            vertex_positions=r_hand_obj,
                            triangle_indices=r_hand_faces,
                            vertex_normals=r_mesh.vertex_normals,
                        ),
                    )
                    
                elif "both" in text_entry.lower():
                    rr.log(
                        f"{sample_prefix}/r_hand",
                        rr.Mesh3D(
                            vertex_positions=r_hand_obj,
                            triangle_indices=r_hand_faces,
                            vertex_normals=r_mesh.vertex_normals,
                        ),
                    )
                    rr.log(
                        f"{sample_prefix}/l_hand",
                        rr.Mesh3D(
                            vertex_positions=l_hand_obj,
                            triangle_indices=l_hand_faces,
                            vertex_normals=l_mesh.vertex_normals
                        ),
                    )



                else:
                    rr.log(
                        f"{sample_prefix}/l_hand",
                        rr.Mesh3D(
                            vertex_positions=l_hand_obj,
                            triangle_indices=l_hand_faces,
                            vertex_normals=l_mesh.vertex_normals
                        ),
                    )

if __name__ == "__main__":
    main()
    
