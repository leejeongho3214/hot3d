import argparse
import json
import inspect
import re
from typing import Optional
import itertools
import numpy as np
# NumPy 1.24+ compatibility for legacy packages (e.g., chumpy)
for _name, _type in [("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str)]:
    if _name not in np.__dict__:
        setattr(np, _name, _type)

import os
import torch
import trimesh
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import rerun as rr
import pickle

from rot import *
from data_loaders.mano_layer import MANOHandModel
from mano import build_mano_aa

CONTACT_DISTANCE_THRESHOLD = 0.01

def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)

def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
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


def _compute_contact_colors(hand_vertices_obj: torch.Tensor, obj_points: torch.Tensor) -> np.ndarray:
    """
    Simple closest-point distance test between hand vertices (in object frame) and object point cloud.
    Returns per-vertex RGB colors with contacts marked red.
    """
    # hand_vertices_obj: (V,3), obj_points: (P,3)
    dists = torch.cdist(hand_vertices_obj.unsqueeze(0), obj_points.unsqueeze(0)).squeeze(0).min(dim=1).values
    contact_mask = dists <= CONTACT_DISTANCE_THRESHOLD
    colors = np.full((hand_vertices_obj.shape[0], 3), 200, dtype=np.uint8)
    colors[contact_mask.cpu().numpy()] = np.array([255, 0, 0], dtype=np.uint8)
    return colors


def _normalize_for_lookup(token: str, keep_underscores: bool = True) -> str:
    token = token.lower().strip()
    token = token.replace("-", "_")
    token = token.replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]", "", token)
    if not keep_underscores:
        token = token.replace("_", "")
    return token


def _extract_object_key(text_entry: str) -> Optional[str]:
    """
    Extract object key from text, then align with obj_pc keys via normalization
    (underscores vs spaces, hyphens, punctuation).
    """
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

object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
obj_pc = dict()

for obj_name in object_model.obj_pcs.keys():
    _, pc, _, _ = object_model(obj_name)
    obj_pc[obj_name] = torch.tensor(pc)

_sample_counter = itertools.count()  # global counter so train/eval don't collide


def main(file_name="acc_ori_train.pkl"):
    rr.init("Input Data", spawn=True)

    with open(os.path.join(home, f"Desktop/hot3d_vis/dataset/{file_name}"), "rb") as f:
        item_list = pickle.load(f)

    for item_name in item_list.keys():
        item = item_list[item_name]
        l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
        r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

        obj_param = item["x_obj"]
        l_hand, r_hand = item["x_lhand"], item["x_rhand"]

        text = item["action"]
        act_id = item["act_id"]

        entry_counts = defaultdict(int)

        for batch_idx in range(len(text)):
            text_entry = text[batch_idx]

            object_key = _extract_object_key(text_entry)
            if object_key is None or object_key not in obj_pc:
                print(f"[WARN] skip entry due to unresolved object: '{text_entry}' -> '{object_key}'")
                continue

            obj_offset = torch.zeros(3, dtype=torch.float32)

            entry_counts[(object_key, text_entry)] += 1
            sanitized_entry = re.sub(r"\s+", "_", text_entry.strip())
            sample_idx = next(_sample_counter)
            log_prefix = f"{object_key}/{sanitized_entry}/{act_id[batch_idx]}"
            base_path = log_prefix

            r_hand_vertices, r_hand_faces = process_hand_result(r_hand_layer, torch.tensor(r_hand[batch_idx]))
            l_hand_vertices, l_hand_faces = process_hand_result(l_hand_layer, torch.tensor(l_hand[batch_idx]))
            obj_vertices, obj_rotmat, obj_trans = process_obj_result(
                obj_pc[object_key], torch.tensor(obj_param[batch_idx])
            )

            r_mesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
            l_mesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)

            # Static object template at a fixed pose for this object key.
            rr.log(
                f"{base_path}/object_pc",
                rr.Points3D(
                    positions=obj_pc[object_key] + obj_offset,
                    radii=0.005,
                    colors=[0, 255, 0],
                    labels=[act_id[batch_idx]],
                ),
                static=True,
            )

            for frame_idx in range(obj_vertices.shape[0]):
                rr.set_time_sequence("frame", frame_idx)

                R = obj_rotmat[frame_idx]
                t = obj_trans[frame_idx]

                def to_obj_frame(verts):
                    return torch.einsum("ij,kj->ki", R.T, verts - t) + obj_offset

                r_hand_obj = to_obj_frame(r_hand_vertices[frame_idx])
                l_hand_obj = to_obj_frame(l_hand_vertices[frame_idx])
                obj_points = obj_pc[object_key] + obj_offset
                r_colors = _compute_contact_colors(r_hand_obj, obj_points)
                l_colors = _compute_contact_colors(l_hand_obj, obj_points)

                if "right" in text_entry.lower():
                    rr.log(
                        f"{base_path}/r_hand",
                        rr.Mesh3D(
                            vertex_positions=r_hand_obj,
                            triangle_indices=r_hand_faces,
                            vertex_normals=r_mesh.vertex_normals,
                            vertex_colors=r_colors,
                        ),
                    )

                elif "both" in text_entry.lower():
                    rr.log(
                        f"{base_path}/r_hand",
                        rr.Mesh3D(
                            vertex_positions=r_hand_obj,
                            triangle_indices=r_hand_faces,
                            vertex_normals=r_mesh.vertex_normals,
                            vertex_colors=r_colors,
                        ),
                    )
                    rr.log(
                        f"{base_path}/l_hand",
                        rr.Mesh3D(
                            vertex_positions=l_hand_obj,
                            triangle_indices=l_hand_faces,
                            vertex_normals=l_mesh.vertex_normals,
                            vertex_colors=l_colors,
                        ),
                    )

                else:
                    rr.log(
                        f"{base_path}/l_hand",
                        rr.Mesh3D(
                            vertex_positions=l_hand_obj,
                            triangle_indices=l_hand_faces,
                            vertex_normals=l_mesh.vertex_normals,
                            vertex_colors=l_colors,
                        ),
                    )


if __name__ == "__main__":
    # main(file_name="acc_ori_train.pkl")
    # main(file_name="acc_ori_eval.pkl")
    main(file_name="acc_ori.pkl")
