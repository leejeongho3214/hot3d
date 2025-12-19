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


def _slugify_action_key(text: str) -> str:
    """Convert the normalized action description into an id-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug


with open(os.path.join(home, "Desktop/hot3d_vis/instance.json"), "r") as f:
    instance_ = json.load(f)

object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
obj_pc = dict()

for obj_name in object_model.obj_pcs.keys():
    _, pc, _, _ = object_model(obj_name)
    obj_pc[obj_name] = torch.tensor(pc)

l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

        
def visualize_rr(recoding_name, file_name, color):
    with open(recoding_name, "rb") as f:
        item = pickle.load(f)

        action_to_index = dict()
        action_to_slug = dict()
        action_to_sample_counts = defaultdict(int)
        
        for x_lhand, x_rhand, x_obj, text, gt_lhand, gt_rhand, gt_obj, gaze_map, cov_map in item:
            for batch_idx in range(len(x_lhand)):
                text_entry = text[batch_idx]
                
                if text_entry == "Discard":
                    continue
                
                action_key = _normalize_action_key(text_entry)
                if action_key not in action_to_index:
                    action_index = len(action_to_index)
                    action_to_index[action_key] = action_index
                    slug = _slugify_action_key(action_key)
                    action_to_slug[action_key] = slug if slug else "action"
                    
                action_index = action_to_index[action_key]
                action_slug = action_to_slug[action_key]
                base_path = f"{file_name}/{action_index}_{action_slug}"
                hand_path = "r_hand" if "right" in text_entry.lower() else "l_hand"
                sample_idx = action_to_sample_counts[action_key]
                action_to_sample_counts[action_key] += 1
                sample_pred_path = f"pred/{base_path}/sample_{sample_idx:03d}"
                sample_gt_path = f"gt/{base_path}/sample_{sample_idx:03d}"

                r_hand_vertices, r_hand_faces = process_hand_result(
                    r_hand_layer, x_rhand[batch_idx]
                )
                l_hand_vertices, l_hand_faces = process_hand_result(
                    l_hand_layer, x_lhand[batch_idx]
                )
                
                obj_vertices = process_obj_result(
                    obj_pc[text_entry.split("of ")[-1].split(" with")[0].lower()],
                    x_obj[batch_idx],
                )

                r_hand_vertices_gt, _ = process_hand_result(r_hand_layer, gt_rhand[batch_idx])
                l_hand_vertices_gt, _ = process_hand_result(l_hand_layer, gt_lhand[batch_idx])
                obj_vertices_gt = process_obj_result(obj_pc[text[batch_idx].split("of ")[-1].split(' with')[0].lower()], gt_obj[batch_idx])

                r_mesh = trimesh.Trimesh(
                    vertices=r_hand_vertices[0], faces=r_hand_faces, process=False
                )
                l_mesh = trimesh.Trimesh(
                    vertices=l_hand_vertices[0], faces=l_hand_faces, process=False
                )

                for frame_idx in range(obj_vertices.shape[0]):
                    rr.set_time_sequence("sample", sample_idx)
                    rr.set_time_sequence("frame", frame_idx)

                    if "right" in text_entry.lower():
                        rr.log(
                            f"{sample_pred_path}/{hand_path}",
                            rr.Mesh3D(
                                vertex_positions=r_hand_vertices[frame_idx],
                                triangle_indices=r_hand_faces,
                                vertex_normals=r_mesh.vertex_normals,
                            ),
                        )

                    else:
                        rr.log(
                            f"{sample_pred_path}/{hand_path}",
                            rr.Mesh3D(
                                vertex_positions=l_hand_vertices[frame_idx],
                                triangle_indices=l_hand_faces,
                                vertex_normals=l_mesh.vertex_normals,
                            ),
                        )

                    rr.log(
                        f"{sample_pred_path}/object",
                        rr.Points3D(
                            positions=obj_vertices[frame_idx],
                            radii=0.005,
                            colors=color,
                            labels=[text_entry],
                        ),
                    )
                    
                    if "right" in text_entry.lower():
                        rr.log(
                            f"{sample_gt_path}/{hand_path}",
                            rr.Mesh3D(
                                vertex_positions=r_hand_vertices_gt[frame_idx],
                                triangle_indices=r_hand_faces,
                                vertex_normals=r_mesh.vertex_normals,
                            ),
                        )

                    else:
                        rr.log(
                            f"{sample_gt_path}/{hand_path}",
                            rr.Mesh3D(
                                vertex_positions=l_hand_vertices_gt[frame_idx],
                                triangle_indices=l_hand_faces,
                                vertex_normals=l_mesh.vertex_normals,
                            ),
                        )

                    rr.log(
                        f"{sample_gt_path}/object",
                        rr.Points3D(
                            positions=obj_vertices_gt[frame_idx],
                            radii=0.005,
                            colors=[121, 121, 121],
                            labels=[text_entry],
                        ),
                    )

                    if gaze_map[batch_idx][frame_idx].sum() != 0 or frame_idx == 0:
                        colors = np.zeros_like(obj_pc[obj_name], dtype=np.uint8)
                        colors[gaze_map[batch_idx][frame_idx] == 1] = [255, 255, 0]
                        colors[gaze_map[batch_idx][frame_idx] == 0] = [0, 0, 255]

                    rr.log(
                        f"{sample_gt_path}/gaze_map",
                        rr.Points3D(
                            positions=obj_vertices_gt[frame_idx],
                            radii=0.005,
                            colors=colors,
                            labels=[text[batch_idx]],
                        ),
                    )


                    colors_cov = np.zeros_like(obj_pc[obj_name], dtype=np.uint8)
                    colors_cov[cov_map[batch_idx] == 1] = [255, 255, 0]
                    colors_cov[cov_map[batch_idx] == 0] = [0, 0, 255]

                    rr.log(
                        f"{sample_gt_path}/cov_map",
                        rr.Points3D(
                            positions=obj_vertices_gt[frame_idx],
                            radii=0.005,
                            colors=colors_cov,
                        )
                        )

def main():
    rr.init("Input Data", spawn=True)

    for i in range(2, 20, 2):
        file_name = f"both_ori_loss_{i}000"
        recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
        visualize_rr(recoding_name, file_name, random_rgb_color())
    
    # file_name = "text2hoi_cov_mug_ori"
    # recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    # visualize_rr(recoding_name, file_name, random_rgb_color())
    
    # file_name = "text2hoi_gaze_mug_ori"
    # recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    # visualize_rr(recoding_name, file_name, random_rgb_color())

if __name__ == "__main__":
    main()