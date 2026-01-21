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


def visualize_rr(recoding_name, idx):
    with open(recoding_name, "rb") as f:
        item = pickle.load(f)

        action_to_sample_counts = defaultdict(int)
        ref_obj_pred = {}
        ref_obj_gt = {}

        for (
            x_lhand,
            x_rhand,
            x_obj,
            text,
            course_lhand,
            course_rhand,
            gt_obj,
            gaze_map,
            cov_map,
            est_cov_map,
        ) in item:
            for batch_idx in range(32):
                text_entry = text
                object_key, action_key = _build_grouping_keys(text_entry)
                base_path = _sanitize_entity_path(object_key)
                text_path = _sanitize_entity_path(text_entry)
                hand_path = "r_hand" if "right" in text_entry.lower() else "l_hand"
                sample_idx = action_to_sample_counts[action_key]
                action_to_sample_counts[action_key] += 1
                sample_pred_path = f"{idx}/original/fine/{base_path}/{text_path}/sample_{sample_idx:03d}"
                sample_gt_path = f"{idx}/original/course/{base_path}/{text_path}/sample_{sample_idx:03d}"

                # if text_entry.split("of ")[-1].split(" with")[0].lower() not in ["mug_white", "flask", "mug_patterned"]:

                if text_entry.split("of ")[-1].split(" with")[0].lower() not in [
                    "mug_white"
                ]:
                    continue

                # r_hand_vertices, r_hand_faces = process_hand_result(
                #     r_hand_layer, x_rhand[batch_idx]
                # )
                # l_hand_vertices, l_hand_faces = process_hand_result(
                #     l_hand_layer, x_lhand[batch_idx]
                # )

                obj_vertices = process_obj_result(
                    obj_pc[text_entry.split("of ")[-1].split(" with")[0].lower()],
                    x_obj[batch_idx],
                )

                r_hand_vertices_gt, r_hand_faces = process_hand_result(
                    r_hand_layer, course_rhand[batch_idx]
                )
                l_hand_vertices_gt, l_hand_faces = process_hand_result(
                    l_hand_layer, course_lhand[batch_idx]
                )
                # obj_vertices_gt = process_obj_result(
                #     obj_pc[text.split("of ")[-1].split(" with")[0].lower()],
                #     gt_obj[batch_idx],
                # )
                obj_vertices_gt = obj_vertices

                if action_key not in ref_obj_gt:
                    ref_obj_gt[action_key] = obj_vertices_gt[0].detach().cpu().numpy()

                if action_key not in ref_obj_pred:
                    ref_obj_pred[action_key] = obj_vertices[0].detach().cpu().numpy()

                r_mesh = trimesh.Trimesh(
                    vertices=r_hand_vertices_gt[0], faces=r_hand_faces, process=False
                )
                l_mesh = trimesh.Trimesh(
                    vertices=l_hand_vertices_gt[0], faces=l_hand_faces, process=False
                )

                for frame_idx in range(obj_vertices.shape[0]):
                    rr.set_time_sequence("sample", sample_idx)
                    rr.set_time_sequence("frame", frame_idx)

                    # if "right" in text_entry.lower():
                    #     rr.log(
                    #         f"{sample_pred_path}/{hand_path}",
                    #         rr.Mesh3D(
                    #             vertex_positions=r_hand_vertices[frame_idx],
                    #             triangle_indices=r_hand_faces,
                    #             vertex_normals=r_mesh.vertex_normals,
                    #         ),
                    #     )

                    # else:
                    #     rr.log(
                    #         f"{sample_pred_path}/{hand_path}",
                    #         rr.Mesh3D(
                    #             vertex_positions=l_hand_vertices[frame_idx],
                    #             triangle_indices=l_hand_faces,
                    #             vertex_normals=l_mesh.vertex_normals,
                    #         ),
                    #     )

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

                    # rr.log(
                    #     f"{sample_gt_path}/object",
                    #     rr.Points3D(
                    #         positions=obj_vertices[frame_idx],
                    #         radii=0.005,
                    #         colors=[121, 121, 121],
                    #         labels=[text_entry],
                    #     ),
                    # )

                    obj_pred_np = obj_vertices[frame_idx].detach().cpu().numpy()
                    obj_gt_np = obj_vertices_gt[frame_idx].detach().cpu().numpy()
                    # r_hand_np = r_hand_vertices[frame_idx].detach().cpu().numpy()
                    # l_hand_np = l_hand_vertices[frame_idx].detach().cpu().numpy()
                    r_hand_gt_np = r_hand_vertices_gt[frame_idx].detach().cpu().numpy()
                    l_hand_gt_np = l_hand_vertices_gt[frame_idx].detach().cpu().numpy()

                    R_pred, t_pred = rigid_transform(
                        obj_pred_np, ref_obj_pred[action_key]
                    )
                    R_gt, t_gt = rigid_transform(obj_gt_np, ref_obj_gt[action_key])

                    # r_hand_aligned = apply_rigid_transform(r_hand_np, R_pred, t_pred)
                    # l_hand_aligned = apply_rigid_transform(l_hand_np, R_pred, t_pred)
                    r_hand_gt_aligned = apply_rigid_transform(r_hand_gt_np, R_gt, t_gt)
                    l_hand_gt_aligned = apply_rigid_transform(l_hand_gt_np, R_gt, t_gt)

                    aligned_pred_path = f"{idx}/aligned/pred/{base_path}/{text_path}/sample_{sample_idx:03d}"
                    aligned_gt_path = f"{idx}/aligned/course/{base_path}/{text_path}/sample_{sample_idx:03d}"

                    # rr.log(
                    #     f"{aligned_pred_path}/object",
                    #     rr.Points3D(
                    #         positions=ref_obj_pred[action_key],
                    #         radii=0.005,
                    #         colors=[160, 160, 160],
                    #         labels=[text_entry],
                    #     ),
                    # )
                    rr.log(
                        f"{aligned_gt_path}/object",
                        rr.Points3D(
                            positions=ref_obj_gt[action_key],
                            radii=0.005,
                            colors=[160, 160, 160],
                            labels=[text_entry],
                        ),
                    )

                    if "right" in text_entry.lower():
                        # rr.log(
                        #     f"{aligned_pred_path}/{hand_path}",
                        #     rr.Mesh3D(
                        #         vertex_positions=r_hand_aligned,
                        #         triangle_indices=r_hand_faces,
                        #         vertex_normals=r_mesh.vertex_normals,
                        #     ),
                        # )
                        rr.log(
                            f"{aligned_gt_path}/{hand_path}",
                            rr.Mesh3D(
                                vertex_positions=r_hand_gt_aligned,
                                triangle_indices=r_hand_faces,
                                vertex_normals=r_mesh.vertex_normals,
                            ),
                        )
                    else:
                        # rr.log(
                        #     f"{aligned_pred_path}/{hand_path}",
                        #     rr.Mesh3D(
                        #         vertex_positions=l_hand_aligned,
                        #         triangle_indices=l_hand_faces,
                        #         vertex_normals=l_mesh.vertex_normals,
                        #     ),
                        # )
                        rr.log(
                            f"{aligned_gt_path}/{hand_path}",
                            rr.Mesh3D(
                                vertex_positions=l_hand_gt_aligned,
                                triangle_indices=l_hand_faces,
                                vertex_normals=l_mesh.vertex_normals,
                            ),
                        )

                    # if gaze_map[batch_idx][frame_idx].sum() != 0 or frame_idx == 0:
                    #     colors = np.zeros_like(obj_pc[obj_name], dtype=np.uint8)
                    #     colors[gaze_map[batch_idx][frame_idx] == 1] = [255, 255, 0]
                    #     colors[gaze_map[batch_idx][frame_idx] == 0] = [0, 0, 255]

                    # rr.log(
                    #     f"{sample_gt_path}/gaze_map",
                    #     rr.Points3D(
                    #         positions=obj_vertices_gt[frame_idx],
                    #         radii=0.005,
                    #         colors=colors,
                    #         labels=[text[batch_idx]],
                    #     ),
                    # )

                    # colors_cov = np.zeros_like(obj_pc[obj_name], dtype=np.uint8)
                    # colors_cov[cov_map[batch_idx] == 1] = [255, 255, 0]
                    # colors_cov[cov_map[batch_idx] == 0] = [0, 0, 255]

                    # rr.log(
                    #     f"{sample_gt_path}/cov_map",
                    #     rr.Points3D(
                    #         positions=obj_vertices_gt[frame_idx],
                    #         radii=0.005,
                    #         colors=colors_cov,
                    #     )
                    #     )

                    colors_cov = np.zeros_like(obj_pc[obj_name], dtype=np.uint8)
                    # colors_cov[:] = [0, 0, 255]
                    colors_cov[est_cov_map[batch_idx] == 1] = [255, 255, 0]
                    colors_cov[est_cov_map[batch_idx] == 0] = [0, 0, 255]

                    # rr.log(
                    #     f"{sample_pred_path}/est_cov_map",
                    #     rr.Points3D(
                    #         positions=obj_vertices[frame_idx],
                    #         radii=0.005,
                    #         colors=colors_cov,
                    #     ),
                    # )


                    rr.log(
                        f"{sample_gt_path}/est_cov_map",
                        rr.Points3D(
                            positions=obj_vertices[frame_idx],
                            radii=0.005,
                            colors=colors_cov,
                        ),
                    )


def main():
    rr.init("Input Data", spawn=True)

    # file_name = f"grab_ori"
    # recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    # visualize_rr(recoding_name, file_name)
    
    file_name = f"grab_mug"
    recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    visualize_rr(recoding_name, file_name)
    
    file_name = f"grab_mug_two"
    recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    visualize_rr(recoding_name, file_name)



if __name__ == "__main__":
    main()
