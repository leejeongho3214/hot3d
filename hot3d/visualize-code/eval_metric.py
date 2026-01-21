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
    hand_joints = out.joints + hand_trans
    hand_faces = hand_layer.faces.copy().astype(np.int16)
    hand_faces = torch.LongTensor(hand_faces)
    return hand_vertices, hand_faces, hand_joints


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


def _to_torch(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


def _compute_cov_map_from_course(
    course_lhand,
    course_rhand,
    x_obj,
    obj_points,
    l_hand_layer,
    r_hand_layer,
    threshold: float = 0.02,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return None


def _compute_contact_from_vertices(
    obj_vertices_np: np.ndarray,
    l_hand_vertices_np: np.ndarray,
    r_hand_vertices_np: np.ndarray,
    threshold: float = 0.02,
) -> np.ndarray:
    nframes = obj_vertices_np.shape[0]
    contact_per_frame = np.zeros((nframes, obj_vertices_np.shape[1]), dtype=bool)

    for frame_idx in range(nframes):
        obj_frame = obj_vertices_np[frame_idx]
        frame_contact = np.zeros(obj_frame.shape[0], dtype=bool)
        if l_hand_vertices_np is not None:
            l_frame = l_hand_vertices_np[min(frame_idx, l_hand_vertices_np.shape[0] - 1)]
            l_dist = np.linalg.norm(obj_frame[:, None, :] - l_frame[None, :, :], axis=2)
            frame_contact |= l_dist.min(axis=1) < threshold
        if r_hand_vertices_np is not None:
            r_frame = r_hand_vertices_np[min(frame_idx, r_hand_vertices_np.shape[0] - 1)]
            r_dist = np.linalg.norm(obj_frame[:, None, :] - r_frame[None, :, :], axis=2)
            frame_contact |= r_dist.min(axis=1) < threshold
        contact_per_frame[frame_idx] = frame_contact
    return contact_per_frame


def _topk_mass(counts: np.ndarray, k_ratio: float = 0.05) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    n = counts.shape[0]
    k = max(1, int(np.ceil(n * k_ratio)))
    topk = np.partition(counts, -k)[-k:]
    return float(topk.sum() / total)


def _gini_index(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = sorted_counts.size
    cum = np.sum((np.arange(1, n + 1) * sorted_counts))
    return float((2.0 * cum) / (n * total) - (n + 1) / n)


def _simpson_diversity(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _coverage_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def _jaccard_diversity_samples(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 0.0
    values = []
    for i in range(len(masks) - 1):
        for j in range(i + 1, len(masks)):
            a = masks[i]
            b = masks[j]
            union = np.logical_or(a, b).sum()
            if union == 0:
                continue
            inter = np.logical_and(a, b).sum()
            values.append(1.0 - (inter / union))
    return float(np.mean(values)) if values else 0.0


def _object_diameter(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    return float(np.linalg.norm(max_xyz - min_xyz))


def _spatial_spread(
    mask: np.ndarray,
    points: np.ndarray,
    max_points: int = 200,
) -> float:
    if mask.sum() < 2:
        return 0.0
    pts = points[mask]
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    mean_pairwise = dists[np.triu_indices(dists.shape[0], k=1)].mean()
    diameter = _object_diameter(points)
    if diameter <= 0:
        return 0.0
    return float(mean_pairwise / diameter)


def _knn_coverage(
    mask: np.ndarray,
    points: np.ndarray,
    k: int = 5,
    max_points: int = 200,
) -> float:
    if mask.sum() < 2:
        return 0.0
    pts = points[mask]
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    kk = min(k, pts.shape[0] - 1)
    nearest = np.partition(dists, kk, axis=1)[:, :kk]
    mean_knn = nearest.mean()
    diameter = _object_diameter(points)
    if diameter <= 0:
        return 0.0
    return float(mean_knn / diameter)


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
        overlap_stats = defaultdict(list)
        reverse_overlap_stats = defaultdict(list)
        diversity_stats = defaultdict(list)
        sample_diversity_masks = defaultdict(list)
        pkl_sample_diversity_masks = defaultdict(list)

        for (
            x_lhand,
            x_rhand,
            x_obj,
            text,
            course_lhand,
            course_rhand,
            _,
            _,
            _,
            cov_map,
            # cov_map,
        ) in item:
            for batch_idx in range(32):
                text_entry = str(text)
                object_key, action_key = _build_grouping_keys(text_entry)
                base_path = _sanitize_entity_path(object_key)
                text_path = _sanitize_entity_path(text_entry)
                hand_path = "r_hand" if "right" in text_entry.lower() else "l_hand"
                sample_idx = action_to_sample_counts[action_key]
                action_to_sample_counts[action_key] += 1
                sample_pred_path = f"{idx}/original/fine/{base_path}/{text_path}/sample_{sample_idx:03d}"
                sample_gt_path = f"{idx}/original/course/{base_path}/{text_path}/sample_{sample_idx:03d}"

                # if text_entry.split("of ")[-1].split(" with")[0].lower() not in ["mug_white", "flask", "mug_patterned"]:

                obj_name = text_entry.split("of ")[-1].split(" with")[0].lower()
                if obj_name not in [
                    "mug_white"
                ]:
                    continue

                # r_hand_vertices, r_hand_faces = process_hand_result(
                #     r_hand_layer, x_rhand[batch_idx]
                # )
                # l_hand_vertices, l_hand_faces = process_hand_result(
                #     l_hand_layer, x_lhand[batch_idx]
                # )

                obj_vertices = process_obj_result(obj_pc[obj_name], x_obj[batch_idx])

                r_hand_vertices_gt, r_hand_faces, r_hand_joints = process_hand_result(
                    r_hand_layer, course_rhand[batch_idx]
                )
                l_hand_vertices_gt, l_hand_faces, l_hand_joints = process_hand_result(
                    l_hand_layer, course_lhand[batch_idx]
                )
                obj_vertices_gt = obj_vertices

                r_mesh = trimesh.Trimesh(
                    vertices=r_hand_vertices_gt[0], faces=r_hand_faces, process=False
                )
                l_mesh = trimesh.Trimesh(
                    vertices=l_hand_vertices_gt[0], faces=l_hand_faces, process=False
                )

                obj_vertices_np = obj_vertices_gt
                l_hand_joints_np = (
                    l_hand_joints
                    if "left" in text_entry.lower()
                    else None
                )
                r_hand_joints_np = (
                    r_hand_joints
                    if "right" in text_entry.lower()
                    else None
                )
                contact_per_frame = _compute_contact_from_vertices(
                    obj_vertices_np,
                    l_hand_joints_np,
                    r_hand_joints_np,
                )
                contact_cumulative = np.cumsum(contact_per_frame, axis=0) > 0
                computed_cov_map = contact_cumulative[-1].astype(np.float32)
                computed_counts = contact_per_frame.sum(axis=0).astype(np.float32)
                pkl_cov_map = cov_map[batch_idx] if cov_map is not None else None

                if pkl_cov_map is not None:
                    pkl_cov_map = np.asarray(pkl_cov_map)
                    if pkl_cov_map.ndim > 1:
                        pkl_cov_map = pkl_cov_map.any(axis=0).astype(np.float32)

                if obj_vertices_np is None:
                    continue

                pkl_mask = None
                if pkl_cov_map is not None and pkl_cov_map.shape[0] == obj_vertices_np.shape[1]:
                    pkl_mask = pkl_cov_map > 0

                for frame_idx in range(contact_cumulative.shape[0]):
                    rr.set_time_sequence("sample", sample_idx)
                    rr.set_time_sequence("frame", frame_idx)
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
                    obj_frame_np = obj_vertices_np[frame_idx]
                    colors_cov = np.zeros((obj_frame_np.shape[0], 3), dtype=np.uint8)
                    colors_cov[:] = [0, 0, 255]

                    computed_mask = contact_cumulative[frame_idx]
                    colors_cov[computed_mask] = [0, 255, 0]

                    if pkl_mask is not None:
                        colors_cov[pkl_mask] = [255, 255, 0]
                        overlap_mask = pkl_mask & computed_mask
                        colors_cov[overlap_mask] = [255, 0, 0]

                    rr.log(
                        f"{sample_gt_path}/contact_map_compare",
                        rr.Points3D(
                            positions=obj_frame_np,
                            radii=0.005,
                            colors=colors_cov,
                        ),
                    )

                if pkl_mask is not None:
                    final_mask = contact_cumulative[-1]
                    overlap_mask = pkl_mask & final_mask
                    denom_pkl = pkl_mask.sum()
                    denom_calc = final_mask.sum()
                    if denom_pkl > 0:
                        overlap_pct = (overlap_mask.sum() / denom_pkl) * 100.0
                        overlap_stats[text_entry].append(float(overlap_pct))
                    if denom_calc > 0:
                        reverse_overlap_pct = (overlap_mask.sum() / denom_calc) * 100.0
                        reverse_overlap_stats[text_entry].append(float(reverse_overlap_pct))

                if computed_counts is not None:
                    diversity_stats[text_entry].append(
                        (
                            _topk_mass(computed_counts, 0.05),
                            _gini_index(computed_counts),
                            _simpson_diversity(computed_counts),
                        )
                    )
                if computed_cov_map is not None:
                    mask = computed_cov_map > 0
                    sample_diversity_masks[text_entry].append(mask)

                if pkl_mask is not None:
                    pkl_sample_diversity_masks[text_entry].append(pkl_mask)

                rr.log(
                    f"{sample_gt_path}/contact_map_compare",
                    rr.Points3D(
                        positions=obj_frame_np,
                        radii=0.005,
                        colors=colors_cov,
                    ),
                )

        if overlap_stats:
            print("Contact overlap (%) per text")
            for text_key, values in overlap_stats.items():
                avg_pct = float(np.mean(values)) if values else 0.0
                reverse_values = reverse_overlap_stats.get(text_key, [])
                avg_reverse = float(np.mean(reverse_values)) if reverse_values else 0.0
                print(f"- {text_key}: pkl<-calc ⬆️  {avg_pct:.2f}% | calc<-pkl  ⬆️  {avg_reverse:.2f}%")

        # if diversity_stats: 
        #     print("Contact diversity per text (top5%, gini, simpson)")
        #     for text_key, values in diversity_stats.items():
        #         if not values:
        #             print(f"- {text_key}: 0.000, 0.000, 0.000")
        #             continue
        #         arr = np.asarray(values, dtype=np.float32)
        #         avg_vals = arr.mean(axis=0)
        #         print(
        #             f"- {text_key}: {avg_vals[0]:.3f}, {avg_vals[1]:.3f}, {avg_vals[2]:.3f}"
        #         )
        # if sample_diversity_masks:
        #     print("Contact diversity across samples per text (coverage, jaccard, gini, simpson)")
        #     for text_key, masks in sample_diversity_masks.items():
        #         if not masks:
        #             print(f"- {text_key}: 0.000, 0.000, 0.000, 0.000")
        #             continue
        #         union_mask = np.logical_or.reduce(masks)
        #         coverage = _coverage_ratio(union_mask)
        #         jaccard = _jaccard_diversity_samples(masks)
        #         freq_counts = np.sum(np.stack(masks, axis=0), axis=0).astype(np.float32)
        #         gini = _gini_index(freq_counts)
        #         simpson = _simpson_diversity(freq_counts)
        #         print(f"- {text_key}: {coverage:.3f}, {jaccard:.3f}, {gini:.3f}, {simpson:.3f}")
        # if pkl_sample_diversity_masks:
        #     print("PKL contact diversity across samples per text (coverage, jaccard, gini, simpson)")
        #     for text_key, masks in pkl_sample_diversity_masks.items():
        #         if not masks:
        #             print(f"- {text_key}: 0.000, 0.000, 0.000, 0.000")
        #             continue
        #         union_mask = np.logical_or.reduce(masks)
        #         coverage = _coverage_ratio(union_mask)
        #         jaccard = _jaccard_diversity_samples(masks)
        #         freq_counts = np.sum(np.stack(masks, axis=0), axis=0).astype(np.float32)
        #         gini = _gini_index(freq_counts)
        #         simpson = _simpson_diversity(freq_counts)
        #         print(f"- {text_key}: {coverage:.3f}, {jaccard:.3f}, {gini:.3f}, {simpson:.3f}")


def main():
    rr.init("Input Data", spawn=True)

    file_name = f"grab_mug_ro"
    recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    visualize_rr(recoding_name, file_name)
    
    # file_name = f"grab_mug"
    # recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    # visualize_rr(recoding_name, file_name)
    
    # file_name = f"grab_mug_two"
    # recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}.pkl"
    # visualize_rr(recoding_name, file_name)


if __name__ == "__main__":
    main()
