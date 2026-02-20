import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
HOT3D_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
for _path in [VIS_ROOT, HOT3D_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mano import build_mano_aa
from data_loaders.mano_layer import MANOHandModel
import pickle
import rerun as rr
from rot import *
import torch
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import trimesh
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from projectaria_tools.core.sophus import SE3
import numpy as np
import itertools
from typing import Optional
import re
import inspect
import json
import argparse

if not hasattr(inspect, "getargspec"):
    # Compatibility for chumpy on Python 3.11+
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

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


def _norm_key(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value


def _parse_object_part_from_text(
    text_entry: str,
) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(text_entry, str):
        return None, None
    match = re.search(
        r"grab\s+(?P<part>.+?)\s+of\s+(?P<object>.+)$",
        text_entry,
        re.IGNORECASE,
    )
    if match:
        part = match.group("part").strip()
        obj = match.group("object").strip()
        obj = re.split(r"\s+with\s+", obj, flags=re.IGNORECASE)[0]
        obj = obj.rstrip(" .,:;")
        return obj, part
    return None, None


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)


def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_gaze_frame_points(gaze_entry, frame_idx):
    if gaze_entry is None:
        return None

    if isinstance(gaze_entry, list):
        if frame_idx >= len(gaze_entry):
            return None
        frame = gaze_entry[frame_idx]
        frame_arr = _to_numpy(frame)
        if frame_arr.ndim == 3 and frame_arr.shape[0] == 2:
            start = frame_arr[0].reshape(-1)
            end = frame_arr[1].reshape(-1)
            return start, end

    if isinstance(gaze_entry, (list, tuple)) and len(gaze_entry) == 2:
        start_list, end_list = gaze_entry
        if frame_idx >= len(start_list) or frame_idx >= len(end_list):
            return None
        return start_list[frame_idx], end_list[frame_idx]

    gaze_arr = _to_numpy(gaze_entry)
    # Support shapes like (T, 2, 3, 1) or (B, T, 2, 3, 1) after indexing batch.
    if gaze_arr.ndim >= 4 and gaze_arr.shape[-1] == 1:
        gaze_arr = np.squeeze(gaze_arr, axis=-1)
    if gaze_arr.ndim == 5 and gaze_arr.shape[2] == 2 and gaze_arr.shape[3] == 3:
        return gaze_arr[0, frame_idx, 0], gaze_arr[0, frame_idx, 1]
    if gaze_arr.ndim == 4 and gaze_arr.shape[1] == 2 and gaze_arr.shape[2] == 3:
        return gaze_arr[frame_idx, 0], gaze_arr[frame_idx, 1]
    if gaze_arr.ndim == 4 and gaze_arr.shape[1] == 2:
        return gaze_arr[frame_idx, 0], gaze_arr[frame_idx, 1]
    if gaze_arr.ndim == 3 and gaze_arr.shape[-1] == 3:
        if gaze_arr.shape[1] == 2:
            return gaze_arr[frame_idx, 0], gaze_arr[frame_idx, 1]
        if gaze_arr.shape[0] == 2:
            return gaze_arr[0, frame_idx], gaze_arr[1, frame_idx]
    return None


def _select_gaze_hits(obj_points, hit_indices):
    if hit_indices is None:
        return None
    if isinstance(hit_indices, torch.Tensor):
        hit_indices = hit_indices.detach().cpu()
    if isinstance(hit_indices, np.ndarray):
        if hit_indices.dtype == bool and hit_indices.shape[0] == obj_points.shape[0]:
            return obj_points[hit_indices]
        if hit_indices.ndim == 1 and hit_indices.shape[0] == obj_points.shape[0]:
            unique_vals = np.unique(hit_indices)
            if set(unique_vals.tolist()).issubset({0, 1}):
                return obj_points[hit_indices.astype(bool)]
        if hit_indices.ndim == 1:
            return obj_points[hit_indices]
    if isinstance(hit_indices, list):
        if len(hit_indices) == 0:
            return obj_points[:0]
        if (
            isinstance(hit_indices[0], (bool, np.bool_))
            and len(hit_indices) == obj_points.shape[0]
        ):
            return obj_points[np.array(hit_indices, dtype=bool)]
        return obj_points[hit_indices]
    return None


def _apply_pose_to_point(point, pose):
    if point is None or pose is None:
        return point
    p = _to_numpy(point).reshape(-1)
    if p.shape[0] != 3:
        return point
    T = _to_numpy(pose)
    if T.shape != (4, 4):
        return point
    ph = np.concatenate([p, [1.0]], axis=0)
    pw = T @ ph
    return pw[:3]


def _apply_pose_to_vector(vec, pose):
    if vec is None or pose is None:
        return vec
    v = _to_numpy(vec).reshape(-1)
    if v.shape[0] != 3:
        return vec
    T = _to_numpy(pose)
    if T.shape != (4, 4):
        return vec
    R = T[:3, :3]
    return R @ v


def _invert_pose(pose):
    if pose is None:
        return None
    T = _to_numpy(pose)
    if T.shape != (4, 4):
        return None
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def _prepare_point_scores(
    values, num_points: int, invert: bool = False
) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = _to_numpy(values)
    if arr.size == 0:
        return None
    arr = np.asarray(arr)
    if arr.ndim > 1:
        if arr.shape[-1] == num_points:
            arr = arr.reshape(-1, num_points).mean(axis=0)
        elif arr.shape[0] == num_points:
            arr = arr.reshape(num_points, -1).mean(axis=1)
        else:
            arr = arr.reshape(-1)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != num_points:
        return None
    if arr.dtype == np.bool_:
        scores = arr.astype(np.float32)
    else:
        scores = arr.astype(np.float32, copy=False)
    if invert:
        scores = 1.0 - scores
    vmin = float(np.min(scores))
    vmax = float(np.max(scores))
    if vmax > vmin:
        scores = (scores - vmin) / (vmax - vmin)
    else:
        scores = np.zeros_like(scores, dtype=np.float32)
    return np.clip(scores, 0.0, 1.0)


def _scores_to_blue_yellow(scores: np.ndarray) -> np.ndarray:
    low_color = np.array([0, 120, 255], dtype=np.float32)
    high_color = np.array([255, 255, 0], dtype=np.float32)
    return (low_color + (high_color - low_color) * scores[:, None]).astype(np.uint8)


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

label_merged_path = os.path.join(home, "label_merged.json")
if not os.path.exists(label_merged_path):
    local_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "label_merged.json"
    )
    local_path = os.path.abspath(local_path)
    if os.path.exists(local_path):
        label_merged_path = local_path
    else:
        local_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "label_merged.json"
        )
        local_path = os.path.abspath(local_path)
        if os.path.exists(local_path):
            label_merged_path = local_path
label_merged = None
if os.path.exists(label_merged_path):
    with open(label_merged_path, "r") as f:
        label_merged = json.load(f)
else:
    label_merged = None
    print(f"[WARN] label_merged.json not found. Tried: {label_merged_path}")

_label_key_map = {}
_label_part_map = {}
if isinstance(label_merged, dict):
    for k, v in label_merged.items():
        _label_key_map[_norm_key(k)] = k
        if isinstance(v, dict):
            _label_part_map[k] = {_norm_key(pk): pk for pk in v.keys()}

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
        if norm_key_flat and (
            norm_key_flat in norm_cand_flat or norm_cand_flat in norm_key_flat
        ):
            return key
    return None


def main(file_name="acc_ori_train.pkl"):
    rr.init("Input Data", spawn=True)

    with open(os.path.join(home, f"Desktop/hot3d_vis/{file_name}"), "rb") as f:
        item = pickle.load(f)

    l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
    r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

    obj_param = item["x_obj"]
    l_hand, r_hand = item["x_lhand"], item["x_rhand"]
    gaze = item.get("gaze")
    gaze_map = item.get("gaze_map")
    afford_raw = item.get("afford_map")
    if afford_raw is None:
        afford_raw = item.get("affordance") or item.get("affordnace")
    cam_pose = item.get("cam_pose")

    text = item["action"]
    act_id = item["act_id"]

    entry_counts = defaultdict(int)

    for batch_idx in range(len(text)):
        text_entry = text[batch_idx]

        object_key = _extract_object_key(text_entry)
        if object_key is None or object_key not in obj_pc:
            print(
                f"[WARN] skip entry due to unresolved object: '{text_entry}' -> '{object_key}'"
            )
            continue

        if object_key != "mug_white":
            continue

        entry_counts[(object_key, text_entry)] += 1
        sanitized_entry = re.sub(r"\s+", "_", text_entry.strip())
        log_prefix = f"{file_name}/{object_key}/{sanitized_entry}/{act_id[batch_idx]}"
        base_path = log_prefix

        r_hand_vertices, r_hand_faces = process_hand_result(
            r_hand_layer, torch.tensor(r_hand[batch_idx])
        )
        l_hand_vertices, l_hand_faces = process_hand_result(
            l_hand_layer, torch.tensor(l_hand[batch_idx])
        )
        obj_vertices = process_obj_result(
            obj_pc[object_key], torch.tensor(obj_param[batch_idx])
        )

        r_mesh = trimesh.Trimesh(
            vertices=r_hand_vertices[0], faces=r_hand_faces, process=False
        )
        l_mesh = trimesh.Trimesh(
            vertices=l_hand_vertices[0], faces=l_hand_faces, process=False
        )

        gaze_entry = gaze[batch_idx] if gaze is not None else None
        gaze_map_entry = gaze_map[batch_idx] if gaze_map is not None else None
        afford_entry = afford_raw[batch_idx] if afford_raw is not None else None
        cam_pose_entry = cam_pose[batch_idx] if cam_pose is not None else None

        afford_colors = None
        afford_scores = None
        if afford_entry is not None:
            afford_scores = _prepare_point_scores(
                afford_entry,
                num_points=obj_vertices.shape[1],
                invert=True,
            )
            if afford_scores is not None:
                afford_colors = _scores_to_blue_yellow(afford_scores)

        part_mask = None
        if label_merged is not None:
            obj_name, part_name = _parse_object_part_from_text(text_entry)
            if obj_name is not None and part_name is not None:
                obj_lookup = _label_key_map.get(_norm_key(obj_name))
                if obj_lookup is not None:
                    part_lookup = _label_part_map.get(obj_lookup, {}).get(
                        _norm_key(part_name)
                    )
                    if part_lookup is not None:
                        part_indices = np.asarray(
                            label_merged[obj_lookup][part_lookup], dtype=np.int64
                        )
                        part_mask = np.zeros(
                            obj_vertices.shape[1], dtype=np.float32)
                        valid = (part_indices >= 0) & (
                            part_indices < part_mask.shape[0]
                        )
                        part_mask[part_indices[valid]] = 1.0

        for frame_idx in range(obj_vertices.shape[0]):
            rr.set_time("frame", sequence=frame_idx)

            if "right" in text_entry.lower():
                rr.log(
                    f"{base_path}/r_hand",
                    rr.Mesh3D(
                        vertex_positions=r_hand_vertices[frame_idx],
                        triangle_indices=r_hand_faces,
                        vertex_normals=r_mesh.vertex_normals,
                    ),
                )

            elif "both" in text_entry.lower():
                rr.log(
                    f"{base_path}/r_hand",
                    rr.Mesh3D(
                        vertex_positions=r_hand_vertices[frame_idx],
                        triangle_indices=r_hand_faces,
                        vertex_normals=r_mesh.vertex_normals,
                    ),
                )
                rr.log(
                    f"{base_path}/l_hand",
                    rr.Mesh3D(
                        vertex_positions=l_hand_vertices[frame_idx],
                        triangle_indices=l_hand_faces,
                        vertex_normals=l_mesh.vertex_normals,
                    ),
                )

            else:
                rr.log(
                    f"{base_path}/l_hand",
                    rr.Mesh3D(
                        vertex_positions=l_hand_vertices[frame_idx],
                        triangle_indices=l_hand_faces,
                        vertex_normals=l_mesh.vertex_normals,
                    ),
                )

            object_colors = None
            obj_points_np = _to_numpy(obj_vertices[frame_idx])

            gaze_frame = _get_gaze_frame_points(gaze_entry, frame_idx)
            if gaze_frame is not None:
                gaze_start, gaze_end = gaze_frame
                if gaze_start is not None and gaze_end is not None:
                    # Stored as [origin, direction] for ray-based gaze.
                    start = _to_numpy(gaze_start).reshape(-1)
                    direction = _to_numpy(gaze_end).reshape(-1)
                    pose = None
                    if cam_pose_entry is not None and frame_idx < len(cam_pose_entry):
                        pose = cam_pose_entry[frame_idx]
                    if pose is not None:
                        start = _apply_pose_to_point(start, pose)
                        direction = _apply_pose_to_vector(direction, pose)
                    if start.shape[0] == 3 and direction.shape[0] == 3:
                        rr.log(
                            f"{base_path}/gaze",
                            rr.Arrows3D(
                                origins=[start],
                                vectors=[direction],
                                colors=[[255, 255, 0]],
                                labels=["gaze_vector"],
                            ),
                        )

            # if gaze_map_entry is not None and frame_idx < len(gaze_map_entry):
            #     gaze_indices = gaze_map_entry[frame_idx]
            #     if isinstance(gaze_indices, torch.Tensor):
            #         gaze_indices = gaze_indices.detach().cpu()
            #     gaze_indices = np.asarray(gaze_indices).reshape(-1)
            #     obj_points_np = _to_numpy(obj_vertices[frame_idx])
            #     if gaze_indices.size >= 0:
            #         colors = np.zeros((obj_points_np.shape[0], 3), dtype=np.uint8)
            #         colors[:] = [0, 255, 0]
            #         if gaze_indices.size > 0:
            #             valid = (gaze_indices >= 0) & (gaze_indices < obj_points_np.shape[0])
            #             colors[gaze_indices[valid]] = [255, 0, 0]
            #         object_colors = colors

            rr.log(
                f"{base_path}/object_pc",
                rr.Points3D(
                    positions=obj_points_np,
                    radii=0.005,
                    colors=object_colors if object_colors is not None else [
                        0, 255, 0],
                    labels=[act_id[batch_idx]],
                ),
            )

            offset_step = 0.25
            if afford_colors is not None:
                rr.log(
                    f"{base_path}/afford_map",
                    rr.Points3D(
                        positions=obj_points_np +
                        np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        radii=0.005,
                        colors=afford_colors,
                        labels=[act_id[batch_idx]],
                    ),
                )

            if afford_scores is not None and part_mask is not None:
                new_scores = 0.8 * afford_scores + 0.2 * part_mask
                new_colors = _scores_to_blue_yellow(new_scores)
                rr.log(
                    f"{base_path}/new_map",
                    rr.Points3D(
                        positions=obj_points_np +
                        np.array([offset_step, 0.0, 0.0], dtype=np.float32),
                        radii=0.005,
                        colors=new_colors,
                        labels=[act_id[batch_idx]],
                    ),
                )


if __name__ == "__main__":
    main(file_name="grab_afford.pkl")
    # main(file_name = "grab_afford_min.pkl")
    # main(file_name = "acc_ori_eval.pkl")
