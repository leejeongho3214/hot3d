import argparse
import inspect
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import rerun as rr
import torch
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT3D_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if HOT3D_ROOT not in sys.path:
    sys.path.insert(0, HOT3D_ROOT)

from interaction_common import (
    ObjectModel,
    _coerce_text,
    _extract_object_key,
    _pose9_sequence,
    _safe_mesh_volume,
    _to_numpy,
    _to_tensor,
    process_hand_result_standard as process_hand_result,
    process_obj_result_text2hoi as process_obj_result,
)
from mano import build_mano_aa
from rot import rot6d_to_rotmat

try:
    from container_cavity import (
        build_container_region,
        cavity_penetration_metrics,
        has_container_spec,
    )
except ModuleNotFoundError:

    def has_container_spec(_object_key):
        return False

    def cavity_penetration_metrics(_obj_mesh, _object_key, _hand_local):
        return None

    def build_container_region(_obj_mesh, _object_key):
        return None


if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

for _name, _type in [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("object", object),
    ("str", str),
    ("complex", complex),
    ("unicode", str),
]:
    if _name not in np.__dict__:
        setattr(np, _name, _type)


HOME = os.path.expanduser("~")
HOT3D_VIS_DIR = os.path.join(HOME, "Desktop", "hot3d_vis")
CONTACT_THRESHOLD_M = 0.01
MIN_CR_FOR_VALID = 0.01
FINGERTIP_JOINT_INDICES = (16, 17, 18, 19, 20)
MIN_CONTACT_KEY_JOINTS = 2


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        dest="input_files",
        action="append",
        help="Result file path or filename under --vis-dir. Repeatable.",
    )
    parser.add_argument(
        "--vis-dir",
        default=HOT3D_VIS_DIR,
        help="Directory containing obj.pkl, meshes, and default result files.",
    )
    parser.add_argument(
        "--data-root",
        default=HOT3D_VIS_DIR,
        help="HOT3D dataset root containing bps_enc.npz/.npy. Defaults to --vis-dir-like local setup.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Log only the last frame of each sequence.",
    )
    parser.add_argument(
        "--hand-visuals",
        nargs="+",
        choices=("mesh", "joints", "vertices"),
        default=("mesh", "joints"),
        help="Select which hand representations to log. Default: mesh joints",
    )
    return parser.parse_args()


def _resolve_input_path(file_name: str, vis_dir: str) -> str:
    expanded = os.path.expanduser(file_name)
    if os.path.isabs(expanded):
        return expanded
    candidate = os.path.join(vis_dir, expanded)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(expanded)


def _load_bps_dict(data_root: Optional[str]) -> dict[str, np.ndarray]:
    if not data_root:
        return {}
    root = os.path.expanduser(data_root)
    for name in ("bps_enc.npz", "bps_enc.npy"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=False)
            return {
                key: np.asarray(data[key], dtype=np.float32).reshape(-1, 3)
                for key in data.files
            }
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
        return {
            key: np.asarray(value, dtype=np.float32).reshape(-1, 3)
            for key, value in raw.items()
        }
    return {}


def _get_batch_item(data, batch_idx: int):
    if data is None:
        return None
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        if data.ndim >= 2 and data.shape[0] > batch_idx:
            return data[batch_idx]
        return None
    if isinstance(data, (list, tuple)):
        return data[batch_idx] if len(data) > batch_idx else None
    return None


def _get_batch_text(text, batch_idx: int) -> str:
    if isinstance(text, (list, tuple)) and len(text) > batch_idx:
        return _coerce_text(text[batch_idx])
    return _coerce_text(text)


def _batch_size(data) -> int:
    if data is None:
        return 0
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        if data.ndim >= 3:
            return int(data.shape[0])
        if data.ndim >= 2:
            return 1
        return 0
    if isinstance(data, (list, tuple)):
        return len(data)
    return 1


def _wrap_single_sample(data):
    if data is None:
        return None
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        return [data] if data.ndim >= 2 else data
    if isinstance(data, (list, tuple)):
        return data
    return [data]


def _dict_get_first(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _batch_hint_size(*values) -> int:
    for value in values:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and value:
            return len(value)
    return 0


def _normalized_object_key(text: str) -> str:
    return _coerce_text(text).strip().lower()


def _sanitize_entity_path(text: str) -> str:
    sanitized = re.sub(r"\s+", "_", _coerce_text(text).strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.\\-]", "_", sanitized)
    return sanitized.strip("._") or "entry"


def _build_grouping_key(text: str) -> tuple[str, str]:
    text = _coerce_text(text).strip()
    object_key = _extract_object_key(text)
    return object_key, f"{object_key}::{text}"


def _set_frame_time(frame_idx: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_idx)
    else:
        rr.set_time("frame", sequence=frame_idx)


def _apply_offset(points, offset_xyz):
    if torch.is_tensor(points):
        offset = torch.tensor(offset_xyz, dtype=points.dtype, device=points.device)
        return points + offset
    return np.asarray(points) + np.asarray(offset_xyz, dtype=np.float32)


def _vertex_colors_like(points, rgb_color):
    num_points = int(
        points.shape[0] if torch.is_tensor(points) else np.asarray(points).shape[0]
    )
    return np.tile(np.asarray(rgb_color, dtype=np.uint8), (num_points, 1))


def _get_batch_cov_map(cov_map, batch_idx: int, batch_size: int):
    if cov_map is None:
        return None
    if isinstance(cov_map, (list, tuple)):
        if len(cov_map) == batch_size:
            return cov_map[batch_idx]
        return cov_map
    if torch.is_tensor(cov_map) or isinstance(cov_map, np.ndarray):
        if batch_size > 1 and cov_map.ndim >= 2 and cov_map.shape[0] == batch_size:
            return cov_map[batch_idx]
        return cov_map
    return cov_map


def _get_batch_gaze_value(data, batch_idx: int, batch_size: int):
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        if len(data) == batch_size:
            return data[batch_idx]
        return data
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        if data.ndim >= 1 and data.shape[0] == batch_size:
            return data[batch_idx]
        return data
    return data


def _get_batch_cam_pose(cam_pose, batch_idx: int, batch_size: int):
    if cam_pose is None:
        return None
    if isinstance(cam_pose, (list, tuple)):
        if len(cam_pose) == batch_size:
            return cam_pose[batch_idx]
        return cam_pose
    if torch.is_tensor(cam_pose) or isinstance(cam_pose, np.ndarray):
        if cam_pose.ndim >= 3 and cam_pose.shape[0] == batch_size:
            return cam_pose[batch_idx]
        return cam_pose
    return cam_pose


def _framewise_point_mask_from_cov_map(
    cov_map, num_points: int
) -> Optional[np.ndarray]:
    if cov_map is None or num_points <= 0:
        return None
    arr = _to_numpy(cov_map)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.full((1, num_points), bool(arr > 0), dtype=bool)
    if arr.ndim == 1 and arr.shape[0] == num_points:
        return (arr > 0).reshape(1, num_points)
    if arr.ndim == 2 and arr.shape[1] == num_points:
        return (arr > 0).astype(bool)

    point_axes = [ax for ax, size in enumerate(arr.shape) if size == num_points]
    if not point_axes:
        return None

    arr = np.moveaxis(arr, point_axes[0], -1)
    if arr.ndim == 1 and arr.shape[0] == num_points:
        return (arr > 0).reshape(1, num_points)

    non_point_shape = arr.shape[:-1]
    if not non_point_shape:
        return (arr > 0).reshape(1, num_points)

    frame_axis = int(np.argmax(non_point_shape))
    arr = np.moveaxis(arr, frame_axis, 0)
    if arr.ndim > 2:
        reduce_axes = tuple(range(1, arr.ndim - 1))
        arr = (arr > 0).any(axis=reduce_axes)
    else:
        arr = arr > 0
    if arr.ndim != 2 or arr.shape[1] != num_points:
        return None
    return arr.astype(bool)


def _framewise_gaze_rays(gaze) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if gaze is None:
        return None, None
    arr = _to_numpy(gaze)
    if arr.size == 0:
        return None, None
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape == (2, 3):
        arr = arr.reshape(1, 2, 3)
    elif arr.ndim == 3 and arr.shape[-2:] == (2, 3):
        pass
    else:
        return None, None

    origins = arr[:, 0, :]
    directions = arr[:, 1, :]
    if origins.ndim != 2 or directions.ndim != 2:
        return None, None
    return origins, directions


def _transform_gaze_to_world(
    gaze,
    cam_pose,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    origins, directions = _framewise_gaze_rays(gaze)
    if origins is None or directions is None:
        return None, None
    if cam_pose is None:
        return origins, directions

    pose = np.asarray(_to_numpy(cam_pose), dtype=np.float32)
    pose = np.squeeze(pose)
    if pose.ndim == 2 and pose.shape == (4, 4):
        pose = pose.reshape(1, 4, 4)
    elif pose.ndim != 3 or pose.shape[-2:] != (4, 4):
        return origins, directions

    frame_count = min(origins.shape[0], directions.shape[0], pose.shape[0])
    if frame_count <= 0:
        return None, None

    origins = origins[:frame_count]
    directions = directions[:frame_count]
    pose = pose[:frame_count]

    origins_h = np.concatenate(
        [origins, np.ones((frame_count, 1), dtype=np.float32)],
        axis=1,
    )
    world_origins = np.einsum("fij,fj->fi", pose, origins_h)[:, :3]
    world_directions = np.einsum("fij,fj->fi", pose[:, :3, :3], directions)
    return world_origins, world_directions


def _gaze_proximity_mask(
    object_points,
    gaze_origins: Optional[np.ndarray],
    gaze_directions: Optional[np.ndarray],
    threshold_m: float = 0.02,
) -> Optional[np.ndarray]:
    if gaze_origins is None or gaze_directions is None:
        return None
    points = np.asarray(_to_numpy(object_points), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return None
    if gaze_origins.ndim != 2 or gaze_directions.ndim != 2:
        return None

    norms = np.linalg.norm(gaze_directions, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-8)
    if not np.any(valid):
        return None
    origins = gaze_origins[valid]
    directions = gaze_directions[valid] / norms[valid, None]

    point_delta = points[None, :, :] - origins[:, None, :]
    proj = np.einsum("gpc,gc->gp", point_delta, directions)
    ray_mask = proj >= 0.0
    rejection = point_delta - proj[..., None] * directions[:, None, :]
    distances = np.linalg.norm(rejection, axis=-1)
    distances = np.where(ray_mask, distances, np.inf)
    min_distances = np.min(distances, axis=0)
    return np.isfinite(min_distances) & (min_distances <= float(threshold_m))


def _final_nonempty_point_mask(point_map, num_points: int) -> Optional[np.ndarray]:
    masks = _framewise_point_mask_from_cov_map(point_map, num_points)
    if masks is None or masks.shape[0] == 0:
        return None
    for idx in range(masks.shape[0] - 1, -1, -1):
        mask = np.asarray(masks[idx], dtype=bool)
        if mask.shape[0] == num_points and np.any(mask):
            return mask
    return None


def _cumulative_point_mask_upto_frame(
    point_map,
    num_points: int,
    frame_idx: int,
    total_frames: int,
) -> Optional[np.ndarray]:
    masks = _framewise_point_mask_from_cov_map(point_map, num_points)
    if masks is None or masks.shape[0] == 0 or total_frames <= 0:
        return None
    if total_frames == 1:
        gaze_frame_idx = masks.shape[0] - 1
    else:
        progress = float(frame_idx + 1) / float(total_frames)
        gaze_frame_idx = int(np.ceil(progress * masks.shape[0])) - 1
    gaze_frame_idx = max(0, min(gaze_frame_idx, masks.shape[0] - 1))
    cumulative = np.any(masks[: gaze_frame_idx + 1], axis=0)
    cumulative = np.asarray(cumulative, dtype=bool)
    if cumulative.shape[0] != num_points or not np.any(cumulative):
        return None
    return cumulative


def _synced_gaze_from_gaze_map(
    gaze,
    gaze_map,
    obj_vertices_world,
    max_origin_frames: int = 30,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    obj_np = np.asarray(_to_numpy(obj_vertices_world), dtype=np.float32)
    if obj_np.ndim != 3 or obj_np.shape[1] <= 0:
        return None, None, None

    target_mask = _final_nonempty_point_mask(gaze_map, int(obj_np.shape[1]))
    if target_mask is None:
        return None, None, None

    gaze_origins, _gaze_dirs = _framewise_gaze_rays(gaze)
    if gaze_origins is None:
        return None, None, target_mask

    valid_origins = gaze_origins[np.isfinite(gaze_origins).all(axis=1)]
    if valid_origins.shape[0] == 0:
        return None, None, target_mask

    origin_frames = min(max_origin_frames, valid_origins.shape[0])
    target_frames = min(max_origin_frames, obj_np.shape[0])
    gaze_origin = np.mean(valid_origins[:origin_frames], axis=0).astype(np.float32)
    target_point = np.mean(obj_np[:target_frames, target_mask], axis=(0, 1)).astype(
        np.float32
    )
    gaze_vector = target_point - gaze_origin
    if not np.isfinite(gaze_origin).all() or not np.isfinite(gaze_vector).all():
        return None, None, target_mask
    if np.linalg.norm(gaze_vector) <= 1e-8:
        return None, None, target_mask
    return gaze_origin, gaze_vector, target_mask


def _sdf_penetration_metrics(
    obj_mesh: trimesh.Trimesh,
    object_key: Optional[str],
    obj_pose_params,
    hand_points_world: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    if hand_points_world.shape[0] == 0:
        return (
            np.zeros((hand_points_world.shape[0],), dtype=bool),
            np.zeros((hand_points_world.shape[0],), dtype=np.float32),
            None,
            np.zeros((hand_points_world.shape[0],), dtype=bool),
            np.zeros((hand_points_world.shape[0],), dtype=bool),
        )
    try:
        hand_points_world = np.asarray(hand_points_world, dtype=np.float64)
        hand_valid = np.all(np.isfinite(hand_points_world), axis=1)
        if hand_valid.sum() == 0:
            return (
                np.zeros((hand_points_world.shape[0],), dtype=bool),
                np.zeros((hand_points_world.shape[0],), dtype=np.float32),
                None,
                np.zeros((hand_points_world.shape[0],), dtype=bool),
                np.zeros((hand_points_world.shape[0],), dtype=bool),
            )

        obj_pose = _pose9_sequence(obj_pose_params)
        last_pose = _to_numpy(obj_pose[-1]).astype(np.float64)
        trans = last_pose[:3]
        rot = (
            _to_numpy(rot6d_to_rotmat(_to_tensor(last_pose[3:9]).reshape(1, 6)))
            .reshape(3, 3)
            .astype(np.float64)
        )
        hand_local = np.einsum("ni,ij->nj", hand_points_world[hand_valid] - trans, rot)
        closest_points_local_valid = None
        solid_valid = np.zeros((hand_local.shape[0],), dtype=bool)
        cavity_valid = np.zeros((hand_local.shape[0],), dtype=bool)
        if has_container_spec(object_key):
            cavity_result = cavity_penetration_metrics(obj_mesh, object_key, hand_local)
            if cavity_result is not None:
                cavity_valid, depth_valid = cavity_result
                inside_valid = cavity_valid
                region = build_container_region(obj_mesh, object_key)
                if region is not None and np.any(cavity_valid):
                    closest_points_local_valid = np.full(
                        (hand_local.shape[0], 3), np.nan, dtype=np.float64
                    )
                    closest_local_inner, _, _ = trimesh.proximity.closest_point(
                        region["inner_mesh"], hand_local[cavity_valid]
                    )
                    closest_points_local_valid[cavity_valid] = closest_local_inner
                solid_valid = np.zeros_like(cavity_valid)
            else:
                signed_distance = np.asarray(
                    trimesh.proximity.signed_distance(obj_mesh, hand_local),
                    dtype=np.float64,
                )
                solid_valid = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
                inside_valid = solid_valid
                depth_valid = np.zeros_like(signed_distance, dtype=np.float32)
                depth_valid[inside_valid] = np.abs(
                    signed_distance[inside_valid]
                ).astype(np.float32)
                cavity_valid = np.zeros_like(inside_valid)
                if np.any(inside_valid):
                    closest_local, _, _ = trimesh.proximity.closest_point(
                        obj_mesh, hand_local[inside_valid]
                    )
                    closest_points_local_valid = np.full(
                        (hand_local.shape[0], 3), np.nan, dtype=np.float64
                    )
                    closest_points_local_valid[inside_valid] = closest_local
        else:
            signed_distance = np.asarray(
                trimesh.proximity.signed_distance(obj_mesh, hand_local),
                dtype=np.float64,
            )
            solid_valid = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
            cavity_valid = np.zeros_like(solid_valid)
            inside_valid = solid_valid
            depth_valid = np.zeros_like(signed_distance, dtype=np.float32)
            depth_valid[inside_valid] = np.abs(signed_distance[inside_valid]).astype(
                np.float32
            )
            if np.any(inside_valid):
                closest_local, _, _ = trimesh.proximity.closest_point(
                    obj_mesh, hand_local[inside_valid]
                )
                closest_points_local_valid = np.full(
                    (hand_local.shape[0], 3), np.nan, dtype=np.float64
                )
                closest_points_local_valid[inside_valid] = closest_local

        inside = np.zeros((hand_points_world.shape[0],), dtype=bool)
        depth = np.zeros((hand_points_world.shape[0],), dtype=np.float32)
        solid = np.zeros((hand_points_world.shape[0],), dtype=bool)
        cavity = np.zeros((hand_points_world.shape[0],), dtype=bool)
        valid_idx = np.flatnonzero(hand_valid)
        inside[valid_idx] = inside_valid
        depth[valid_idx] = depth_valid
        solid[valid_idx] = solid_valid
        cavity[valid_idx] = cavity_valid
        closest_points_world = None
        if closest_points_local_valid is not None:
            closest_points_world = np.full(
                (hand_points_world.shape[0], 3), np.nan, dtype=np.float32
            )
            local_valid_idx = np.flatnonzero(hand_valid)
            closest_local_subset = closest_points_local_valid
            valid_rows = np.all(np.isfinite(closest_local_subset), axis=1)
            if np.any(valid_rows):
                closest_world_subset = (
                    np.einsum("ni,ji->nj", closest_local_subset[valid_rows], rot)
                    + trans[None, :]
                )
                closest_points_world[local_valid_idx[valid_rows]] = (
                    closest_world_subset.astype(np.float32)
                )
        return inside, depth, closest_points_world, solid, cavity
    except Exception:
        return (
            np.zeros((hand_points_world.shape[0],), dtype=bool),
            np.zeros((hand_points_world.shape[0],), dtype=np.float32),
            None,
            np.zeros((hand_points_world.shape[0],), dtype=bool),
            np.zeros((hand_points_world.shape[0],), dtype=bool),
        )


def _sample_metrics(
    obj_seq: np.ndarray,
    object_key: Optional[str],
    obj_pose_params,
    obj_mesh: Optional[trimesh.Trimesh],
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    l_faces: np.ndarray,
    r_faces: np.ndarray,
    use_left: bool,
    use_right: bool,
    contact_threshold: float = CONTACT_THRESHOLD_M,
):
    if obj_seq.shape[0] == 0:
        return None

    obj_last = obj_seq[-1]
    hand_last_parts = []
    hand_volume_parts = []
    hand_part_specs = []

    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        l_last = l_seq[-1]
        hand_last_parts.append(l_last)
        hand_volume_parts.append((l_last, l_faces))
        hand_part_specs.append(("left", int(l_last.shape[0])))
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        r_last = r_seq[-1]
        hand_last_parts.append(r_last)
        hand_volume_parts.append((r_last, r_faces))
        hand_part_specs.append(("right", int(r_last.shape[0])))
    if not hand_last_parts:
        return None

    hand_last = np.concatenate(hand_last_parts, axis=0)
    dists = np.linalg.norm(hand_last[:, None, :] - obj_last[None, :, :], axis=2).min(
        axis=1
    )
    contact_mask = dists < contact_threshold
    cr = float(contact_mask.mean())

    if obj_mesh is not None:
        (
            inside_mask,
            inside_depth_m,
            closest_surface_world,
            solid_mask,
            cavity_mask,
        ) = _sdf_penetration_metrics(obj_mesh, object_key, obj_pose_params, hand_last)
    else:
        inside_mask = np.zeros((hand_last.shape[0],), dtype=bool)
        inside_depth_m = np.zeros((hand_last.shape[0],), dtype=np.float32)
        closest_surface_world = None
        solid_mask = np.zeros((hand_last.shape[0],), dtype=bool)
        cavity_mask = np.zeros((hand_last.shape[0],), dtype=bool)
    id_mm = (
        float(inside_depth_m[inside_mask].mean() * 1000.0) if inside_mask.any() else 0.0
    )
    id_max_mm = (
        float(inside_depth_m[inside_mask].max() * 1000.0) if inside_mask.any() else 0.0
    )
    id_max_hand_point = None
    id_max_object_point = None
    if inside_mask.any():
        inside_indices = np.flatnonzero(inside_mask)
        max_local_idx = int(np.argmax(inside_depth_m[inside_indices]))
        max_idx = int(inside_indices[max_local_idx])
        if (
            closest_surface_world is not None
            and max_idx < closest_surface_world.shape[0]
            and np.all(np.isfinite(closest_surface_world[max_idx]))
        ):
            id_max_hand_point = hand_last[max_idx].astype(np.float32)
            id_max_object_point = closest_surface_world[max_idx].astype(np.float32)
    iv_m3 = 0.0
    cursor = 0
    for verts, faces in hand_volume_parts:
        n = verts.shape[0]
        local_inside = inside_mask[cursor : cursor + n]
        cursor += n
        mesh_vol = _safe_mesh_volume(verts, faces)
        if mesh_vol <= 0.0 or n <= 0:
            continue
        iv_m3 += (mesh_vol / float(n)) * float(local_inside.sum())

    return {
        "vr_percent": 100.0 if cr >= MIN_CR_FOR_VALID else 0.0,
        "vr_pass_count": 1 if cr >= MIN_CR_FOR_VALID else 0,
        "vr_total_count": 1,
        "cr_percent": 100.0 * cr,
        "iv_cm3": float(iv_m3 * 1e6),
        "id_mm": id_mm,
        "id_max_mm": id_max_mm,
        "inside_mask": inside_mask,
        "solid_mask": solid_mask,
        "cavity_mask": cavity_mask,
        "hand_part_specs": hand_part_specs,
        "id_max_hand_point": id_max_hand_point,
        "id_max_object_point": id_max_object_point,
        "id_vertex_count": int(np.count_nonzero(inside_mask)),
    }


def _metrics_label(
    variant_name: str,
    metrics: dict,
    gt_metrics: Optional[dict],
    meta: Optional[dict] = None,
) -> str:
    lines = [
        f"{variant_name.upper()}",
        f"VR (%) {metrics['vr_percent']:.0f} ({metrics['vr_pass_count']}/{metrics['vr_total_count']})",
        f"IV (cm^3) {metrics['iv_cm3']:.2f}",
        f"ID (mm) {metrics['id_mm']:.2f}",
        f"ID_max (mm) {metrics['id_max_mm']:.2f}",
    ]
    return "\n".join(lines)


def _meta_line_points(
    meta: Optional[dict],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[float]]:
    if not meta:
        return None, None, None, None
    metric_name = None
    metric_value = None
    hand_point = meta.get("id_max_hand_point")
    object_point = meta.get("id_max_object_point")
    if hand_point is not None and object_point is not None:
        metric_name = "id_max"
        if "id_max_mm" in meta:
            metric_value = float(meta["id_max_mm"])
    else:
        hand_point = meta.get("pen_max_hand_point")
        object_point = meta.get("pen_max_object_point")
        if hand_point is not None and object_point is not None:
            metric_name = "pen_max"
            if "pen_max_mm" in meta:
                metric_value = float(meta["pen_max_mm"])
    if hand_point is None or object_point is None:
        return None, None, metric_name, metric_value
    try:
        hand_point_np = np.asarray(hand_point, dtype=np.float32).reshape(3)
        object_point_np = np.asarray(object_point, dtype=np.float32).reshape(3)
    except Exception:
        return None, None, metric_name, metric_value
    if not np.all(np.isfinite(hand_point_np)) or not np.all(
        np.isfinite(object_point_np)
    ):
        return None, None, metric_name, metric_value
    return hand_point_np, object_point_np, metric_name, metric_value


def _load_items_from_recording(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        raw = np.load(path, allow_pickle=True)
        payload = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
        if not isinstance(payload, dict):
            raise ValueError(f"npy payload must be a dict: {path}")
        if "save_list" in payload:
            save_list = payload["save_list"]
            return (
                save_list.tolist() if isinstance(save_list, np.ndarray) else save_list
            )

        motion = np.asarray(payload["motion"])
        motion = np.squeeze(motion)
        if motion.ndim == 2:
            motion = motion[None, ...]
        if motion.ndim < 3:
            raise ValueError(f"unsupported motion shape: {tuple(motion.shape)}")

        sample_axis = 0
        if "num_samples" in payload:
            try:
                hint = int(np.asarray(payload["num_samples"]).item())
                matches = [
                    ax for ax, size in enumerate(motion.shape) if int(size) == hint
                ]
                if matches:
                    sample_axis = matches[0]
            except Exception:
                pass
        motion = np.moveaxis(motion, sample_axis, 0)

        candidate_axes = [
            ax for ax, size in enumerate(motion.shape[1:], start=1) if size >= 207
        ]
        if not candidate_axes:
            raise ValueError(f"motion has no feature axis: {tuple(motion.shape)}")
        feat_axis = next(
            (ax for ax in candidate_axes if motion.shape[ax] == 207), candidate_axes[0]
        )
        motion = np.moveaxis(motion, feat_axis, -1).reshape(
            motion.shape[0], -1, motion.shape[feat_axis]
        )

        texts = payload.get("text")
        lengths = payload.get("lengths")
        rebuilt = []
        for i, seq in enumerate(motion):
            seq_len = seq.shape[0]
            if lengths is not None:
                try:
                    seq_len = int(np.asarray(lengths).reshape(-1)[i])
                except Exception:
                    pass
            seq = seq[: max(1, min(seq_len, seq.shape[0]))]
            rebuilt.append(
                [
                    [seq[:, 198:207]],
                    [seq[:, :99]],
                    [seq[:, 99:198]],
                    (
                        _coerce_text(np.asarray(texts, dtype=object)[i])
                        if texts is not None
                        else ""
                    ),
                    None,
                    None,
                    None,
                    None,
                ]
            )
        return rebuilt

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "save_list" in data.files:
            save_list = data["save_list"]
            return (
                save_list.tolist() if isinstance(save_list, np.ndarray) else save_list
            )
        if not all(key in data.files for key in ("x_lhand", "x_rhand", "text")):
            raise ValueError(f"npz missing expected keys: {path}")
        x_obj_arr = data["x_obj"] if "x_obj" in data.files else data["obj_points"]
        rebuilt = []
        for i in range(len(data["x_lhand"])):
            rebuilt.append(
                [
                    [x_obj_arr[i]],
                    [data["x_lhand"][i]],
                    [data["x_rhand"][i]],
                    data["text"][i],
                    None,
                    None,
                    None,
                    None,
                ]
            )
        return rebuilt

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        if "save_list" in payload:
            save_list = payload["save_list"]
            return (
                save_list.tolist() if isinstance(save_list, np.ndarray) else save_list
            )
        if any(
            key in payload
            for key in (
                "variants",
                "x_obj",
                "obj_params",
                "x_lhand",
                "lhand_params",
                "x_rhand",
                "rhand_params",
                "text",
                "texts",
            )
        ):
            return [payload]
        return list(payload.values())
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    return payload


def _object_vertices_from_source(x_obj_source, object_key: str, obj_pc_source=None):
    x_obj_arr = _to_numpy(x_obj_source)
    canonical_obj_pc = obj_pc_source
    if canonical_obj_pc is None:
        if object_key in BPS_PC:
            canonical_obj_pc = torch.as_tensor(BPS_PC[object_key], dtype=torch.float32)
        else:
            canonical_obj_pc = OBJ_PC[object_key]
    else:
        canonical_obj_pc = torch.as_tensor(
            _to_numpy(canonical_obj_pc), dtype=torch.float32
        )
    if x_obj_arr.ndim in (1, 2) and x_obj_arr.shape[-1] >= 9:
        return process_obj_result(canonical_obj_pc, _pose9_sequence(x_obj_source))
    if x_obj_arr.ndim == 2 and x_obj_arr.shape[1] == 3:
        return torch.as_tensor(x_obj_arr[None, ...], dtype=torch.float32)
    if x_obj_arr.ndim == 3 and x_obj_arr.shape[-1] == 3:
        return torch.as_tensor(x_obj_arr, dtype=torch.float32)
    raise ValueError(f"unsupported x_obj shape: {tuple(x_obj_arr.shape)}")


def _object_vertices_from_meta_or_source(
    x_obj_source,
    object_key: str,
    object_meta: Optional[dict],
):
    if isinstance(object_meta, dict):
        for key in (
            "transf_obj_pc",
            "transformed_obj_pc",
            "obj_pc_world",
            "obj_vertices_world",
        ):
            value = object_meta.get(key)
            if value is None:
                continue
            value_np = _to_numpy(value)
            if value_np.ndim == 2 and value_np.shape[1] == 3:
                return torch.as_tensor(value_np[None, ...], dtype=torch.float32)
            if value_np.ndim == 3 and value_np.shape[-1] == 3:
                return torch.as_tensor(value_np, dtype=torch.float32)
    if isinstance(object_meta, dict):
        obj_pc_source = object_meta.get("obj_pc_org")
        if obj_pc_source is not None:
            return _object_vertices_from_source(
                x_obj_source, object_key, obj_pc_source=obj_pc_source
            )
        if object_key in OBJ_PC:
            return _object_vertices_from_source(
                x_obj_source, object_key, obj_pc_source=None
            )
        raise ValueError("object_meta is missing obj_pc_org")
    return _object_vertices_from_source(x_obj_source, object_key, obj_pc_source=None)


def _object_meta_has_world_vertices(object_meta: Optional[dict]) -> bool:
    if not isinstance(object_meta, dict):
        return False
    for key in (
        "transf_obj_pc",
        "transformed_obj_pc",
        "obj_pc_world",
        "obj_vertices_world",
    ):
        value = object_meta.get(key)
        if value is None:
            continue
        value_np = _to_numpy(value)
        if (value_np.ndim == 2 and value_np.shape[1] == 3) or (
            value_np.ndim == 3 and value_np.shape[-1] == 3
        ):
            return True
    return False


def _validate_pen_max_object_alignment(
    eval_meta: Optional[dict],
    obj_vertices,
    recording_name: str,
    text_entry: str,
    batch_idx: int,
) -> None:
    if not eval_meta:
        return
    pen_max_info = eval_meta.get("id_info")
    if not isinstance(pen_max_info, dict):
        pen_max_info = eval_meta.get("pen_max_info")
    if not isinstance(pen_max_info, dict):
        return
    line = pen_max_info.get("line")
    if not isinstance(line, dict):
        return
    object_point = line.get("object_point")
    object_vertex_idx = pen_max_info.get("object_vertex_idx")
    if object_point is None or object_vertex_idx is None:
        return
    try:
        object_point_np = np.asarray(object_point, dtype=np.float32).reshape(3)
        object_vertex_idx = int(object_vertex_idx)
    except Exception:
        return
    obj_vertices_np = _to_numpy(obj_vertices)
    if (
        obj_vertices_np.ndim != 2
        or obj_vertices_np.shape[1] != 3
        or object_vertex_idx < 0
        or object_vertex_idx >= obj_vertices_np.shape[0]
    ):
        return
    transformed_point = np.asarray(
        obj_vertices_np[object_vertex_idx], dtype=np.float32
    ).reshape(3)
    if not np.all(np.isfinite(object_point_np)) or not np.all(
        np.isfinite(transformed_point)
    ):
        return
    diff_m = float(np.linalg.norm(object_point_np - transformed_point))
    if diff_m > 1e-4:
        print(
            f"[WARN] eval object mismatch in {os.path.basename(recording_name)} "
            f"batch={batch_idx} text='{text_entry}' idx={object_vertex_idx} "
            f"diff_mm={diff_m * 1000.0:.3f}"
        )


def _validate_pen_max_hand_alignment(
    eval_meta: Optional[dict],
    l_hand_vertices,
    r_hand_vertices,
    recording_name: str,
    text_entry: str,
    batch_idx: int,
) -> None:
    if not eval_meta:
        return
    pen_max_info = eval_meta.get("id_info")
    if not isinstance(pen_max_info, dict):
        pen_max_info = eval_meta.get("pen_max_info")
    if not isinstance(pen_max_info, dict):
        return
    line = pen_max_info.get("line")
    if not isinstance(line, dict):
        return
    hand_point = line.get("hand_vertex_point")
    if hand_point is None:
        hand_point = line.get("hand_joint_point")
    hand_joint_idx = pen_max_info.get("hand_vertex_idx")
    if hand_joint_idx is None:
        hand_joint_idx = pen_max_info.get("hand_joint_idx")
    hand_side = str(pen_max_info.get("hand_side", "")).strip().lower()
    if hand_point is None or hand_joint_idx is None:
        return
    try:
        hand_point_np = np.asarray(hand_point, dtype=np.float32).reshape(3)
        hand_joint_idx = int(hand_joint_idx)
    except Exception:
        return
    if hand_side in {"left", "l", "l_hand"}:
        joints_np = _to_numpy(l_hand_vertices)
    elif hand_side in {"right", "r", "r_hand"}:
        joints_np = _to_numpy(r_hand_vertices)
    else:
        return
    if (
        joints_np.ndim != 2
        or joints_np.shape[1] != 3
        or hand_joint_idx < 0
        or hand_joint_idx >= joints_np.shape[0]
    ):
        return
    rendered_point = np.asarray(joints_np[hand_joint_idx], dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(hand_point_np)) or not np.all(
        np.isfinite(rendered_point)
    ):
        return
    diff_m = float(np.linalg.norm(hand_point_np - rendered_point))
    if diff_m > 1e-4:
        print(
            f"[WARN] eval hand mismatch in {os.path.basename(recording_name)} "
            f"batch={batch_idx} text='{text_entry}' side={hand_side} vertex={hand_joint_idx} "
            f"diff_mm={diff_m * 1000.0:.3f}"
        )


def _looks_like_text_field(value) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        if not value:
            return True
        return all(isinstance(item, str) for item in value)
    return False


def _looks_like_eval_meta_list(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and (
        "contact" in first
        or "penetration_ok" in first
        or "success" in first
        or "ID" in first
        or "iv_cm3" in first
        or "pen_max_mm" in first
        or "id_info" in first
        or "iv_info" in first
        or "pen_vertex_info" in first
    )


def _looks_like_object_meta_list(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and ("object_name" in first or "obj_pc_org" in first)


def _looks_like_cam_pose(value) -> bool:
    try:
        arr = np.asarray(_to_numpy(value))
    except Exception:
        return False
    return arr.ndim >= 2 and arr.shape[-2:] == (4, 4)


def _sequence_frame_count(value) -> int:
    if value is None:
        return 0
    try:
        arr = np.asarray(_to_numpy(value))
    except Exception:
        return 0
    if arr.ndim == 0:
        return 0
    if arr.ndim == 1:
        return 1 if arr.shape[0] > 0 else 0
    return int(arr.shape[0])


def _zeros_hand_params(nframes: int):
    return np.zeros((max(0, int(nframes)), 99), dtype=np.float32)


def _zeros_hand_vertices(nframes: int):
    return np.zeros((max(0, int(nframes)), 778, 3), dtype=np.float32)


def _zeros_hand_joints(nframes: int):
    return np.zeros((max(0, int(nframes)), 21, 3), dtype=np.float32)


def _resolve_hand_world_geometry(
    stored_vertices_source,
    stored_joints_source,
    hand_params_source,
    hand_layer,
):
    if stored_vertices_source is not None and stored_joints_source is not None:
        vertices = torch.as_tensor(
            _to_numpy(stored_vertices_source), dtype=torch.float32
        )
        joints = torch.as_tensor(_to_numpy(stored_joints_source), dtype=torch.float32)
        return vertices, joints

    if hand_params_source is None:
        return None, None

    hand_params = torch.as_tensor(_to_numpy(hand_params_source), dtype=torch.float32)
    if hand_params.ndim != 2 or hand_params.shape[-1] != 99:
        return None, None
    if hand_params.shape[0] == 0:
        empty_vertices = torch.zeros((0, 778, 3), dtype=torch.float32)
        empty_joints = torch.zeros((0, 21, 3), dtype=torch.float32)
        return empty_vertices, empty_joints

    vertices, joints, _ = process_hand_result(hand_layer, hand_params)
    return (
        torch.as_tensor(_to_numpy(vertices), dtype=torch.float32),
        torch.as_tensor(_to_numpy(joints), dtype=torch.float32),
    )


def _normalize_debug_variant_dict(
    variant,
    default_name: str,
    default_hand_color,
    default_object_color,
):
    hand_variant = variant.get("hand_variant")
    if not isinstance(hand_variant, dict):
        return None
    x_obj = _dict_get_first(hand_variant, ("x_obj",))
    hand_params = _dict_get_first(hand_variant, ("hand_params",))
    hand_vertices = _dict_get_first(hand_variant, ("hand_vertices",))
    hand_joints = _dict_get_first(hand_variant, ("hand_joints",))
    obj_vertices = _dict_get_first(hand_variant, ("obj_vertices",))
    if (
        x_obj is None
        or hand_params is None
        or hand_vertices is None
        or hand_joints is None
    ):
        return None

    active_hand = (
        _coerce_text(
            variant.get("active_hand")
            or hand_variant.get("active_hand")
            or variant.get("hand_side")
            or variant.get("hand")
        )
        .strip()
        .lower()
    )
    if active_hand not in {"left", "right"}:
        active_hand = "right"

    nframes = max(
        _sequence_frame_count(x_obj),
        _sequence_frame_count(hand_params),
        _sequence_frame_count(hand_vertices),
        _sequence_frame_count(hand_joints),
        _sequence_frame_count(obj_vertices),
    )
    if nframes <= 0:
        return None

    x_lhand = _zeros_hand_params(nframes)
    x_rhand = _zeros_hand_params(nframes)
    lhand_vertices_world = _zeros_hand_vertices(nframes)
    rhand_vertices_world = _zeros_hand_vertices(nframes)
    lhand_joints_world = _zeros_hand_joints(nframes)
    rhand_joints_world = _zeros_hand_joints(nframes)

    if active_hand == "left":
        x_lhand = hand_params
        lhand_vertices_world = hand_vertices
        lhand_joints_world = hand_joints
    else:
        x_rhand = hand_params
        rhand_vertices_world = hand_vertices
        rhand_joints_world = hand_joints

    return {
        "name": _coerce_text(
            variant.get("display_name") or variant.get("name") or default_name
        ),
        "x_obj": x_obj,
        "x_lhand": x_lhand,
        "x_rhand": x_rhand,
        "lhand_vertices_world": lhand_vertices_world,
        "rhand_vertices_world": rhand_vertices_world,
        "lhand_joints_world": lhand_joints_world,
        "rhand_joints_world": rhand_joints_world,
        "target_sampled_vertices": variant.get("target_sampled_vertices"),
        "pred_sampled_vertices": variant.get("pred_sampled_vertices"),
        "active_hand": active_hand,
        "object_vertices_world": obj_vertices,
        "hand_color": tuple(variant.get("hand_color", default_hand_color)),
        "object_color": tuple(variant.get("object_color", default_object_color)),
    }


def _normalize_variant_dict(
    variant,
    default_name: str = "pred",
    default_hand_color=(235, 87, 87),
    default_object_color=(64, 176, 166),
):
    if not isinstance(variant, dict):
        return None
    if "hand_variant" in variant:
        normalized = _normalize_debug_variant_dict(
            variant,
            default_name=default_name,
            default_hand_color=default_hand_color,
            default_object_color=default_object_color,
        )
        if normalized is not None:
            return normalized
    x_obj = _dict_get_first(
        variant,
        ("x_obj", "obj_params", "coarse_x_obj", "pred_x_obj", "object", "obj"),
    )
    x_lhand = _dict_get_first(
        variant,
        ("x_lhand", "lhand_params", "coarse_x_lhand", "pred_x_lhand", "lhand"),
    )
    x_rhand = _dict_get_first(
        variant,
        ("x_rhand", "rhand_params", "coarse_x_rhand", "pred_x_rhand", "rhand"),
    )
    if x_obj is None or x_lhand is None or x_rhand is None:
        return None
    return {
        "name": _coerce_text(variant.get("name") or default_name),
        "x_obj": x_obj,
        "x_lhand": x_lhand,
        "x_rhand": x_rhand,
        "lhand_vertices_world": _dict_get_first(
            variant, ("lhand_vertices_world", "left_hand_vertices_world")
        ),
        "rhand_vertices_world": _dict_get_first(
            variant, ("rhand_vertices_world", "right_hand_vertices_world")
        ),
        "lhand_joints_world": _dict_get_first(
            variant, ("lhand_joints_world", "left_hand_joints_world")
        ),
        "rhand_joints_world": _dict_get_first(
            variant, ("rhand_joints_world", "right_hand_joints_world")
        ),
        "target_sampled_vertices": _dict_get_first(
            variant, ("target_sampled_vertices", "sampled_vertices_target")
        ),
        "pred_sampled_vertices": _dict_get_first(
            variant, ("pred_sampled_vertices", "sampled_vertices_pred")
        ),
        "hand_color": tuple(variant.get("hand_color", default_hand_color)),
        "object_color": tuple(variant.get("object_color", default_object_color)),
    }


def _parse_dict_entry(entry: dict):
    text = _dict_get_first(entry, ("text", "texts", "caption", "captions", "prompt"))
    meta_list = _dict_get_first(
        entry, ("meta_list", "eval_meta_list", "eval_meta", "eval", "metrics")
    )
    object_meta_list = _dict_get_first(
        entry, ("object_meta_list", "object_meta", "obj_meta", "object_infos")
    )
    contact_list = _dict_get_first(entry, ("contact_list", "contacts"))
    pen_max_list = _dict_get_first(entry, ("pen_max_list", "pen_max", "pen_values"))
    cov_map = _dict_get_first(
        entry,
        ("cov_map", "gt_cov_map", "contact_map", "object_contact_map"),
    )
    gaze_map = _dict_get_first(entry, ("gaze_map", "eye_gaze_map"))
    gaze = _dict_get_first(entry, ("gaze", "eye_gaze"))
    cam_pose = _dict_get_first(entry, ("cam_pose", "camera_pose", "cam_poses"))
    batch_hint = _batch_hint_size(
        text, meta_list, object_meta_list, contact_list, pen_max_list, cov_map
    )
    variants = []

    raw_variants = entry.get("variants")
    if isinstance(raw_variants, (list, tuple)):
        for raw_variant in raw_variants:
            normalized = _normalize_variant_dict(raw_variant)
            if normalized is not None:
                if _coerce_text(normalized.get("name", "")).strip().lower() in {
                    "gt",
                    "gt_debug",
                }:
                    continue
                variants.append(normalized)

    if not variants:
        pred_variant = _normalize_variant_dict(entry, default_name="pred")
        if pred_variant is not None:
            if batch_hint <= 1:
                pred_variant["x_obj"] = _wrap_single_sample(pred_variant["x_obj"])
                pred_variant["x_lhand"] = _wrap_single_sample(pred_variant["x_lhand"])
                pred_variant["x_rhand"] = _wrap_single_sample(pred_variant["x_rhand"])
            variants.append(pred_variant)

    if not variants:
        coarse_variant = _normalize_variant_dict(
            {
                "name": "coarse",
                "x_obj": _dict_get_first(entry, ("coarse_x_obj",)),
                "x_lhand": _dict_get_first(entry, ("coarse_x_lhand",)),
                "x_rhand": _dict_get_first(entry, ("coarse_x_rhand",)),
            },
            default_name="coarse",
        )
        refined_variant = _normalize_variant_dict(
            {
                "name": "refined",
                "x_obj": _dict_get_first(
                    entry, ("x_obj", "refined_x_obj", "pred_x_obj")
                ),
                "x_lhand": _dict_get_first(
                    entry, ("x_lhand", "refined_x_lhand", "pred_x_lhand")
                ),
                "x_rhand": _dict_get_first(
                    entry, ("x_rhand", "refined_x_rhand", "pred_x_rhand")
                ),
            },
            default_name="refined",
        )
        for variant in (coarse_variant, refined_variant):
            if variant is not None:
                if batch_hint <= 1:
                    variant["x_obj"] = _wrap_single_sample(variant["x_obj"])
                    variant["x_lhand"] = _wrap_single_sample(variant["x_lhand"])
                    variant["x_rhand"] = _wrap_single_sample(variant["x_rhand"])
                variants.append(variant)

    if not variants:
        return None

    if text is None:
        if _batch_size(variants[0]["x_lhand"]) > 1:
            text = [""] * _batch_size(variants[0]["x_lhand"])
        else:
            text = ""

    if isinstance(meta_list, dict):
        meta = meta_list
        meta_list = None
    else:
        meta = _dict_get_first(entry, ("meta",))

    if object_meta_list is None:
        object_name = _dict_get_first(entry, ("object_name", "obj_name"))
        object_vertices_world = None
        if variants:
            object_vertices_world = variants[0].get("object_vertices_world")
        gt_variant_raw = entry.get("gt_variant")
        if object_vertices_world is None and isinstance(gt_variant_raw, dict):
            hand_variant = gt_variant_raw.get("hand_variant")
            if isinstance(hand_variant, dict):
                object_vertices_world = hand_variant.get("obj_vertices")
        if object_name is not None or object_vertices_world is not None:
            object_meta_list = {
                "object_name": object_name,
                "obj_vertices_world": object_vertices_world,
            }

    if meta is None:
        active_hand = _coerce_text(entry.get("active_hand")).strip().lower()
        if active_hand in {"left", "right"}:
            meta = {
                "eval_hands": {
                    "left": active_hand == "left",
                    "right": active_hand == "right",
                }
            }

    return {
        "text": text,
        "meta": meta,
        "meta_list": meta_list,
        "contact_list": contact_list,
        "pen_max_list": pen_max_list,
        "cov_map": cov_map,
        "gaze_map": gaze_map,
        "gaze": gaze,
        "cam_pose": cam_pose,
        "object_meta_list": object_meta_list,
        "variants": variants,
    }


def _parse_entry(entry):
    if isinstance(entry, dict):
        return _parse_dict_entry(entry)
    if not isinstance(entry, (list, tuple)):
        return None
    cam_pose = (
        entry[-1] if len(entry) > 11 and _looks_like_cam_pose(entry[-1]) else None
    )
    if len(entry) >= 11:
        if _looks_like_text_field(entry[3]) and (
            _looks_like_object_meta_list(entry[4])
            or _looks_like_object_meta_list(entry[10])
        ):
            object_meta_list = (
                entry[10] if _looks_like_object_meta_list(entry[10]) else entry[4]
            )
            return {
                "text": entry[3],
                "meta_list": None,
                "contact_list": None,
                "pen_max_list": None,
                "gaze_map": None,
                "gaze": None,
                "cam_pose": cam_pose,
                "object_meta_list": object_meta_list,
                "variants": [
                    {
                        "name": "pred",
                        "x_obj": entry[0],
                        "x_lhand": entry[1],
                        "x_rhand": entry[2],
                        "hand_color": (235, 87, 87),
                        "object_color": (64, 176, 166),
                    },
                ],
            }
        if _looks_like_eval_meta_list(entry[4]) and _looks_like_object_meta_list(
            entry[10]
        ):
            return {
                "text": entry[3],
                "meta_list": entry[4],
                "contact_list": entry[5],
                "pen_max_list": entry[6],
                "gaze_map": None,
                "gaze": None,
                "cam_pose": cam_pose,
                "object_meta_list": entry[10],
                "variants": [
                    {
                        "name": "pred",
                        "x_obj": entry[0],
                        "x_lhand": entry[1],
                        "x_rhand": entry[2],
                        "hand_color": (235, 87, 87),
                        "object_color": (64, 176, 166),
                    },
                ],
            }
        # New format:
        # [coarse_lhand, coarse_rhand, coarse_obj, refined_obj,
        #  refined_lhand, refined_rhand, text, gaze_map, gaze, cov_map, gt_x_obj]
        if _looks_like_text_field(entry[6]):
            return {
                "text": entry[6],
                "gaze_map": entry[7],
                "gaze": entry[8],
                "cam_pose": cam_pose,
                "cov_map": entry[9],
                "variants": [
                    {
                        "name": "coarse",
                        "x_obj": entry[2],
                        "x_lhand": entry[0],
                        "x_rhand": entry[1],
                    },
                    {
                        "name": "refined",
                        "x_obj": entry[3],
                        "x_lhand": entry[4],
                        "x_rhand": entry[5],
                    },
                ],
            }

        # Legacy format:
        # [fine_lhand, fine_rhand, x_obj, text, coarse_lhand, coarse_rhand, ...]
        if _looks_like_text_field(entry[3]):
            return {
                "text": entry[3],
                "gaze_map": None,
                "gaze": None,
                "cam_pose": cam_pose,
                "variants": [
                    {
                        "name": "coarse",
                        "x_obj": entry[2],
                        "x_lhand": entry[4],
                        "x_rhand": entry[5],
                    },
                    {
                        "name": "refined",
                        "x_obj": entry[2],
                        "x_lhand": entry[0],
                        "x_rhand": entry[1],
                    },
                ],
            }
    if len(entry) in (7, 8):
        if _looks_like_text_field(entry[3]):
            return {
                "text": entry[3],
                "meta_list": None,
                "contact_list": None,
                "pen_max_list": None,
                "gaze_map": entry[4],
                "gaze": entry[5] if len(entry) > 5 else None,
                "cam_pose": (
                    entry[7]
                    if len(entry) > 7 and _looks_like_cam_pose(entry[7])
                    else None
                ),
                "cov_map": entry[6] if len(entry) > 6 else None,
                "variants": [
                    {
                        "name": "pred",
                        "x_obj": entry[0],
                        "x_lhand": entry[1],
                        "x_rhand": entry[2],
                    }
                ],
            }
        meta_list = entry[4] if len(entry) > 4 else None
        contact_list = entry[5] if len(entry) > 5 else None
        pen_max_list = entry[6] if len(entry) > 6 else None
        return {
            "text": entry[3],
            "meta_list": meta_list,
            "contact_list": contact_list,
            "pen_max_list": pen_max_list,
            "gaze_map": None,
            "gaze": None,
            "cam_pose": None,
            "cov_map": None,
            "variants": [
                {
                    "name": "pred",
                    "x_obj": entry[0],
                    "x_lhand": entry[1],
                    "x_rhand": entry[2],
                }
            ],
        }
    if len(entry) == 10:
        if _looks_like_eval_meta_list(entry[4]):
            return {
                "text": entry[3],
                "meta_list": entry[4],
                "contact_list": entry[5],
                "pen_max_list": entry[6],
                "gaze_map": None,
                "gaze": None,
                "cam_pose": None,
                "cov_map": None,
                "variants": [
                    {
                        "name": "pred",
                        "x_obj": entry[0],
                        "x_lhand": entry[1],
                        "x_rhand": entry[2],
                        "hand_color": (235, 87, 87),
                        "object_color": (64, 176, 166),
                    },
                ],
            }
        return {
            "text": entry[3],
            "meta": None,
            "gaze_map": entry[4],
            "gaze": entry[5],
            "cam_pose": (
                entry[7] if len(entry) > 7 and _looks_like_cam_pose(entry[7]) else None
            ),
            "cov_map": entry[6] if len(entry) > 6 else None,
            "variants": [
                {
                    "name": "pred",
                    "x_obj": entry[0],
                    "x_lhand": entry[1],
                    "x_rhand": entry[2],
                    "hand_color": (235, 87, 87),
                    "object_color": (64, 176, 166),
                },
            ],
        }
    if len(entry) == 9 and isinstance(entry[-1], dict):
        return {
            "text": entry[3],
            "meta": entry[8],
            "gaze_map": None,
            "gaze": None,
            "cam_pose": None,
            "cov_map": None,
            "variants": [
                {
                    "name": "pred",
                    "x_obj": entry[0],
                    "x_lhand": entry[1],
                    "x_rhand": entry[2],
                    "hand_color": (235, 87, 87),
                    "object_color": (64, 176, 166),
                },
            ],
        }
    return None


def _hand_names(text_entry: str):
    text_lower = text_entry.lower()
    if "both" in text_lower:
        return ("l_hand", "r_hand")
    if "right" in text_lower:
        return ("r_hand",)
    return ("l_hand",)


def _resolve_eval_hands_from_text(text: str) -> tuple[Optional[bool], Optional[bool]]:
    text = str(text).lower()
    if "both hands" in text:
        return True, True
    if "left hand" in text and "right hand" not in text:
        return True, False
    if "right hand" in text and "left hand" not in text:
        return False, True
    return None, None


def _hand_has_meaningful_motion(
    hand_x,
    root_norm_thresh: float = 0.05,
    motion_span_thresh: float = 0.02,
) -> bool:
    hand_x_np = _to_numpy(hand_x).astype(np.float32)
    root = hand_x_np[:, :3]
    root_norm = np.linalg.norm(root, axis=1)
    motion_span = np.linalg.norm(root.max(axis=0) - root.min(axis=0))
    return bool(root_norm.max() > root_norm_thresh or motion_span > motion_span_thresh)


def _eval_hand_selection(
    eval_meta: Optional[dict], text_entry: str, left_hand_x=None, right_hand_x=None
) -> tuple[bool, bool]:
    if isinstance(eval_meta, dict):
        eval_hands = eval_meta.get("eval_hands")
        if isinstance(eval_hands, dict):
            left = bool(eval_hands.get("left", False))
            right = bool(eval_hands.get("right", False))
            if left or right:
                return left, right
    use_left, use_right = _resolve_eval_hands_from_text(text_entry)
    if use_left is None:
        use_left = (
            _hand_has_meaningful_motion(left_hand_x)
            if left_hand_x is not None
            else False
        )
    if use_right is None:
        use_right = (
            _hand_has_meaningful_motion(right_hand_x)
            if right_hand_x is not None
            else False
        )
    return bool(use_left), bool(use_right)


def _log_hand_mesh(
    sample_path: str,
    hand_name: str,
    run_id: str,
    vertices,
    faces,
    mesh_color=(170, 170, 170),
    vertex_colors=None,
) -> None:
    vertices_np = _to_numpy(vertices)
    if not np.isfinite(vertices_np).all():
        return
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces, process=False)
    rr.log(
        f"{sample_path}/{hand_name}/{run_id}",
        rr.Mesh3D(
            vertex_positions=vertices_np.astype(np.float32),
            triangle_indices=faces,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=(
                vertex_colors
                if vertex_colors is not None
                else _vertex_colors_like(vertices, mesh_color)
            ),
        ),
    )


def _log_hand_vertices(
    sample_path: str,
    hand_name: str,
    run_id: str,
    vertices,
    point_color=(170, 170, 170),
    vertex_colors=None,
    radius: float = 0.0015,
) -> None:
    vertices_np = _to_numpy(vertices)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        return
    if not np.isfinite(vertices_np).all():
        return
    rr.log(
        f"{sample_path}/{hand_name}/{run_id}",
        rr.Points3D(
            positions=vertices_np.astype(np.float32),
            radii=[radius] * int(vertices_np.shape[0]),
            colors=(
                vertex_colors
                if vertex_colors is not None
                else _vertex_colors_like(vertices, point_color)
            ),
        ),
    )


def _log_hand_joints(
    sample_path: str,
    hand_name: str,
    run_id: str,
    joints,
    joint_color=(170, 170, 170),
    highlight_joint_indices: Optional[set[int]] = None,
    highlight_color=(255, 215, 0),
    radius: float = 0.0025,
    labels: Optional[list[str]] = None,
) -> None:
    joints_np = _to_numpy(joints)
    if joints_np.ndim != 2 or joints_np.shape[1] != 3:
        return
    if not np.isfinite(joints_np).all():
        return
    colors = np.tile(np.asarray(joint_color, dtype=np.uint8), (joints_np.shape[0], 1))
    if highlight_joint_indices:
        for joint_idx in highlight_joint_indices:
            if 0 <= int(joint_idx) < joints_np.shape[0]:
                colors[int(joint_idx)] = np.asarray(highlight_color, dtype=np.uint8)
    rr.log(
        f"{sample_path}/{hand_name}/{run_id}",
        rr.Points3D(
            positions=joints_np.astype(np.float32),
            radii=[radius] * int(joints_np.shape[0]),
            colors=colors,
            labels=labels,
        ),
    )


def _contact_joint_index_sets_from_distance(
    l_hand_joints,
    r_hand_joints,
    obj_vertices,
    threshold_m: float = 0.01,
) -> tuple[set[int], set[int]]:
    def collect(joints_xyz, obj_xyz) -> set[int]:
        joints_np = _to_numpy(joints_xyz)
        obj_np = _to_numpy(obj_xyz)
        if (
            joints_np.ndim != 2
            or obj_np.ndim != 2
            or joints_np.shape[1] != 3
            or obj_np.shape[1] != 3
            or joints_np.shape[0] == 0
            or obj_np.shape[0] == 0
        ):
            return set()
        dists = np.linalg.norm(joints_np[:, None, :] - obj_np[None, :, :], axis=2).min(
            axis=1
        )
        return {
            int(idx)
            for idx, dist in enumerate(dists.tolist())
            if np.isfinite(dist) and float(dist) <= float(threshold_m)
        }

    return collect(l_hand_joints, obj_vertices), collect(r_hand_joints, obj_vertices)


def _log_offset_anchor(label: str, offset_xyz, color, anchor_id: str) -> None:
    rr.log(
        f"anchors/{anchor_id}",
        rr.Points3D(
            positions=np.asarray(
                [[offset_xyz[0], offset_xyz[1] + 0.25, offset_xyz[2]]], dtype=np.float32
            ),
            radii=[0.006],
            colors=[list(color)],
            labels=[label],
        ),
        static=True,
    )


def _log_result_group_anchor(group_name: str, color) -> None:
    rr.log(
        f"vr/{group_name}/_group_label",
        rr.Points3D(
            positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            radii=[0.001],
            colors=[list(color)],
            labels=[group_name],
        ),
        static=True,
    )


def _color_vertices_by_index_set(
    points,
    base_color,
    highlight_indices: Optional[set[int]] = None,
    highlight_color=(255, 215, 0),
):
    points_np = _to_numpy(points)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        return _vertex_colors_like(points, base_color)
    colors = np.tile(np.asarray(base_color, dtype=np.uint8), (points_np.shape[0], 1))
    if highlight_indices:
        for idx in highlight_indices:
            idx = int(idx)
            if 0 <= idx < points_np.shape[0]:
                colors[idx] = np.asarray(highlight_color, dtype=np.uint8)
    return colors


def _highlight_single_vertex(
    colors,
    vertex_idx: Optional[int],
    highlight_color=(80, 255, 80),
):
    colors_np = np.asarray(colors, dtype=np.uint8).copy()
    if vertex_idx is None:
        return colors_np
    try:
        vertex_idx = int(vertex_idx)
    except Exception:
        return colors_np
    if 0 <= vertex_idx < colors_np.shape[0]:
        colors_np[vertex_idx] = np.asarray(highlight_color, dtype=np.uint8)
    return colors_np


def _highlight_vertex_indices(
    colors,
    vertex_indices,
    highlight_color=(255, 128, 0),
):
    colors_np = np.asarray(colors, dtype=np.uint8).copy()
    if vertex_indices is None:
        return colors_np
    if isinstance(vertex_indices, np.ndarray):
        vertex_indices = vertex_indices.tolist()
    if not isinstance(vertex_indices, (list, tuple, set)):
        return colors_np
    for vertex_idx in vertex_indices:
        try:
            vertex_idx = int(vertex_idx)
        except Exception:
            continue
        if 0 <= vertex_idx < colors_np.shape[0]:
            colors_np[vertex_idx] = np.asarray(highlight_color, dtype=np.uint8)
    return colors_np


def _split_mask_by_part(
    mask: np.ndarray, hand_part_specs: list[tuple[str, int]]
) -> dict[str, np.ndarray]:
    out = {}
    cursor = 0
    for part_name, part_len in hand_part_specs:
        out[part_name] = mask[cursor : cursor + part_len]
        cursor += part_len
    return out


def _hand_point_colors(
    base_color,
    solid_local_mask: Optional[np.ndarray],
    cavity_local_mask: Optional[np.ndarray],
    nverts: int,
    pen_local_mask: Optional[np.ndarray] = None,
):
    colors = np.tile(np.asarray(base_color, dtype=np.uint8), (nverts, 1))
    if solid_local_mask is not None and solid_local_mask.shape[0] == nverts:
        colors[solid_local_mask.astype(bool)] = np.array([255, 215, 0], dtype=np.uint8)
    if cavity_local_mask is not None and cavity_local_mask.shape[0] == nverts:
        colors[cavity_local_mask.astype(bool)] = np.array([0, 255, 255], dtype=np.uint8)
    if pen_local_mask is not None and pen_local_mask.shape[0] == nverts:
        colors[pen_local_mask.astype(bool)] = np.array([255, 128, 0], dtype=np.uint8)
    return colors


def _log_metric_label(
    variant_path: str,
    run_id: str,
    position_xyz,
    label: str,
    color,
    static: bool = False,
) -> None:
    rr.log(
        f"{variant_path}/metrics/labels/{run_id}",
        rr.Points3D(
            positions=np.asarray([position_xyz], dtype=np.float32),
            radii=[0.004],
            colors=[list(color)],
            labels=[label],
        ),
        static=static,
    )


def _meta_label(meta: Optional[dict]) -> Optional[str]:
    if not meta:
        return None
    lines = []
    if "success" in meta:
        lines.append(f"success: {bool(meta['success'])}")
    if "end_contact" in meta:
        lines.append(f"end_contact: {bool(meta['end_contact'])}")
    if "id_max_mm" in meta:
        lines.append(f"id_max_m: {float(meta['id_max_mm']) / 1000.0:.4f}")
    elif "pen_max_mm" in meta:
        lines.append(f"pen_max_m: {float(meta['pen_max_mm']) / 1000.0:.4f}")
    if "id_max_threshold_mm" in meta:
        lines.append(f"id_thresh_m: {float(meta['id_max_threshold_mm']) / 1000.0:.4f}")
    elif "pen_max_threshold_mm" in meta:
        lines.append(
            f"pen_thresh_m: {float(meta['pen_max_threshold_mm']) / 1000.0:.4f}"
        )
    hand_point, object_point, metric_name, metric_value = _meta_line_points(meta)
    if hand_point is not None and object_point is not None:
        if metric_value is not None:
            label_name = "id_max_m" if metric_name == "id_max" else "pen_max_m"
            lines.append(f"{label_name}: {metric_value / 1000.0:.4f}")
    reasons = meta.get("failure_reasons")
    if reasons:
        if isinstance(reasons, (list, tuple)):
            lines.append("failure: " + ", ".join(str(x) for x in reasons))
        else:
            lines.append(f"failure: {reasons}")
    return "\n".join(lines) if lines else None


def _get_batch_meta_value(values, batch_idx: int):
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        if len(values) <= batch_idx:
            return None
        return values[batch_idx]
    return values if batch_idx == 0 else None


def _normalize_eval_meta(
    raw_meta,
    raw_contact=None,
    raw_pen_max_mm=None,
) -> Optional[dict]:
    meta = {}
    if isinstance(raw_meta, dict):
        meta.update(raw_meta)
    elif raw_meta is not None:
        return None

    if raw_contact is not None and "contact" not in meta:
        meta["contact"] = bool(raw_contact)
    if raw_pen_max_mm is not None and "pen_max_mm" not in meta:
        try:
            meta["pen_max_mm"] = float(raw_pen_max_mm)
        except Exception:
            pass

    pen_vertex_info = meta.get("pen_vertex_info")
    if isinstance(pen_vertex_info, dict):
        if "hand_side" in pen_vertex_info and "hand_side" not in meta:
            meta["hand_side"] = pen_vertex_info["hand_side"]
        if "hand_vertex_idx" in pen_vertex_info and "hand_vertex_idx" not in meta:
            meta["hand_vertex_idx"] = pen_vertex_info["hand_vertex_idx"]
        if "object_vertex_idx" in pen_vertex_info and "object_vertex_idx" not in meta:
            meta["object_vertex_idx"] = pen_vertex_info["object_vertex_idx"]

    return meta if meta else None


def _eval_metric_label(eval_meta: Optional[dict]) -> Optional[str]:
    if not eval_meta:
        return None
    lines = ["EVAL"]
    if "success" in eval_meta:
        lines.append(f"eval_success: {bool(eval_meta['success'])}")
    if "vr_percent" in eval_meta:
        try:
            lines.append(f"eval_vr_percent: {float(eval_meta['vr_percent']):.1f}")
        except Exception:
            pass
    if "cr_percent" in eval_meta:
        try:
            lines.append(f"eval_cr_percent: {float(eval_meta['cr_percent']):.1f}")
        except Exception:
            pass
    if "iv_cm3" in eval_meta:
        try:
            lines.append(f"eval_iv_cm3: {float(eval_meta['iv_cm3']):.4f}")
        except Exception:
            pass
    if "id_mm" in eval_meta:
        try:
            lines.append(f"eval_id_m: {float(eval_meta['id_mm']) / 1000.0:.4f}")
        except Exception:
            pass
    if "id_max_mm" in eval_meta:
        try:
            lines.append(f"eval_id_max_m: {float(eval_meta['id_max_mm']) / 1000.0:.4f}")
        except Exception:
            pass
    if "iv_success_threshold_cm3" in eval_meta:
        try:
            lines.append(
                "eval_iv_thresh_cm3: "
                f"{float(eval_meta['iv_success_threshold_cm3']):.4f}"
            )
        except Exception:
            pass
    contact_joint_indices = eval_meta.get("contact_joint_indices")
    if isinstance(contact_joint_indices, dict):
        left_n = len(contact_joint_indices.get("left", []) or [])
        right_n = len(contact_joint_indices.get("right", []) or [])
        lines.append(f"contact_joints: L={left_n} R={right_n}")
    joint_signed_distance_info = eval_meta.get("joint_signed_distance_info")
    if isinstance(joint_signed_distance_info, dict):
        signed_values = []
        for hand_side in ("left", "right"):
            entries = joint_signed_distance_info.get(hand_side)
            if not isinstance(entries, (list, tuple)):
                continue
            for item in entries:
                if not isinstance(item, dict):
                    continue
                try:
                    signed_values.append(float(item["signed_distance_m"]))
                except Exception:
                    continue
        if signed_values:
            lines.append(
                "joint_sd_m[min,max]: "
                f"{min(signed_values):.4f}, {max(signed_values):.4f}"
            )
    return "\n".join(lines) if len(lines) > 1 else None


def _contact_joint_index_sets_from_eval_meta(
    eval_meta: Optional[dict],
    use_left: bool,
    use_right: bool,
) -> tuple[set[int], set[int]]:
    left_indices: set[int] = set()
    right_indices: set[int] = set()
    if not eval_meta:
        return left_indices, right_indices
    contact_joint_indices = eval_meta.get("contact_joint_indices")
    if not isinstance(contact_joint_indices, dict):
        return left_indices, right_indices

    def collect(raw) -> set[int]:
        out: set[int] = set()
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple, set)):
            for idx in raw:
                try:
                    out.add(int(idx))
                except Exception:
                    pass
        return out

    if use_left:
        left_indices = collect(contact_joint_indices.get("left"))
    if use_right:
        right_indices = collect(contact_joint_indices.get("right"))
    return left_indices, right_indices


def _joint_distance_labels_from_eval_meta(
    eval_meta: Optional[dict],
    use_left: bool,
    use_right: bool,
    left_joint_count: int,
    right_joint_count: int,
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    if not eval_meta:
        return None, None
    joint_info = eval_meta.get("joint_signed_distance_info")
    if not isinstance(joint_info, dict):
        return None, None

    def collect(raw_entries, enabled: bool, joint_count: int) -> Optional[list[str]]:
        if not enabled:
            return None
        labels = ["" for _ in range(joint_count)]
        if not isinstance(raw_entries, (list, tuple)):
            return labels
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            try:
                joint_idx = int(item["hand_joint_idx"])
                signed_distance_m = float(item["signed_distance_m"])
            except Exception:
                continue
            if 0 <= joint_idx < joint_count:
                labels[joint_idx] = f"j{joint_idx} {signed_distance_m:+.4f}m"
        return labels

    left_labels = collect(joint_info.get("left"), use_left, left_joint_count)
    right_labels = collect(joint_info.get("right"), use_right, right_joint_count)
    return left_labels, right_labels


def _iv_interior_hand_vertex_index_sets(
    eval_meta: Optional[dict],
) -> tuple[set[int], set[int]]:
    if not eval_meta:
        return set(), set()
    iv_info = eval_meta.get("iv_info")
    if not isinstance(iv_info, dict):
        return set(), set()

    def collect(raw) -> set[int]:
        out: set[int] = set()
        if isinstance(raw, dict):
            raw = raw.get("interior_hand_vertex_indices")
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple, set)):
            for idx in raw:
                try:
                    out.add(int(idx))
                except Exception:
                    pass
        return out

    if "left" in iv_info or "right" in iv_info:
        return collect(iv_info.get("left")), collect(iv_info.get("right"))

    hand_side = str(iv_info.get("hand_side", "")).strip().lower()
    indices = collect(iv_info)
    if hand_side in {"left", "l", "l_hand"}:
        return indices, set()
    if hand_side in {"right", "r", "r_hand"}:
        return set(), indices
    return set(), set()


def _contact_threshold_vertex_index_sets(
    eval_meta: Optional[dict],
    use_left: bool,
    use_right: bool,
) -> tuple[set[int], set[int], set[int]]:
    left_hand_indices: set[int] = set()
    right_hand_indices: set[int] = set()
    object_indices: set[int] = set()
    if not eval_meta:
        return left_hand_indices, right_hand_indices, object_indices
    contact_joint_info = eval_meta.get("contact_joint_info")
    if isinstance(contact_joint_info, dict):
        for hand_side, enabled in [("left", use_left), ("right", use_right)]:
            if not enabled:
                continue
            entries = contact_joint_info.get(hand_side)
            if not isinstance(entries, (list, tuple)):
                continue
            for item in entries:
                if not isinstance(item, dict):
                    continue
                object_vertex_idx = item.get("nearest_object_vertex_idx")
                if object_vertex_idx is None:
                    object_vertex_idx = item.get("object_vertex_idx")
                try:
                    object_indices.add(int(object_vertex_idx))
                except Exception:
                    pass
        return left_hand_indices, right_hand_indices, object_indices
    container = eval_meta.get("contact_threshold_indices")
    if not isinstance(container, dict):
        return left_hand_indices, right_hand_indices, object_indices
    for hand_side, enabled, target in [
        ("left", use_left, left_hand_indices),
        ("right", use_right, right_hand_indices),
    ]:
        if not enabled:
            continue
        info = container.get(hand_side)
        if not isinstance(info, dict):
            continue
        hand_vertex_indices = info.get("hand_vertex_indices")
        object_vertex_indices = info.get("object_vertex_indices")
        if isinstance(hand_vertex_indices, (list, tuple)):
            for idx in hand_vertex_indices:
                try:
                    target.add(int(idx))
                except Exception:
                    pass
        if isinstance(object_vertex_indices, (list, tuple)):
            for idx in object_vertex_indices:
                try:
                    object_indices.add(int(idx))
                except Exception:
                    pass
    return left_hand_indices, right_hand_indices, object_indices


def _penetrated_object_vertex_indices(eval_meta: Optional[dict]) -> list[int]:
    object_indices: list[int] = []
    seen: set[int] = set()
    if not eval_meta:
        return object_indices

    direct_indices = eval_meta.get("penetrated_object_vertex_indices")
    if isinstance(direct_indices, np.ndarray):
        direct_indices = direct_indices.tolist()
    if isinstance(direct_indices, (list, tuple)):
        for idx in direct_indices:
            try:
                idx = int(idx)
            except Exception:
                continue
            if idx not in seen:
                seen.add(idx)
                object_indices.append(idx)
        return object_indices

    def collect(value) -> None:
        if isinstance(value, dict):
            if "object_vertex_idx" in value:
                try:
                    idx = int(value["object_vertex_idx"])
                except Exception:
                    idx = None
                if idx is not None and idx not in seen:
                    seen.add(idx)
                    object_indices.append(idx)
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(eval_meta.get("penetration_vertex_info"))
    if not object_indices:
        collect(eval_meta.get("pen_vertex_info"))
    return object_indices


def _pen_max_vertex_and_object_indices(
    eval_meta: Optional[dict],
) -> tuple[Optional[str], Optional[int], Optional[int]]:
    if not eval_meta:
        return None, None, None
    pen_max_info = eval_meta.get("id_info")
    if not isinstance(pen_max_info, dict):
        pen_max_info = eval_meta.get("pen_max_info")
    if not isinstance(pen_max_info, dict):
        return None, None, None
    hand_side = pen_max_info.get("hand_side")
    try:
        hand_vertex_idx = (
            int(pen_max_info["hand_vertex_idx"])
            if pen_max_info.get("hand_vertex_idx") is not None
            else None
        )
    except Exception:
        hand_vertex_idx = None
    try:
        object_vertex_idx = (
            int(pen_max_info["object_vertex_idx"])
            if pen_max_info.get("object_vertex_idx") is not None
            else None
        )
    except Exception:
        object_vertex_idx = None
    return (
        str(hand_side).strip().lower() if hand_side is not None else None,
        hand_vertex_idx,
        object_vertex_idx,
    )


def _vr_result_group_from_final_frame(
    final_l_joints,
    final_r_joints,
    final_obj_pos,
    use_left: bool,
    use_right: bool,
    threshold_m: float = CONTACT_THRESHOLD_M,
) -> tuple[str, set[int], set[int]]:
    def collect(joints_xyz, enabled: bool) -> set[int]:
        if not enabled:
            return set()
        joints_np = _to_numpy(joints_xyz)
        obj_np = _to_numpy(final_obj_pos)
        if (
            joints_np.ndim != 2
            or obj_np.ndim != 2
            or joints_np.shape[1] != 3
            or obj_np.shape[1] != 3
            or joints_np.shape[0] == 0
            or obj_np.shape[0] == 0
        ):
            return set()
        valid_tip_indices = [
            idx for idx in FINGERTIP_JOINT_INDICES if 0 <= idx < joints_np.shape[0]
        ]
        if not valid_tip_indices:
            return set()
        tip_xyz = joints_np[valid_tip_indices]
        dists = np.linalg.norm(tip_xyz[:, None, :] - obj_np[None, :, :], axis=2).min(
            axis=1
        )
        return {
            int(joint_idx)
            for joint_idx, dist in zip(valid_tip_indices, dists.tolist())
            if np.isfinite(dist) and float(dist) <= float(threshold_m)
        }

    left_contact_joint_indices = collect(final_l_joints, use_left)
    right_contact_joint_indices = collect(final_r_joints, use_right)
    result_group = (
        "success"
        if (len(left_contact_joint_indices) + len(right_contact_joint_indices))
        >= MIN_CONTACT_KEY_JOINTS
        else "fail"
    )
    return result_group, left_contact_joint_indices, right_contact_joint_indices


def _resolve_eval_pen_points(
    eval_meta: Optional[dict],
    l_hand_vertices,
    r_hand_vertices,
    obj_vertices,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    if not eval_meta:
        return None, None, None
    pen_max_info = eval_meta.get("id_info")
    if not isinstance(pen_max_info, dict):
        pen_max_info = eval_meta.get("pen_max_info")
    if isinstance(pen_max_info, dict):
        line = pen_max_info.get("line")
        if isinstance(line, dict):
            object_point = line.get("object_point")
            hand_point = line.get("hand_vertex_point")
            if hand_point is None:
                hand_point = line.get("hand_joint_point")
            if object_point is not None and hand_point is not None:
                try:
                    object_point_np = np.asarray(
                        object_point, dtype=np.float32
                    ).reshape(3)
                    hand_point_np = np.asarray(hand_point, dtype=np.float32).reshape(3)
                except Exception:
                    object_point_np = None
                    hand_point_np = None
                if (
                    object_point_np is not None
                    and hand_point_np is not None
                    and np.all(np.isfinite(object_point_np))
                    and np.all(np.isfinite(hand_point_np))
                ):
                    metric_value = None
                    if "distance_m" in pen_max_info:
                        try:
                            metric_value = float(pen_max_info["distance_m"])
                        except Exception:
                            metric_value = None
                    elif "distance_mm" in pen_max_info:
                        try:
                            metric_value = float(pen_max_info["distance_mm"]) / 1000.0
                        except Exception:
                            metric_value = None
                    elif "ID" in eval_meta:
                        try:
                            metric_value = float(eval_meta["ID"]) / 1000.0
                        except Exception:
                            metric_value = None
                    elif "pen_max_mm" in eval_meta:
                        try:
                            metric_value = float(eval_meta["pen_max_mm"]) / 1000.0
                        except Exception:
                            metric_value = None
                    return hand_point_np, object_point_np, metric_value
        hand_side = str(pen_max_info.get("hand_side", "")).strip().lower()
        hand_vertex_idx = pen_max_info.get("hand_vertex_idx")
        if hand_vertex_idx is None:
            hand_vertex_idx = pen_max_info.get("hand_joint_idx")
        object_vertex_idx = pen_max_info.get("object_vertex_idx")
        try:
            hand_vertex_idx = (
                int(hand_vertex_idx) if hand_vertex_idx is not None else None
            )
            object_vertex_idx = (
                int(object_vertex_idx) if object_vertex_idx is not None else None
            )
        except Exception:
            hand_vertex_idx = None
            object_vertex_idx = None
        if hand_vertex_idx is not None and object_vertex_idx is not None:
            if hand_side in {"left", "l", "l_hand"}:
                hand_points = _to_numpy(l_hand_vertices)
            elif hand_side in {"right", "r", "r_hand"}:
                hand_points = _to_numpy(r_hand_vertices)
            else:
                hand_points = None
            obj_vertices_np = _to_numpy(obj_vertices)
            if (
                hand_points is not None
                and hand_points.ndim == 2
                and obj_vertices_np.ndim == 2
                and 0 <= hand_vertex_idx < hand_points.shape[0]
                and 0 <= object_vertex_idx < obj_vertices_np.shape[0]
            ):
                hand_point_np = np.asarray(
                    hand_points[hand_vertex_idx], dtype=np.float32
                ).reshape(3)
                object_point_np = np.asarray(
                    obj_vertices_np[object_vertex_idx], dtype=np.float32
                ).reshape(3)
                if np.all(np.isfinite(hand_point_np)) and np.all(
                    np.isfinite(object_point_np)
                ):
                    metric_value = None
                    if "distance_m" in pen_max_info:
                        try:
                            metric_value = float(pen_max_info["distance_m"])
                        except Exception:
                            metric_value = None
                    elif "distance_mm" in pen_max_info:
                        try:
                            metric_value = float(pen_max_info["distance_mm"]) / 1000.0
                        except Exception:
                            metric_value = None
                    elif "ID" in eval_meta:
                        try:
                            metric_value = float(eval_meta["ID"]) / 1000.0
                        except Exception:
                            metric_value = None
                    elif "pen_max_mm" in eval_meta:
                        try:
                            metric_value = float(eval_meta["pen_max_mm"]) / 1000.0
                        except Exception:
                            metric_value = None
                    return hand_point_np, object_point_np, metric_value
    hand_side = str(eval_meta.get("hand_side", "")).strip().lower()
    hand_vertex_idx = eval_meta.get("hand_vertex_idx")
    object_vertex_idx = eval_meta.get("object_vertex_idx")
    if hand_vertex_idx is None or object_vertex_idx is None:
        return None, None, None
    try:
        hand_vertex_idx = int(hand_vertex_idx)
        object_vertex_idx = int(object_vertex_idx)
    except Exception:
        return None, None, None

    if hand_side in {"left", "l", "l_hand"}:
        hand_vertices = _to_numpy(l_hand_vertices)
    elif hand_side in {"right", "r", "r_hand"}:
        hand_vertices = _to_numpy(r_hand_vertices)
    else:
        return None, None, None

    obj_vertices_np = _to_numpy(obj_vertices)
    if (
        hand_vertex_idx < 0
        or object_vertex_idx < 0
        or hand_vertex_idx >= hand_vertices.shape[0]
        or object_vertex_idx >= obj_vertices_np.shape[0]
    ):
        return None, None, None

    hand_point = np.asarray(hand_vertices[hand_vertex_idx], dtype=np.float32).reshape(3)
    object_point = np.asarray(
        obj_vertices_np[object_vertex_idx], dtype=np.float32
    ).reshape(3)
    if not np.all(np.isfinite(hand_point)) or not np.all(np.isfinite(object_point)):
        return None, None, None

    metric_value = None
    if "ID" in eval_meta:
        try:
            metric_value = float(eval_meta["ID"]) / 1000.0
        except Exception:
            metric_value = None
    elif "pen_max_mm" in eval_meta:
        try:
            metric_value = float(eval_meta["pen_max_mm"]) / 1000.0
        except Exception:
            metric_value = None
    return hand_point, object_point, metric_value


def _resolve_pen_max_hand_vertex_index(
    eval_meta: Optional[dict],
    l_hand_vertices,
    r_hand_vertices,
    l_hand_joints=None,
    r_hand_joints=None,
) -> tuple[Optional[str], Optional[int]]:
    if not eval_meta:
        return None, None
    pen_max_info = eval_meta.get("id_info")
    if not isinstance(pen_max_info, dict):
        pen_max_info = eval_meta.get("pen_max_info")
    if not isinstance(pen_max_info, dict):
        return None, None
    line = pen_max_info.get("line")
    hand_side = str(pen_max_info.get("hand_side", "")).strip().lower()
    hand_vertex_idx = pen_max_info.get("hand_vertex_idx")
    if hand_vertex_idx is not None:
        try:
            hand_vertex_idx = int(hand_vertex_idx)
        except Exception:
            hand_vertex_idx = None
    hand_joint_idx = pen_max_info.get("hand_joint_idx")
    hand_point = line.get("hand_vertex_point") if isinstance(line, dict) else None
    if hand_point is None and isinstance(line, dict):
        hand_point = line.get("hand_joint_point")
    hand_point_np = None
    if hand_point is not None:
        try:
            hand_point_np = np.asarray(hand_point, dtype=np.float32).reshape(3)
        except Exception:
            hand_point_np = None
    if hand_side in {"left", "l", "l_hand"}:
        hand_vertices_np = _to_numpy(l_hand_vertices)
        hand_joints_np = _to_numpy(l_hand_joints) if l_hand_joints is not None else None
        resolved_side = "left"
    elif hand_side in {"right", "r", "r_hand"}:
        hand_vertices_np = _to_numpy(r_hand_vertices)
        hand_joints_np = _to_numpy(r_hand_joints) if r_hand_joints is not None else None
        resolved_side = "right"
    else:
        return None, None
    if (
        hand_vertex_idx is not None
        and hand_vertices_np.ndim == 2
        and 0 <= hand_vertex_idx < hand_vertices_np.shape[0]
    ):
        return resolved_side, hand_vertex_idx
    if (
        hand_point_np is None
        and hand_joint_idx is not None
        and hand_joints_np is not None
    ):
        try:
            hand_joint_idx = int(hand_joint_idx)
        except Exception:
            hand_joint_idx = None
        if (
            hand_joint_idx is not None
            and hand_joints_np.ndim == 2
            and 0 <= hand_joint_idx < hand_joints_np.shape[0]
        ):
            hand_point_np = np.asarray(
                hand_joints_np[hand_joint_idx], dtype=np.float32
            ).reshape(3)
    if (
        hand_vertices_np.ndim != 2
        or hand_vertices_np.shape[1] != 3
        or hand_vertices_np.shape[0] == 0
        or hand_point_np is None
        or not np.all(np.isfinite(hand_point_np))
    ):
        return None, None
    dists = np.linalg.norm(hand_vertices_np - hand_point_np[None, :], axis=1)
    if dists.size == 0 or not np.isfinite(dists).any():
        return None, None
    return resolved_side, int(np.nanargmin(dists))


def _resolve_contact_joint_links(
    eval_meta: Optional[dict],
    l_hand_vertices,
    r_hand_vertices,
    obj_vertices,
    use_left: bool = True,
    use_right: bool = True,
) -> list[dict]:
    if not eval_meta:
        return []
    contact_vertex_info = eval_meta.get("contact_joint_info")
    if not isinstance(contact_vertex_info, dict):
        contact_vertex_info = eval_meta.get("contact_vertex_info")
    if not isinstance(contact_vertex_info, dict):
        return []

    obj_vertices_np = _to_numpy(obj_vertices)
    left_vertices_np = (
        _to_numpy(l_hand_vertices) if l_hand_vertices is not None else None
    )
    right_vertices_np = (
        _to_numpy(r_hand_vertices) if r_hand_vertices is not None else None
    )

    links = []
    for hand_side, hand_vertices_np, line_color, joint_color in [
        ("left", left_vertices_np, [64, 196, 255], [0, 255, 255]),
        ("right", right_vertices_np, [255, 176, 64], [255, 64, 0]),
    ]:
        if hand_side == "left" and not use_left:
            continue
        if hand_side == "right" and not use_right:
            continue
        if hand_vertices_np is None:
            continue
        entries = contact_vertex_info.get(hand_side)
        if not isinstance(entries, (list, tuple)):
            continue
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                continue
            try:
                hand_vertex_idx = int(
                    item["hand_vertex_idx"]
                    if "hand_vertex_idx" in item
                    else item["hand_joint_idx"]
                )
                object_vertex_idx = int(
                    item["object_vertex_idx"]
                    if "object_vertex_idx" in item
                    else item["nearest_object_vertex_idx"]
                )
            except Exception:
                continue
            if (
                hand_vertex_idx < 0
                or object_vertex_idx < 0
                or hand_vertex_idx >= hand_vertices_np.shape[0]
                or object_vertex_idx >= obj_vertices_np.shape[0]
            ):
                continue
            hand_point = np.asarray(
                hand_vertices_np[hand_vertex_idx], dtype=np.float32
            ).reshape(3)
            object_point = np.asarray(
                obj_vertices_np[object_vertex_idx], dtype=np.float32
            ).reshape(3)
            if not np.all(np.isfinite(hand_point)) or not np.all(
                np.isfinite(object_point)
            ):
                continue
            distance = item.get("signed_distance_m")
            if distance is None:
                distance = item.get("distance")
            try:
                distance = float(distance) if distance is not None else None
            except Exception:
                distance = None
            links.append(
                {
                    "hand_side": hand_side,
                    "entry_idx": idx,
                    "hand_vertex_idx": hand_vertex_idx,
                    "object_vertex_idx": object_vertex_idx,
                    "distance": distance,
                    "hand_point": hand_point,
                    "object_point": object_point,
                    "line_color": line_color,
                    "joint_color": joint_color,
                    "object_color": [255, 255, 0],
                }
            )
    return links


def _log_contact_joint_links(
    variant_path: str,
    run_id: str,
    links: list[dict],
    static: bool = False,
) -> None:
    for link in links:
        hand_side = link["hand_side"]
        entry_idx = link["entry_idx"]
        hand_point = link["hand_point"]
        object_point = link["object_point"]
        link_name = f"eval_contact_{hand_side}_{entry_idx:03d}"
        rr.log(
            f"{variant_path}/metrics/{link_name}_line/{run_id}",
            rr.LineStrips3D(
                [[object_point.tolist(), hand_point.tolist()]],
                colors=[link["line_color"]],
                radii=0.001,
            ),
            static=static,
        )
        rr.log(
            f"{variant_path}/metrics/{link_name}_points/{run_id}",
            rr.Points3D(
                positions=np.asarray([object_point, hand_point], dtype=np.float32),
                radii=[0.0035, 0.0035],
                colors=[link["object_color"], link["joint_color"]],
                labels=[
                    "",
                    (
                        f"{hand_side} joint {link['hand_vertex_idx']} d={link['distance']:.4f}m"
                        if link["distance"] is not None
                        else f"{hand_side} joint {link['hand_vertex_idx']}"
                    ),
                ],
            ),
            static=static,
        )


def _resolve_fingertip_distance_links(
    eval_meta: Optional[dict],
    l_hand_joints,
    r_hand_joints,
    obj_vertices,
    use_left: bool = True,
    use_right: bool = True,
) -> list[dict]:
    if not eval_meta:
        return []
    fingertip_info = eval_meta.get("fingertip_distance_info")
    if not isinstance(fingertip_info, dict):
        return []

    obj_vertices_np = _to_numpy(obj_vertices)
    left_joints_np = _to_numpy(l_hand_joints) if l_hand_joints is not None else None
    right_joints_np = _to_numpy(r_hand_joints) if r_hand_joints is not None else None

    links = []
    for hand_side, joints_np, line_color, point_color in [
        ("left", left_joints_np, [32, 220, 255], [32, 220, 255]),
        ("right", right_joints_np, [255, 96, 32], [255, 96, 32]),
    ]:
        if hand_side == "left" and not use_left:
            continue
        if hand_side == "right" and not use_right:
            continue
        if joints_np is None:
            continue
        entries = fingertip_info.get(hand_side)
        if not isinstance(entries, (list, tuple)):
            continue
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                continue
            try:
                hand_joint_idx = int(item["hand_joint_idx"])
                object_vertex_idx = int(item["object_vertex_idx"])
            except Exception:
                continue
            if (
                joints_np.ndim != 2
                or obj_vertices_np.ndim != 2
                or hand_joint_idx < 0
                or object_vertex_idx < 0
                or hand_joint_idx >= joints_np.shape[0]
                or object_vertex_idx >= obj_vertices_np.shape[0]
            ):
                continue
            hand_point = np.asarray(
                joints_np[hand_joint_idx], dtype=np.float32
            ).reshape(3)
            object_point = np.asarray(
                obj_vertices_np[object_vertex_idx], dtype=np.float32
            ).reshape(3)
            if not np.all(np.isfinite(hand_point)) or not np.all(
                np.isfinite(object_point)
            ):
                continue
            distance_m = item.get("distance_m")
            try:
                distance_m = float(distance_m) if distance_m is not None else None
            except Exception:
                distance_m = None
            if distance_m is None:
                distance_mm = item.get("distance_mm")
                try:
                    distance_m = (
                        float(distance_mm) / 1000.0 if distance_mm is not None else None
                    )
                except Exception:
                    distance_m = None
            links.append(
                {
                    "hand_side": hand_side,
                    "entry_idx": idx,
                    "hand_joint_idx": hand_joint_idx,
                    "object_vertex_idx": object_vertex_idx,
                    "distance_m": distance_m,
                    "hand_point": hand_point,
                    "object_point": object_point,
                    "line_color": line_color,
                    "joint_color": point_color,
                    "object_color": [255, 255, 0],
                }
            )
    return links


def _log_fingertip_distance_links(
    variant_path: str,
    run_id: str,
    links: list[dict],
    static: bool = False,
) -> None:
    for link in links:
        hand_side = link["hand_side"]
        entry_idx = link["entry_idx"]
        hand_point = link["hand_point"]
        object_point = link["object_point"]
        link_name = f"eval_fingertip_{hand_side}_{entry_idx:03d}"
        rr.log(
            f"{variant_path}/metrics/{link_name}_line/{run_id}",
            rr.LineStrips3D(
                [[object_point.tolist(), hand_point.tolist()]],
                colors=[link["line_color"]],
                radii=0.001,
            ),
            static=static,
        )
        distance_label = (
            f"{link['distance_m']:.4f}m" if link["distance_m"] is not None else None
        )
        rr.log(
            f"{variant_path}/metrics/{link_name}_points/{run_id}",
            rr.Points3D(
                positions=np.asarray([object_point, hand_point], dtype=np.float32),
                radii=[0.0035, 0.0035],
                colors=[link["object_color"], link["joint_color"]],
                labels=[
                    (
                        f"{hand_side} obj_v {link['object_vertex_idx']} {distance_label}"
                        if distance_label is not None
                        else f"{hand_side} obj_v {link['object_vertex_idx']}"
                    ),
                    (
                        f"{hand_side} tip_j {link['hand_joint_idx']} {distance_label}"
                        if distance_label is not None
                        else f"{hand_side} tip_j {link['hand_joint_idx']}"
                    ),
                ],
            ),
            static=static,
        )


def _log_meta_id_max_line(
    variant_path: str,
    run_id: str,
    object_point,
    hand_point,
    metric_name: str = "id_max",
    metric_value_mm: Optional[float] = None,
    static: bool = False,
):
    if object_point is None or hand_point is None:
        return
    object_point = np.asarray(object_point, dtype=np.float32)
    hand_point = np.asarray(hand_point, dtype=np.float32)
    rr.log(
        f"{variant_path}/metrics/{metric_name}_line/{run_id}",
        rr.LineStrips3D(
            [
                [
                    object_point.tolist(),
                    hand_point.tolist(),
                ]
            ],
            colors=[[255, 0, 0]],
            radii=0.001,
        ),
        static=static,
    )
    rr.log(
        f"{variant_path}/metrics/{metric_name}_points/{run_id}",
        rr.Points3D(
            positions=np.asarray([object_point, hand_point], dtype=np.float32),
            radii=[0.0035, 0.0035],
            colors=[[255, 255, 0], [255, 0, 255]],
            labels=[
                (
                    f"{metric_name} object {metric_value_mm:.2f} mm"
                    if metric_value_mm is not None
                    else f"{metric_name} object"
                ),
                (
                    f"{metric_name} hand {metric_value_mm:.2f} mm"
                    if metric_value_mm is not None
                    else f"{metric_name} hand"
                ),
            ],
        ),
        static=static,
    )


def visualize_rr(
    recording_name,
    run_id,
    offset_xyz=(0.0, 0.0, 0.0),
    run_color=(121, 121, 121),
    selected_objects: Optional[set[str]] = None,
    object_slot_lookup: Optional[dict[str, int]] = None,
    object_offset_step: float = 1.2,
    final_only: bool = False,
):
    rr.log(f"{_sanitize_entity_path(run_id)}", rr.Clear.recursive())
    items = _load_items_from_recording(recording_name)
    sample_counts = defaultdict(int)
    object_counts = defaultdict(int)
    if object_slot_lookup is None:
        observed_objects = []
        observed_set = set()
        for entry in items:
            parsed = _parse_entry(entry)
            if parsed is None:
                continue
            text = parsed["text"]
            object_meta_list = parsed.get("object_meta_list")
            primary_lhand = (
                parsed["variants"][0]["x_lhand"] if parsed.get("variants") else None
            )
            batch_size = _batch_size(primary_lhand)
            for batch_idx in range(batch_size):
                text_entry = _get_batch_text(text, batch_idx)
                object_meta = _get_batch_meta_value(object_meta_list, batch_idx)
                stored_object_name = (
                    object_meta.get("object_name")
                    if isinstance(object_meta, dict)
                    else None
                )
                object_key = (
                    _normalized_object_key(stored_object_name)
                    if stored_object_name is not None
                    else _normalized_object_key(_build_grouping_key(text_entry)[0])
                )
                if object_key not in observed_set:
                    observed_set.add(object_key)
                    observed_objects.append(object_key)
        object_slot_lookup = {
            object_key: slot for slot, object_key in enumerate(observed_objects)
        }

    for entry in items:
        parsed = _parse_entry(entry)
        if parsed is None:
            entry_len = (
                len(entry) if isinstance(entry, (list, tuple)) else type(entry).__name__
            )
            print(f"[WARN] unsupported entry format {entry_len} in {recording_name}")
            continue

        text = parsed["text"]
        meta = parsed.get("meta")
        meta_list = parsed.get("meta_list")
        contact_list = parsed.get("contact_list")
        pen_max_list = parsed.get("pen_max_list")
        cov_map = parsed.get("cov_map")
        gaze_map = parsed.get("gaze_map")
        gaze = parsed.get("gaze")
        cam_pose = parsed.get("cam_pose")
        object_meta_list = parsed.get("object_meta_list")
        variants = parsed["variants"]
        if not variants:
            continue

        primary_lhand = variants[0]["x_lhand"]
        batch_size = _batch_size(primary_lhand)
        if batch_size <= 0:
            print(f"[WARN] unsupported hand batch container in {recording_name}")
            continue

        for batch_idx in range(batch_size):
            text_entry = _get_batch_text(text, batch_idx)
            eval_meta = _normalize_eval_meta(
                _get_batch_meta_value(meta_list, batch_idx),
                raw_contact=_get_batch_meta_value(contact_list, batch_idx),
                raw_pen_max_mm=_get_batch_meta_value(pen_max_list, batch_idx),
            )
            cov_map_sample = _get_batch_cov_map(cov_map, batch_idx, batch_size)
            gaze_map_sample = _get_batch_gaze_value(gaze_map, batch_idx, batch_size)
            gaze_sample = _get_batch_gaze_value(gaze, batch_idx, batch_size)
            cam_pose_sample = _get_batch_cam_pose(cam_pose, batch_idx, batch_size)
            object_meta = _get_batch_meta_value(object_meta_list, batch_idx)
            stored_object_name = None
            if isinstance(object_meta, dict):
                stored_object_name = object_meta.get("object_name")
            object_name_for_group = (
                _normalized_object_key(stored_object_name)
                if stored_object_name is not None
                else None
            )
            object_key, action_key = _build_grouping_key(text_entry)
            if object_name_for_group:
                object_key = object_name_for_group
                action_key = f"{object_key}::{_coerce_text(text_entry).strip()}"
            else:
                object_key = _normalized_object_key(object_key)
            if selected_objects is not None and object_key not in selected_objects:
                continue
            has_object_meta = object_meta_list is not None
            if has_object_meta and not isinstance(object_meta, dict):
                print(
                    f"[WARN] missing object_meta for '{text_entry}' batch {batch_idx}"
                )
                continue
            if has_object_meta and isinstance(object_meta, dict):
                if (
                    object_meta.get("obj_pc_org") is None
                    and not _object_meta_has_world_vertices(object_meta)
                    and object_key not in OBJ_PC
                ):
                    print(
                        f"[WARN] object_meta missing obj_pc_org/world points for '{text_entry}' batch {batch_idx}"
                    )
                    continue
            if not has_object_meta and object_key not in OBJ_PC:
                print(
                    f"[WARN] unresolved object key: '{object_key}' from text '{text_entry}'"
                )
                continue

            sample_idx = sample_counts[action_key]
            sample_counts[action_key] += 1
            object_slot_idx = (
                0
                if object_slot_lookup is None
                else object_slot_lookup.get(object_key, 0)
            )
            sample_offset = (
                float(offset_xyz[0]),
                float(offset_xyz[1]),
                float(offset_xyz[2] + object_slot_idx * object_offset_step),
            )

            for variant in variants:
                x_obj = variant["x_obj"]
                x_lhand = variant["x_lhand"]
                x_rhand = variant["x_rhand"]
                lhand_vertices_world = variant.get("lhand_vertices_world")
                rhand_vertices_world = variant.get("rhand_vertices_world")
                lhand_joints_world = variant.get("lhand_joints_world")
                rhand_joints_world = variant.get("rhand_joints_world")
                target_sampled_vertices = variant.get("target_sampled_vertices")
                pred_sampled_vertices = variant.get("pred_sampled_vertices")
                variant_name = variant["name"]
                if str(variant_name).strip().lower() in {"gt", "gt_debug"}:
                    continue
                hand_color = tuple(
                    variant.get(
                        "hand_color", variant.get("mesh_color", (170, 170, 170))
                    )
                )
                object_color = tuple(variant.get("object_color", run_color))

                x_obj_source = _get_batch_item(x_obj, batch_idx)
                if x_obj_source is None:
                    print(
                        f"[WARN] missing x_obj pose for '{text_entry}' ({variant_name}) at batch {batch_idx}"
                    )
                    continue

                try:
                    obj_vertices = _object_vertices_from_meta_or_source(
                        x_obj_source, object_key, object_meta
                    )
                    _validate_pen_max_object_alignment(
                        eval_meta,
                        obj_vertices[max(0, int(obj_vertices.shape[0]) - 1)],
                        recording_name,
                        text_entry,
                        batch_idx,
                    )
                    l_vertices_source = _get_batch_item(lhand_vertices_world, batch_idx)
                    r_vertices_source = _get_batch_item(rhand_vertices_world, batch_idx)
                    l_joints_source = _get_batch_item(lhand_joints_world, batch_idx)
                    r_joints_source = _get_batch_item(rhand_joints_world, batch_idx)
                    x_lhand_source = _get_batch_item(x_lhand, batch_idx)
                    x_rhand_source = _get_batch_item(x_rhand, batch_idx)
                    l_hand_vertices, l_hand_joints = _resolve_hand_world_geometry(
                        l_vertices_source,
                        l_joints_source,
                        x_lhand_source,
                        L_HAND_LAYER,
                    )
                    r_hand_vertices, r_hand_joints = _resolve_hand_world_geometry(
                        r_vertices_source,
                        r_joints_source,
                        x_rhand_source,
                        R_HAND_LAYER,
                    )
                    if (
                        l_hand_vertices is None
                        or l_hand_joints is None
                        or r_hand_vertices is None
                        or r_hand_joints is None
                    ):
                        raise ValueError(
                            "variant is missing usable hand geometry/params"
                        )

                    l_hand_faces = torch.as_tensor(
                        L_HAND_LAYER.faces.copy().astype(np.int64)
                    )
                    r_hand_faces = torch.as_tensor(
                        R_HAND_LAYER.faces.copy().astype(np.int64)
                    )
                    _validate_pen_max_hand_alignment(
                        eval_meta,
                        l_hand_vertices[max(0, int(l_hand_vertices.shape[0]) - 1)],
                        r_hand_vertices[max(0, int(r_hand_vertices.shape[0]) - 1)],
                        recording_name,
                        text_entry,
                        batch_idx,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] failed to build sample for '{text_entry}' ({variant_name}) at batch {batch_idx}: {exc}"
                    )
                    continue

                max_frames = min(
                    int(l_hand_vertices.shape[0]),
                    int(r_hand_vertices.shape[0]),
                    int(obj_vertices.shape[0]),
                )
                synced_gaze_origin = None
                synced_gaze_vector = None
                synced_gaze_mask = None
                if gaze_sample is not None:
                    if cam_pose_sample is not None:
                        world_gaze_origins, world_gaze_directions = (
                            _transform_gaze_to_world(gaze_sample, cam_pose_sample)
                        )
                        if (
                            world_gaze_origins is not None
                            and world_gaze_directions is not None
                            and world_gaze_origins.shape[0] > 0
                        ):
                            gaze_frame_limit = min(
                                max_frames, world_gaze_origins.shape[0]
                            )
                            if gaze_frame_limit > 0:
                                synced_gaze_origin = np.asarray(
                                    world_gaze_origins[gaze_frame_limit - 1],
                                    dtype=np.float32,
                                )
                                synced_gaze_vector = np.asarray(
                                    world_gaze_directions[gaze_frame_limit - 1],
                                    dtype=np.float32,
                                )
                    else:
                        (
                            synced_gaze_origin,
                            synced_gaze_vector,
                            synced_gaze_mask,
                        ) = _synced_gaze_from_gaze_map(
                            gaze_sample,
                            gaze_map_sample,
                            obj_vertices[:max_frames],
                        )
                use_left, use_right = _eval_hand_selection(
                    eval_meta,
                    text_entry,
                    left_hand_x=_to_tensor(x_lhand[batch_idx]),
                    right_hand_x=_to_tensor(x_rhand[batch_idx]),
                )

                if max_frames > 0 and variant_name == "pred":
                    final_obj_pos = _apply_offset(
                        obj_vertices[max_frames - 1], sample_offset
                    )
                    final_l_joints = _apply_offset(
                        l_hand_joints[max_frames - 1], sample_offset
                    )
                    final_r_joints = _apply_offset(
                        r_hand_joints[max_frames - 1], sample_offset
                    )
                    left_contact_joint_indices, right_contact_joint_indices = (
                        _contact_joint_index_sets_from_eval_meta(
                            eval_meta,
                            use_left=use_left,
                            use_right=use_right,
                        )
                    )
                    if (
                        not left_contact_joint_indices
                        and not right_contact_joint_indices
                    ):
                        _, left_contact_joint_indices, right_contact_joint_indices = (
                            _vr_result_group_from_final_frame(
                                final_l_joints,
                                final_r_joints,
                                final_obj_pos,
                                use_left=use_left,
                                use_right=use_right,
                                threshold_m=CONTACT_THRESHOLD_M,
                            )
                        )
                else:
                    left_contact_joint_indices, right_contact_joint_indices = (
                        set(),
                        set(),
                    )
                left_joint_distance_labels, right_joint_distance_labels = (
                    _joint_distance_labels_from_eval_meta(
                        eval_meta,
                        use_left=use_left,
                        use_right=use_right,
                        left_joint_count=int(l_hand_joints.shape[1]),
                        right_joint_count=int(r_hand_joints.shape[1]),
                    )
                )
                object_counts[object_key] += 1
                render_hand_names = set()
                if use_left:
                    render_hand_names.add("l_hand")
                if use_right:
                    render_hand_names.add("r_hand")
                enabled_hand_visuals = {
                    str(item).strip().lower()
                    for item in getattr(_CLI_ARGS, "hand_visuals", ())
                }
                sample_path = (
                    f"{_sanitize_entity_path(run_id)}/"
                    f"{_sanitize_entity_path(object_key)}/"
                    f"{_sanitize_entity_path(text_entry)}/sample_{sample_idx:03d}"
                )
                variant_path = f"{sample_path}/{variant_name}"
                frame_indices = (
                    [max_frames - 1]
                    if final_only and max_frames > 0
                    else range(max_frames)
                )
                for frame_idx in frame_indices:
                    _set_frame_time(frame_idx)
                    l_pos = _apply_offset(l_hand_vertices[frame_idx], sample_offset)
                    r_pos = _apply_offset(r_hand_vertices[frame_idx], sample_offset)
                    l_joint_pos = _apply_offset(l_hand_joints[frame_idx], sample_offset)
                    r_joint_pos = _apply_offset(r_hand_joints[frame_idx], sample_offset)
                    obj_pos = _apply_offset(obj_vertices[frame_idx], sample_offset)

                    for hand_name in sorted(render_hand_names):
                        if hand_name == "l_hand":
                            if "mesh" in enabled_hand_visuals:
                                _log_hand_mesh(
                                    variant_path,
                                    "l_hand_mesh",
                                    run_id,
                                    l_pos,
                                    l_hand_faces,
                                    mesh_color=hand_color,
                                    vertex_colors=None,
                                )
                            if "vertices" in enabled_hand_visuals:
                                _log_hand_vertices(
                                    variant_path,
                                    "l_hand_vertices",
                                    run_id,
                                    l_pos,
                                    point_color=hand_color,
                                )
                            if "joints" in enabled_hand_visuals:
                                _log_hand_joints(
                                    variant_path,
                                    hand_name,
                                    run_id,
                                    l_joint_pos,
                                    joint_color=hand_color,
                                    highlight_joint_indices=left_contact_joint_indices,
                                    highlight_color=(0, 255, 255),
                                    radius=0.0025,
                                    labels=(
                                        left_joint_distance_labels
                                        if variant_name == "pred"
                                        and frame_idx == max_frames - 1
                                        else None
                                    ),
                                )
                        else:
                            if "mesh" in enabled_hand_visuals:
                                _log_hand_mesh(
                                    variant_path,
                                    "r_hand_mesh",
                                    run_id,
                                    r_pos,
                                    r_hand_faces,
                                    mesh_color=hand_color,
                                    vertex_colors=None,
                                )
                            if "vertices" in enabled_hand_visuals:
                                _log_hand_vertices(
                                    variant_path,
                                    "r_hand_vertices",
                                    run_id,
                                    r_pos,
                                    point_color=hand_color,
                                )
                            if "joints" in enabled_hand_visuals:
                                _log_hand_joints(
                                    variant_path,
                                    hand_name,
                                    run_id,
                                    r_joint_pos,
                                    joint_color=hand_color,
                                    highlight_joint_indices=right_contact_joint_indices,
                                    highlight_color=(255, 176, 64),
                                    radius=0.0025,
                                    labels=(
                                        right_joint_distance_labels
                                        if variant_name == "pred"
                                        and frame_idx == max_frames - 1
                                        else None
                                    ),
                                )

                    obj_np = _to_numpy(obj_pos)
                    if np.isfinite(obj_np).all():
                        cov_mask_frames = _framewise_point_mask_from_cov_map(
                            cov_map_sample,
                            int(obj_np.shape[0]),
                        )
                        rr.log(
                            f"{variant_path}/obj/{run_id}",
                            rr.Points3D(
                                positions=obj_pos,
                                radii=0.0025,
                                colors=[list(object_color)],
                            ),
                        )
                        if cov_mask_frames is not None and cov_mask_frames.shape[0] > 0:
                            cov_mask = cov_mask_frames[
                                min(frame_idx, cov_mask_frames.shape[0] - 1)
                            ]
                            cov_mask = np.asarray(cov_mask, dtype=bool)
                            if cov_mask.shape[0] == obj_np.shape[0] and np.any(
                                cov_mask
                            ):
                                rr.log(
                                    f"{variant_path}/obj_contact/{run_id}",
                                    rr.Points3D(
                                        positions=obj_pos[cov_mask],
                                        radii=0.0035,
                                        colors=[[255, 220, 0]],
                                    ),
                                )
                        target_pts = _get_batch_item(target_sampled_vertices, batch_idx)
                        pred_pts = _get_batch_item(pred_sampled_vertices, batch_idx)
                        if target_pts is not None:
                            target_pts = _to_numpy(target_pts)
                            if (
                                target_pts.ndim == 3
                                and target_pts.shape[-1] == 3
                                and target_pts.shape[0] > frame_idx
                            ):
                                rr.log(
                                    f"{variant_path}/sampled_target/{run_id}",
                                    rr.Points3D(
                                        positions=_apply_offset(
                                            target_pts[frame_idx], sample_offset
                                        ),
                                        radii=0.003,
                                        colors=[[0, 0, 0]],
                                    ),
                                )
                        if pred_pts is not None:
                            pred_pts = _to_numpy(pred_pts)
                            if (
                                pred_pts.ndim == 3
                                and pred_pts.shape[-1] == 3
                                and pred_pts.shape[0] > frame_idx
                            ):
                                rr.log(
                                    f"{variant_path}/sampled_pred/{run_id}",
                                    rr.Points3D(
                                        positions=_apply_offset(
                                            pred_pts[frame_idx], sample_offset
                                        ),
                                        radii=0.0032,
                                        colors=[[255, 209, 102]],
                                    ),
                                )
                        if (
                            synced_gaze_origin is not None
                            and synced_gaze_vector is not None
                            and np.isfinite(synced_gaze_origin).all()
                            and np.isfinite(synced_gaze_vector).all()
                            and np.linalg.norm(synced_gaze_vector) > 1e-8
                        ):
                            gaze_origin = np.asarray(
                                synced_gaze_origin, dtype=np.float32
                            ).reshape(3)
                            gaze_direction = np.asarray(
                                synced_gaze_vector, dtype=np.float32
                            ).reshape(3)
                            gaze_origin = gaze_origin + np.asarray(
                                sample_offset, dtype=np.float32
                            )
                            rr.log(
                                f"{variant_path}/gaze/{run_id}",
                                rr.Arrows3D(
                                    origins=[gaze_origin],
                                    vectors=[gaze_direction],
                                    colors=[[255, 255, 0]],
                                    radii=0.002,
                                    labels=["gaze"],
                                ),
                            )
                            rr.log(
                                f"{variant_path}/gaze_origin/{run_id}",
                                rr.Points3D(
                                    positions=np.asarray(
                                        [gaze_origin], dtype=np.float32
                                    ),
                                    radii=[0.004],
                                    colors=[[255, 255, 0]],
                                    labels=["gaze_origin"],
                                ),
                            )
                            cumulative_gaze_mask = _cumulative_point_mask_upto_frame(
                                gaze_map_sample,
                                int(obj_np.shape[0]),
                                frame_idx,
                                max_frames,
                            )
                            if (
                                cumulative_gaze_mask is not None
                                and cumulative_gaze_mask.shape[0] == obj_np.shape[0]
                            ):
                                rr.log(
                                    f"{variant_path}/obj_gaze_map/{run_id}",
                                    rr.Points3D(
                                        positions=obj_pos[cumulative_gaze_mask],
                                        radii=0.0045,
                                        colors=_vertex_colors_like(
                                            obj_pos[cumulative_gaze_mask], (255, 255, 0)
                                        ),
                                    ),
                                )
                            gaze_near_mask = _gaze_proximity_mask(
                                obj_pos,
                                np.asarray([gaze_origin], dtype=np.float32),
                                np.asarray([gaze_direction], dtype=np.float32),
                                threshold_m=0.02,
                            )
                            if gaze_near_mask is not None and np.any(gaze_near_mask):
                                rr.log(
                                    f"{variant_path}/obj_gaze_near/{run_id}",
                                    rr.Points3D(
                                        positions=obj_pos[gaze_near_mask],
                                        radii=0.004,
                                        colors=_vertex_colors_like(
                                            obj_pos[gaze_near_mask], (255, 255, 0)
                                        ),
                                    ),
                                )
                        if variant_name == "pred":
                            if frame_idx == max_frames - 1:
                                contact_links = _resolve_contact_joint_links(
                                    eval_meta,
                                    l_joint_pos,
                                    r_joint_pos,
                                    obj_pos,
                                    use_left=use_left,
                                    use_right=use_right,
                                )
                                if contact_links:
                                    _log_contact_joint_links(
                                        variant_path,
                                        run_id,
                                        contact_links,
                                        static=False,
                                    )
    object_summary = ", ".join(
        f"{obj}={count}" for obj, count in sorted(object_counts.items())
    )
    print(f"[OBJECT-GROUP] {os.path.basename(recording_name)} | {object_summary}")


def main():
    args = _parse_args()
    # If the script is launched via an absolute Python path without activating
    # the env, ensure sibling CLI tools (e.g. `rerun`) are still discoverable.
    python_bin_dir = os.path.dirname(sys.executable)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin_dir and python_bin_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([python_bin_dir, *path_entries])

    try:
        rr.init("Input Data", spawn=True)
    except RuntimeError as exc:
        if "Failed to find Rerun Viewer executable in PATH" not in str(exc):
            raise
        print("Rerun Viewer executable not found in PATH; continuing with spawn=False.")
        rr.init("Input Data", spawn=False)

    offset_step = 1.0
    object_offset_step = 1.2
    run_colors = [
        (235, 87, 87),
        (47, 128, 237),
        (39, 174, 96),
        (242, 153, 74),
        (0, 163, 136),
        (155, 81, 224),
        (96, 125, 139),
    ]

    selected_objects = None
    object_slot_lookup = None

    input_files = [
        # "s_cov_map.pkl",
        # "us_cov_map.pkl",
        "s_bps_bim_cano.pkl",
        "s_bps_bim_cano_dist.pkl",
        # "s_gaze_cov_point++.pkl",
    ]
    for run_idx, file_name in enumerate(input_files):
        input_path = _resolve_input_path(file_name, args.vis_dir)
        if not os.path.exists(input_path):
            print(f"[WARN] input file not found: {input_path}")
            continue
        offset_xyz = (run_idx * offset_step, 0.0, 0.0)
        run_color = run_colors[run_idx % len(run_colors)]
        run_name = os.path.basename(input_path)
        _log_offset_anchor(run_name, offset_xyz, run_color, f"run_{run_idx}")
        visualize_rr(
            input_path,
            run_name,
            offset_xyz=offset_xyz,
            run_color=run_color,
            selected_objects=selected_objects,
            object_slot_lookup=object_slot_lookup,
            object_offset_step=object_offset_step,
            final_only=args.final_only,
        )


_CLI_ARGS = _parse_args()
OBJECT_MODEL = ObjectModel(
    os.path.join(os.path.expanduser(_CLI_ARGS.vis_dir), "obj.pkl")
)
BPS_PC = _load_bps_dict(_CLI_ARGS.data_root)
OBJ_PC = {}
OBJ_MESH = {}
for _obj_name in OBJECT_MODEL.obj_pcs.keys():
    _, _pc, _, _obj_path = OBJECT_MODEL(_obj_name)
    OBJ_PC[_obj_name] = torch.tensor(_pc)
    mesh_path = os.path.join(os.path.expanduser(_CLI_ARGS.vis_dir), _obj_path)
    try:
        OBJ_MESH[_obj_name] = trimesh.load(mesh_path, force="mesh")
    except Exception as ex:
        print(f"[WARN] failed to load object mesh for {_obj_name}: {ex}")
L_HAND_LAYER = build_mano_aa(is_rhand=False, flat_hand=False)
R_HAND_LAYER = build_mano_aa(is_rhand=True, flat_hand=False)


if __name__ == "__main__":
    main()
