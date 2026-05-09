import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import torch
import trimesh

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None
try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree = None
try:
    import tqdm
except ModuleNotFoundError:

    class _TqdmFallback:
        @staticmethod
        def tqdm(iterable=None, total=None, desc=None, leave=True):
            if iterable is not None:
                return iterable

            class _DummyPbar:
                def update(self, _n=1):
                    return None

                def close(self):
                    return None

            return _DummyPbar()

    tqdm = _TqdmFallback()

try:
    import rerun as rr
except ModuleNotFoundError:
    rr = None

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT3D_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if HOT3D_ROOT not in sys.path:
    sys.path.insert(0, HOT3D_ROOT)

from interaction_common import (
    ObjectModel,
    _extract_object_key,
    _pose9_sequence,
    _safe_mesh_volume,
    _sequence_length,
    _slice_last_frames,
    _to_numpy,
    _to_torch,
    process_hand_result_standard as process_hand_result,
    process_obj_result_standard as process_obj_result,
)
from mano import build_mano_aa
from rot import rot6d_to_axis_angle, rot6d_to_rotmat

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]


MIN_CONTACT_KEY_JOINTS = 2
IV_SUCCESS_THRESHOLD_CM3 = 100.0
_MISSING_OBJ_MESH_WARNED = False
LATETHOI_CONTACT_THRESHOLD_M = 0.005
_MESH_CACHE_VERSION = 2
_OBJECT_PC_SAMPLE_SEED = 0


def _diag_extent(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return 0.0
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return 0.0
    ext = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(ext))


def _align_mesh_scale_to_object_pc(
    mesh: trimesh.Trimesh,
    obj_pc,
    object_key: str,
    ratio_threshold: float = 3.0,
) -> trimesh.Trimesh:
    pc_diag = _diag_extent(np.asarray(obj_pc, dtype=np.float64))
    mesh_diag = _diag_extent(np.asarray(mesh.vertices, dtype=np.float64))
    if pc_diag <= 0.0 or mesh_diag <= 0.0:
        return mesh
    ratio = mesh_diag / pc_diag
    if (1.0 / ratio_threshold) <= ratio <= ratio_threshold:
        return mesh
    scale = pc_diag / mesh_diag
    mesh_aligned = mesh.copy()
    mesh_aligned.apply_scale(float(scale))
    print(
        f"[WARN] mesh scale mismatch for '{object_key}': "
        f"mesh/pc diag ratio={ratio:.4f}. applying scale {scale:.6f}."
    )
    return mesh_aligned


def _prepare_mesh_for_proximity(
    mesh: trimesh.Trimesh,
    object_key: str,
    max_faces: int = 20000,
) -> trimesh.Trimesh:
    if mesh is None or not hasattr(mesh, "faces"):
        return mesh
    try:
        face_count = int(mesh.faces.shape[0])
    except Exception:
        return mesh
    if face_count <= max_faces:
        return mesh
    try:
        face_idx = np.linspace(0, face_count - 1, max_faces, dtype=np.int64)
        faces_sampled = np.asarray(mesh.faces[face_idx], dtype=np.int64)
        unique_vertices, inverse = np.unique(
            faces_sampled.reshape(-1), return_inverse=True
        )
        vertices_sampled = np.asarray(mesh.vertices[unique_vertices], dtype=np.float64)
        faces_remapped = inverse.reshape(-1, 3).astype(np.int64)
        sampled_mesh = trimesh.Trimesh(
            vertices=vertices_sampled,
            faces=faces_remapped,
            process=False,
        )
        print(
            f"[WARN] heavy mesh '{object_key}' face-sampled for metrics: "
            f"{face_count} -> {int(sampled_mesh.faces.shape[0])} faces, "
            f"{int(mesh.vertices.shape[0])} -> {int(sampled_mesh.vertices.shape[0])} vertices."
        )
        return sampled_mesh
    except Exception as ex:
        print(
            f"[WARN] failed to face-sample heavy mesh '{object_key}' "
            f"({face_count} faces): {ex}"
        )
    return mesh


def _object_mesh_cache_dir(obj_pkl_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(obj_pkl_path)), ".mesh_cache")


def _object_mesh_cache_path(
    obj_pkl_path: str,
    mesh_path: str,
    object_key: str,
    max_faces: int,
) -> str:
    stat = os.stat(mesh_path)
    token = "|".join(
        [
            str(_MESH_CACHE_VERSION),
            os.path.abspath(mesh_path),
            str(int(stat.st_mtime_ns)),
            str(int(stat.st_size)),
            str(int(max_faces)),
        ]
    )
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    safe_key = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in object_key
    )
    return os.path.join(
        _object_mesh_cache_dir(obj_pkl_path),
        f"{safe_key}_{digest}.pkl",
    )


def _load_cached_object_mesh(
    cache_path: str,
    object_key: str,
    expected_pc_diag: float,
) -> Optional[trimesh.Trimesh]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        if float(payload.get("pc_diag", -1.0)) != float(expected_pc_diag):
            return None
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return None
        if faces.ndim != 2 or faces.shape[1] != 3:
            return None
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as ex:
        print(f"[WARN] failed to read mesh cache for '{object_key}': {ex}")
        return None


def _write_cached_object_mesh(
    cache_path: str,
    mesh: trimesh.Trimesh,
    pc_diag: float,
    object_key: str,
) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            "pc_diag": float(pc_diag),
            "vertices": np.asarray(mesh.vertices, dtype=np.float64),
            "faces": np.asarray(mesh.faces, dtype=np.int64),
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"[WARN] failed to write mesh cache for '{object_key}': {ex}")


def _resolve_object_mesh_path(
    obj_pkl_path: str,
    obj_path_value,
) -> tuple[Optional[str], list[str]]:
    raw_path = os.path.expanduser(str(obj_path_value))
    tried = []

    if os.path.isabs(raw_path):
        tried.append(raw_path)
        return (raw_path if os.path.exists(raw_path) else None), tried

    obj_pkl_dir = os.path.dirname(os.path.abspath(obj_pkl_path))
    candidates = [
        os.path.join(obj_pkl_dir, raw_path),
        os.path.join(obj_pkl_dir, os.path.basename(raw_path)),
        os.path.abspath(raw_path),
    ]

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        norm = os.path.abspath(os.path.expanduser(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        ordered_candidates.append(norm)

    tried.extend(ordered_candidates)
    for candidate in ordered_candidates:
        if os.path.exists(candidate):
            return candidate, tried
    return None, tried


def _load_object_mesh(
    obj_pkl_path: str,
    obj_path_value,
    obj_pc,
    object_key: str,
    max_faces: int = 20000,
) -> Optional[trimesh.Trimesh]:
    mesh_path, tried_paths = _resolve_object_mesh_path(obj_pkl_path, obj_path_value)
    if mesh_path is None:
        print(
            f"[WARN] failed to resolve object mesh for '{object_key}' from obj_path "
            f"'{obj_path_value}'. tried: {tried_paths}"
        )
        return None
    pc_diag = _diag_extent(np.asarray(obj_pc, dtype=np.float64))
    cache_path = _object_mesh_cache_path(
        obj_pkl_path, mesh_path, object_key=object_key, max_faces=max_faces
    )
    cached_mesh = _load_cached_object_mesh(
        cache_path, object_key=object_key, expected_pc_diag=pc_diag
    )
    if cached_mesh is not None:
        return cached_mesh
    try:
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"expected Trimesh, got {type(loaded)!r}")
        mesh = trimesh.Trimesh(
            vertices=np.asarray(loaded.vertices, dtype=np.float64),
            faces=np.asarray(loaded.faces, dtype=np.int64),
            process=False,
        )
        mesh = _align_mesh_scale_to_object_pc(mesh, obj_pc, object_key)
        mesh = _prepare_mesh_for_proximity(
            mesh, object_key=object_key, max_faces=max_faces
        )
        _write_cached_object_mesh(
            cache_path, mesh, pc_diag=pc_diag, object_key=object_key
        )
        return mesh
    except Exception as ex:
        print(
            f"[WARN] failed to load object mesh for '{object_key}' from "
            f"'{mesh_path}': {ex}"
        )
        return None


def _build_proxy_mesh_from_object_pc(
    obj_pc,
    object_key: str,
) -> Optional[trimesh.Trimesh]:
    pts = np.asarray(obj_pc, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        print(
            f"[WARN] invalid object point cloud for proxy mesh '{object_key}': {pts.shape}"
        )
        return None
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < 4:
        print(
            f"[WARN] insufficient object points for proxy mesh '{object_key}': {pts.shape[0]}"
        )
        return None

    def _finalize_proxy_mesh(mesh, method: str) -> Optional[trimesh.Trimesh]:
        if mesh is None or not hasattr(mesh, "faces"):
            return None
        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or faces.ndim != 2
                or faces.shape[1] != 3
                or faces.shape[0] == 0
            ):
                return None
            proxy_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
            )
            components = proxy_mesh.split(only_watertight=False)
            if components:
                proxy_mesh = max(components, key=lambda m: int(m.faces.shape[0]))
            face_count = int(proxy_mesh.faces.shape[0])
            print(
                f"[INFO] proxy mesh for '{object_key}' built from object point cloud "
                f"using {method}: {pts.shape[0]} points -> {face_count} faces."
            )
            return proxy_mesh
        except Exception:
            return None

    if o3d is not None and cKDTree is not None:
        try:
            tree = cKDTree(pts)
            k = min(9, pts.shape[0])
            dists, _ = tree.query(pts, k=k)
            if dists.ndim == 1:
                dists = dists[:, None]
            neighbor_dists = dists[:, 1:] if dists.shape[1] > 1 else dists[:, :0]
            finite_neighbor_dists = neighbor_dists[np.isfinite(neighbor_dists)]
            if finite_neighbor_dists.size > 0:
                base_spacing = float(np.median(finite_neighbor_dists))
                if base_spacing > 0.0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    for alpha_scale in (2.0, 2.5, 3.0, 3.5):
                        alpha = base_spacing * alpha_scale
                        if not np.isfinite(alpha) or alpha <= 0.0:
                            continue
                        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                            pcd, alpha
                        )
                        vertices = np.asarray(mesh_o3d.vertices)
                        faces = np.asarray(mesh_o3d.triangles)
                        if vertices.size == 0 or faces.size == 0:
                            continue
                        proxy_mesh = _finalize_proxy_mesh(
                            trimesh.Trimesh(
                                vertices=vertices,
                                faces=faces,
                                process=False,
                            ),
                            method=f"alpha-shape(alpha={alpha:.6f})",
                        )
                        if proxy_mesh is not None:
                            return proxy_mesh
        except Exception as ex:
            print(
                f"[WARN] alpha-shape proxy mesh reconstruction failed for "
                f"'{object_key}': {ex}"
            )
    try:
        point_cloud = trimesh.points.PointCloud(pts)
        proxy_mesh = point_cloud.convex_hull
        if proxy_mesh is None or not hasattr(proxy_mesh, "faces"):
            print(
                f"[WARN] proxy mesh reconstruction returned no faces for '{object_key}'"
            )
            return None
        proxy_mesh = _finalize_proxy_mesh(proxy_mesh, method="convex-hull")
        if proxy_mesh is not None:
            return proxy_mesh
        print(
            f"[WARN] convex-hull proxy mesh reconstruction returned invalid mesh "
            f"for '{object_key}'"
        )
        return None
    except Exception as ex:
        print(f"[WARN] failed to build proxy mesh for '{object_key}': {ex}")
        return None


def _object_pc_proxy_cache_key(obj_pc, object_key: str) -> Optional[tuple]:
    pts = np.asarray(obj_pc, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    pts = np.ascontiguousarray(pts)
    digest = hashlib.sha1(pts.view(np.uint8)).hexdigest()
    return (str(object_key).lower(), tuple(pts.shape), pts.dtype.str, digest)


def _get_or_build_proxy_mesh_from_object_pc(
    obj_pc,
    object_key: str,
    proxy_cache: dict,
) -> Optional[trimesh.Trimesh]:
    cache_key = _object_pc_proxy_cache_key(obj_pc, object_key)
    if cache_key is None:
        return _build_proxy_mesh_from_object_pc(obj_pc, object_key=object_key)
    if cache_key not in proxy_cache:
        proxy_cache[cache_key] = _build_proxy_mesh_from_object_pc(
            obj_pc, object_key=object_key
        )
    return proxy_cache[cache_key]


def _sample_object_pc_from_mesh(
    mesh: trimesh.Trimesh,
    object_key: str,
    count: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if mesh is None or count <= 0:
        return None, None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
    ):
        print(f"[WARN] invalid mesh for object pc resampling '{object_key}'")
        return None, None
    rng_state = np.random.get_state()
    np.random.seed(_OBJECT_PC_SAMPLE_SEED)
    try:
        samples, face_idx = trimesh.sample.sample_surface(mesh, int(count))
    except Exception as ex:
        print(f"[WARN] failed to resample object pc from mesh for '{object_key}': {ex}")
        return None, None
    finally:
        np.random.set_state(rng_state)
    samples = np.asarray(samples, dtype=np.float32)
    face_idx = np.asarray(face_idx, dtype=np.int64)
    if samples.ndim != 2 or samples.shape != (int(count), 3):
        print(
            f"[WARN] unexpected sampled object pc shape for '{object_key}': {samples.shape}"
        )
        return None, None
    normals = None
    try:
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        if face_normals.ndim == 2 and face_normals.shape[1] == 3:
            normals = face_normals[face_idx]
    except Exception:
        normals = None
    print(
        f"[INFO] object pc for '{object_key}' replaced with {count} mesh-sampled surface points."
    )
    return samples, normals


def _latethoi_intersect_vox_volume_m3(
    obj_vertices: np.ndarray,
    obj_faces: np.ndarray,
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    pitch: float = 0.005,
) -> float:
    try:
        obj_mesh = trimesh.Trimesh(
            vertices=np.asarray(obj_vertices, dtype=np.float64),
            faces=np.asarray(obj_faces, dtype=np.int64),
            process=False,
        )
        hand_mesh = trimesh.Trimesh(
            vertices=np.asarray(hand_vertices, dtype=np.float64),
            faces=np.asarray(hand_faces, dtype=np.int64),
            process=False,
        )
        obj_vox = obj_mesh.voxelized(pitch=float(pitch))
        obj_points = np.asarray(obj_vox.points, dtype=np.float64)
        if obj_points.ndim != 2 or obj_points.shape[0] == 0:
            return 0.0
        inside = np.asarray(hand_mesh.contains(obj_points), dtype=bool)
        return float(inside.sum()) * float(pitch**3)
    except Exception:
        return 0.0


def _transform_object_mesh_to_world(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_pose_params,
) -> Optional[trimesh.Trimesh]:
    if obj_mesh is None:
        return None
    try:
        vertices_local = np.asarray(obj_mesh.vertices, dtype=np.float64)
        faces = np.asarray(obj_mesh.faces, dtype=np.int64)
        if (
            vertices_local.ndim != 2
            or vertices_local.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[1] != 3
            or vertices_local.shape[0] == 0
        ):
            return None
        obj_pose = _pose9_sequence(obj_pose_params)
        last_pose = _to_numpy(obj_pose[-1]).astype(np.float64)
        trans = last_pose[:3]
        rot = (
            _to_numpy(rot6d_to_rotmat(_to_torch(last_pose[3:9]).reshape(1, 6)))
            .reshape(3, 3)
            .astype(np.float64)
        )
        vertices_world = np.einsum("ni,ji->nj", vertices_local, rot) + trans[None, :]
        return trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    except Exception:
        return None


def _object_overlap_region(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_pose_params,
    obj_points_world: np.ndarray,
    hand_mesh_parts: list[tuple[np.ndarray, np.ndarray]],
    voxel_pitch: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, float]:
    object_point_mask = np.zeros((int(obj_points_world.shape[0]),), dtype=bool)
    overlap_voxel_points = np.zeros((0, 3), dtype=np.float32)
    if obj_mesh is None or not hand_mesh_parts:
        return object_point_mask, overlap_voxel_points, 0.0

    obj_mesh_world = _transform_object_mesh_to_world(obj_mesh, obj_pose_params)
    if obj_mesh_world is None:
        return object_point_mask, overlap_voxel_points, 0.0

    hand_meshes = []
    for verts, faces in hand_mesh_parts:
        verts_np = np.asarray(verts, dtype=np.float64)
        faces_np = np.asarray(faces, dtype=np.int64)
        if (
            verts_np.ndim != 2
            or verts_np.shape[1] != 3
            or verts_np.shape[0] == 0
            or faces_np.ndim != 2
            or faces_np.shape[1] != 3
            or faces_np.shape[0] == 0
        ):
            continue
        hand_meshes.append(
            trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        )
    if not hand_meshes:
        return object_point_mask, overlap_voxel_points, 0.0

    try:
        obj_points_world = np.asarray(obj_points_world, dtype=np.float64)
        valid_points = (
            obj_points_world.ndim == 2
            and obj_points_world.shape[1] == 3
            and obj_points_world.shape[0] > 0
        )
        if valid_points:
            finite_mask = np.all(np.isfinite(obj_points_world), axis=1)
            valid_idx = np.flatnonzero(finite_mask)
            if valid_idx.size > 0:
                points_valid = obj_points_world[valid_idx]
                inside_any = np.zeros((points_valid.shape[0],), dtype=bool)
                for hand_mesh in hand_meshes:
                    inside_any |= np.asarray(
                        hand_mesh.contains(points_valid), dtype=bool
                    )
                object_point_mask[valid_idx] = inside_any
    except Exception:
        pass

    try:
        obj_vox = obj_mesh_world.voxelized(pitch=float(voxel_pitch))
        obj_voxel_points = np.asarray(obj_vox.points, dtype=np.float64)
        if obj_voxel_points.ndim == 2 and obj_voxel_points.shape[0] > 0:
            inside_any = np.zeros((obj_voxel_points.shape[0],), dtype=bool)
            for hand_mesh in hand_meshes:
                inside_any |= np.asarray(
                    hand_mesh.contains(obj_voxel_points), dtype=bool
                )
            overlap_voxel_points = np.asarray(
                obj_voxel_points[inside_any], dtype=np.float32
            )
            overlap_volume_m3 = float(overlap_voxel_points.shape[0]) * float(
                voxel_pitch**3
            )
            return object_point_mask, overlap_voxel_points, overlap_volume_m3
    except Exception:
        pass

    return object_point_mask, overlap_voxel_points, 0.0


def _selected_hands(text: str) -> tuple[bool, bool]:
    t = str(text).lower()
    use_right = "right" in t
    use_left = "left" in t
    if "both" in t:
        use_left = True
        use_right = True
    return use_left, use_right


def _slice_unused_hand_outputs(seq, joints, faces, keep: bool):
    if keep:
        return seq, joints, faces
    seq_empty = seq[:0] if seq is not None else None
    joints_empty = joints[:0] if joints is not None else None
    faces_empty = faces[:0] if faces is not None else None
    return seq_empty, joints_empty, faces_empty


def _normalize_motion_npy_array(motion, nsamples_hint=None) -> np.ndarray:
    """
    Normalize motion array to shape [N, T, D].
    Supports common layouts such as [N,T,D], [N,D,T], [N,D,1,T].
    """
    motion = np.asarray(motion)
    if motion.ndim < 2:
        raise ValueError(f"unsupported motion shape: {tuple(motion.shape)}")
    motion = np.squeeze(motion)
    if motion.ndim == 2:
        motion = motion[None, ...]
    if motion.ndim < 3:
        raise ValueError(f"unsupported motion shape: {tuple(motion.shape)}")

    sample_axis = 0
    if nsamples_hint is not None:
        try:
            hint = int(np.asarray(nsamples_hint).item())
            matches = [ax for ax, sz in enumerate(motion.shape) if int(sz) == hint]
            if matches:
                sample_axis = matches[0]
        except Exception:
            pass
    motion = np.moveaxis(motion, sample_axis, 0)  # [N, ...]

    tail_shape = motion.shape[1:]
    candidate_axes = [ax for ax, sz in enumerate(tail_shape, start=1) if sz >= 207]
    if not candidate_axes:
        raise ValueError(
            f"motion has no feature axis (>=207) after normalization: {tuple(motion.shape)}"
        )
    if any(motion.shape[ax] == 207 for ax in candidate_axes):
        feat_axis = next(ax for ax in candidate_axes if motion.shape[ax] == 207)
    else:
        feat_axis = min(candidate_axes, key=lambda ax: abs(int(motion.shape[ax]) - 207))

    motion = np.moveaxis(motion, feat_axis, -1)  # [N, ..., D]
    d = motion.shape[-1]
    motion = motion.reshape(motion.shape[0], -1, d)  # [N, T, D]
    return motion


def _load_items_from_path(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "save_list" in payload:
            save_list = payload["save_list"]
            if isinstance(save_list, np.ndarray):
                save_list = save_list.tolist()
            return save_list
        return payload
    if ext == ".npy":
        raw = np.load(path, allow_pickle=True)
        payload = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
        if isinstance(payload, dict) and "save_list" in payload:
            save_list = payload["save_list"]
            if isinstance(save_list, np.ndarray):
                save_list = save_list.tolist()
            return save_list
        if not isinstance(payload, dict) or "motion" not in payload:
            raise ValueError(
                f"unsupported npy payload; expected dict with 'motion' or 'save_list': {path}"
            )

        motion = _normalize_motion_npy_array(
            payload["motion"], nsamples_hint=payload.get("num_samples", None)
        )
        if motion.shape[-1] < 207:
            raise ValueError(
                f"motion last dim must be >=207 (lhand99+rhand99+obj9), got {tuple(motion.shape)}"
            )
        text_arr = payload.get("text", None)
        lengths_arr = payload.get("lengths", None)
        nsamples = min(
            int(payload.get("num_samples", motion.shape[0])), motion.shape[0]
        )

        x_obj_list, x_lhand_list, x_rhand_list, text_list = [], [], [], []
        text_np = np.asarray(text_arr, dtype=object) if text_arr is not None else None
        lengths_np = (
            np.asarray(lengths_arr).reshape(-1) if lengths_arr is not None else None
        )
        for i in range(nsamples):
            seq = motion[i]
            t = int(seq.shape[0])
            if lengths_np is not None and i < lengths_np.shape[0]:
                try:
                    t = int(lengths_np[i])
                except Exception:
                    t = int(seq.shape[0])
            t = max(1, min(t, int(seq.shape[0])))
            seq = seq[:t]

            # diffh2o motion layout: [:99]=lhand, [99:198]=rhand, [198:207]=obj
            x_lhand_list.append(seq[:, :99].astype(np.float32, copy=False))
            x_rhand_list.append(seq[:, 99:198].astype(np.float32, copy=False))
            x_obj_list.append(seq[:, 198:207].astype(np.float32, copy=False))
            if text_np is not None and i < text_np.shape[0]:
                text_list.append(str(text_np[i]))
            else:
                text_list.append("")

        return [[x_obj_list, x_lhand_list, x_rhand_list, text_list, None, None, None]]

    raise ValueError(f"unsupported input extension: {path}")


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
        or "pen_max_mm" in first
        or "pen_vertex_info" in first
        or "contact_vertex_info" in first
    )


def _looks_like_text_field(value) -> bool:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if not isinstance(value, str):
        return False
    return len(value.strip()) > 0


def _looks_like_object_meta_list(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and (
        "object_name" in first or "obj_pc_org" in first or "data_id" in first
    )


def _last_frame_contact_joint_mask(
    obj_last: np.ndarray,
    l_joints: np.ndarray,
    r_joints: np.ndarray,
    use_left: bool,
    use_right: bool,
) -> tuple[np.ndarray, np.ndarray]:
    left_mask = np.zeros((0,), dtype=bool)
    right_mask = np.zeros((0,), dtype=bool)
    if use_left and l_joints is not None and l_joints.shape[0] > 0:
        joints_last = np.asarray(l_joints[-1], dtype=np.float32)
        if (
            joints_last.ndim == 2
            and joints_last.shape[1] == 3
            and obj_last.shape[0] > 0
        ):
            dists = np.linalg.norm(
                joints_last[:, None, :] - obj_last[None, :, :], axis=2
            ).min(axis=1)
            left_mask = dists <= CONTACT_THRESHOLD_FOR_VALID
    if use_right and r_joints is not None and r_joints.shape[0] > 0:
        joints_last = np.asarray(r_joints[-1], dtype=np.float32)
        if (
            joints_last.ndim == 2
            and joints_last.shape[1] == 3
            and obj_last.shape[0] > 0
        ):
            dists = np.linalg.norm(
                joints_last[:, None, :] - obj_last[None, :, :], axis=2
            ).min(axis=1)
            right_mask = dists <= CONTACT_THRESHOLD_FOR_VALID
    return left_mask, right_mask


CONTACT_THRESHOLD_FOR_VALID = 0.02


def _last_frame_contact_joint_distances(
    obj_last: np.ndarray,
    joints: np.ndarray,
) -> np.ndarray:
    if joints is None or joints.shape[0] == 0 or obj_last.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    joints_last = np.asarray(joints[-1], dtype=np.float32)
    if joints_last.ndim != 2 or joints_last.shape[1] != 3:
        return np.zeros((0,), dtype=np.float32)
    dists = np.linalg.norm(joints_last[:, None, :] - obj_last[None, :, :], axis=2).min(
        axis=1
    )
    return dists.astype(np.float32)


def _contact_joint_indices(
    l_joints: np.ndarray,
    r_joints: np.ndarray,
    use_left: bool,
    use_right: bool,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[set[int], set[int]]:
    left_idx: set[int] = set()
    right_idx: set[int] = set()
    if use_left and l_joints is not None and l_joints.shape[0] > 0:
        for i, keep in enumerate(np.asarray(left_mask, dtype=bool).tolist()):
            if keep:
                left_idx.add(int(i))
    if use_right and r_joints is not None and r_joints.shape[0] > 0:
        for i, keep in enumerate(np.asarray(right_mask, dtype=bool).tolist()):
            if keep:
                right_idx.add(int(i))
    return left_idx, right_idx


def _set_eval_sample_time(sample_idx: int) -> None:
    if rr is None:
        return
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("sample", sample_idx)
    else:
        rr.set_time("sample", sequence=sample_idx)


def _log_rerun_status(message: str) -> None:
    if rr is None:
        return
    try:
        rr.log("eval/status", rr.TextLog(str(message)))
    except Exception:
        pass


def _start_rerun_visualization() -> bool:
    if rr is None:
        print("[WARN] rerun is not installed; skipping visualization.")
        return False
    try:
        python_bin_dir = os.path.dirname(os.path.abspath(sys.executable))
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if python_bin_dir and python_bin_dir not in path_entries:
            os.environ["PATH"] = os.pathsep.join([python_bin_dir, *path_entries])
        viewer_path = shutil.which("rerun")
        try:
            rr.init("Eval Metrics", spawn=True)
        except RuntimeError as exc:
            if "Failed to find Rerun Viewer executable in PATH" not in str(exc):
                raise
            print(
                "Rerun Viewer executable not found in PATH; continuing with spawn=False."
            )
            rr.init("Eval Metrics", spawn=False)
        rr.log("eval", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        _log_rerun_status(
            "Rerun connected. Metric ranking samples will appear under "
            "eval/<file>/<metric>/<Top|Bottom>/<sample> after each file finishes."
        )
        print(
            "[INFO] Rerun viewer started"
            + (f" via {viewer_path}" if viewer_path else "")
            + ". Ranking entities appear after each input file finishes."
        )
        return True
    except Exception as ex:
        print(f"[WARN] failed to start/connect Rerun viewer: {ex}")
        return False


def _vertex_colors_like(points: np.ndarray, rgb_color) -> np.ndarray:
    return np.tile(np.asarray(rgb_color, dtype=np.uint8), (int(points.shape[0]), 1))


def _log_eval_hand_mesh(
    path: str, vertices, faces, mesh_color, vertex_colors=None
) -> None:
    if rr is None:
        return
    vertices_np = _to_numpy(vertices).astype(np.float32)
    faces_np = _to_numpy(faces)
    if (
        vertices_np.ndim != 2
        or vertices_np.shape[1] != 3
        or not np.isfinite(vertices_np).all()
    ):
        return
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=vertices_np,
            triangle_indices=faces_np,
            vertex_normals=mesh.vertex_normals.astype(np.float32),
            vertex_colors=(
                vertex_colors
                if vertex_colors is not None
                else _vertex_colors_like(vertices_np, mesh_color)
            ),
        ),
    )


def _log_eval_hand_vertices(
    path: str,
    vertices,
    point_color,
    vertex_colors=None,
    radius: float = 0.0018,
) -> None:
    if rr is None:
        return
    vertices_np = _to_numpy(vertices).astype(np.float32)
    if (
        vertices_np.ndim != 2
        or vertices_np.shape[1] != 3
        or not np.isfinite(vertices_np).all()
    ):
        return
    rr.log(
        path,
        rr.Points3D(
            positions=vertices_np,
            radii=[radius] * int(vertices_np.shape[0]),
            colors=(
                np.asarray(vertex_colors, dtype=np.uint8)
                if vertex_colors is not None
                else _vertex_colors_like(vertices_np, point_color)
            ),
        ),
    )


def _log_eval_points(
    path: str, points, color, radius: float, labels=None, colors=None
) -> None:
    if rr is None:
        return
    points_np = _to_numpy(points).astype(np.float32)
    if (
        points_np.ndim != 2
        or points_np.shape[1] != 3
        or not np.isfinite(points_np).all()
    ):
        return
    rr.log(
        path,
        rr.Points3D(
            positions=points_np,
            radii=[radius] * int(points_np.shape[0]),
            colors=(
                np.asarray(colors, dtype=np.uint8)
                if colors is not None
                else _vertex_colors_like(points_np, color)
            ),
            labels=labels,
        ),
    )


def _log_eval_metric_line(
    path: str, hand_point, object_point, metric_value_mm: Optional[float]
) -> None:
    if rr is None:
        return
    rr.log(f"{path}/id_max_line", rr.Clear.recursive())
    rr.log(f"{path}/id_max_points", rr.Clear.recursive())
    if hand_point is None or object_point is None:
        return
    hand_point = np.asarray(hand_point, dtype=np.float32).reshape(3)
    object_point = np.asarray(object_point, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(hand_point)) or not np.all(np.isfinite(object_point)):
        return
    rr.log(
        f"{path}/id_max_line",
        rr.LineStrips3D(
            [[object_point.tolist(), hand_point.tolist()]],
            colors=[[255, 64, 64]],
            radii=0.0015,
        ),
    )
    labels = [
        "id_max object",
        (
            f"id_max hand {metric_value_mm:.2f} mm"
            if metric_value_mm is not None
            else "id_max hand"
        ),
    ]
    rr.log(
        f"{path}/id_max_points",
        rr.Points3D(
            positions=np.asarray([object_point, hand_point], dtype=np.float32),
            radii=[0.004, 0.004],
            colors=[[255, 255, 0], [255, 64, 64]],
            labels=labels,
        ),
    )


def _log_eval_metric_lines(
    path: str,
    hand_points,
    object_points,
    color,
    radius: float,
    label_prefix: str,
    metric_values_mm=None,
) -> None:
    if rr is None:
        return
    rr.log(path, rr.Clear.recursive())
    rr.log(f"{path}_points", rr.Clear.recursive())
    if hand_points is None or object_points is None:
        return
    hand_points = np.asarray(hand_points, dtype=np.float32)
    object_points = np.asarray(object_points, dtype=np.float32)
    if (
        hand_points.ndim != 2
        or object_points.ndim != 2
        or hand_points.shape != object_points.shape
        or hand_points.shape[1] != 3
    ):
        return
    valid = np.all(np.isfinite(hand_points), axis=1) & np.all(
        np.isfinite(object_points), axis=1
    )
    if not np.any(valid):
        return
    hand_valid = hand_points[valid]
    object_valid = object_points[valid]
    metric_values = None
    if metric_values_mm is not None:
        metric_values = np.asarray(metric_values_mm, dtype=np.float32)
        if metric_values.ndim == 1 and metric_values.shape[0] == valid.shape[0]:
            metric_values = metric_values[valid]
        elif metric_values.ndim != 1 or metric_values.shape[0] != hand_valid.shape[0]:
            metric_values = None
    if metric_values is None:
        metric_values = np.linalg.norm(hand_valid - object_valid, axis=1) * 1000.0
    strips = [
        [object_valid[i].tolist(), hand_valid[i].tolist()]
        for i in range(int(hand_valid.shape[0]))
    ]
    rr.log(
        path,
        rr.LineStrips3D(
            strips,
            colors=[list(color)] * len(strips),
            radii=radius,
        ),
    )
    rr.log(
        f"{path}_points",
        rr.Points3D(
            positions=np.concatenate([object_valid, hand_valid], axis=0),
            radii=[0.0025] * int(object_valid.shape[0])
            + [0.0025] * int(hand_valid.shape[0]),
            colors=[list(color)] * int(object_valid.shape[0])
            + [list(color)] * int(hand_valid.shape[0]),
            labels=[f"{label_prefix}_obj {value:.2f} mm" for value in metric_values]
            + [f"{label_prefix}_hand {value:.2f} mm" for value in metric_values],
        ),
    )


def _vertex_volume_radius(per_vertex_volume_m3: Optional[float]) -> Optional[float]:
    if per_vertex_volume_m3 is None or per_vertex_volume_m3 <= 0.0:
        return None
    return float(((3.0 * float(per_vertex_volume_m3)) / (4.0 * np.pi)) ** (1.0 / 3.0))


def _log_eval_penetration_volume(
    path: str,
    hand_points,
    per_vertex_volume_m3: Optional[float],
    color=(255, 80, 80, 180),
) -> None:
    if rr is None:
        return
    points = np.asarray(hand_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return
    radius = _vertex_volume_radius(per_vertex_volume_m3)
    if radius is None:
        return
    rr.log(
        path,
        rr.Points3D(
            positions=points,
            radii=[radius] * int(points.shape[0]),
            colors=[list(color)] * int(points.shape[0]),
        ),
    )


def _log_eval_file_summary(
    file_name: str,
    split_name: str,
    overall: dict,
) -> None:
    if rr is None:
        return
    safe_file = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(file_name)
    )
    lines = [
        f"# {file_name}",
        f"- split: {split_name}",
        f"- samples: {int(overall.get('samples', 0) or 0)}",
        f"- VR: {_format_percent_with_count(overall.get('valid_contact_rate', 0.0), overall.get('valid_samples'), overall.get('samples'))}",
        f"- CR: {_format_percent_from_ratio(overall.get('cr', 0.0))}",
        f"- SR: {_format_percent_from_ratio(overall.get('success_rate', 0.0))}",
        f"- IV: {_format_float(overall.get('iv_cm3'), digits=4)} cm^3",
        f"- ID: {_format_float(overall.get('id_mm'), digits=4)} mm",
        f"- ID_max: {_format_float(overall.get('id_max_mm'), digits=4)} mm",
    ]
    if overall.get("cr_gt") is not None:
        lines.extend(
            [
                "",
                "## GT Values",
                f"- GT CR: {_format_percent_from_ratio(overall.get('cr_gt', 0.0))}",
                f"- GT ID: {_format_float(overall.get('id_gt'), digits=4)} mm",
                f"- GT ID_max: {_format_float(overall.get('id_max_gt'), digits=4)} mm",
                f"- GT IV: {_format_float(overall.get('iv_gt'), digits=4)} cm^3",
            ]
        )
    rr.log(
        f"eval_summary/{split_name}/{safe_file}",
        rr.TextDocument("\n".join(lines), media_type="text/markdown"),
        static=True,
    )


def _format_vr_joint_distances_mm(row: dict, side: str) -> str:
    if side == "left":
        indices = row.get("contact_joint_indices_left", set()) or set()
        distances_m = row.get("left_joint_distances_m")
        prefix = "L"
    elif side == "right":
        indices = row.get("contact_joint_indices_right", set()) or set()
        distances_m = row.get("right_joint_distances_m")
        prefix = "R"
    else:
        return ""
    if distances_m is None:
        return ""
    dists = np.asarray(distances_m, dtype=np.float32)
    entries = []
    for idx in sorted(int(i) for i in indices):
        if 0 <= idx < dists.shape[0] and np.isfinite(dists[idx]):
            entries.append(f"{prefix}{idx}:{float(dists[idx]) * 1000.0:.2f}mm")
    return ",".join(entries)


def _vr_joint_label(
    idx: int,
    prefix: str,
    highlight_indices,
    distances_m,
) -> str:
    if idx not in (highlight_indices or set()):
        return ""
    if distances_m is None:
        return f"{prefix}{idx}"
    dists = np.asarray(distances_m, dtype=np.float32)
    if 0 <= idx < dists.shape[0] and np.isfinite(dists[idx]):
        return f"{prefix}{idx} {float(dists[idx]) * 1000.0:.2f}mm"
    return f"{prefix}{idx}"


def _eval_metric_label(row: dict) -> str:
    valid_contact = bool(row.get("valid_contact", False))
    success = bool(row.get("success", False))
    iv_cm3 = row.get("iv_cm3")
    pen_proxy_cm3 = row.get("penetration_proxy_cm3")
    failure_reasons = []
    if not valid_contact:
        left_n = len(row.get("contact_joint_indices_left", set()) or [])
        right_n = len(row.get("contact_joint_indices_right", set()) or [])
        failure_reasons.append(
            f"VR fail: contact joints {left_n + right_n}/{MIN_CONTACT_KEY_JOINTS}"
        )
        joint_distance_lines = []
        for side, raw_dists in (
            ("L", row.get("left_joint_distances_m")),
            ("R", row.get("right_joint_distances_m")),
        ):
            if raw_dists is None:
                continue
            dists = np.asarray(raw_dists, dtype=np.float32)
            if dists.size == 0:
                continue
            nearest = sorted(
                ((int(idx), float(dist)) for idx, dist in enumerate(dists)),
                key=lambda item: item[1],
            )[:5]
            joint_distance_lines.append(
                f"{side} joint dist mm: "
                + ", ".join(f"{idx}:{dist * 1000.0:.1f}" for idx, dist in nearest)
            )
    if iv_cm3 is not None and float(iv_cm3) > float(IV_SUCCESS_THRESHOLD_CM3):
        failure_reasons.append(
            f"IV fail: {float(iv_cm3):.2f}>{IV_SUCCESS_THRESHOLD_CM3:.2f} cm^3"
        )
    if not failure_reasons and not success:
        failure_reasons.append("SR fail")

    lines = [
        f"rank={int(row['rank'])}" if row.get("rank") is not None else None,
        (
            f"{row['ranking_bucket']} {row['ranking_metric_name']}="
            f"{float(row['ranking_metric_value']):.6f}"
            if row.get("ranking_metric_name") is not None
            and row.get("ranking_metric_value") is not None
            else None
        ),
        f"sample={int(row.get('sample_idx', -1))}",
        f"CR={float(row.get('cr', 0.0)):.4f}",
        f"VR={int(valid_contact)}",
        f"SR={int(success)}",
        f"IV={float(iv_cm3):.2f} cm^3" if iv_cm3 is not None else "IV=NA",
        (
            f"PenProxy={float(pen_proxy_cm3):.2f} cm^3"
            if pen_proxy_cm3 is not None
            else "PenProxy=NA"
        ),
        f"ID={float(row['id_mm']):.2f} mm" if row.get("id_mm") is not None else "ID=NA",
        (
            f"IDmax={float(row['id_max_mm']):.2f} mm"
            if row.get("id_max_mm") is not None
            else "IDmax=NA"
        ),
    ]
    lines = [line for line in lines if line is not None]
    lines.extend(f"FAIL: {reason}" for reason in failure_reasons)
    lines.extend(joint_distance_lines if not valid_contact else [])
    return "\n".join(lines)


def _log_eval_sample_visualization(
    file_name: str,
    split_name: str,
    text: str,
    sample_idx: int,
    object_key: str,
    obj_last: np.ndarray,
    l_last: np.ndarray,
    r_last: np.ndarray,
    l_joints_last: np.ndarray,
    r_joints_last: np.ndarray,
    l_faces: np.ndarray,
    r_faces: np.ndarray,
    metric: dict,
    use_left: bool,
    use_right: bool,
    view_name: str = "pred",
    rank: Optional[int] = None,
    path_prefix: Optional[str] = None,
    sample_label: Optional[str] = None,
    hand_style: str = "mesh",
) -> None:
    if rr is None:
        return
    if sample_label is not None:
        sample_token = sample_label
    else:
        sample_token = (
            f"rank_{int(rank):04d}_sample_{int(sample_idx):04d}"
            if rank is not None
            else f"sample_{int(sample_idx):04d}"
        )
    if path_prefix is not None:
        sample_path = f"{path_prefix}/{sample_token}"
    else:
        safe_file = os.path.splitext(os.path.basename(file_name))[0]
        sample_path = (
            f"eval/{view_name}/{split_name}/{safe_file}/{object_key}/{sample_token}"
        )
    _set_eval_sample_time(int(rank) if rank is not None else int(sample_idx))
    obj_last = np.asarray(obj_last, dtype=np.float32)
    _log_eval_points(
        f"{sample_path}/object",
        obj_last,
        color=(64, 176, 166),
        radius=0.0025,
    )

    left_highlight = metric.get("contact_joint_indices_left", set())
    right_highlight = metric.get("contact_joint_indices_right", set())
    if not use_left:
        rr.log(f"{sample_path}/left_hand", rr.Clear.recursive())
    if not use_right:
        rr.log(f"{sample_path}/right_hand", rr.Clear.recursive())
    if use_left and l_last is not None and l_last.shape[0] > 0:
        colors = _vertex_colors_like(l_last, (235, 87, 87))
        pred_contact_mask = metric.get("pred_contact_mask")
        if pred_contact_mask is not None:
            local_mask = np.asarray(pred_contact_mask[: l_last.shape[0]], dtype=bool)
            colors[local_mask] = np.asarray([255, 215, 0], dtype=np.uint8)
        local_inside = metric.get("inside_mask_left")
        if local_inside is not None:
            local_inside = np.asarray(local_inside, dtype=bool)
            colors[local_inside] = np.asarray([0, 220, 255], dtype=np.uint8)
        if hand_style == "vertices":
            rr.log(f"{sample_path}/left_hand/mesh", rr.Clear.recursive())
            _log_eval_hand_vertices(
                f"{sample_path}/left_hand/vertices",
                l_last,
                (235, 87, 87),
                colors,
            )
        else:
            rr.log(f"{sample_path}/left_hand/vertices", rr.Clear.recursive())
            _log_eval_hand_mesh(
                f"{sample_path}/left_hand/mesh", l_last, l_faces, (235, 87, 87), colors
            )
        left_joint_colors = _vertex_colors_like(l_joints_last, (235, 87, 87))
        for idx in left_highlight:
            if 0 <= int(idx) < left_joint_colors.shape[0]:
                left_joint_colors[int(idx)] = np.asarray([255, 215, 0], dtype=np.uint8)
        _log_eval_points(
            f"{sample_path}/left_hand/joints",
            l_joints_last,
            (235, 87, 87),
            0.003,
            colors=left_joint_colors,
        )

    if use_right and r_last is not None and r_last.shape[0] > 0:
        colors = _vertex_colors_like(r_last, (83, 109, 254))
        pred_contact_mask = metric.get("pred_contact_mask")
        if pred_contact_mask is not None:
            local_mask = np.asarray(pred_contact_mask[-r_last.shape[0] :], dtype=bool)
            colors[local_mask] = np.asarray([255, 215, 0], dtype=np.uint8)
        local_inside = metric.get("inside_mask_right")
        if local_inside is not None:
            local_inside = np.asarray(local_inside, dtype=bool)
            colors[local_inside] = np.asarray([0, 220, 255], dtype=np.uint8)
        if hand_style == "vertices":
            rr.log(f"{sample_path}/right_hand/mesh", rr.Clear.recursive())
            _log_eval_hand_vertices(
                f"{sample_path}/right_hand/vertices",
                r_last,
                (83, 109, 254),
                colors,
            )
        else:
            rr.log(f"{sample_path}/right_hand/vertices", rr.Clear.recursive())
            _log_eval_hand_mesh(
                f"{sample_path}/right_hand/mesh",
                r_last,
                r_faces,
                (83, 109, 254),
                colors,
            )
        right_joint_colors = _vertex_colors_like(r_joints_last, (83, 109, 254))
        for idx in right_highlight:
            if 0 <= int(idx) < right_joint_colors.shape[0]:
                right_joint_colors[int(idx)] = np.asarray([255, 215, 0], dtype=np.uint8)
        _log_eval_points(
            f"{sample_path}/right_hand/joints",
            r_joints_last,
            (83, 109, 254),
            0.003,
            colors=right_joint_colors,
        )

    label_anchor = (
        obj_last.mean(axis=0) if obj_last.size > 0 else np.zeros((3,), dtype=np.float32)
    )
    label_anchor = label_anchor + np.asarray([0.0, 0.10, 0.0], dtype=np.float32)
    _log_eval_points(
        f"{sample_path}/metrics/label",
        np.asarray([label_anchor], dtype=np.float32),
        (255, 255, 255),
        0.004,
        labels=[
            _eval_metric_label(
                {**metric, "sample_idx": sample_idx, "text": text, "rank": rank}
            )
        ],
    )
    _log_eval_metric_line(
        f"{sample_path}/metrics",
        metric.get("id_max_hand_point"),
        metric.get("id_max_object_point"),
        metric.get("id_max_mm"),
    )
    _log_eval_metric_lines(
        f"{sample_path}/metrics/id_lines",
        metric.get("id_hand_points"),
        metric.get("id_object_points"),
        color=(64, 255, 128),
        radius=0.0008,
        label_prefix="id",
        metric_values_mm=metric.get("id_depths_mm"),
    )
    _log_eval_penetration_volume(
        f"{sample_path}/metrics/penetration_volume",
        metric.get("id_hand_points"),
        metric.get("penetration_vertex_volume_m3"),
    )
    _log_eval_points(
        f"{sample_path}/metrics/object_overlap_voxels",
        metric.get("object_overlap_voxel_points"),
        color=(255, 64, 160),
        radius=0.0018,
    )


def _log_gt_metric_rankings(
    gt_rank_visualizations: list[dict],
    split_name: str,
    topk: int,
    hand_style: str = "mesh",
) -> None:
    if rr is None or not gt_rank_visualizations or topk <= 0:
        return

    split_items = [
        item for item in gt_rank_visualizations if item.get("split") == split_name
    ]
    if not split_items:
        return

    metric_specs = [
        ("CR", "cr", True),
        ("IV", "iv_cm3", False),
        ("ID", "id_mm", False),
        ("ID_max", "id_max_mm", False),
    ]

    for metric_name, metric_key, higher_is_better in metric_specs:
        valid_items = [
            item
            for item in split_items
            if item.get(metric_key) is not None
            and np.isfinite(float(item.get(metric_key)))
        ]
        if not valid_items:
            continue
        ranked = sorted(
            valid_items,
            key=lambda item: float(item[metric_key]),
            reverse=higher_is_better,
        )
        top_items = ranked[:topk]
        bottom_items = ranked[-topk:]
        bottom_items = list(reversed(bottom_items))

        for bucket_name, bucket_items in (("Top", top_items), ("Bottom", bottom_items)):
            split_label = str(split_name).capitalize()
            path_prefix = f"eval/{split_label}.G.T/{metric_name}/{bucket_name}"
            for rank, item in enumerate(bucket_items):
                _log_eval_sample_visualization(
                    *item["args"],
                    view_name="gt_ranked",
                    rank=rank,
                    path_prefix=path_prefix,
                    sample_label=(
                        f"Sample_{int(item.get('sample_idx', rank)):04d}_rank_{rank:02d}"
                    ),
                    hand_style=hand_style,
                )


def _log_pred_metric_rankings(
    pred_rank_visualizations: list[dict],
    split_name: str,
    topk: int,
    hand_style: str = "mesh",
) -> None:
    if rr is None or not pred_rank_visualizations or topk <= 0:
        return

    metric_specs = [
        ("CR", "cr", True),
        ("VR", "valid_contact", True),
        ("SR", "success", True),
        ("IV", "iv_cm3", False),
        ("ID", "id_mm", False),
        ("ID_max", "id_max_mm", False),
    ]
    for metric_name, metric_key, higher_is_better in metric_specs:
        valid_items = [
            item
            for item in pred_rank_visualizations
            if item.get(metric_key) is not None
            and np.isfinite(float(item.get(metric_key)))
        ]
        if not valid_items:
            continue
        file_name = str(valid_items[0].get("file_name", "result"))
        ranked = sorted(
            valid_items,
            key=lambda item: float(item[metric_key]),
            reverse=higher_is_better,
        )
        top_items = ranked[:topk]
        bottom_items = ranked[-topk:]
        bottom_items = list(reversed(bottom_items))
        for bucket_name, bucket_items in (("Top", top_items), ("Bottom", bottom_items)):
            safe_file = os.path.splitext(
                os.path.basename(str(bucket_items[0]["file_name"]))
            )[0]
            path_prefix = f"eval/{safe_file}/{metric_name}/{bucket_name}"
            for rank, item in enumerate(bucket_items):
                args = list(item["args"])
                args[12] = {
                    **args[12],
                    "ranking_metric_name": metric_name,
                    "ranking_metric_value": float(item[metric_key]),
                    "ranking_bucket": bucket_name,
                }
                _log_eval_sample_visualization(
                    *args,
                    view_name="pred_ranked",
                    rank=rank,
                    path_prefix=path_prefix,
                    sample_label=f"Sample_{int(item.get('sample_idx', rank)):04d}_rank_{rank:02d}",
                    hand_style=hand_style,
                )
        print(
            f"[INFO] Rerun ranking logged: {file_name} / {metric_name} "
            f"(top={len(top_items)}, bottom={len(bottom_items)})"
        )
        _log_rerun_status(
            f"Logged {file_name} / {metric_name}: "
            f"top={len(top_items)}, bottom={len(bottom_items)}"
        )


def _iter_samples_from_record(record):
    if not isinstance(record, (list, tuple)):
        return

    gt_lhand_params = None
    gt_rhand_params = None
    gt_obj_params = None
    gt_cov_map = None
    object_meta_list = None
    if len(record) >= 10 and _looks_like_eval_meta_list(record[4]):
        # New save format:
        # [x_obj, x_lhand, x_rhand, text, eval_meta_list, contact_list, pen_max_list,
        #  gt_x_obj, gt_x_lhand, gt_x_rhand, object_meta_list?]
        x_obj, course_lhand, course_rhand, text = record[:4]
        gt_obj_params = record[7] if len(record) > 7 else None
        gt_lhand_params = record[8] if len(record) > 8 else None
        gt_rhand_params = record[9] if len(record) > 9 else None
        if len(record) > 10 and _looks_like_object_meta_list(record[10]):
            object_meta_list = record[10]
    elif len(record) == 11:
        # HOT3D diffh2o save format:
        # [x_obj, x_lhand, x_rhand, text, object_meta_list, ..., ..., ..., ..., ..., object_meta_list]
        if _looks_like_object_meta_list(record[4]) or _looks_like_object_meta_list(
            record[10]
        ):
            x_obj = record[0]
            course_lhand = record[1]
            course_rhand = record[2]
            text = record[3]
            object_meta_list = (
                record[10] if _looks_like_object_meta_list(record[10]) else record[4]
            )
        # New format also handled in texthoi_vis:
        # [coarse_lhand, coarse_rhand, coarse_obj, refined_obj,
        #  refined_lhand, refined_rhand, text, gaze_map, gaze, cov_map, gt_x_obj]
        elif _looks_like_text_field(record[6]):
            x_obj = record[3]
            course_lhand = record[4]
            course_rhand = record[5]
            text = record[6]
            gt_cov_map = record[9]
            gt_obj_params = record[10]
        else:
            (
                _fine_lhand,
                _fine_rhand,
                x_obj,
                text,
                course_lhand,
                course_rhand,
                gt_obj_params,
                _cond_enc,
                gt_cov_map,
                _est_cov_map,
                _extra,
            ) = record
    elif len(record) == 10:
        (
            x_obj,
            course_lhand,
            course_rhand,
            text,
            _gaze_map,
            _gaze,
            gt_cov_map,
            gt_obj_params,
            gt_lhand_params,
            gt_rhand_params,
        ) = record
    elif len(record) == 8:
        (
            x_obj,
            course_lhand,
            course_rhand,
            text,
            _gaze_map,
            _gaze,
            gt_cov_map,
            gt_obj_params,
        ) = record
    elif len(record) == 7:
        (
            x_obj,
            course_lhand,
            course_rhand,
            text,
            _gaze_map,
            _gaze,
            gt_cov_map,
        ) = record
        gt_obj_params = None
    else:
        return

    def _batch_size(data) -> int:
        if data is None:
            return 0
        if isinstance(data, (list, tuple)):
            return len(data)
        if torch.is_tensor(data) or isinstance(data, np.ndarray):
            if data.ndim >= 3:
                return int(data.shape[0])
            if data.ndim >= 2:
                return 1
            return 0
        return 1

    def _batch_item(data, idx: int):
        if data is None:
            return None
        if isinstance(data, (list, tuple)):
            return data[idx] if len(data) > idx else None
        if torch.is_tensor(data) or isinstance(data, np.ndarray):
            if data.ndim >= 3:
                return data[idx] if data.shape[0] > idx else None
            if data.ndim >= 2:
                return data if idx == 0 else None
            return data if idx == 0 else None
        return data

    candidate_sizes = [
        _batch_size(x_obj),
        _batch_size(course_lhand),
        _batch_size(course_rhand),
        _batch_size(text),
        _batch_size(object_meta_list),
    ]
    batch_size = max(candidate_sizes) if any(s > 0 for s in candidate_sizes) else 0
    for i in range(batch_size):
        x_obj_i = _batch_item(x_obj, i)
        l_i = _batch_item(course_lhand, i)
        r_i = _batch_item(course_rhand, i)
        if x_obj_i is None or l_i is None or r_i is None:
            continue
        text_entry = _batch_item(text, i)
        if text_entry is None:
            text_entry = text
        yield {
            "text": str(text_entry),
            "obj_params": x_obj_i,
            "lhand_params": l_i,
            "rhand_params": r_i,
            "gt_cov_map": _batch_item(gt_cov_map, i),
            "gt_obj_params": _batch_item(gt_obj_params, i),
            "gt_lhand_params": _batch_item(gt_lhand_params, i),
            "gt_rhand_params": _batch_item(gt_rhand_params, i),
            "object_meta": _batch_item(object_meta_list, i),
            "sample_idx": i,
        }


def _mesh_penetration_metrics(
    obj_mesh: trimesh.Trimesh,
    obj_surface_points: Optional[np.ndarray],
    obj_surface_normals: Optional[np.ndarray],
    obj_pose_params,
    hand_points_world: np.ndarray,
    penetration_region_threshold: float = 0.01,
    compute_closest_points: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hand_points_world.shape[0] == 0:
        return (
            np.zeros((hand_points_world.shape[0],), dtype=bool),
            np.zeros((hand_points_world.shape[0],), dtype=np.float32),
            np.full((hand_points_world.shape[0], 3), np.nan, dtype=np.float32),
        )
    try:
        hand_points_world = np.asarray(hand_points_world, dtype=np.float64)
        hand_valid = np.all(np.isfinite(hand_points_world), axis=1)
        if hand_valid.sum() == 0:
            return (
                np.zeros((hand_points_world.shape[0],), dtype=bool),
                np.zeros((hand_points_world.shape[0],), dtype=np.float32),
                np.full((hand_points_world.shape[0], 3), np.nan, dtype=np.float32),
            )
        obj_pose = _pose9_sequence(obj_pose_params)
        last_pose = _to_numpy(obj_pose[-1]).astype(np.float64)
        trans = last_pose[:3]
        rot = (
            _to_numpy(rot6d_to_rotmat(_to_torch(last_pose[3:9]).reshape(1, 6)))
            .reshape(3, 3)
            .astype(np.float64)
        )
        hand_local = np.einsum("ni,ij->nj", hand_points_world[hand_valid] - trans, rot)
        inside_valid = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
        depth_valid = np.zeros((hand_local.shape[0],), dtype=np.float32)
        closest_local_valid = None
        if obj_surface_points is not None and obj_surface_normals is not None:
            surface_points = np.asarray(obj_surface_points, dtype=np.float64)
            surface_normals = np.asarray(obj_surface_normals, dtype=np.float64)
            valid_surface_shape = (
                surface_points.ndim == 2
                and surface_points.shape[1] == 3
                and surface_normals.shape == surface_points.shape
                and surface_points.shape[0] > 0
            )
            if valid_surface_shape:
                normal_lengths = np.linalg.norm(surface_normals, axis=1)
                valid_surface = (
                    np.all(np.isfinite(surface_points), axis=1)
                    & np.all(np.isfinite(surface_normals), axis=1)
                    & np.isfinite(normal_lengths)
                    & (normal_lengths > 1e-8)
                )
                surface_points = surface_points[valid_surface]
                surface_normals = surface_normals[valid_surface]
                normal_lengths = normal_lengths[valid_surface]
            if valid_surface_shape and surface_points.shape[0] > 0:
                nearest_surface_idx = np.argmin(
                    np.linalg.norm(
                        hand_local[:, None, :] - surface_points[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                nearest_surface_points = surface_points[nearest_surface_idx]
                nearest_surface_normals = surface_normals[nearest_surface_idx]
                nearest_surface_normals = (
                    nearest_surface_normals / normal_lengths[nearest_surface_idx, None]
                )
                surface_signed = np.sum(
                    (hand_local - nearest_surface_points) * nearest_surface_normals,
                    axis=1,
                )
                inside_valid = inside_valid & (surface_signed < 0.0)
                depth_valid[inside_valid] = np.abs(surface_signed[inside_valid]).astype(
                    np.float32
                )
                if compute_closest_points:
                    closest_local_valid = np.full(
                        (hand_local.shape[0], 3), np.nan, dtype=np.float64
                    )
                    closest_local_valid[inside_valid] = (
                        hand_local[inside_valid]
                        - surface_signed[inside_valid, None]
                        * nearest_surface_normals[inside_valid]
                    )

                # Restrict penetration depth to the locally penetrated entry region.
                # Without this, thin objects can underestimate depth because an inside
                # point may be closer to the opposite wall than to the surface it
                # actually penetrated through.
                region_threshold = float(max(penetration_region_threshold, 1e-4))
                entry_surface_mask = (~inside_valid) & (
                    np.abs(surface_signed) <= region_threshold
                )
                candidate_surface_idx = np.unique(
                    nearest_surface_idx[entry_surface_mask]
                )

                if candidate_surface_idx.size == 0:
                    expanded_entry_mask = (~inside_valid) & (
                        np.abs(surface_signed) <= (region_threshold * 2.0)
                    )
                    candidate_surface_idx = np.unique(
                        nearest_surface_idx[expanded_entry_mask]
                    )

                if candidate_surface_idx.size > 0 and np.any(inside_valid):
                    candidate_surface_points = surface_points[candidate_surface_idx]
                    candidate_surface_normals = surface_normals[candidate_surface_idx]
                    candidate_normal_lengths = np.linalg.norm(
                        candidate_surface_normals, axis=1, keepdims=True
                    )
                    candidate_surface_normals = (
                        candidate_surface_normals / candidate_normal_lengths
                    )

                    inside_indices = np.flatnonzero(inside_valid)
                    inside_points = hand_local[inside_indices]
                    rel = (
                        inside_points[:, None, :] - candidate_surface_points[None, :, :]
                    )
                    candidate_signed = np.sum(
                        rel * candidate_surface_normals[None, :, :], axis=2
                    )
                    candidate_depth = np.abs(candidate_signed)
                    candidate_depth[candidate_signed >= 0.0] = np.inf

                    best_candidate_idx = np.argmin(candidate_depth, axis=1)
                    best_candidate_depth = candidate_depth[
                        np.arange(candidate_depth.shape[0]), best_candidate_idx
                    ]
                    valid_candidate = np.isfinite(best_candidate_depth)

                    if np.any(valid_candidate):
                        chosen_normals = candidate_surface_normals[
                            best_candidate_idx[valid_candidate]
                        ]
                        chosen_signed = candidate_signed[
                            np.arange(candidate_signed.shape[0])[valid_candidate],
                            best_candidate_idx[valid_candidate],
                        ]
                        chosen_points = candidate_surface_points[
                            best_candidate_idx[valid_candidate]
                        ]
                        target_inside_indices = inside_indices[valid_candidate]
                        depth_valid[target_inside_indices] = np.abs(
                            chosen_signed
                        ).astype(np.float32)
                        if compute_closest_points:
                            if closest_local_valid is None:
                                closest_local_valid = np.full(
                                    (hand_local.shape[0], 3),
                                    np.nan,
                                    dtype=np.float64,
                                )
                            closest_local_valid[target_inside_indices] = (
                                inside_points[valid_candidate]
                                - chosen_signed[:, None] * chosen_normals
                            )
        if closest_local_valid is None and np.any(inside_valid):
            closest_local, _, _ = trimesh.proximity.closest_point(
                obj_mesh, hand_local[inside_valid]
            )
            closest_local = np.asarray(closest_local, dtype=np.float64)
            depth_valid[inside_valid] = np.linalg.norm(
                hand_local[inside_valid] - closest_local,
                axis=1,
            ).astype(np.float32)
            if compute_closest_points:
                closest_local_valid = np.full(
                    (hand_local.shape[0], 3), np.nan, dtype=np.float64
                )
                closest_local_valid[inside_valid] = closest_local

        closest_points_world = np.full(
            (hand_points_world.shape[0], 3), np.nan, dtype=np.float32
        )
        if closest_local_valid is not None and np.any(np.isfinite(closest_local_valid)):
            finite_rows = np.all(np.isfinite(closest_local_valid), axis=1)
            if np.any(finite_rows):
                closest_world_subset = (
                    np.einsum("ni,ji->nj", closest_local_valid[finite_rows], rot)
                    + trans[None, :]
                ).astype(np.float32)
                valid_idx = np.flatnonzero(hand_valid)
                closest_points_world[valid_idx[finite_rows]] = closest_world_subset
        inside = np.zeros((hand_points_world.shape[0],), dtype=bool)
        depth = np.zeros((hand_points_world.shape[0],), dtype=np.float32)
        valid_idx = np.flatnonzero(hand_valid)
        inside[valid_idx] = inside_valid
        depth[valid_idx] = depth_valid
        return inside, depth, closest_points_world
    except Exception as ex:
        raise RuntimeError(
            "trimesh penetration query failed; install required dependencies "
            "(e.g. rtree) and verify mesh validity."
        ) from ex


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

    point_axis = point_axes[0]
    arr_p = np.moveaxis(arr, point_axis, -1)  # [..., N]
    if arr_p.ndim == 1 and arr_p.shape[0] == num_points:
        return (arr_p > 0).reshape(1, num_points)

    non_point_shape = arr_p.shape[:-1]
    if len(non_point_shape) == 0:
        return (arr_p > 0).reshape(1, num_points)
    frame_axis = int(np.argmax(non_point_shape))
    arr_fp = np.moveaxis(arr_p, frame_axis, 0)  # [T, ..., N]
    if arr_fp.ndim > 2:
        reduce_axes = tuple(range(1, arr_fp.ndim - 1))
        arr_fp = (arr_fp > 0).any(axis=reduce_axes)
    else:
        arr_fp = arr_fp > 0
    if arr_fp.ndim != 2 or arr_fp.shape[1] != num_points:
        return None
    return arr_fp.astype(bool)


def _contact_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "gt_precision": float(precision),
        "gt_recall": float(recall),
        "gt_f1": float(f1),
        "gt_iou": float(iou),
    }


def _latethoi_lastframe_hand_metrics(
    obj_seq: np.ndarray,
    obj_pose_params,
    obj_mesh: Optional[trimesh.Trimesh],
    hand_seq: np.ndarray,
    hand_faces: np.ndarray,
    contact_threshold: float = LATETHOI_CONTACT_THRESHOLD_M,
) -> Optional[dict]:
    if (
        obj_mesh is None
        or obj_seq is None
        or hand_seq is None
        or obj_seq.shape[0] == 0
        or hand_seq.shape[0] == 0
        or hand_faces is None
        or len(hand_faces) == 0
    ):
        return None

    try:
        obj_pose = _pose9_sequence(obj_pose_params)
        last_pose = _to_numpy(obj_pose[-1]).astype(np.float64)
        trans = last_pose[:3]
        rot = (
            _to_numpy(rot6d_to_rotmat(_to_torch(last_pose[3:9]).reshape(1, 6)))
            .reshape(3, 3)
            .astype(np.float64)
        )
        obj_last = np.asarray(obj_seq[-1], dtype=np.float64)
        hand_last = np.asarray(hand_seq[-1], dtype=np.float64)
        if hand_last.ndim != 2 or hand_last.shape[1] != 3 or hand_last.shape[0] == 0:
            return None

        hand_local = np.einsum("ni,ij->nj", hand_last - trans, rot)
        inside_mask = np.asarray(obj_mesh.contains(hand_local), dtype=bool)

        if not inside_mask.any():
            volume_m3 = 0.0
            max_depth_m = 0.0
            contact_ratio = 0.0
        else:
            closest_local, closest_dist, _ = trimesh.proximity.closest_point(
                obj_mesh, hand_local
            )
            closest_dist = np.asarray(closest_dist, dtype=np.float64)
            max_depth_m = (
                float(np.max(closest_dist[inside_mask])) if inside_mask.any() else 0.0
            )
            volume_m3 = _latethoi_intersect_vox_volume_m3(
                obj_last,
                np.asarray(obj_mesh.faces, dtype=np.int64),
                hand_last,
                np.asarray(hand_faces, dtype=np.int64),
                pitch=0.005,
            )
            contact_ratio = float(np.mean(closest_dist < float(contact_threshold)))

        return {
            "inter_volume_mean": float(volume_m3),
            "inter_volume_contact": float(volume_m3),
            "inter_volume_max": float(volume_m3),
            "inter_depth_mean": float(max_depth_m),
            "inter_depth_contact": float(max_depth_m),
            "inter_depth_max": float(max_depth_m),
            "contact_ratio_mean": float(contact_ratio),
            "contact_ratio_contact": float(contact_ratio),
            "contact_ratio_max": float(contact_ratio),
            "contact_ratio_off_ground": float(contact_ratio),
        }
    except Exception:
        return None


def _latethoi_lastframe_sample_metrics(
    obj_seq: np.ndarray,
    obj_pose_params,
    obj_mesh: Optional[trimesh.Trimesh],
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    l_faces: np.ndarray,
    r_faces: np.ndarray,
    use_left: bool,
    use_right: bool,
) -> dict:
    per_hand = {}
    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        left_metrics = _latethoi_lastframe_hand_metrics(
            obj_seq, obj_pose_params, obj_mesh, l_seq, l_faces
        )
        if left_metrics is not None:
            per_hand["lhand"] = left_metrics
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        right_metrics = _latethoi_lastframe_hand_metrics(
            obj_seq, obj_pose_params, obj_mesh, r_seq, r_faces
        )
        if right_metrics is not None:
            per_hand["rhand"] = right_metrics

    if not per_hand:
        return {
            "rhand": {},
            "lhand": {},
            "off_ground_contact_ratio": None,
            "off_ground_contact_rate": None,
            "jerk": None,
            "hand_metric_mean": {},
        }

    metric_keys = sorted({k for hand in per_hand.values() for k in hand.keys()})
    hand_metric_mean = {}
    for key in metric_keys:
        vals = [
            float(hand[key]) for hand in per_hand.values() if hand.get(key) is not None
        ]
        hand_metric_mean[key] = float(np.mean(vals)) if vals else None

    return {
        "rhand": per_hand.get("rhand", {}),
        "lhand": per_hand.get("lhand", {}),
        "off_ground_contact_ratio": None,
        "off_ground_contact_rate": None,
        "jerk": None,
        "hand_metric_mean": hand_metric_mean,
    }


def _text2hoi_hand_metrics(
    obj_points: np.ndarray,
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    hand_joints: np.ndarray,
    contact_threshold: float = 0.02,
) -> Optional[dict]:
    obj_points = np.asarray(obj_points, dtype=np.float64)
    hand_vertices = np.asarray(hand_vertices, dtype=np.float64)
    hand_faces = np.asarray(hand_faces, dtype=np.int64)
    hand_joints = np.asarray(hand_joints, dtype=np.float64)
    if (
        obj_points.ndim != 2
        or obj_points.shape[1] != 3
        or obj_points.shape[0] == 0
        or hand_vertices.ndim != 2
        or hand_vertices.shape[1] != 3
        or hand_vertices.shape[0] == 0
        or hand_faces.ndim != 2
        or hand_faces.shape[1] != 3
    ):
        return None
    try:
        hand_mesh = trimesh.Trimesh(
            vertices=hand_vertices, faces=hand_faces, process=False
        )
        hand_normals = np.asarray(hand_mesh.vertex_normals, dtype=np.float64)
        nearest_idx = np.argmin(
            np.linalg.norm(obj_points[:, None, :] - hand_vertices[None, :, :], axis=2),
            axis=1,
        )
        nearest_hand_vertices = hand_vertices[nearest_idx]
        nearest_hand_normals = hand_normals[nearest_idx]
        nn_vector = nearest_hand_vertices - obj_points
        interior = np.sum(nn_vector * nearest_hand_normals, axis=1) > 0.0
        distances = np.linalg.norm(nn_vector, axis=1)
        penetration_loss_m = (
            float(np.mean(distances[interior])) if np.any(interior) else 0.0
        )
        penetration_max_m = (
            float(np.max(distances[interior])) if np.any(interior) else 0.0
        )

        if (
            hand_joints.ndim == 2
            and hand_joints.shape[1] == 3
            and hand_joints.shape[0] > 0
        ):
            joint_dist = np.linalg.norm(
                obj_points[:, None, :] - hand_joints[None, :, :], axis=2
            )
            contact_obj_mask = np.any(joint_dist < float(contact_threshold), axis=1)
            contact_joint_mask = np.any(joint_dist < float(contact_threshold), axis=0)
            contact_obj_ratio = float(np.mean(contact_obj_mask))
            contact_joint_ratio = float(np.mean(contact_joint_mask))
        else:
            contact_obj_ratio = 0.0
            contact_joint_ratio = 0.0

        return {
            "penetration_loss_m": penetration_loss_m,
            "penetration_max_m": penetration_max_m,
            "interior_object_ratio": float(np.mean(interior)),
            "contact_object_ratio": contact_obj_ratio,
            "contact_joint_ratio": contact_joint_ratio,
        }
    except Exception:
        return None


def _text2hoi_sample_metrics(
    obj_seq: np.ndarray,
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    l_joints: np.ndarray,
    r_joints: np.ndarray,
    l_faces: np.ndarray,
    r_faces: np.ndarray,
    use_left: bool,
    use_right: bool,
) -> dict:
    if obj_seq is None or obj_seq.shape[0] == 0:
        return {"rhand": {}, "lhand": {}, "hand_metric_mean": {}}
    obj_last = np.asarray(obj_seq[-1], dtype=np.float64)
    per_hand = {}
    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        left = _text2hoi_hand_metrics(
            obj_last,
            l_seq[-1],
            l_faces,
            l_joints[-1] if l_joints is not None and l_joints.shape[0] > 0 else None,
        )
        if left is not None:
            per_hand["lhand"] = left
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        right = _text2hoi_hand_metrics(
            obj_last,
            r_seq[-1],
            r_faces,
            r_joints[-1] if r_joints is not None and r_joints.shape[0] > 0 else None,
        )
        if right is not None:
            per_hand["rhand"] = right
    metric_keys = [
        "penetration_loss_m",
        "penetration_max_m",
        "interior_object_ratio",
        "contact_object_ratio",
        "contact_joint_ratio",
    ]
    hand_metric_mean = {}
    for key in metric_keys:
        vals = [metrics[key] for metrics in per_hand.values() if key in metrics]
        hand_metric_mean[key] = float(np.mean(vals)) if vals else None
    return {
        "rhand": per_hand.get("rhand", {}),
        "lhand": per_hand.get("lhand", {}),
        "hand_metric_mean": hand_metric_mean,
    }


def _diffh2o_native_hand_metrics(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_params,
    hand_seq: np.ndarray,
    max_frames: int,
    contact_threshold: float = 0.005,
) -> Optional[dict]:
    if (
        obj_mesh is None
        or obj_params is None
        or hand_seq is None
        or hand_seq.shape[0] == 0
    ):
        return None
    nframes = min(_sequence_length(obj_params), int(hand_seq.shape[0]))
    if nframes <= 0:
        return None
    indices = _sampled_eval_indices(nframes, max_frames)
    pose = _pose9_sequence(_slice_frame_indices(obj_params, indices))
    hand_eval = np.asarray(hand_seq[indices], dtype=np.float64)
    if hand_eval.ndim != 3 or hand_eval.shape[2] != 3:
        return None

    iv_counts = []
    id_vals = []
    cr_vals = []
    for frame_idx in range(len(indices)):
        pose_t = _to_numpy(pose[frame_idx]).astype(np.float64)
        trans = pose_t[:3]
        rot = (
            _to_numpy(rot6d_to_rotmat(_to_torch(pose_t[3:9]).reshape(1, 6)))
            .reshape(3, 3)
            .astype(np.float64)
        )
        hand_world = hand_eval[frame_idx]
        if hand_world.ndim != 2 or hand_world.shape[1] != 3 or hand_world.shape[0] == 0:
            continue
        hand_local = np.einsum("ni,ij->nj", hand_world - trans, rot)
        inside_mask = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
        try:
            _closest_local, closest_dist, _closest_tri = trimesh.proximity.closest_point(
                obj_mesh, hand_local
            )
            closest_dist = np.asarray(closest_dist, dtype=np.float64)
        except Exception:
            closest_dist = np.full((hand_local.shape[0],), np.inf, dtype=np.float64)
        iv_counts.append(float(np.count_nonzero(inside_mask)))
        id_vals.append(
            float(np.max(closest_dist[inside_mask])) if np.any(inside_mask) else 0.0
        )
        cr_vals.append(float(np.mean(closest_dist < float(contact_threshold))))

    if not iv_counts:
        return None
    return {
        "iv_count_mean": float(np.mean(iv_counts)),
        "iv_count_max": float(np.max(iv_counts)),
        "inter_depth_mean": float(np.mean(id_vals)) if id_vals else 0.0,
        "inter_depth_max": float(np.max(id_vals)) if id_vals else 0.0,
        "contact_ratio_mean": float(np.mean(cr_vals)) if cr_vals else 0.0,
        "contact_ratio_max": float(np.max(cr_vals)) if cr_vals else 0.0,
    }


def _diffh2o_native_sample_metrics(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_params,
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    use_left: bool,
    use_right: bool,
    max_frames: int,
) -> dict:
    per_hand = {}
    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        left = _diffh2o_native_hand_metrics(
            obj_mesh, obj_params, l_seq, max_frames=max_frames
        )
        if left is not None:
            per_hand["lhand"] = left
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        right = _diffh2o_native_hand_metrics(
            obj_mesh, obj_params, r_seq, max_frames=max_frames
        )
        if right is not None:
            per_hand["rhand"] = right
    metric_keys = [
        "iv_count_mean",
        "iv_count_max",
        "inter_depth_mean",
        "inter_depth_max",
        "contact_ratio_mean",
        "contact_ratio_max",
    ]
    hand_metric_mean = {}
    for key in metric_keys:
        vals = [metrics[key] for metrics in per_hand.values() if key in metrics]
        hand_metric_mean[key] = float(np.mean(vals)) if vals else None
    return {
        "rhand": per_hand.get("rhand", {}),
        "lhand": per_hand.get("lhand", {}),
        "hand_metric_mean": hand_metric_mean,
    }


def _sample_metrics(
    obj_seq: np.ndarray,
    obj_pose_params,
    obj_mesh: Optional[trimesh.Trimesh],
    obj_surface_points: Optional[np.ndarray],
    obj_surface_normals: Optional[np.ndarray],
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    l_joints: np.ndarray,
    r_joints: np.ndarray,
    l_faces: np.ndarray,
    r_faces: np.ndarray,
    use_left: bool,
    use_right: bool,
    contact_threshold: float,
    compute_id: bool = True,
    compute_iv: bool = True,
    compute_closest_points: bool = True,
) -> Optional[dict]:
    if obj_seq.shape[0] == 0:
        return None

    # last-frame interaction for CR/ID/IV
    obj_last = obj_seq[-1]
    hand_last_parts = []
    hand_volume_parts = []

    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        l_last = l_seq[-1]
        hand_last_parts.append((l_last, "left"))
        hand_volume_parts.append((l_last, l_faces))
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        r_last = r_seq[-1]
        hand_last_parts.append((r_last, "right"))
        hand_volume_parts.append((r_last, r_faces))

    if not hand_last_parts:
        return None

    hand_last = np.concatenate([x[0] for x in hand_last_parts], axis=0)
    left_count = (
        int(l_seq[-1].shape[0])
        if use_left and l_seq is not None and l_seq.shape[0] > 0
        else 0
    )
    right_count = (
        int(r_seq[-1].shape[0])
        if use_right and r_seq is not None and r_seq.shape[0] > 0
        else 0
    )

    obj_mesh_for_metrics = obj_mesh
    if obj_mesh_for_metrics is not None:
        inside_mask, inside_depth_m, closest_surface_points = _mesh_penetration_metrics(
            obj_mesh_for_metrics,
            obj_surface_points,
            obj_surface_normals,
            obj_pose_params,
            hand_last,
            penetration_region_threshold=max(float(contact_threshold) * 2.0, 0.01),
            compute_closest_points=bool(compute_closest_points or compute_id),
        )
    else:
        global _MISSING_OBJ_MESH_WARNED
        if compute_id or compute_iv:
            if not _MISSING_OBJ_MESH_WARNED:
                print(
                    "[WARN] object mesh missing from obj.pkl mapping; IV/ID cannot be computed for those samples."
                )
                _MISSING_OBJ_MESH_WARNED = True
        inside_mask = np.zeros((hand_last.shape[0],), dtype=bool)
        inside_depth_m = np.zeros((hand_last.shape[0],), dtype=np.float32)
        closest_surface_points = np.full(
            (hand_last.shape[0], 3), np.nan, dtype=np.float32
        )
    inside_hand_points = (
        hand_last[inside_mask].astype(np.float32)
        if inside_mask.any()
        else np.zeros((0, 3), dtype=np.float32)
    )
    object_penetration_mask = np.zeros((obj_last.shape[0],), dtype=bool)
    object_overlap_voxel_points = np.zeros((0, 3), dtype=np.float32)
    overlap_volume_cm3 = None
    if obj_mesh_for_metrics is not None and bool(compute_closest_points):
        (
            object_penetration_mask,
            object_overlap_voxel_points,
            overlap_volume_m3,
        ) = _object_overlap_region(
            obj_mesh_for_metrics,
            obj_pose_params,
            obj_last,
            hand_volume_parts,
        )
        overlap_volume_cm3 = float(overlap_volume_m3 * 1e6)

    if inside_hand_points.shape[0] > 0:
        inside_depths_mm = (inside_depth_m[inside_mask] * 1000.0).astype(np.float32)
        inside_object_points = closest_surface_points[inside_mask].astype(np.float32)
        finite_object_points = np.all(np.isfinite(inside_object_points), axis=1)
        if (
            bool(compute_closest_points)
            and not np.all(finite_object_points)
            and object_overlap_voxel_points.shape[0] > 0
        ):
            missing_hand_points = inside_hand_points[~finite_object_points]
            nearest_overlap_idx = np.argmin(
                np.linalg.norm(
                    missing_hand_points[:, None, :]
                    - object_overlap_voxel_points[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            inside_object_points[~finite_object_points] = np.asarray(
                object_overlap_voxel_points[nearest_overlap_idx], dtype=np.float32
            )
    else:
        inside_object_points = np.zeros((0, 3), dtype=np.float32)
        inside_depths_mm = np.zeros((0,), dtype=np.float32)
    penetration_vertex_volume_m3 = None
    penetration_proxy_cm3 = None

    # CR: ratio of last-frame hand vertices within threshold of object vertices.
    if hand_last.shape[0] == 0:
        pred_contact_mask = np.zeros((0,), dtype=bool)
        cr = 0.0
    else:
        dists = np.linalg.norm(
            hand_last[:, None, :] - obj_last[None, :, :], axis=2
        ).min(axis=1)
        pred_contact_mask = dists <= contact_threshold
        cr = float(pred_contact_mask.mean())

    left_contact_mask, right_contact_mask = _last_frame_contact_joint_mask(
        obj_last,
        l_joints,
        r_joints,
        use_left,
        use_right,
    )
    left_joint_distances_m = (
        _last_frame_contact_joint_distances(obj_last, l_joints)
        if use_left
        else np.zeros((0,), dtype=np.float32)
    )
    right_joint_distances_m = (
        _last_frame_contact_joint_distances(obj_last, r_joints)
        if use_right
        else np.zeros((0,), dtype=np.float32)
    )
    valid_contact = (
        int(np.count_nonzero(left_contact_mask))
        + int(np.count_nonzero(right_contact_mask))
    ) >= MIN_CONTACT_KEY_JOINTS
    left_contact_joint_indices, right_contact_joint_indices = _contact_joint_indices(
        l_joints,
        r_joints,
        use_left,
        use_right,
        left_contact_mask,
        right_contact_mask,
    )

    id_max_hand_point = None
    id_max_object_point = None
    if inside_hand_points.shape[0] > 0 and inside_depths_mm.shape[0] > 0:
        max_local_idx = int(np.argmax(inside_depths_mm))
        id_max_hand_point = inside_hand_points[max_local_idx].astype(np.float32)
        if inside_object_points.shape[0] > max_local_idx and np.all(
            np.isfinite(inside_object_points[max_local_idx])
        ):
            id_max_object_point = inside_object_points[max_local_idx].astype(np.float32)

    # 4) ID [mm]: optional because mesh signed-distance queries dominate runtime.
    if compute_id and obj_mesh_for_metrics is not None:
        id_mm = float(inside_depths_mm.mean()) if inside_depths_mm.shape[0] > 0 else 0.0
        id_max_mm = (
            float(inside_depths_mm.max()) if inside_depths_mm.shape[0] > 0 else 0.0
        )
    else:
        id_mm = None
        id_max_mm = None

    # 3) IV [cm^3]: optional because trimesh volume/proxy work is not needed
    # when only contact and penetration depth metrics are requested.
    iv_cm3 = None
    if compute_iv:
        iv_m3 = 0.0
        cursor = 0
        for verts, faces in hand_volume_parts:
            n = verts.shape[0]
            local_inside = inside_mask[cursor : cursor + n]
            cursor += n
            mesh_vol = _safe_mesh_volume(verts, faces)
            if mesh_vol <= 0.0 or n <= 0:
                continue
            per_vertex_volume_m3 = mesh_vol / float(n)
            iv_m3 += per_vertex_volume_m3 * float(local_inside.sum())
            if penetration_vertex_volume_m3 is None:
                penetration_vertex_volume_m3 = per_vertex_volume_m3
        iv_cm3 = float(iv_m3 * 1e6)
        if use_left and use_right:
            iv_cm3 *= 0.5
        penetration_proxy_cm3 = iv_cm3

    return {
        "iv_cm3": iv_cm3,
        "id_mm": id_mm,
        "id_max_mm": id_max_mm,
        "cr": cr,
        "valid_contact": bool(valid_contact),
        "success": (
            bool(valid_contact)
            if iv_cm3 is None
            else bool(valid_contact)
            and float(iv_cm3) <= float(IV_SUCCESS_THRESHOLD_CM3)
        ),
        "pred_contact_mask": pred_contact_mask,
        "inside_mask": inside_mask,
        "inside_mask_left": inside_mask[:left_count].copy(),
        "inside_mask_right": inside_mask[left_count : left_count + right_count].copy(),
        "object_penetration_mask": object_penetration_mask,
        "left_contact_mask": left_contact_mask,
        "right_contact_mask": right_contact_mask,
        "left_joint_distances_m": left_joint_distances_m,
        "right_joint_distances_m": right_joint_distances_m,
        "contact_joint_indices_left": left_contact_joint_indices,
        "contact_joint_indices_right": right_contact_joint_indices,
        "id_max_hand_point": id_max_hand_point,
        "id_max_object_point": id_max_object_point,
        "id_hand_points": inside_hand_points,
        "id_object_points": inside_object_points,
        "id_depths_mm": inside_depths_mm,
        "penetration_proxy_cm3": penetration_proxy_cm3,
        "penetration_vertex_volume_m3": penetration_vertex_volume_m3,
        "object_overlap_voxel_points": object_overlap_voxel_points,
        "object_overlap_volume_cm3": overlap_volume_cm3,
    }


def _frame_rotmat_from_params(params, frame_idx: int = 0) -> Optional[np.ndarray]:
    t = _to_torch(params)
    if t.ndim == 1:
        if t.shape[0] < 9:
            return None
        t = t.unsqueeze(0)
    if t.ndim != 2 or t.shape[1] < 9 or t.shape[0] <= 0:
        return None
    idx = frame_idx if frame_idx >= 0 else (t.shape[0] + frame_idx)
    idx = max(0, min(idx, t.shape[0] - 1))
    rot6d = t[idx, 3:9].reshape(1, 6)
    rot = rot6d_to_rotmat(rot6d).reshape(3, 3)
    return _to_numpy(rot).astype(np.float64)


def _relative_hand_object_rotmats_first_frame(
    obj_params,
    lhand_params,
    rhand_params,
    use_left: bool,
    use_right: bool,
) -> list[np.ndarray]:
    return _relative_hand_object_rotmats_at_frame(
        obj_params, lhand_params, rhand_params, use_left, use_right, frame_idx=0
    )


def _relative_hand_object_rotmats_at_frame(
    obj_params,
    lhand_params,
    rhand_params,
    use_left: bool,
    use_right: bool,
    frame_idx: int,
) -> list[np.ndarray]:
    out = []
    obj_rot = _frame_rotmat_from_params(obj_params, frame_idx=frame_idx)
    if obj_rot is None:
        return out
    if use_left:
        l_rot = _frame_rotmat_from_params(lhand_params, frame_idx=frame_idx)
        if l_rot is not None:
            out.append(obj_rot.T @ l_rot)
    if use_right:
        r_rot = _frame_rotmat_from_params(rhand_params, frame_idx=frame_idx)
        if r_rot is not None:
            out.append(obj_rot.T @ r_rot)
    return out


def _rotation_geodesic_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    rel = r1.T @ r2
    cos_theta = (np.trace(rel) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _mean_pairwise_rotation_distance_deg(rotmats: list[np.ndarray]) -> float:
    n = len(rotmats)
    if n < 2:
        return 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(_rotation_geodesic_deg(rotmats[i], rotmats[j]))
    return float(np.mean(vals)) if vals else 0.0


def _object_relative_rotation_diversity_map(
    rows: list[dict], rot_key: str = "rel_rot_first"
) -> dict[str, float]:
    by_obj = {}
    for r in rows:
        obj = r.get("object")
        rot = r.get(rot_key)
        if obj is None or rot is None:
            continue
        if isinstance(rot, list):
            vals = rot
        else:
            vals = [rot]
        for v in vals:
            if v is not None:
                by_obj.setdefault(obj, []).append(v)
    return {obj: _mean_pairwise_rotation_distance_deg(v) for obj, v in by_obj.items()}


def _object_relative_rotation_diversity(
    rows: list[dict], rot_key: str = "rel_rot_first"
) -> tuple[float, int]:
    per_obj = _object_relative_rotation_diversity_map(rows, rot_key=rot_key)
    if not per_obj:
        return 0.0, 0
    return float(np.mean(list(per_obj.values()))), int(len(per_obj))


def _wrist_trajectory_array(
    lhand_params,
    rhand_params,
    use_left: bool,
    use_right: bool,
    nframes: int,
) -> Optional[np.ndarray]:
    if nframes <= 0:
        return None
    traj = np.full((nframes, 2, 3), np.nan, dtype=np.float32)
    has_any = False

    if use_left:
        l = _to_numpy(_to_torch(lhand_params))
        if l.ndim == 1:
            l = l[None, :]
        if l.ndim == 2 and l.shape[0] > 0 and l.shape[1] >= 3:
            t = min(nframes, l.shape[0])
            traj[:t, 0, :] = l[:t, :3]
            has_any = True
    if use_right:
        r = _to_numpy(_to_torch(rhand_params))
        if r.ndim == 1:
            r = r[None, :]
        if r.ndim == 2 and r.shape[0] > 0 and r.shape[1] >= 3:
            t = min(nframes, r.shape[0])
            traj[:t, 1, :] = r[:t, :3]
            has_any = True

    return traj if has_any else None


def _canonicalize_wrist_trajectory(
    wrist_traj_world: Optional[np.ndarray], obj_params
) -> Optional[np.ndarray]:
    if wrist_traj_world is None:
        return None
    wrist_traj_world = np.asarray(wrist_traj_world, dtype=np.float32)
    if wrist_traj_world.ndim != 3 or wrist_traj_world.shape[-1] != 3:
        return None

    obj_pose = _pose9_sequence(obj_params)
    obj_pose_np = _to_numpy(obj_pose).astype(np.float64)
    if obj_pose_np.ndim != 2 or obj_pose_np.shape[1] < 9 or obj_pose_np.shape[0] == 0:
        return None

    t = min(wrist_traj_world.shape[0], obj_pose_np.shape[0])
    canonical = np.full_like(wrist_traj_world, np.nan, dtype=np.float32)
    obj_trans = obj_pose_np[:t, :3]
    obj_rot = _to_numpy(rot6d_to_rotmat(_to_torch(obj_pose_np[:t, 3:9]))).reshape(
        t, 3, 3
    )

    for frame_idx in range(t):
        frame = wrist_traj_world[frame_idx]
        valid = np.all(np.isfinite(frame), axis=1)
        if not np.any(valid):
            continue
        canonical[frame_idx, valid] = np.einsum(
            "ni,ij->nj",
            frame[valid].astype(np.float64) - obj_trans[frame_idx],
            obj_rot[frame_idx],
        ).astype(np.float32)
    return canonical


def _canonical_wrist_trajectory_array(
    lhand_params,
    rhand_params,
    obj_params,
    use_left: bool,
    use_right: bool,
    nframes: int,
) -> Optional[np.ndarray]:
    wrist_world = _wrist_trajectory_array(
        lhand_params, rhand_params, use_left, use_right, nframes
    )
    return _canonicalize_wrist_trajectory(wrist_world, obj_params)


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / norm


def _object_axis_alignment_from_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.eye(3, dtype=np.float32)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 3:
        return np.eye(3, dtype=np.float32)

    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        return np.eye(3, dtype=np.float32)
    order = np.argsort(eigvals)[::-1]
    axes = np.asarray(eigvecs[:, order], dtype=np.float64)
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        return np.eye(3, dtype=np.float32)

    # Choose deterministic axis signs from the point-cloud extent so different
    # objects share a more stable common orientation.
    for axis_idx in range(2):
        axis = _normalize_vec(axes[:, axis_idx])
        if not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-12:
            return np.eye(3, dtype=np.float32)
        proj = centered @ axis
        pos_extent = float(np.max(proj))
        neg_extent = float(abs(np.min(proj)))
        if neg_extent > pos_extent:
            axis = -axis
        axes[:, axis_idx] = axis

    third_axis = np.cross(axes[:, 0], axes[:, 1])
    third_axis = _normalize_vec(third_axis)
    if np.linalg.norm(third_axis) <= 1e-12:
        return np.eye(3, dtype=np.float32)
    axes[:, 2] = third_axis
    if float(np.linalg.det(axes)) < 0.0:
        axes[:, 2] *= -1.0
    return axes.astype(np.float32)


def _apply_object_axis_alignment(
    wrist_traj: Optional[np.ndarray],
    axis_alignment: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if wrist_traj is None:
        return None
    traj = np.asarray(wrist_traj, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        return None
    if axis_alignment is None:
        return traj.copy()
    align = np.asarray(axis_alignment, dtype=np.float32)
    if align.shape != (3, 3) or not np.isfinite(align).all():
        return traj.copy()
    out = np.full_like(traj, np.nan, dtype=np.float32)
    valid = np.all(np.isfinite(traj), axis=2)
    for frame_idx in range(traj.shape[0]):
        if np.any(valid[frame_idx]):
            out[frame_idx, valid[frame_idx]] = traj[frame_idx, valid[frame_idx]] @ align
    return out


def _active_wrist_centroid(traj: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(traj), axis=2)
    out = np.full((traj.shape[0], 3), np.nan, dtype=np.float32)
    for frame_idx in range(traj.shape[0]):
        if np.any(valid[frame_idx]):
            out[frame_idx] = np.mean(traj[frame_idx, valid[frame_idx]], axis=0)
    return out


def _wrist_trajectory_distance_m(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    t = min(int(a.shape[0]), int(b.shape[0]))
    if t <= 0:
        return None

    a_t = np.asarray(a[:t], dtype=np.float32)
    b_t = np.asarray(b[:t], dtype=np.float32)
    vals = []
    shared_hands = min(a_t.shape[1], b_t.shape[1])
    for hand_idx in range(shared_hands):
        a_hand = a_t[:, hand_idx, :]
        b_hand = b_t[:, hand_idx, :]
        valid = np.all(np.isfinite(a_hand), axis=1) & np.all(
            np.isfinite(b_hand), axis=1
        )
        if np.any(valid):
            vals.extend(np.linalg.norm(a_hand[valid] - b_hand[valid], axis=1).tolist())
    if vals:
        return float(np.mean(vals))

    # If one sample is left-hand-only and another is right-hand-only, there is no
    # shared hand slot. Compare the active wrist centroid for each frame instead
    # of implicitly using an inactive zero-valued hand.
    a_center = _active_wrist_centroid(a_t)
    b_center = _active_wrist_centroid(b_t)
    valid = np.all(np.isfinite(a_center), axis=1) & np.all(
        np.isfinite(b_center), axis=1
    )
    if not np.any(valid):
        return None
    return float(np.linalg.norm(a_center[valid] - b_center[valid], axis=1).mean())


def _mean_pairwise_wrist_distance_m(trajs: list[np.ndarray]) -> float:
    n = len(trajs)
    if n < 2:
        return 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = _wrist_trajectory_distance_m(trajs[i], trajs[j])
            if dist is not None:
                vals.append(dist)
    return float(np.mean(vals)) if vals else 0.0


def _sample_diversity_m(
    rows: list[dict], traj_key: str = "wrist_traj"
) -> tuple[float, int]:
    by_text = {}
    for r in rows:
        text = r.get("text")
        wrist_traj = r.get(traj_key)
        if text is None or wrist_traj is None:
            continue
        by_text.setdefault(text, []).append(wrist_traj)
    per_text = [
        _mean_pairwise_wrist_distance_m(trajs)
        for trajs in by_text.values()
        if len(trajs) >= 2
    ]
    if not per_text:
        return 0.0, 0
    return float(np.mean(per_text)), int(len(per_text))


def _latethoi_last_hand_feature(
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    use_left: bool,
    use_right: bool,
) -> Optional[np.ndarray]:
    if (not use_left or l_seq is None or l_seq.shape[0] == 0) and (
        not use_right or r_seq is None or r_seq.shape[0] == 0
    ):
        return None
    feat = np.full((2, 778, 3), np.nan, dtype=np.float32)
    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        l_last = np.asarray(l_seq[-1], dtype=np.float32)
        if l_last.ndim == 2 and l_last.shape[1] == 3:
            feat[0, : min(778, l_last.shape[0]), :] = l_last[:778]
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        r_last = np.asarray(r_seq[-1], dtype=np.float32)
        if r_last.ndim == 2 and r_last.shape[1] == 3:
            feat[1, : min(778, r_last.shape[0]), :] = r_last[:778]
    return feat


def _latethoi_last_frame_feature_distance(
    a: np.ndarray, b: np.ndarray
) -> Optional[float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != (2, 778, 3) or b.shape != (2, 778, 3):
        return None
    valid = np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)
    if not np.any(valid):
        return None
    diff = a - b
    diff_norm = np.linalg.norm(diff, axis=2)
    vals = diff_norm[valid]
    if vals.size == 0:
        return None
    denom = np.sqrt(float(vals.size))
    if denom <= 0.0:
        return None
    return float(vals.sum() / denom)


def _latethoi_last_frame_mean_pairwise_distance(
    feats: list[np.ndarray],
) -> float:
    n = len(feats)
    if n < 2:
        return 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = _latethoi_last_frame_feature_distance(feats[i], feats[j])
            if dist is not None:
                vals.append(dist)
    return float(np.mean(vals)) if vals else 0.0


def _latethoi_last_frame_sample_diversity(
    rows: list[dict],
) -> tuple[float, int]:
    by_text = {}
    for row in rows:
        text = row.get("text")
        feat = row.get("latethoi_last_hand_feat")
        if text is None or feat is None:
            continue
        by_text.setdefault(str(text), []).append(feat)
    per_text = [
        _latethoi_last_frame_mean_pairwise_distance(feats)
        for feats in by_text.values()
        if len(feats) >= 2
    ]
    if not per_text:
        return 0.0, 0
    return float(np.mean(per_text)), int(len(per_text))


def _latethoi_last_frame_overall_diversity(
    rows: list[dict],
) -> tuple[float, int]:
    feats = [
        row["latethoi_last_hand_feat"]
        for row in rows
        if row.get("latethoi_last_hand_feat") is not None
    ]
    if len(feats) < 2:
        return 0.0, len(feats)
    return float(_latethoi_last_frame_mean_pairwise_distance(feats)), int(len(feats))


def _overall_diversity_m(
    rows: list[dict], traj_key: str = "wrist_traj"
) -> tuple[float, int]:
    trajs = [r[traj_key] for r in rows if r.get(traj_key) is not None]
    if len(trajs) < 2:
        return 0.0, len(trajs)
    return float(_mean_pairwise_wrist_distance_m(trajs)), int(len(trajs))


def _pairwise_wrist_distances_m(trajs: list[np.ndarray]) -> np.ndarray:
    n = len(trajs)
    if n < 2:
        return np.zeros((0,), dtype=np.float32)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = _wrist_trajectory_distance_m(trajs[i], trajs[j])
            if dist is not None:
                vals.append(dist)
    return np.asarray(vals, dtype=np.float32)


def _sample_diversity_values_by_text(
    rows: list[dict], traj_key: str = "wrist_traj"
) -> dict[str, float]:
    by_text: dict[str, list[np.ndarray]] = {}
    for row in rows:
        text = row.get("text")
        wrist_traj = row.get(traj_key)
        if text is None or wrist_traj is None:
            continue
        by_text.setdefault(str(text), []).append(wrist_traj)
    return {
        text: float(_mean_pairwise_wrist_distance_m(trajs))
        for text, trajs in by_text.items()
        if len(trajs) >= 2
    }


def _per_object_diversity(
    rows: list[dict], traj_key: str = "wrist_traj"
) -> dict[str, dict]:
    by_object: dict[str, list[dict]] = {}
    for row in rows:
        object_key = row.get("object")
        if object_key is None or row.get(traj_key) is None:
            continue
        by_object.setdefault(str(object_key), []).append(row)
    out = {}
    for object_key, object_rows in by_object.items():
        sample_diversity_m, sample_diversity_prompts = _sample_diversity_m(
            object_rows, traj_key=traj_key
        )
        overall_diversity_m, overall_diversity_samples = _overall_diversity_m(
            object_rows, traj_key=traj_key
        )
        out[object_key] = {
            "object": object_key,
            "samples": int(len(object_rows)),
            "sample_diversity_m": sample_diversity_m,
            "sample_diversity_prompts": sample_diversity_prompts,
            "overall_diversity_m": overall_diversity_m,
            "overall_diversity_samples": overall_diversity_samples,
        }
    return out


def _plot_diversity_visualizations(
    all_results: list[dict], summary_rows: list[dict], output_dir: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(out_dir, exist_ok=True)

    def _set_equal_3d_axes(ax, pts: np.ndarray) -> None:
        pts = np.asarray(pts, dtype=np.float64)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] == 0:
            return
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins) / 2.0), 1e-4)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def _hands_for_row(row: dict) -> list[tuple[int, str, str]]:
        use_left, use_right = _selected_hands(str(row.get("text", "")))
        hands = []
        if use_left:
            hands.append((0, "left", "#c0392b"))
        if use_right:
            hands.append((1, "right", "#2980b9"))
        return hands

    def _draw_wrist_trajectories(
        rows: list[dict],
        ax3d,
        ax_xy,
        traj_key: str = "wrist_traj",
        max_rows: int = 80,
    ) -> None:
        rows = [row for row in rows if row.get(traj_key) is not None]
        if not rows:
            ax3d.set_visible(False)
            ax_xy.set_visible(False)
            return
        if len(rows) > max_rows:
            idx = np.linspace(0, len(rows) - 1, max_rows, dtype=np.int64)
            rows = [rows[int(i)] for i in idx]

        all_points = []
        for row in rows:
            traj = np.asarray(row[traj_key], dtype=np.float64)
            if traj.ndim != 3 or traj.shape[2] != 3:
                continue
            for hand_idx, hand_name, color in _hands_for_row(row):
                pts = traj[:, hand_idx, :]
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if pts.shape[0] == 0:
                    continue
                all_points.append(pts)
                ax3d.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=color,
                    alpha=0.34,
                    linewidth=1.2,
                )
                ax3d.scatter(
                    pts[0:1, 0], pts[0:1, 1], pts[0:1, 2], color=color, s=10, alpha=0.8
                )
                ax_xy.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.34, linewidth=1.2)
                ax_xy.scatter(pts[0, 0], pts[0, 1], color=color, s=10, alpha=0.8)

        if all_points:
            stacked = np.concatenate(all_points, axis=0)
            _set_equal_3d_axes(ax3d, stacked)
            ax_xy.set_aspect("equal", adjustable="box")
        ax3d.set_title("3D wrist trajectories")
        ax3d.set_xlabel("x (m)")
        ax3d.set_ylabel("y (m)")
        ax3d.set_zlabel("z (m)")
        ax_xy.set_xlabel("x (m)")
        ax_xy.set_ylabel("y (m)")
        ax_xy.grid(alpha=0.25)

    def _plot_wrist_trajectories(
        rows: list[dict],
        title: str,
        path: str,
        traj_key: str = "wrist_traj",
        max_rows: int = 80,
    ) -> None:
        fig = plt.figure(figsize=(12, 5.6))
        ax3d = fig.add_subplot(121, projection="3d")
        ax_xy = fig.add_subplot(122)
        _draw_wrist_trajectories(
            rows, ax3d, ax_xy, traj_key=traj_key, max_rows=max_rows
        )
        if ax3d.get_visible():
            ax3d.set_title("3D wrist trajectories")
        if ax_xy.get_visible():
            ax_xy.set_title("Top-down x-y projection")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    plot_rows = [
        row
        for row in summary_rows
        if row.get("file_name") != "ALL_FILES" and not row.get("is_gt_row", False)
    ]
    if plot_rows:
        labels = [str(row.get("file_name", "")) for row in plot_rows]
        sd = np.asarray(
            [float(row.get("sample_diversity_m", 0.0)) for row in plot_rows]
        )
        od = np.asarray(
            [float(row.get("overall_diversity_m", 0.0)) for row in plot_rows]
        )
        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 1.15), 5.2))
        ax.bar(x - width / 2.0, sd, width, label="SD: within same prompt")
        ax.bar(x + width / 2.0, od, width, label="OD: across all samples")
        ax.set_ylabel("Mean wrist trajectory distance (m)")
        ax.set_title("SD vs OD")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sd_od_by_file.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 1.15), 4.8))
        ax.bar(x, od - sd, color="#7f8c8d")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("OD - SD (m)")
        ax.set_title("How Much Overall Diversity Exceeds Same-Prompt Diversity")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "od_minus_sd_by_file.png"), dpi=180)
        plt.close(fig)

    for result in all_results:
        file_stem = os.path.splitext(str(result.get("file_name", "result")))[0]
        rows = result.get("per_sample_rows", [])
        summary_row = next(
            (
                row
                for row in summary_rows
                if row.get("file_name") == result.get("file_name")
                and not row.get("is_gt_row", False)
            ),
            None,
        )
        sd_by_text = _sample_diversity_values_by_text(rows)
        od_dist = _pairwise_wrist_distances_m(
            [
                row["wrist_traj_aligned"]
                for row in rows
                if row.get("wrist_traj_aligned") is not None
            ]
        )

        if sd_by_text or od_dist.size:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            sd_vals = np.asarray(list(sd_by_text.values()), dtype=np.float32)
            if sd_vals.size:
                axes[0].hist(
                    sd_vals,
                    bins=min(20, max(5, sd_vals.size)),
                    color="#3498db",
                    alpha=0.85,
                )
                axes[0].axvline(
                    float(sd_vals.mean()), color="black", linestyle="--", linewidth=1.2
                )
            axes[0].set_title("Per-prompt SD distribution")
            axes[0].set_xlabel("Mean pairwise distance within prompt (m)")
            axes[0].set_ylabel("Prompt count")
            axes[0].grid(axis="y", alpha=0.25)

            if od_dist.size:
                axes[1].hist(od_dist, bins=30, color="#e67e22", alpha=0.85)
                axes[1].axvline(
                    float(od_dist.mean()), color="black", linestyle="--", linewidth=1.2
                )
            axes[1].set_title("Overall pairwise distance distribution")
            axes[1].set_xlabel("Pairwise distance across samples (m)")
            axes[1].set_ylabel("Pair count")
            axes[1].grid(axis="y", alpha=0.25)
            fig.suptitle(file_stem)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"{file_stem}_diversity_distributions.png"),
                dpi=180,
            )
            plt.close(fig)

        if sd_by_text:
            ranked = sorted(sd_by_text.items(), key=lambda x: x[1], reverse=True)
            top = ranked[:10]
            bottom = ranked[-10:] if len(ranked) > 10 else []
            shown = top + bottom
            labels = [
                (text[:54] + "...") if len(text) > 57 else text for text, _ in shown
            ]
            values = [value for _, value in shown]
            colors = ["#c0392b"] * len(top) + ["#27ae60"] * len(bottom)
            fig, ax = plt.subplots(figsize=(11, max(4.5, len(shown) * 0.38)))
            y = np.arange(len(shown))
            ax.barh(y, values, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("SD within prompt (m)")
            ax.set_title(f"{file_stem}: highest and lowest prompt-level SD")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"{file_stem}_prompt_sd_ranked.png"), dpi=180
            )
            plt.close(fig)

            by_text: dict[str, list[dict]] = {}
            for row in rows:
                if row.get("wrist_traj") is not None:
                    by_text.setdefault(str(row.get("text", "")), []).append(row)
            top_text, top_sd = ranked[0]
            low_text, low_sd = ranked[-1]
            _plot_wrist_trajectories(
                by_text.get(top_text, []),
                f"{file_stem}: highest-SD prompt trajectories (SD={top_sd:.3f}m)\n{top_text}",
                os.path.join(out_dir, f"{file_stem}_top_prompt_wrist_trajectories.png"),
                traj_key="wrist_traj_aligned",
                max_rows=80,
            )
            _plot_wrist_trajectories(
                by_text.get(low_text, []),
                f"{file_stem}: lowest-SD prompt trajectories (SD={low_sd:.3f}m)\n{low_text}",
                os.path.join(out_dir, f"{file_stem}_low_prompt_wrist_trajectories.png"),
                traj_key="wrist_traj_aligned",
                max_rows=80,
            )

            overall_od = (
                float(summary_row.get("overall_diversity_m", 0.0))
                if summary_row is not None
                else 0.0
            )
            overall_sd = (
                float(summary_row.get("sample_diversity_m", 0.0))
                if summary_row is not None
                else 0.0
            )
            fig = plt.figure(figsize=(17, 9.2))
            axes = [
                fig.add_subplot(2, 3, 1, projection="3d"),
                fig.add_subplot(2, 3, 2, projection="3d"),
                fig.add_subplot(2, 3, 3, projection="3d"),
                fig.add_subplot(2, 3, 4),
                fig.add_subplot(2, 3, 5),
                fig.add_subplot(2, 3, 6),
            ]
            _draw_wrist_trajectories(
                by_text.get(top_text, []),
                axes[0],
                axes[3],
                traj_key="wrist_traj_aligned",
                max_rows=80,
            )
            _draw_wrist_trajectories(
                by_text.get(low_text, []),
                axes[1],
                axes[4],
                traj_key="wrist_traj_aligned",
                max_rows=80,
            )
            _draw_wrist_trajectories(
                rows,
                axes[2],
                axes[5],
                traj_key="wrist_traj_aligned",
                max_rows=100,
            )
            if axes[0].get_visible():
                axes[0].set_title(f"Highest SD prompt\nSD={top_sd:.3f}m")
            if axes[3].get_visible():
                axes[3].set_title("Top-down x-y")
            if axes[1].get_visible():
                axes[1].set_title(f"Lowest SD prompt\nSD={low_sd:.3f}m")
            if axes[4].get_visible():
                axes[4].set_title("Top-down x-y")
            if axes[2].get_visible():
                axes[2].set_title(f"Overall sample set\nOD={overall_od:.3f}m")
            if axes[5].get_visible():
                axes[5].set_title(
                    f"Top-down x-y\nSD={overall_sd:.3f}m, OD={overall_od:.3f}m"
                )
            fig.suptitle(
                f"{file_stem}: SD/OD trajectory comparison\n"
                f"left=highest SD prompt | middle=lowest SD prompt | right=overall OD sample set"
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"{file_stem}_sd_od_trajectory_comparison.png"),
                dpi=180,
            )
            plt.close(fig)

        _plot_wrist_trajectories(
            rows,
            f"{file_stem}: overall wrist trajectories",
            os.path.join(out_dir, f"{file_stem}_overall_wrist_trajectories.png"),
            traj_key="wrist_traj_aligned",
            max_rows=100,
        )

    print(f"Saved diversity visualizations: {out_dir}")


def _write_physics_distribution_markdown(
    md_path: str,
    all_results: list[dict],
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        return f"\n## Physics Distributions\n\n_Plot generation skipped: {ex}_\n"

    md_dir = os.path.dirname(os.path.abspath(md_path))
    assets_dir = os.path.join(
        md_dir, f"{os.path.splitext(os.path.basename(md_path))[0]}_assets"
    )
    os.makedirs(assets_dir, exist_ok=True)

    metric_specs = [
        ("cr", "CR", "", "#1f77b4"),
        ("iv_cm3", "IV (cm^3)", "cm^3", "#d62728"),
        ("id_mm", "ID (mm)", "mm", "#2ca02c"),
        ("id_max_mm", "ID_max (mm)", "mm", "#9467bd"),
    ]

    sections = ["\n## Physics Distributions\n"]
    for result in all_results:
        file_name = str(result.get("file_name", "result"))
        rows = result.get("per_sample_rows", [])
        if not rows:
            continue
        safe_name = os.path.splitext(os.path.basename(file_name))[0]
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_name
        )
        values_by_metric: dict[str, np.ndarray] = {}
        for key, _label, _unit, _color in metric_specs:
            vals = []
            for row in rows:
                value = row.get(key)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except Exception:
                    continue
                if np.isfinite(value_f):
                    vals.append(value_f)
            values_by_metric[key] = np.asarray(vals, dtype=np.float64)

        if not any(arr.size > 0 for arr in values_by_metric.values()):
            continue

        stats_lines = []
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
        axes = axes.reshape(-1)
        for ax, (key, label, unit, color) in zip(axes, metric_specs):
            vals = values_by_metric[key]
            if vals.size == 0:
                ax.text(0.5, 0.5, "No valid values", ha="center", va="center")
                ax.set_title(label)
                ax.set_axis_off()
                continue
            bins = min(30, max(8, int(np.sqrt(vals.size))))
            ax.hist(vals, bins=bins, color=color, alpha=0.82, edgecolor="white")
            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2)
            ax.axvline(median_val, color="#555555", linestyle=":", linewidth=1.1)
            ax.set_title(label)
            ax.set_ylabel("Sample count")
            if unit:
                ax.set_xlabel(unit)
            ax.grid(axis="y", alpha=0.25)
            stats_lines.append(
                f"- `{label}`: n={vals.size}, mean={mean_val:.4f}, median={median_val:.4f}, min={float(np.min(vals)):.4f}, max={float(np.max(vals)):.4f}"
            )
        fig.suptitle(f"{file_name}: per-sample physics metric distributions")
        fig.tight_layout()
        hist_image_name = f"{safe_name}_physics_histograms.png"
        hist_image_path = os.path.join(assets_dir, hist_image_name)
        fig.savefig(hist_image_path, dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
        axes = axes.reshape(-1)
        for ax, (key, label, unit, color) in zip(axes, metric_specs):
            vals = values_by_metric[key]
            if vals.size == 0:
                ax.text(0.5, 0.5, "No valid values", ha="center", va="center")
                ax.set_title(label)
                ax.set_axis_off()
                continue
            flierprops = dict(
                marker="o",
                markersize=3.5,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.5,
            )
            boxprops = dict(facecolor=color, alpha=0.5, edgecolor=color)
            medianprops = dict(color="black", linewidth=1.4)
            whiskerprops = dict(color=color, linewidth=1.2)
            capprops = dict(color=color, linewidth=1.2)
            ax.boxplot(
                vals,
                vert=True,
                patch_artist=True,
                widths=0.45,
                flierprops=flierprops,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
            )
            ax.set_title(label)
            ax.set_xticks([1])
            ax.set_xticklabels(["samples"])
            ax.grid(axis="y", alpha=0.25)
            if unit:
                ax.set_ylabel(unit)
        fig.suptitle(f"{file_name}: per-sample physics metric box plots")
        fig.tight_layout()
        box_image_name = f"{safe_name}_physics_boxplots.png"
        box_image_path = os.path.join(assets_dir, box_image_name)
        fig.savefig(box_image_path, dpi=180)
        plt.close(fig)

        hist_rel_path = os.path.relpath(hist_image_path, md_dir)
        box_rel_path = os.path.relpath(box_image_path, md_dir)
        sections.append(f"\n### {file_name}\n")
        sections.extend(line + "\n" for line in stats_lines)
        sections.append("\nHistogram:\n")
        sections.append(f"\n![]({hist_rel_path})\n")
        sections.append("\nBox plot:\n")
        sections.append(f"\n![]({box_rel_path})\n")
    return "".join(sections)


def _save_resampled_object_visualizations(
    object_rows: list[dict], output_dir: str
) -> None:
    if not object_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"[INFO] saving {len(object_rows)} resampled object visualizations to {out_dir}"
    )

    def _set_equal_3d_axes(ax, pts: np.ndarray) -> None:
        pts = np.asarray(pts, dtype=np.float64)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] == 0:
            return
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins) / 2.0), 1e-4)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def _scatter_pc(ax, pts: np.ndarray, title: str, color: str) -> None:
        pts = np.asarray(pts, dtype=np.float64)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] == 0:
            ax.set_visible(False)
            return
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=6,
            c=color,
            alpha=0.9,
            linewidths=0.0,
        )
        _set_equal_3d_axes(ax, pts)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    def _plot_mesh(
        ax,
        vertices: np.ndarray,
        faces: np.ndarray,
        title: str,
        face_color: str = "#95a5a6",
        edge_color: str = "#2c3e50",
        alpha: float = 0.28,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        valid_vertices = (
            vertices.ndim == 2
            and vertices.shape[1] == 3
            and np.all(np.isfinite(vertices))
        )
        valid_faces = faces.ndim == 2 and faces.shape[1] == 3 and faces.shape[0] > 0
        if not valid_vertices or not valid_faces:
            ax.set_visible(False)
            return
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            color=face_color,
            edgecolor=edge_color,
            linewidth=0.12,
            alpha=alpha,
            antialiased=True,
            shade=False,
        )
        _set_equal_3d_axes(ax, vertices)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    def _optional_array(row: dict, key: str, dtype, shape_tail: tuple[int, ...]):
        value = row.get(key)
        if value is None:
            return np.zeros((0,) + shape_tail, dtype=dtype)
        return np.asarray(value, dtype=dtype)

    for row in tqdm.tqdm(object_rows, desc="resampled-object-png", leave=False):
        object_key = str(row.get("object_key", "object"))
        orig_pc = np.asarray(row.get("original_pc"), dtype=np.float32)
        used_pc = np.asarray(row.get("used_pc"), dtype=np.float32)
        mesh_vertices = np.asarray(row.get("mesh_vertices"), dtype=np.float32)
        mesh_faces = np.asarray(row.get("mesh_faces"), dtype=np.int64)
        orig_proxy_vertices = _optional_array(
            row, "orig_proxy_vertices", np.float32, (3,)
        )
        orig_proxy_faces = _optional_array(row, "orig_proxy_faces", np.int64, (3,))
        used_proxy_vertices = _optional_array(
            row, "used_proxy_vertices", np.float32, (3,)
        )
        used_proxy_faces = _optional_array(row, "used_proxy_faces", np.int64, (3,))
        if (
            orig_pc.ndim != 2
            or orig_pc.shape[1] != 3
            or used_pc.ndim != 2
            or used_pc.shape[1] != 3
        ):
            continue
        fig = plt.figure(figsize=(22.0, 5.4))
        ax_orig = fig.add_subplot(141, projection="3d")
        ax_mesh = fig.add_subplot(142, projection="3d")
        ax_orig_proxy = fig.add_subplot(143, projection="3d")
        ax_used_proxy = fig.add_subplot(144, projection="3d")
        _scatter_pc(
            ax_orig,
            orig_pc,
            f"obj.pkl point cloud ({orig_pc.shape[0]} pts)",
            "#7f8c8d",
        )
        _plot_mesh(
            ax_mesh,
            mesh_vertices,
            mesh_faces,
            f"resolved mesh ({mesh_faces.shape[0]} faces)",
        )
        if orig_proxy_vertices.ndim == 2 and orig_proxy_vertices.shape[1] == 3:
            _plot_mesh(
                ax_orig_proxy,
                orig_proxy_vertices,
                orig_proxy_faces,
                f"proxy from obj.pkl pc ({orig_proxy_faces.shape[0]} faces)",
                face_color="#95a5a6",
                edge_color="#2c3e50",
                alpha=0.28,
            )
        _scatter_pc(
            ax_orig_proxy,
            orig_pc,
            "",
            "#7f8c8d",
        )
        if used_proxy_vertices.ndim == 2 and used_proxy_vertices.shape[1] == 3:
            _plot_mesh(
                ax_used_proxy,
                used_proxy_vertices,
                used_proxy_faces,
                f"proxy from mesh-sampled pc ({used_proxy_faces.shape[0]} faces)",
                face_color="#a8dadc",
                edge_color="#1d3557",
                alpha=0.28,
            )
        _scatter_pc(
            ax_used_proxy,
            used_pc,
            f"mesh-sampled point cloud ({used_pc.shape[0]} pts)",
            "#16a085",
        )
        combined_pts = [orig_pc.astype(np.float32), used_pc.astype(np.float32)]
        if mesh_vertices.ndim == 2 and mesh_vertices.shape[1] == 3:
            combined_pts.append(mesh_vertices.astype(np.float32))
        if orig_proxy_vertices.ndim == 2 and orig_proxy_vertices.shape[1] == 3:
            combined_pts.append(orig_proxy_vertices.astype(np.float32))
        if used_proxy_vertices.ndim == 2 and used_proxy_vertices.shape[1] == 3:
            combined_pts.append(used_proxy_vertices.astype(np.float32))
        combined_pts = np.concatenate(combined_pts, axis=0)
        _set_equal_3d_axes(ax_orig, combined_pts)
        _set_equal_3d_axes(ax_mesh, combined_pts)
        _set_equal_3d_axes(ax_orig_proxy, combined_pts)
        _set_equal_3d_axes(ax_used_proxy, combined_pts)
        fig.suptitle(f"{object_key}\nproxy meshes used for IV/ID comparison")
        fig.tight_layout()
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in object_key
        )
        fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=180)
        plt.close(fig)

    print(f"[INFO] saved resampled object visualizations: {out_dir}")


def _mean_optional(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _aggregate(rows: list[dict], min_cr_for_valid: float) -> dict:
    if not rows:
        return {
            "samples": 0,
            "iv_cm3": None,
            "id_mm": None,
            "id_max_mm": None,
            "cr": 0.0,
            "success_rate": 0.0,
            "success_samples": 0,
            "valid_contact_rate": 0.0,
            "valid_samples": 0,
            "iv_cm3_valid": None,
            "id_mm_valid": None,
            "id_max_mm_valid": None,
            "gt_samples": 0,
            "cr_gt": 0.0,
            "success_rate_gt": 0.0,
            "success_gt_samples": 0,
            "gt_iou": 0.0,
            "gt_precision": 0.0,
            "gt_recall": 0.0,
            "gt_f1": 0.0,
            "id_gt": None,
            "id_max_gt": None,
            "id_err_to_gt": None,
            "id_max_err_to_gt": None,
            "id_gt_samples": 0,
            "iv_gt": None,
            "iv_gt_samples": 0,
            "iv_gt_valid": None,
            "id_gt_valid": None,
            "id_max_gt_valid": None,
            "iv_err_to_gt_valid": None,
            "id_err_to_gt_valid": None,
            "id_max_err_to_gt_valid": None,
            "valid_gt_samples": 0,
            "relrot_diversity_deg": 0.0,
            "relrot_object_count": 0,
            "relrot_diversity_gt_deg": 0.0,
            "relrot_gt_object_count": 0,
            "sample_diversity_m": 0.0,
            "sample_diversity_prompts": 0,
            "object_avg_sample_diversity_m": 0.0,
            "object_avg_sample_diversity_objects": 0,
            "overall_diversity_local_m": 0.0,
            "overall_diversity_local_samples": 0,
            "overall_diversity_m": 0.0,
            "overall_diversity_samples": 0,
            "object_avg_overall_diversity_m": 0.0,
            "object_avg_overall_diversity_objects": 0,
        }
    relrot_diversity_deg, relrot_object_count = _object_relative_rotation_diversity(
        rows, rot_key="rel_rot_first"
    )
    relrot_diversity_gt_deg, relrot_gt_object_count = (
        _object_relative_rotation_diversity(rows, rot_key="rel_rot_first_gt")
    )
    sample_diversity_m, sample_diversity_prompts = _sample_diversity_m(rows)
    overall_diversity_local_m, overall_diversity_local_samples = _overall_diversity_m(
        rows
    )
    overall_diversity_m, overall_diversity_samples = _overall_diversity_m(
        rows, traj_key="wrist_traj_aligned"
    )
    per_object_diversity = _per_object_diversity(rows)
    valid_rows = [r for r in rows if bool(r.get("valid_contact", False))]
    success_rows = [r for r in rows if bool(r.get("success", False))]
    gt_rows = [r for r in rows if r.get("gt_iou") is not None]
    id_gt_rows = [r for r in rows if r.get("id_gt") is not None]
    iv_gt_rows = [r for r in rows if r.get("iv_gt") is not None]
    gt_valid_rows = [r for r in rows if bool(r.get("valid_contact_gt", False))]
    gt_success_rows = [r for r in rows if bool(r.get("success_gt", False))]
    pred_valid_gt_rows = [r for r in valid_rows if r.get("id_gt") is not None]
    pred_valid_gt_iv_rows = [
        r
        for r in valid_rows
        if r.get("iv_gt") is not None and r.get("iv_cm3") is not None
    ]
    return {
        "samples": len(rows),
        "iv_cm3": _mean_optional(r.get("iv_cm3") for r in rows),
        "id_mm": _mean_optional(r.get("id_mm") for r in rows),
        "id_max_mm": _mean_optional(r.get("id_max_mm") for r in rows),
        "cr": float(np.mean([r["cr"] for r in rows])),
        "success_rate": float(len(success_rows) / len(rows)),
        "success_samples": int(len(success_rows)),
        "valid_contact_rate": float(len(valid_rows) / len(rows)),
        "valid_samples": int(len(valid_rows)),
        "iv_cm3_valid": _mean_optional(r.get("iv_cm3") for r in valid_rows),
        "id_mm_valid": _mean_optional(r.get("id_mm") for r in valid_rows),
        "id_max_mm_valid": _mean_optional(r.get("id_max_mm") for r in valid_rows),
        "gt_samples": int(len(gt_rows)),
        "cr_gt": float(np.mean([r["cr_gt"] for r in gt_rows])) if gt_rows else 0.0,
        "success_rate_gt": (
            float(len(gt_success_rows) / len(gt_rows)) if gt_rows else 0.0
        ),
        "success_gt_samples": int(len(gt_success_rows)),
        "gt_iou": float(np.mean([r["gt_iou"] for r in gt_rows])) if gt_rows else 0.0,
        "gt_precision": (
            float(np.mean([r["gt_precision"] for r in gt_rows])) if gt_rows else 0.0
        ),
        "gt_recall": (
            float(np.mean([r["gt_recall"] for r in gt_rows])) if gt_rows else 0.0
        ),
        "gt_f1": float(np.mean([r["gt_f1"] for r in gt_rows])) if gt_rows else 0.0,
        "cr_err_to_gt": (
            float(np.mean([abs(r["cr"] - r["cr_gt"]) for r in gt_rows]))
            if gt_rows
            else 0.0
        ),
        "id_gt": _mean_optional(r.get("id_gt") for r in id_gt_rows),
        "id_max_gt": _mean_optional(r.get("id_max_gt") for r in id_gt_rows),
        "id_err_to_gt": _mean_optional(
            abs(r["id_mm"] - r["id_gt"])
            for r in id_gt_rows
            if r.get("id_mm") is not None and r.get("id_gt") is not None
        ),
        "id_max_err_to_gt": _mean_optional(
            abs(r["id_max_mm"] - r["id_max_gt"])
            for r in id_gt_rows
            if r.get("id_max_mm") is not None and r.get("id_max_gt") is not None
        ),
        "id_gt_samples": int(len(id_gt_rows)),
        "iv_gt": _mean_optional(r.get("iv_gt") for r in iv_gt_rows),
        "iv_gt_samples": int(len(iv_gt_rows)),
        "iv_gt_valid": _mean_optional(r.get("iv_gt") for r in gt_valid_rows),
        "id_gt_valid": _mean_optional(r.get("id_gt") for r in gt_valid_rows),
        "id_max_gt_valid": _mean_optional(r.get("id_max_gt") for r in gt_valid_rows),
        "iv_err_to_gt_valid": _mean_optional(
            abs(r["iv_cm3"] - r["iv_gt"]) for r in pred_valid_gt_iv_rows
        ),
        "id_err_to_gt_valid": _mean_optional(
            abs(r["id_mm"] - r["id_gt"])
            for r in pred_valid_gt_rows
            if r.get("id_mm") is not None and r.get("id_gt") is not None
        ),
        "id_max_err_to_gt_valid": _mean_optional(
            abs(r["id_max_mm"] - r["id_max_gt"])
            for r in pred_valid_gt_rows
            if r.get("id_max_mm") is not None and r.get("id_max_gt") is not None
        ),
        "valid_gt_samples": int(len(gt_valid_rows)),
        "relrot_diversity_deg": relrot_diversity_deg,
        "relrot_object_count": relrot_object_count,
        "relrot_diversity_gt_deg": relrot_diversity_gt_deg,
        "relrot_gt_object_count": relrot_gt_object_count,
        "sample_diversity_m": sample_diversity_m,
        "sample_diversity_prompts": sample_diversity_prompts,
        "object_avg_sample_diversity_m": _mean_optional(
            item.get("sample_diversity_m") for item in per_object_diversity.values()
        )
        or 0.0,
        "object_avg_sample_diversity_objects": int(len(per_object_diversity)),
        "overall_diversity_local_m": overall_diversity_local_m,
        "overall_diversity_local_samples": overall_diversity_local_samples,
        "overall_diversity_m": overall_diversity_m,
        "overall_diversity_samples": overall_diversity_samples,
        "object_avg_overall_diversity_m": _mean_optional(
            item.get("overall_diversity_m") for item in per_object_diversity.values()
        )
        or 0.0,
        "object_avg_overall_diversity_objects": int(len(per_object_diversity)),
    }


def _latethoi_lastframe_aggregate(rows: list[dict]) -> dict:
    sample_diversity, sample_diversity_prompts = _latethoi_last_frame_sample_diversity(
        rows
    )
    overall_diversity, overall_diversity_samples = (
        _latethoi_last_frame_overall_diversity(rows)
    )
    metric_keys = [
        "inter_volume_mean",
        "inter_volume_contact",
        "inter_volume_max",
        "inter_depth_mean",
        "inter_depth_contact",
        "inter_depth_max",
        "contact_ratio_mean",
        "contact_ratio_contact",
        "contact_ratio_max",
        "contact_ratio_off_ground",
    ]
    out = {
        "samples": int(len(rows)),
        "sample_diversity_last_frame": float(sample_diversity),
        "sample_diversity_prompts": int(sample_diversity_prompts),
        "overall_diversity_last_frame": float(overall_diversity),
        "overall_diversity_samples": int(overall_diversity_samples),
        "iv_cm3": _mean_optional(row.get("iv_cm3") for row in rows),
        "id_mm": _mean_optional(row.get("id_mm") for row in rows),
        "id_max_mm": _mean_optional(row.get("id_max_mm") for row in rows),
        "off_ground_contact_ratio": None,
        "off_ground_contact_rate": None,
        "jerk": None,
    }
    for key in metric_keys:
        vals = []
        for row in rows:
            metrics = row.get("latethoi_lastframe", {})
            hand_metric_mean = (
                metrics.get("hand_metric_mean", {}) if isinstance(metrics, dict) else {}
            )
            value = hand_metric_mean.get(key)
            if value is not None:
                vals.append(float(value))
        out[key] = float(np.mean(vals)) if vals else None
    return out


def _latethoi_lastframe_gt_reference_row(
    split_name: str,
    rows: list[dict],
    source_file_name: Optional[str] = None,
) -> Optional[dict]:
    total_gt_samples = int(len(rows))
    rows = _filtered_gt_reference_rows(rows)
    if not rows:
        return None

    metric_keys = [
        "inter_volume_mean",
        "inter_volume_contact",
        "inter_volume_max",
        "inter_depth_mean",
        "inter_depth_contact",
        "inter_depth_max",
        "contact_ratio_mean",
        "contact_ratio_contact",
        "contact_ratio_max",
        "contact_ratio_off_ground",
    ]
    gt_label = f"{split_name.capitalize()}_G.T"
    if source_file_name:
        gt_label = f"{gt_label} (from {source_file_name})"

    out = {
        "split": split_name,
        "file_name": gt_label,
        "samples": total_gt_samples,
        "sample_diversity_last_frame": 0.0,
        "sample_diversity_prompts": 0,
        "overall_diversity_last_frame": 0.0,
        "overall_diversity_samples": 0,
        "iv_cm3": _mean_optional(row.get("iv_gt") for row in rows),
        "id_mm": _mean_optional(row.get("id_gt") for row in rows),
        "id_max_mm": _mean_optional(row.get("id_max_gt") for row in rows),
        "off_ground_contact_ratio": None,
        "off_ground_contact_rate": None,
        "jerk": None,
        "is_gt_row": True,
    }
    for key in metric_keys:
        vals = []
        for row in rows:
            metrics = row.get("latethoi_lastframe_gt", {})
            hand_metric_mean = (
                metrics.get("hand_metric_mean", {}) if isinstance(metrics, dict) else {}
            )
            value = hand_metric_mean.get(key)
            if value is not None:
                vals.append(float(value))
        out[key] = float(np.mean(vals)) if vals else None
    return out


def _latethoi_lastframe_summary_rows(all_results: list[dict]) -> list[dict]:
    grouped_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    fallback_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    gt_source_file: dict[str, Optional[str]] = {"seen": None, "unseen": None}
    for result in all_results:
        file_name = result.get("file_name", "")
        split = _split_tag_from_file_name(file_name)
        split_rows = result.get("per_sample_rows", [])
        if split not in grouped_rows or not split_rows:
            continue
        if not _allow_gt_source_file(file_name):
            continue
        if not fallback_rows[split]:
            fallback_rows[split] = list(split_rows)
            if gt_source_file[split] is None:
                gt_source_file[split] = file_name
        has_gt_hand = any(bool(r.get("gt_hand_available")) for r in split_rows)
        if has_gt_hand and not grouped_rows[split]:
            grouped_rows[split] = list(split_rows)
            gt_source_file[split] = file_name
    for split in grouped_rows:
        if not grouped_rows[split]:
            grouped_rows[split] = fallback_rows[split]

    rows_by_split: dict[str, list[dict]] = {"seen": [], "unseen": [], "other": []}
    for split_name, file_name, split_rows in _aggregate_results_by_split(all_results):
        agg = _latethoi_lastframe_aggregate(split_rows)
        agg["split"] = split_name
        agg["file_name"] = file_name
        agg["is_gt_row"] = False
        rows_by_split.setdefault(split_name, []).append(agg)

    rows = []
    for split_name in ("seen", "unseen", "other"):
        split_rows = rows_by_split.get(split_name, [])
        if not split_rows:
            continue
        if split_name in {"seen", "unseen"} and grouped_rows.get(split_name):
            gt_row = _latethoi_lastframe_gt_reference_row(
                split_name,
                grouped_rows.get(split_name, []),
                source_file_name=gt_source_file.get(split_name),
            )
            if gt_row is not None and int(gt_row.get("samples", 0)) > 0:
                rows.append(gt_row)
        rows.extend(split_rows)
    return rows


def _text2hoi_aggregate(rows: list[dict]) -> dict:
    metric_keys = [
        "penetration_loss_m",
        "penetration_max_m",
        "interior_object_ratio",
        "contact_object_ratio",
        "contact_joint_ratio",
    ]
    out = {"samples": int(len(rows))}
    for key in metric_keys:
        vals = []
        for row in rows:
            metrics = row.get("text2hoi", {})
            hand_metric_mean = (
                metrics.get("hand_metric_mean", {}) if isinstance(metrics, dict) else {}
            )
            value = hand_metric_mean.get(key)
            if value is not None:
                vals.append(float(value))
        out[key] = float(np.mean(vals)) if vals else None
    return out


def _text2hoi_summary_rows(all_results: list[dict]) -> list[dict]:
    rows = []
    for split_name, file_name, split_rows in _aggregate_results_by_split(all_results):
        agg = _text2hoi_aggregate(split_rows)
        agg["split"] = split_name
        agg["file_name"] = file_name
        rows.append(agg)
    return rows


def _diffh2o_native_aggregate(rows: list[dict]) -> dict:
    sample_diversity, sample_diversity_prompts = _sample_diversity_m(rows)
    overall_diversity, overall_diversity_samples = _overall_diversity_m(
        rows, traj_key="wrist_traj"
    )
    metric_keys = [
        "iv_count_mean",
        "iv_count_max",
        "inter_depth_mean",
        "inter_depth_max",
        "contact_ratio_mean",
        "contact_ratio_max",
    ]
    out = {
        "samples": int(len(rows)),
        "sample_diversity_m": float(sample_diversity),
        "sample_diversity_prompts": int(sample_diversity_prompts),
        "overall_diversity_m": float(overall_diversity),
        "overall_diversity_samples": int(overall_diversity_samples),
    }
    for key in metric_keys:
        vals = []
        for row in rows:
            metrics = row.get("diffh2o_native", {})
            hand_metric_mean = (
                metrics.get("hand_metric_mean", {}) if isinstance(metrics, dict) else {}
            )
            value = hand_metric_mean.get(key)
            if value is not None:
                vals.append(float(value))
        out[key] = float(np.mean(vals)) if vals else None
    return out


def _diffh2o_native_summary_rows(all_results: list[dict]) -> list[dict]:
    grouped_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    fallback_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    gt_source_file: dict[str, Optional[str]] = {"seen": None, "unseen": None}
    for result in all_results:
        file_name = result.get("file_name", "")
        split = _split_tag_from_file_name(file_name)
        split_rows = result.get("per_sample_rows", [])
        if split not in grouped_rows or not split_rows:
            continue
        if not _allow_gt_source_file(file_name):
            continue
        if not fallback_rows[split]:
            fallback_rows[split] = list(split_rows)
            if gt_source_file[split] is None:
                gt_source_file[split] = file_name
        has_gt = any(row.get("wrist_traj_gt") is not None for row in split_rows)
        if has_gt and not grouped_rows[split]:
            grouped_rows[split] = list(split_rows)
            gt_source_file[split] = file_name
    for split in grouped_rows:
        if not grouped_rows[split]:
            grouped_rows[split] = fallback_rows[split]

    rows_by_split: dict[str, list[dict]] = {"seen": [], "unseen": [], "other": []}
    for split_name, file_name, split_rows in _aggregate_results_by_split(all_results):
        agg = _diffh2o_native_aggregate(split_rows)
        agg["split"] = split_name
        agg["file_name"] = file_name
        agg["is_gt_row"] = False
        rows_by_split.setdefault(split_name, []).append(agg)

    rows = []
    for split_name in ("seen", "unseen", "other"):
        split_rows = rows_by_split.get(split_name, [])
        if not split_rows:
            continue
        if split_name in {"seen", "unseen"} and grouped_rows.get(split_name):
            gt_rows = [
                row
                for row in grouped_rows[split_name]
                if row.get("wrist_traj_gt") is not None
            ]
            if gt_rows:
                gt_agg = _diffh2o_native_aggregate_from_gt(gt_rows)
                gt_label = f"{split_name.capitalize()}_G.T"
                if gt_source_file.get(split_name):
                    gt_label = f"{gt_label} (from {gt_source_file[split_name]})"
                gt_agg["split"] = split_name
                gt_agg["file_name"] = gt_label
                gt_agg["samples"] = int(len(gt_rows))
                gt_agg["is_gt_row"] = True
                rows.append(gt_agg)
        rows.extend(split_rows)
    return rows


def _diffh2o_native_aggregate_from_gt(rows: list[dict]) -> dict:
    sample_diversity, sample_diversity_prompts = _sample_diversity_m(
        rows, traj_key="wrist_traj_gt"
    )
    overall_diversity, overall_diversity_samples = _overall_diversity_m(
        rows, traj_key="wrist_traj_gt"
    )
    return {
        "samples": int(len(rows)),
        "sample_diversity_m": float(sample_diversity),
        "sample_diversity_prompts": int(sample_diversity_prompts),
        "overall_diversity_m": float(overall_diversity),
        "overall_diversity_samples": int(overall_diversity_samples),
        "iv_count_mean": None,
        "iv_count_max": None,
        "inter_depth_mean": None,
        "inter_depth_max": None,
        "contact_ratio_mean": None,
        "contact_ratio_max": None,
    }


def _project_metric_aggregate(rows: list[dict], row_key: str) -> dict:
    metric_keys = [
        "inter_volume_mean",
        "inter_volume_contact",
        "inter_volume_max",
        "inter_depth_mean",
        "inter_depth_contact",
        "inter_depth_max",
        "contact_ratio_mean",
        "contact_ratio_contact",
        "contact_ratio_max",
        "contact_ratio_off_ground",
        "off_ground_contact_ratio",
        "off_ground_contact_rate",
        "jerk",
        "jerk_pos",
        "jerk_ang",
    ]
    out = {"samples": int(len(rows)), "computed_samples": 0}
    computed = 0
    for key in metric_keys:
        vals = []
        for row in rows:
            metrics = row.get(row_key)
            if not isinstance(metrics, dict):
                continue
            hand_metric_mean = metrics.get("hand_metric_mean", {})
            value = None
            if key in ("off_ground_contact_ratio", "off_ground_contact_rate"):
                value = metrics.get(key)
            elif isinstance(hand_metric_mean, dict):
                value = hand_metric_mean.get(key)
            if value is not None and np.isfinite(float(value)):
                vals.append(float(value))
        out[key] = float(np.mean(vals)) if vals else None
    for row in rows:
        if isinstance(row.get(row_key), dict):
            computed += 1
    out["computed_samples"] = int(computed)
    return out


def _project_metric_summary_rows(all_results: list[dict], row_key: str) -> list[dict]:
    metric_gt_key = f"{row_key}_gt"
    base_rows = []
    grouped_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    fallback_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    gt_source_file: dict[str, Optional[str]] = {"seen": None, "unseen": None}

    for split_name, file_name, split_rows in _aggregate_results_by_split(all_results):
        agg = _project_metric_aggregate(split_rows, row_key)
        agg["split"] = split_name
        agg["file_name"] = file_name
        agg["is_gt_row"] = False
        base_rows.append(agg)
        if split_name not in grouped_rows or not split_rows:
            continue
        if not _allow_gt_source_file(file_name):
            continue
        if not fallback_rows[split_name]:
            fallback_rows[split_name] = list(split_rows)
            if gt_source_file[split_name] is None:
                gt_source_file[split_name] = file_name
        has_gt_metric = any(isinstance(row.get(metric_gt_key), dict) for row in split_rows)
        if has_gt_metric and not grouped_rows[split_name]:
            grouped_rows[split_name] = list(split_rows)
            gt_source_file[split_name] = file_name

    for split_name in grouped_rows:
        if not grouped_rows[split_name]:
            grouped_rows[split_name] = fallback_rows[split_name]

    rows_by_split: dict[str, list[dict]] = {"seen": [], "unseen": [], "other": []}
    for row in base_rows:
        rows_by_split.setdefault(str(row.get("split", "other")), []).append(row)

    out = []
    split_order = ("seen", "unseen", "other")
    for split_name in split_order:
        split_rows = rows_by_split.get(split_name, [])
        if not split_rows:
            continue
        if split_name in {"seen", "unseen"} and grouped_rows.get(split_name):
            gt_agg = _project_metric_aggregate(grouped_rows[split_name], metric_gt_key)
            if int(gt_agg.get("computed_samples", 0)) > 0:
                gt_label = f"{split_name.capitalize()}_G.T"
                if gt_source_file.get(split_name):
                    gt_label = f"{gt_label} (from {gt_source_file[split_name]})"
                gt_agg["split"] = split_name
                gt_agg["file_name"] = gt_label
                gt_agg["samples"] = int(len(grouped_rows[split_name]))
                gt_agg["is_gt_row"] = True
                out.append(gt_agg)
        out.extend(split_rows)
    return out


def _build_latethoi_project_table_rows(summary_rows: list[dict]) -> list[list[str]]:
    rows = []
    for row in summary_rows:
        rows.append(
            [
                str(row.get("split", "")),
                str(row.get("file_name", "")),
                f"{int(row.get('computed_samples', 0))}/{int(row.get('samples', 0))}",
                _format_float(row.get("inter_volume_mean"), digits=8),
                _format_float(row.get("inter_depth_mean"), digits=6),
                _format_float(row.get("inter_depth_max"), digits=6),
                _format_float(row.get("contact_ratio_mean"), digits=4),
                _format_float(row.get("contact_ratio_contact"), digits=4),
                _format_float(row.get("contact_ratio_off_ground"), digits=4),
                _format_float(row.get("jerk"), digits=6),
            ]
        )
    return rows


def _build_diffh2o_project_table_rows(summary_rows: list[dict]) -> list[list[str]]:
    rows = []
    for row in summary_rows:
        rows.append(
            [
                str(row.get("split", "")),
                str(row.get("file_name", "")),
                f"{int(row.get('computed_samples', 0))}/{int(row.get('samples', 0))}",
                _format_float(row.get("inter_volume_mean"), digits=8),
                _format_float(row.get("inter_depth_mean"), digits=6),
                _format_float(row.get("inter_depth_max"), digits=6),
                _format_float(row.get("contact_ratio_mean"), digits=4),
                _format_float(row.get("contact_ratio_contact"), digits=4),
                _format_float(row.get("jerk_pos"), digits=6),
                _format_float(row.get("jerk_ang"), digits=6),
            ]
        )
    return rows


def _bimart_metric_aggregate(rows: list[dict], row_key: str) -> dict:
    keys = ["jitter", "penetration_1cm", "contact"]
    out = {"samples": int(len(rows)), "computed_samples": 0}
    for key in keys:
        vals = []
        for row in rows:
            metrics = row.get(row_key)
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(key)
            if value is not None and np.isfinite(float(value)):
                vals.append(float(value))
        out[key] = float(np.mean(vals)) if vals else None
    out["computed_samples"] = int(
        sum(1 for row in rows if isinstance(row.get(row_key), dict))
    )
    return out


def _bimart_summary_rows(all_results: list[dict]) -> list[dict]:
    base_rows = []
    grouped_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    fallback_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    gt_source_file: dict[str, Optional[str]] = {"seen": None, "unseen": None}
    gt_apd_by_split: dict[str, Optional[float]] = {"seen": None, "unseen": None}

    for result in all_results:
        file_name = str(result.get("file_name", ""))
        split_name = _split_tag_from_file_name(file_name)
        split_rows = result.get("per_sample_rows", [])
        if split_rows:
            agg = _bimart_metric_aggregate(split_rows, "bimart")
            agg["split"] = split_name
            agg["file_name"] = file_name
            agg["apd_multi"] = (
                result.get("bimart_file_metrics", {}) or {}
            ).get("apd_multi")
            agg["is_gt_row"] = False
            base_rows.append(agg)
        if split_name not in grouped_rows or not split_rows:
            continue
        if not _allow_gt_source_file(file_name):
            continue
        if not fallback_rows[split_name]:
            fallback_rows[split_name] = list(split_rows)
            if gt_source_file[split_name] is None:
                gt_source_file[split_name] = file_name
            gt_apd_by_split[split_name] = (
                result.get("bimart_file_metrics_gt", {}) or {}
            ).get("apd_multi")
        has_gt_metric = any(isinstance(row.get("bimart_gt"), dict) for row in split_rows)
        if has_gt_metric and not grouped_rows[split_name]:
            grouped_rows[split_name] = list(split_rows)
            gt_source_file[split_name] = file_name
            gt_apd_by_split[split_name] = (
                result.get("bimart_file_metrics_gt", {}) or {}
            ).get("apd_multi")

    for split_name in grouped_rows:
        if not grouped_rows[split_name]:
            grouped_rows[split_name] = fallback_rows[split_name]

    rows_by_split: dict[str, list[dict]] = {"seen": [], "unseen": [], "other": []}
    for row in base_rows:
        rows_by_split.setdefault(str(row.get("split", "other")), []).append(row)

    out = []
    for split_name in ("seen", "unseen", "other"):
        split_rows = rows_by_split.get(split_name, [])
        if not split_rows:
            continue
        if split_name in {"seen", "unseen"} and grouped_rows.get(split_name):
            gt_agg = _bimart_metric_aggregate(grouped_rows[split_name], "bimart_gt")
            if int(gt_agg.get("computed_samples", 0)) > 0:
                gt_label = f"{split_name.capitalize()}_G.T"
                if gt_source_file.get(split_name):
                    gt_label = f"{gt_label} (from {gt_source_file[split_name]})"
                gt_agg["split"] = split_name
                gt_agg["file_name"] = gt_label
                gt_agg["samples"] = int(len(grouped_rows[split_name]))
                gt_agg["apd_multi"] = gt_apd_by_split.get(split_name)
                gt_agg["is_gt_row"] = True
                out.append(gt_agg)
        out.extend(split_rows)
    return out


def _build_bimart_table_rows(summary_rows: list[dict]) -> list[list[str]]:
    rows = []
    for row in summary_rows:
        rows.append(
            [
                str(row.get("split", "")),
                str(row.get("file_name", "")),
                f"{int(row.get('computed_samples', 0))}/{int(row.get('samples', 0))}",
                _format_float(row.get("apd_multi"), digits=5),
                _format_float(row.get("jitter"), digits=5),
                _format_float(row.get("penetration_1cm"), digits=4),
                _format_float(row.get("contact"), digits=4),
            ]
        )
    return rows


_LATETHOI_PHYSIC_EVALUATOR = None
_DIFFH2O_METRIC_CLASS = None
_PROJECT_METRIC_WARNED: set[str] = set()


def _warn_project_metric_once(key: str, message: str) -> None:
    if key in _PROJECT_METRIC_WARNED:
        return
    _PROJECT_METRIC_WARNED.add(key)
    print(message)


def _import_latethoi_physic_evaluator(repo_dir: str):
    global _LATETHOI_PHYSIC_EVALUATOR
    if _LATETHOI_PHYSIC_EVALUATOR is not None:
        return _LATETHOI_PHYSIC_EVALUATOR
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    inserted = False
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        inserted = True
    try:
        module = importlib.import_module("engine.evaluation.physic_metrics")
        _LATETHOI_PHYSIC_EVALUATOR = module.PhysicEvaluation()
        return _LATETHOI_PHYSIC_EVALUATOR
    finally:
        if inserted:
            try:
                sys.path.remove(repo_dir)
            except ValueError:
                pass


def _import_diffh2o_metric_class(repo_dir: str):
    global _DIFFH2O_METRIC_CLASS
    if _DIFFH2O_METRIC_CLASS is not None:
        return _DIFFH2O_METRIC_CLASS
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    inserted = False
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        inserted = True
    try:
        module = importlib.import_module("eval.metrics.metrics")
        _DIFFH2O_METRIC_CLASS = module.ObjectContactMetrics
        return _DIFFH2O_METRIC_CLASS
    finally:
        if inserted:
            try:
                sys.path.remove(repo_dir)
            except ValueError:
                pass


def _slice_frame_indices(params, indices: np.ndarray):
    t = _to_torch(params)
    if t.ndim == 1:
        return t.unsqueeze(0)
    idx = torch.as_tensor(indices, dtype=torch.long, device=t.device)
    return t.index_select(0, idx)


def _sampled_eval_indices(nframes: int, max_frames: int) -> np.ndarray:
    nframes = int(nframes)
    max_frames = int(max(1, max_frames))
    if nframes <= 1:
        return np.zeros((1,), dtype=np.int64)
    count = min(nframes, max_frames)
    return np.linspace(0, nframes - 1, num=count).round().astype(np.int64)


def _object_mesh_sequence_from_params(
    obj_mesh: Optional[trimesh.Trimesh], obj_params, indices: np.ndarray
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if obj_mesh is None:
        return None, None, None
    vertices = np.asarray(obj_mesh.vertices, dtype=np.float32)
    faces = np.asarray(obj_mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
        return None, None, None
    pose = _pose9_sequence(_slice_frame_indices(obj_params, indices))
    trans = pose[:, :3]
    rot6d = pose[:, 3:9]
    rotmat = rot6d_to_rotmat(rot6d).reshape(-1, 3, 3)
    vertices_t = torch.as_tensor(vertices, dtype=torch.float32, device=rotmat.device)
    seq = torch.einsum("tij,kj->tki", rotmat, vertices_t) + trans.unsqueeze(1)
    axis_angle = rot6d_to_axis_angle(rot6d).reshape(-1, 3)
    obj_poses = torch.cat([trans, axis_angle], dim=-1)
    return _to_numpy(seq), faces, _to_numpy(obj_poses)


def _aggregate_two_hand_project_metrics(left: Optional[dict], right: Optional[dict]) -> dict:
    parts = [p for p in (left, right) if isinstance(p, dict)]
    if not parts:
        return {}
    keys = sorted({k for part in parts for k in part.keys()})
    out = {}
    for key in keys:
        vals = []
        for part in parts:
            value = part.get(key)
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                vals.append(float(value))
        out[key] = float(np.mean(vals)) if vals else None
    return out


def _compute_latethoi_project_metrics(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_params,
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    l_faces: np.ndarray,
    r_faces: np.ndarray,
    use_left: bool,
    use_right: bool,
    repo_dir: str,
    max_frames: int,
) -> Optional[dict]:
    indices = _sampled_eval_indices(
        min(
            _sequence_length(obj_params),
            l_seq.shape[0] if use_left and l_seq is not None else 10**9,
            r_seq.shape[0] if use_right and r_seq is not None else 10**9,
        ),
        max_frames,
    )
    obj_v, obj_f, _obj_poses = _object_mesh_sequence_from_params(
        obj_mesh, obj_params, indices
    )
    if obj_v is None or obj_f is None:
        return None
    evaluator = _import_latethoi_physic_evaluator(repo_dir)
    l_v = l_seq[indices] if use_left and l_seq is not None and l_seq.shape[0] else None
    r_v = r_seq[indices] if use_right and r_seq is not None and r_seq.shape[0] else None
    if r_v is None:
        r_v = np.zeros((len(indices), 0, 3), dtype=np.float32)
        r_f = np.zeros((0, 3), dtype=np.int64)
    else:
        r_f = r_faces
    l_f = l_faces if l_v is not None else None
    result = evaluator(
        obj_v.astype(np.float32),
        obj_f.astype(np.int64),
        r_v.astype(np.float32),
        r_f.astype(np.int64),
        None if l_v is None else l_v.astype(np.float32),
        None if l_f is None else l_f.astype(np.int64),
        eval_n_frames=int(len(indices)),
    )
    right = result.get("rhand", {}) if isinstance(result, dict) else {}
    left = result.get("lhand", {}) if isinstance(result, dict) else {}
    mean = _aggregate_two_hand_project_metrics(left if use_left else None, right if use_right else None)
    out = dict(result) if isinstance(result, dict) else {}
    out["hand_metric_mean"] = mean
    return out


def _compute_diffh2o_project_metrics(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_params,
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    l_faces: np.ndarray,
    r_faces: np.ndarray,
    use_left: bool,
    use_right: bool,
    repo_dir: str,
    max_frames: int,
) -> Optional[dict]:
    indices = _sampled_eval_indices(
        min(
            _sequence_length(obj_params),
            l_seq.shape[0] if use_left and l_seq is not None else 10**9,
            r_seq.shape[0] if use_right and r_seq is not None else 10**9,
        ),
        max_frames,
    )
    obj_v, obj_f, obj_poses = _object_mesh_sequence_from_params(
        obj_mesh, obj_params, indices
    )
    if obj_v is None or obj_f is None or obj_poses is None:
        return None
    metric_class = _import_diffh2o_metric_class(repo_dir)

    def _run_one(hand_v: np.ndarray, hand_f: np.ndarray) -> dict:
        class _NoopPool:
            def close(self):
                return None

            def join(self):
                return None

        metric_module = sys.modules.get(metric_class.__module__)
        multiprocessing_module = getattr(metric_module, "multiprocessing", None)
        original_pool = (
            getattr(multiprocessing_module, "Pool", None)
            if multiprocessing_module is not None
            else None
        )
        if multiprocessing_module is not None:
            multiprocessing_module.Pool = lambda *args, **kwargs: _NoopPool()
        try:
            metric = metric_class(device="cpu", use_multiprocessing=False)
        finally:
            if multiprocessing_module is not None and original_pool is not None:
                multiprocessing_module.Pool = original_pool
        try:
            metric.compute_metrics(
                sbj_vertices=torch.as_tensor(hand_v, dtype=torch.float32),
                sbj_faces=torch.as_tensor(hand_f, dtype=torch.long),
                obj_vertices=torch.as_tensor(obj_v, dtype=torch.float32),
                obj_faces=torch.as_tensor(obj_f, dtype=torch.long),
                obj_poses=torch.as_tensor(obj_poses, dtype=torch.float32),
                start_id=0,
                end_id=int(len(indices)),
            )
            return {
                **getattr(metric, "volume", {}),
                **getattr(metric, "depth", {}),
                **getattr(metric, "contact_ratio", {}),
                **getattr(metric, "jerk", {}),
            }
        finally:
            pool = getattr(metric, "pool", None)
            if pool is not None:
                try:
                    pool.close()
                    pool.join()
                except Exception:
                    pass

    left = None
    right = None
    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        left = _run_one(l_seq[indices].astype(np.float32), l_faces.astype(np.int64))
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        right = _run_one(r_seq[indices].astype(np.float32), r_faces.astype(np.int64))
    return {
        "lhand": left or {},
        "rhand": right or {},
        "hand_metric_mean": _aggregate_two_hand_project_metrics(left, right),
    }


def _concat_two_hand_vertices(
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    use_left: bool,
    use_right: bool,
) -> np.ndarray:
    parts = []
    if use_left and l_seq is not None and l_seq.shape[0] > 0:
        parts.append(np.asarray(l_seq, dtype=np.float32))
    if use_right and r_seq is not None and r_seq.shape[0] > 0:
        parts.append(np.asarray(r_seq, dtype=np.float32))
    if not parts:
        return np.zeros((0, 0, 3), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    t = min(int(parts[0].shape[0]), int(parts[1].shape[0]))
    return np.concatenate([parts[0][:t], parts[1][:t]], axis=1).astype(np.float32)


def _nearest_surface_signed_values(
    hand_local: np.ndarray,
    obj_surface_points: Optional[np.ndarray],
    obj_surface_normals: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if obj_surface_points is None or obj_surface_normals is None:
        return None
    surface_points = np.asarray(obj_surface_points, dtype=np.float64)
    surface_normals = np.asarray(obj_surface_normals, dtype=np.float64)
    if (
        surface_points.ndim != 2
        or surface_points.shape[1] != 3
        or surface_normals.shape != surface_points.shape
        or surface_points.shape[0] == 0
    ):
        return None
    normal_lengths = np.linalg.norm(surface_normals, axis=1)
    valid_surface = (
        np.all(np.isfinite(surface_points), axis=1)
        & np.all(np.isfinite(surface_normals), axis=1)
        & np.isfinite(normal_lengths)
        & (normal_lengths > 1e-8)
    )
    surface_points = surface_points[valid_surface]
    surface_normals = surface_normals[valid_surface]
    normal_lengths = normal_lengths[valid_surface]
    if surface_points.shape[0] == 0:
        return None
    nearest_surface_idx = np.argmin(
        np.linalg.norm(hand_local[:, None, :] - surface_points[None, :, :], axis=2),
        axis=1,
    )
    nearest_surface_points = surface_points[nearest_surface_idx]
    nearest_surface_normals = (
        surface_normals[nearest_surface_idx] / normal_lengths[nearest_surface_idx, None]
    )
    return np.sum(
        (hand_local - nearest_surface_points) * nearest_surface_normals,
        axis=1,
    )


def _compute_bimart_sample_metrics(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_params,
    obj_surface_points: Optional[np.ndarray],
    obj_surface_normals: Optional[np.ndarray],
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    use_left: bool,
    use_right: bool,
) -> Optional[dict]:
    hand_seq = _concat_two_hand_vertices(l_seq, r_seq, use_left, use_right)
    if hand_seq.ndim != 3 or hand_seq.shape[0] == 0 or hand_seq.shape[1] == 0:
        return None

    jitter = 0.0
    if hand_seq.shape[0] >= 3:
        acceleration = np.diff(np.diff(hand_seq, axis=0), axis=0)
        jitter = float(np.mean(np.linalg.norm(acceleration, axis=-1)) * 100.0)

    if obj_mesh is None:
        return {
            "jitter": jitter,
            "penetration_1cm": None,
            "contact": None,
            "moving_frames": 0,
            "contact_frames": 0,
            "penetration_1cm_frames": 0,
        }

    seq_len = min(int(hand_seq.shape[0]), int(_sequence_length(obj_params)))
    if seq_len <= 0:
        return None
    indices = np.arange(seq_len, dtype=np.int64)
    obj_v_seq, _obj_faces, _obj_poses = _object_mesh_sequence_from_params(
        obj_mesh, obj_params, indices
    )
    if obj_v_seq is None or obj_v_seq.shape[0] == 0:
        return None
    seq_len = min(seq_len, int(obj_v_seq.shape[0]))
    hand_seq = hand_seq[:seq_len]
    obj_v_seq = np.asarray(obj_v_seq[:seq_len], dtype=np.float64)
    obj_pose = _to_numpy(
        _pose9_sequence(_slice_frame_indices(obj_params, indices[:seq_len]))
    ).astype(np.float64)

    moving_frames = 0
    contact_frames = 0
    penetration_1cm_frames = 0
    for frame_idx in range(max(0, seq_len - 1)):
        if np.max(np.abs(obj_v_seq[frame_idx + 1] - obj_v_seq[frame_idx])) <= 1e-5:
            continue
        moving_frames += 1
        hand_world = np.asarray(hand_seq[frame_idx], dtype=np.float64)
        pose = obj_pose[frame_idx]
        trans = pose[:3]
        rot = _to_numpy(rot6d_to_rotmat(_to_torch(pose[3:9]).reshape(1, 6))).reshape(
            3, 3
        ).astype(np.float64)
        hand_local = np.einsum("ni,ij->nj", hand_world - trans, rot)
        signed = _nearest_surface_signed_values(
            hand_local, obj_surface_points, obj_surface_normals
        )
        if signed is not None:
            if np.any(signed < 0.005):
                contact_frames += 1
            inside = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
            if np.any(inside & (signed < -0.01)):
                penetration_1cm_frames += 1
        else:
            inside = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
            if np.any(inside):
                contact_frames += 1
                try:
                    _closest_local, dist_local, _ = trimesh.proximity.closest_point(
                        obj_mesh, hand_local[inside]
                    )
                    if np.any(np.asarray(dist_local, dtype=np.float64) > 0.01):
                        penetration_1cm_frames += 1
                except Exception:
                    pass

    return {
        "jitter": jitter,
        "penetration_1cm": (
            float(penetration_1cm_frames / moving_frames * 100.0)
            if moving_frames > 0
            else None
        ),
        "contact": (
            float(contact_frames / moving_frames * 100.0)
            if moving_frames > 0
            else None
        ),
        "moving_frames": int(moving_frames),
        "contact_frames": int(contact_frames),
        "penetration_1cm_frames": int(penetration_1cm_frames),
    }


def _compute_bimart_apd(groups: dict[str, list[np.ndarray]]) -> Optional[float]:
    apd_vals = []
    for seqs in groups.values():
        valid = []
        for seq in seqs:
            arr = np.asarray(seq, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0:
                valid.append(arr)
        if len(valid) < 2:
            continue
        min_t = min(int(seq.shape[0]) for seq in valid)
        min_v = min(int(seq.shape[1]) for seq in valid)
        if min_t <= 0 or min_v <= 0:
            continue
        stacked = np.stack([seq[:min_t, :min_v] for seq in valid], axis=0)
        errors = np.linalg.norm(
            stacked[:, None, :, :, :] - stacked[None, :, :, :, :], axis=-1
        )
        mpjpe_matrix = np.mean(errors, axis=(2, 3))
        upper = np.triu_indices(stacked.shape[0], k=1)
        if upper[0].size == 0:
            continue
        apd_vals.append(float(np.mean(mpjpe_matrix[upper]) * 100.0))
    if not apd_vals:
        return None
    return float(np.mean(apd_vals))


def evaluate_file(
    path: str,
    obj_pc_by_key: dict,
    obj_pc_normals_by_key: dict,
    obj_mesh_by_key: dict,
    object_axis_alignment_by_key: dict,
    l_hand_layer,
    r_hand_layer,
    contact_threshold: float,
    text_filter: Optional[set[str]],
    min_cr_for_valid: float,
    compute_id: bool,
    compute_iv: bool,
    compare_gt: bool,
    visualize_eval: bool = False,
    visualize_hand_style: str = "mesh",
    sample_idx_filter: Optional[set[int]] = None,
    visualize_gt_vr_fail_only: bool = False,
    visualize_gt_topk: int = 10,
    visualize_metric_rankings: bool = False,
    visualize_metric_topk: int = 5,
    run_project_native_metrics: bool = True,
    latethoi_repo_dir: Optional[str] = None,
    diffh2o_repo_dir: Optional[str] = None,
    project_native_eval_frames: int = 50,
    run_bimart_metrics: bool = True,
) -> dict:
    items = _load_items_from_path(path)
    total_samples = sum(
        1 for record in items for _ in _iter_samples_from_record(record)
    )

    per_sample_rows = []
    gt_fail_visualizations = []
    gt_rank_visualizations = []
    pred_rank_visualizations = []
    bimart_apd_groups: dict[str, list[np.ndarray]] = {}
    bimart_apd_groups_gt: dict[str, list[np.ndarray]] = {}
    sample_pbar = tqdm.tqdm(
        total=total_samples,
        desc=f"samples:{os.path.basename(path)}",
        leave=False,
    )
    for record in items:
        for sample in _iter_samples_from_record(record):
            sample_pbar.update(1)
            if (
                sample_idx_filter is not None
                and int(sample.get("sample_idx", -1)) not in sample_idx_filter
            ):
                continue
            text = sample["text"]
            text_norm = text.strip().lower()
            if text_filter is not None and text_norm not in text_filter:
                continue

            obj_key = _extract_object_key(text)
            object_meta = sample.get("object_meta")
            if obj_key not in obj_pc_by_key and isinstance(object_meta, dict):
                meta_obj_name = object_meta.get("object_name")
                if meta_obj_name is not None:
                    meta_key = str(meta_obj_name).strip().lower()
                    if meta_key in obj_pc_by_key:
                        obj_key = meta_key
            if obj_key not in obj_pc_by_key:
                print(f"[WARN] unresolved object key: '{obj_key}' from text '{text}'")
                continue

            use_left, use_right = _selected_hands(text)
            if not use_left and not use_right:
                print(
                    f"[WARN] skip sample without explicit hand in text: "
                    f"(text='{text}', sample_idx={sample.get('sample_idx', -1)})"
                )
                continue
            # Align usable frame count across streams, then reconstruct only the
            # tail frames needed for the new metrics (VR uses last 3; others last 1).
            nframes = _sequence_length(sample["obj_params"])
            if use_left:
                nframes = min(nframes, _sequence_length(sample["lhand_params"]))
            if use_right:
                nframes = min(nframes, _sequence_length(sample["rhand_params"]))
            if nframes <= 0:
                continue
            metric_frames = min(nframes, 3)

            try:
                obj_seq = _to_numpy(
                    process_obj_result(
                        obj_pc_by_key[obj_key],
                        _slice_last_frames(sample["obj_params"], metric_frames),
                    )
                )
                if use_left:
                    l_seq_t, l_joints_t, l_faces_t = process_hand_result(
                        l_hand_layer,
                        _slice_last_frames(sample["lhand_params"], metric_frames),
                    )
                    l_seq = _to_numpy(l_seq_t)
                    l_joints = _to_numpy(l_joints_t)
                    l_faces = _to_numpy(l_faces_t)
                else:
                    l_seq = np.zeros((0, 0, 3), dtype=np.float32)
                    l_joints = np.zeros((0, 0, 3), dtype=np.float32)
                    l_faces = np.zeros((0, 3), dtype=np.int64)

                if use_right:
                    r_seq_t, r_joints_t, r_faces_t = process_hand_result(
                        r_hand_layer,
                        _slice_last_frames(sample["rhand_params"], metric_frames),
                    )
                    r_seq = _to_numpy(r_seq_t)
                    r_joints = _to_numpy(r_joints_t)
                    r_faces = _to_numpy(r_faces_t)
                else:
                    r_seq = np.zeros((0, 0, 3), dtype=np.float32)
                    r_joints = np.zeros((0, 0, 3), dtype=np.float32)
                    r_faces = np.zeros((0, 3), dtype=np.int64)
            except Exception as ex:
                print(
                    f"[WARN] skip sample due to reconstruction failure in {os.path.basename(path)} "
                    f"(text='{text}', sample_idx={sample.get('sample_idx', -1)}): {ex}"
                )
                continue
            if (
                not np.isfinite(obj_seq).all()
                or not np.isfinite(l_seq).all()
                or not np.isfinite(r_seq).all()
                or not np.isfinite(l_joints).all()
                or not np.isfinite(r_joints).all()
            ):
                print(
                    f"[WARN] skip sample with non-finite values in {os.path.basename(path)} "
                    f"(text='{text}', sample_idx={sample.get('sample_idx', -1)})"
                )
                continue

            try:
                metric = _sample_metrics(
                    obj_seq,
                    sample["obj_params"],
                    obj_mesh_by_key.get(obj_key),
                    _to_numpy(obj_pc_by_key.get(obj_key)),
                    obj_pc_normals_by_key.get(obj_key),
                    l_seq,
                    r_seq,
                    l_joints,
                    r_joints,
                    l_faces,
                    r_faces,
                    use_left,
                    use_right,
                    contact_threshold,
                    compute_id=compute_id,
                    compute_iv=compute_iv,
                    compute_closest_points=bool(
                        visualize_eval or visualize_metric_rankings
                    ),
                )
            except Exception as ex:
                print(
                    f"[WARN] skip sample due to metric failure in {os.path.basename(path)} "
                    f"(text='{text}', sample_idx={sample.get('sample_idx', -1)}, obj='{obj_key}'): {ex}"
                )
                continue
            if metric is None:
                continue

            bimart_metrics = None
            bimart_metrics_gt = None
            bimart_l_seq = np.zeros((0, 0, 3), dtype=np.float32)
            bimart_r_seq = np.zeros((0, 0, 3), dtype=np.float32)
            if run_bimart_metrics:
                try:
                    if use_left:
                        bimart_l_seq_t, _bimart_l_joints_t, _bimart_l_faces_t = (
                            process_hand_result(l_hand_layer, sample["lhand_params"])
                        )
                        bimart_l_seq = _to_numpy(bimart_l_seq_t)
                    else:
                        bimart_l_seq = np.zeros((0, 0, 3), dtype=np.float32)
                    if use_right:
                        bimart_r_seq_t, _bimart_r_joints_t, _bimart_r_faces_t = (
                            process_hand_result(r_hand_layer, sample["rhand_params"])
                        )
                        bimart_r_seq = _to_numpy(bimart_r_seq_t)
                    else:
                        bimart_r_seq = np.zeros((0, 0, 3), dtype=np.float32)
                    bimart_metrics = _compute_bimart_sample_metrics(
                        obj_mesh_by_key.get(obj_key),
                        sample["obj_params"],
                        _to_numpy(obj_pc_by_key.get(obj_key)),
                        obj_pc_normals_by_key.get(obj_key),
                        bimart_l_seq,
                        bimart_r_seq,
                        use_left,
                        use_right,
                    )
                    bimart_hand_seq = _concat_two_hand_vertices(
                        bimart_l_seq, bimart_r_seq, use_left, use_right
                    )
                    if bimart_hand_seq.shape[0] > 0 and bimart_hand_seq.shape[1] > 0:
                        bimart_apd_groups.setdefault(str(text), []).append(
                            bimart_hand_seq
                        )
                except Exception as ex:
                    _warn_project_metric_once(
                        f"bimart-metric:{os.path.basename(path)}",
                        f"[WARN] BimArt metric adapter failed for "
                        f"{os.path.basename(path)}; later samples will be skipped "
                        f"silently if the same issue repeats. First error: {ex}",
                    )

            gt_extra = {
                "cr_gt": None,
                "gt_iou": None,
                "gt_precision": None,
                "gt_recall": None,
                "gt_f1": None,
                "id_gt": None,
                "id_max_gt": None,
                "iv_gt": None,
            }
            nframes_gt_rel = None
            metric_gt = None
            latethoi_lastframe_gt = None
            gt_obj_seq = None
            gt_l_seq = None
            gt_r_seq = None
            gt_l_joints = None
            gt_r_joints = None
            gt_l_faces = None
            gt_r_faces = None
            gt_obj_params = sample.get("gt_obj_params")
            gt_lhand_params = sample.get("gt_lhand_params")
            gt_rhand_params = sample.get("gt_rhand_params")
            gt_hand_available = bool(
                gt_lhand_params is not None or gt_rhand_params is not None
            )
            wrist_traj_gt = None
            wrist_traj_gt_aligned = None
            if compare_gt and gt_obj_params is not None:
                try:
                    nframes_gt = _sequence_length(gt_obj_params)
                    if use_left:
                        nframes_gt = min(
                            nframes_gt,
                            _sequence_length(
                                gt_lhand_params
                                if gt_lhand_params is not None
                                else sample["lhand_params"]
                            ),
                        )
                    if use_right:
                        nframes_gt = min(
                            nframes_gt,
                            _sequence_length(
                                gt_rhand_params
                                if gt_rhand_params is not None
                                else sample["rhand_params"]
                            ),
                        )
                    if nframes_gt <= 0:
                        raise ValueError("GT has no valid frames")
                    metric_frames_gt = min(nframes_gt, 3)
                    gt_obj_seq = _to_numpy(
                        process_obj_result(
                            obj_pc_by_key[obj_key],
                            _slice_last_frames(gt_obj_params, metric_frames_gt),
                        )
                    )
                    gt_l_seq = l_seq
                    gt_r_seq = r_seq
                    gt_l_faces = l_faces
                    gt_r_faces = r_faces
                    gt_l_params_for_traj = sample["lhand_params"]
                    gt_r_params_for_traj = sample["rhand_params"]
                    gt_l_joints = l_joints
                    gt_r_joints = r_joints
                    if use_left and gt_lhand_params is not None:
                        gt_l_seq_t, gt_l_joints_t, gt_l_faces_t = process_hand_result(
                            l_hand_layer,
                            _slice_last_frames(gt_lhand_params, metric_frames_gt),
                        )
                        gt_l_seq = _to_numpy(gt_l_seq_t)
                        gt_l_joints = _to_numpy(gt_l_joints_t)
                        gt_l_faces = _to_numpy(gt_l_faces_t)
                        gt_l_params_for_traj = gt_lhand_params
                    if use_right and gt_rhand_params is not None:
                        gt_r_seq_t, gt_r_joints_t, gt_r_faces_t = process_hand_result(
                            r_hand_layer,
                            _slice_last_frames(gt_rhand_params, metric_frames_gt),
                        )
                        gt_r_seq = _to_numpy(gt_r_seq_t)
                        gt_r_joints = _to_numpy(gt_r_joints_t)
                        gt_r_faces = _to_numpy(gt_r_faces_t)
                        gt_r_params_for_traj = gt_rhand_params
                    nframes_gt_metric = gt_obj_seq.shape[0]
                    if use_left:
                        nframes_gt_metric = min(nframes_gt_metric, gt_l_seq.shape[0])
                    if use_right:
                        nframes_gt_metric = min(nframes_gt_metric, gt_r_seq.shape[0])
                    if nframes_gt_metric > 0:
                        nframes_gt_rel = int(nframes_gt)
                        metric_gt = _sample_metrics(
                            gt_obj_seq[:nframes_gt_metric],
                            sample["gt_obj_params"],
                            obj_mesh_by_key.get(obj_key),
                            _to_numpy(obj_pc_by_key.get(obj_key)),
                            obj_pc_normals_by_key.get(obj_key),
                            gt_l_seq[:nframes_gt_metric],
                            gt_r_seq[:nframes_gt_metric],
                            gt_l_joints[:nframes_gt_metric],
                            gt_r_joints[:nframes_gt_metric],
                            gt_l_faces,
                            gt_r_faces,
                            use_left,
                            use_right,
                            contact_threshold,
                            compute_id=compute_id,
                            compute_iv=compute_iv,
                            compute_closest_points=bool(
                                visualize_eval or visualize_metric_rankings
                            ),
                        )
                        wrist_traj_gt = _canonical_wrist_trajectory_array(
                            gt_l_params_for_traj,
                            gt_r_params_for_traj,
                            gt_obj_params,
                            use_left,
                            use_right,
                            nframes_gt,
                        )
                        wrist_traj_gt_aligned = _apply_object_axis_alignment(
                            wrist_traj_gt,
                            object_axis_alignment_by_key.get(obj_key),
                        )
                        if metric_gt is not None:
                            latethoi_lastframe_gt = _latethoi_lastframe_sample_metrics(
                                gt_obj_seq[:nframes_gt_metric],
                                sample["gt_obj_params"],
                                obj_mesh_by_key.get(obj_key),
                                gt_l_seq[:nframes_gt_metric],
                                gt_r_seq[:nframes_gt_metric],
                                gt_l_faces,
                                gt_r_faces,
                                use_left,
                                use_right,
                            )
                            gt_extra["cr_gt"] = float(metric_gt["cr"])
                            gt_extra["id_gt"] = (
                                float(metric_gt["id_mm"])
                                if metric_gt.get("id_mm") is not None
                                else None
                            )
                            gt_extra["id_max_gt"] = (
                                float(metric_gt["id_max_mm"])
                                if metric_gt.get("id_max_mm") is not None
                                else None
                            )
                            gt_extra["iv_gt"] = (
                                float(metric_gt["iv_cm3"])
                                if metric_gt.get("iv_cm3") is not None
                                else None
                            )
                            gt_extra["valid_contact_gt"] = bool(
                                metric_gt["valid_contact"]
                            )
                            gt_extra["success_gt"] = bool(metric_gt["success"])
                            gt_extra.update(
                                _contact_binary_metrics(
                                    metric["pred_contact_mask"],
                                    metric_gt["pred_contact_mask"],
                                )
                            )
                            if run_bimart_metrics:
                                try:
                                    gt_bimart_l_seq = bimart_l_seq
                                    gt_bimart_r_seq = bimart_r_seq
                                    if use_left and gt_lhand_params is not None:
                                        gt_bimart_l_seq_t, _gt_bimart_l_joints_t, _gt_bimart_l_faces_t = process_hand_result(
                                            l_hand_layer, gt_lhand_params
                                        )
                                        gt_bimart_l_seq = _to_numpy(gt_bimart_l_seq_t)
                                    if use_right and gt_rhand_params is not None:
                                        gt_bimart_r_seq_t, _gt_bimart_r_joints_t, _gt_bimart_r_faces_t = process_hand_result(
                                            r_hand_layer, gt_rhand_params
                                        )
                                        gt_bimart_r_seq = _to_numpy(gt_bimart_r_seq_t)
                                    bimart_metrics_gt = _compute_bimart_sample_metrics(
                                        obj_mesh_by_key.get(obj_key),
                                        sample["gt_obj_params"],
                                        _to_numpy(obj_pc_by_key.get(obj_key)),
                                        obj_pc_normals_by_key.get(obj_key),
                                        gt_bimart_l_seq,
                                        gt_bimart_r_seq,
                                        use_left,
                                        use_right,
                                    )
                                    gt_bimart_hand_seq = _concat_two_hand_vertices(
                                        gt_bimart_l_seq,
                                        gt_bimart_r_seq,
                                        use_left,
                                        use_right,
                                    )
                                    if (
                                        gt_bimart_hand_seq.shape[0] > 0
                                        and gt_bimart_hand_seq.shape[1] > 0
                                    ):
                                        bimart_apd_groups_gt.setdefault(
                                            str(text), []
                                        ).append(gt_bimart_hand_seq)
                                except Exception as ex:
                                    _warn_project_metric_once(
                                        f"bimart-metric-gt:{os.path.basename(path)}",
                                        f"[WARN] GT BimArt metric adapter failed for "
                                        f"{os.path.basename(path)}; later GT samples will be skipped "
                                        f"silently if the same issue repeats. First error: {ex}",
                                    )
                except Exception:
                    pass

            wrist_traj = _canonical_wrist_trajectory_array(
                sample["lhand_params"],
                sample["rhand_params"],
                sample["obj_params"],
                use_left,
                use_right,
                nframes,
            )
            latethoi_lastframe = _latethoi_lastframe_sample_metrics(
                obj_seq,
                sample["obj_params"],
                obj_mesh_by_key.get(obj_key),
                l_seq,
                r_seq,
                l_faces,
                r_faces,
                use_left,
                use_right,
            )
            text2hoi_metrics = _text2hoi_sample_metrics(
                obj_seq,
                l_seq,
                r_seq,
                l_joints,
                r_joints,
                l_faces,
                r_faces,
                use_left,
                use_right,
            )
            diffh2o_native_metrics = _diffh2o_native_sample_metrics(
                obj_mesh_by_key.get(obj_key),
                sample["obj_params"],
                l_seq,
                r_seq,
                use_left,
                use_right,
                max_frames=project_native_eval_frames,
            )
            latethoi_project_metrics = None
            diffh2o_project_metrics = None
            latethoi_project_metrics_gt = None
            diffh2o_project_metrics_gt = None
            if run_project_native_metrics:
                try:
                    project_indices = _sampled_eval_indices(
                        nframes, project_native_eval_frames
                    )
                    project_obj_params = _slice_frame_indices(
                        sample["obj_params"], project_indices
                    )
                    if use_left:
                        project_l_seq_t, _project_l_joints_t, project_l_faces_t = (
                            process_hand_result(
                                l_hand_layer,
                                _slice_frame_indices(
                                    sample["lhand_params"], project_indices
                                ),
                            )
                        )
                        project_l_seq = _to_numpy(project_l_seq_t)
                        project_l_faces = _to_numpy(project_l_faces_t)
                    else:
                        project_l_seq = np.zeros((0, 0, 3), dtype=np.float32)
                        project_l_faces = np.zeros((0, 3), dtype=np.int64)
                    if use_right:
                        project_r_seq_t, _project_r_joints_t, project_r_faces_t = (
                            process_hand_result(
                                r_hand_layer,
                                _slice_frame_indices(
                                    sample["rhand_params"], project_indices
                                ),
                            )
                        )
                        project_r_seq = _to_numpy(project_r_seq_t)
                        project_r_faces = _to_numpy(project_r_faces_t)
                    else:
                        project_r_seq = np.zeros((0, 0, 3), dtype=np.float32)
                        project_r_faces = np.zeros((0, 3), dtype=np.int64)
                    if latethoi_repo_dir:
                        latethoi_project_metrics = _compute_latethoi_project_metrics(
                            obj_mesh_by_key.get(obj_key),
                            project_obj_params,
                            project_l_seq,
                            project_r_seq,
                            project_l_faces,
                            project_r_faces,
                            use_left,
                            use_right,
                            latethoi_repo_dir,
                            int(len(project_indices)),
                        )
                    if diffh2o_repo_dir:
                        diffh2o_project_metrics = _compute_diffh2o_project_metrics(
                            obj_mesh_by_key.get(obj_key),
                            project_obj_params,
                            project_l_seq,
                            project_r_seq,
                            project_l_faces,
                            project_r_faces,
                            use_left,
                            use_right,
                            diffh2o_repo_dir,
                            int(len(project_indices)),
                        )
                except Exception as ex:
                    _warn_project_metric_once(
                        f"project-metric:{os.path.basename(path)}",
                        f"[WARN] project-native metric adapter failed for "
                        f"{os.path.basename(path)}; later samples will be skipped "
                        f"silently if the same issue repeats. First error: {ex}",
                    )
                if compare_gt and gt_obj_params is not None:
                    try:
                        gt_project_nframes = _sequence_length(gt_obj_params)
                        gt_l_project_params = sample["lhand_params"]
                        gt_r_project_params = sample["rhand_params"]
                        if use_left and gt_lhand_params is not None:
                            gt_project_nframes = min(
                                gt_project_nframes, _sequence_length(gt_lhand_params)
                            )
                            gt_l_project_params = gt_lhand_params
                        elif use_left:
                            gt_project_nframes = min(
                                gt_project_nframes,
                                _sequence_length(sample["lhand_params"]),
                            )
                        if use_right and gt_rhand_params is not None:
                            gt_project_nframes = min(
                                gt_project_nframes, _sequence_length(gt_rhand_params)
                            )
                            gt_r_project_params = gt_rhand_params
                        elif use_right:
                            gt_project_nframes = min(
                                gt_project_nframes,
                                _sequence_length(sample["rhand_params"]),
                            )
                        if gt_project_nframes > 0:
                            gt_project_indices = _sampled_eval_indices(
                                gt_project_nframes, project_native_eval_frames
                            )
                            gt_project_obj_params = _slice_frame_indices(
                                gt_obj_params, gt_project_indices
                            )
                            if use_left:
                                gt_project_l_seq_t, _gt_project_l_joints_t, gt_project_l_faces_t = (
                                    process_hand_result(
                                        l_hand_layer,
                                        _slice_frame_indices(
                                            gt_l_project_params, gt_project_indices
                                        ),
                                    )
                                )
                                gt_project_l_seq = _to_numpy(gt_project_l_seq_t)
                                gt_project_l_faces = _to_numpy(gt_project_l_faces_t)
                            else:
                                gt_project_l_seq = np.zeros((0, 0, 3), dtype=np.float32)
                                gt_project_l_faces = np.zeros((0, 3), dtype=np.int64)
                            if use_right:
                                gt_project_r_seq_t, _gt_project_r_joints_t, gt_project_r_faces_t = (
                                    process_hand_result(
                                        r_hand_layer,
                                        _slice_frame_indices(
                                            gt_r_project_params, gt_project_indices
                                        ),
                                    )
                                )
                                gt_project_r_seq = _to_numpy(gt_project_r_seq_t)
                                gt_project_r_faces = _to_numpy(gt_project_r_faces_t)
                            else:
                                gt_project_r_seq = np.zeros((0, 0, 3), dtype=np.float32)
                                gt_project_r_faces = np.zeros((0, 3), dtype=np.int64)
                            if latethoi_repo_dir:
                                latethoi_project_metrics_gt = _compute_latethoi_project_metrics(
                                    obj_mesh_by_key.get(obj_key),
                                    gt_project_obj_params,
                                    gt_project_l_seq,
                                    gt_project_r_seq,
                                    gt_project_l_faces,
                                    gt_project_r_faces,
                                    use_left,
                                    use_right,
                                    latethoi_repo_dir,
                                    int(len(gt_project_indices)),
                                )
                            if diffh2o_repo_dir:
                                diffh2o_project_metrics_gt = _compute_diffh2o_project_metrics(
                                    obj_mesh_by_key.get(obj_key),
                                    gt_project_obj_params,
                                    gt_project_l_seq,
                                    gt_project_r_seq,
                                    gt_project_l_faces,
                                    gt_project_r_faces,
                                    use_left,
                                    use_right,
                                    diffh2o_repo_dir,
                                    int(len(gt_project_indices)),
                                )
                    except Exception as ex:
                        _warn_project_metric_once(
                            f"project-metric-gt:{os.path.basename(path)}",
                            f"[WARN] GT project-native metric adapter failed for "
                            f"{os.path.basename(path)}; later GT samples will be skipped "
                            f"silently if the same issue repeats. First error: {ex}",
                        )
            row = {
                "text": text,
                "object": obj_key,
                "sample_idx": int(sample["sample_idx"]),
                "valid_contact": bool(metric["valid_contact"]),
                "wrist_traj": wrist_traj,
                "wrist_traj_aligned": _apply_object_axis_alignment(
                    wrist_traj,
                    object_axis_alignment_by_key.get(obj_key),
                ),
                "wrist_traj_gt": wrist_traj_gt,
                "wrist_traj_gt_aligned": wrist_traj_gt_aligned,
                "latethoi_last_hand_feat": _latethoi_last_hand_feature(
                    l_seq,
                    r_seq,
                    use_left,
                    use_right,
                ),
                "latethoi_lastframe": latethoi_lastframe,
                "diffh2o_native": diffh2o_native_metrics,
                "latethoi_project": latethoi_project_metrics,
                "diffh2o_project": diffh2o_project_metrics,
                "latethoi_project_gt": latethoi_project_metrics_gt,
                "diffh2o_project_gt": diffh2o_project_metrics_gt,
                "bimart": bimart_metrics,
                "bimart_gt": bimart_metrics_gt,
                "latethoi_lastframe_gt": latethoi_lastframe_gt,
                "text2hoi": text2hoi_metrics,
                "gt_hand_available": gt_hand_available,
                "rel_rot_first": _relative_hand_object_rotmats_first_frame(
                    sample["obj_params"],
                    sample["lhand_params"],
                    sample["rhand_params"],
                    use_left,
                    use_right,
                ),
                "rel_rot_first_gt": (
                    _relative_hand_object_rotmats_first_frame(
                        sample["gt_obj_params"],
                        (
                            sample["gt_lhand_params"]
                            if sample.get("gt_lhand_params") is not None
                            else sample["lhand_params"]
                        ),
                        (
                            sample["gt_rhand_params"]
                            if sample.get("gt_rhand_params") is not None
                            else sample["rhand_params"]
                        ),
                        use_left,
                        use_right,
                    )
                    if sample.get("gt_obj_params") is not None
                    else None
                ),
                "rel_rot_last": _relative_hand_object_rotmats_at_frame(
                    sample["obj_params"],
                    sample["lhand_params"],
                    sample["rhand_params"],
                    use_left,
                    use_right,
                    frame_idx=int(nframes - 1),
                ),
                "rel_rot_last_gt": (
                    _relative_hand_object_rotmats_at_frame(
                        sample["gt_obj_params"],
                        (
                            sample["gt_lhand_params"]
                            if sample.get("gt_lhand_params") is not None
                            else sample["lhand_params"]
                        ),
                        (
                            sample["gt_rhand_params"]
                            if sample.get("gt_rhand_params") is not None
                            else sample["rhand_params"]
                        ),
                        use_left,
                        use_right,
                        frame_idx=(
                            int(nframes_gt_rel - 1)
                            if nframes_gt_rel is not None
                            else int(nframes - 1)
                        ),
                    )
                    if sample.get("gt_obj_params") is not None
                    else None
                ),
                "rel_rot_last_right_only": _relative_hand_object_rotmats_at_frame(
                    sample["obj_params"],
                    sample["lhand_params"],
                    sample["rhand_params"],
                    use_left=False,
                    use_right=True,
                    frame_idx=int(nframes - 1),
                ),
                "rel_rot_last_gt_right_only": (
                    _relative_hand_object_rotmats_at_frame(
                        sample["gt_obj_params"],
                        (
                            sample["gt_lhand_params"]
                            if sample.get("gt_lhand_params") is not None
                            else sample["lhand_params"]
                        ),
                        (
                            sample["gt_rhand_params"]
                            if sample.get("gt_rhand_params") is not None
                            else sample["rhand_params"]
                        ),
                        use_left=False,
                        use_right=True,
                        frame_idx=(
                            int(nframes_gt_rel - 1)
                            if nframes_gt_rel is not None
                            else int(nframes - 1)
                        ),
                    )
                    if sample.get("gt_obj_params") is not None
                    else None
                ),
                **gt_extra,
                **metric,
            }
            per_sample_rows.append(row)
            if visualize_eval or visualize_metric_rankings:
                latethoi_mean = (
                    latethoi_lastframe.get("hand_metric_mean", {})
                    if isinstance(latethoi_lastframe, dict)
                    else {}
                )
                text2hoi_mean = (
                    text2hoi_metrics.get("hand_metric_mean", {})
                    if isinstance(text2hoi_metrics, dict)
                    else {}
                )
                pred_rank_visualizations.append(
                    {
                        "file_name": os.path.basename(path),
                        "sample_idx": int(sample["sample_idx"]),
                        "split": _split_tag_from_file_name(os.path.basename(path)),
                        "cr": float(metric.get("cr", 0.0)),
                        "valid_contact": float(
                            bool(metric.get("valid_contact", False))
                        ),
                        "success": float(bool(metric.get("success", False))),
                        "iv_cm3": (
                            None
                            if metric.get("iv_cm3") is None
                            else float(metric.get("iv_cm3"))
                        ),
                        "id_mm": (
                            None
                            if metric.get("id_mm") is None
                            else float(metric.get("id_mm"))
                        ),
                        "id_max_mm": (
                            None
                            if metric.get("id_max_mm") is None
                            else float(metric.get("id_max_mm"))
                        ),
                        "latethoi_inter_volume_mean_m3": latethoi_mean.get(
                            "inter_volume_mean"
                        ),
                        "latethoi_inter_depth_mean_m": latethoi_mean.get(
                            "inter_depth_mean"
                        ),
                        "latethoi_contact_ratio_mean": latethoi_mean.get(
                            "contact_ratio_mean"
                        ),
                        "text2hoi_penetration_loss_m": text2hoi_mean.get(
                            "penetration_loss_m"
                        ),
                        "text2hoi_penetration_max_m": text2hoi_mean.get(
                            "penetration_max_m"
                        ),
                        "text2hoi_interior_object_ratio": text2hoi_mean.get(
                            "interior_object_ratio"
                        ),
                        "text2hoi_contact_object_ratio": text2hoi_mean.get(
                            "contact_object_ratio"
                        ),
                        "text2hoi_contact_joint_ratio": text2hoi_mean.get(
                            "contact_joint_ratio"
                        ),
                        "args": (
                            os.path.basename(path),
                            _split_tag_from_file_name(os.path.basename(path)),
                            text,
                            int(sample["sample_idx"]),
                            obj_key,
                            obj_seq[-1],
                            (
                                l_seq[-1]
                                if l_seq is not None and l_seq.shape[0] > 0
                                else np.zeros((0, 3), dtype=np.float32)
                            ),
                            (
                                r_seq[-1]
                                if r_seq is not None and r_seq.shape[0] > 0
                                else np.zeros((0, 3), dtype=np.float32)
                            ),
                            (
                                l_joints[-1]
                                if l_joints is not None and l_joints.shape[0] > 0
                                else np.zeros((0, 3), dtype=np.float32)
                            ),
                            (
                                r_joints[-1]
                                if r_joints is not None and r_joints.shape[0] > 0
                                else np.zeros((0, 3), dtype=np.float32)
                            ),
                            l_faces,
                            r_faces,
                            {
                                **metric,
                                "sample_idx": int(sample["sample_idx"]),
                            },
                            use_left,
                            use_right,
                        ),
                    }
                )
            if (
                (visualize_eval or visualize_metric_rankings)
                and compare_gt
                and metric_gt is not None
                and gt_obj_seq is not None
            ):
                vis_args = (
                    os.path.basename(path),
                    _split_tag_from_file_name(os.path.basename(path)),
                    text,
                    int(sample["sample_idx"]),
                    obj_key,
                    gt_obj_seq[-1],
                    (
                        gt_l_seq[-1]
                        if gt_l_seq is not None and gt_l_seq.shape[0] > 0
                        else np.zeros((0, 3), dtype=np.float32)
                    ),
                    (
                        gt_r_seq[-1]
                        if gt_r_seq is not None and gt_r_seq.shape[0] > 0
                        else np.zeros((0, 3), dtype=np.float32)
                    ),
                    (
                        gt_l_joints[-1]
                        if gt_l_joints is not None and gt_l_joints.shape[0] > 0
                        else np.zeros((0, 3), dtype=np.float32)
                    ),
                    (
                        gt_r_joints[-1]
                        if gt_r_joints is not None and gt_r_joints.shape[0] > 0
                        else np.zeros((0, 3), dtype=np.float32)
                    ),
                    gt_l_faces if gt_l_faces is not None else l_faces,
                    gt_r_faces if gt_r_faces is not None else r_faces,
                    {
                        **metric_gt,
                        "sample_idx": int(sample["sample_idx"]),
                    },
                    use_left,
                    use_right,
                )
                gt_rank_visualizations.append(
                    {
                        "sample_idx": int(sample["sample_idx"]),
                        "split": _split_tag_from_file_name(os.path.basename(path)),
                        "cr": float(metric_gt.get("cr", 0.0)),
                        "iv_cm3": (
                            None
                            if metric_gt.get("iv_cm3") is None
                            else float(metric_gt.get("iv_cm3"))
                        ),
                        "id_mm": (
                            None
                            if metric_gt.get("id_mm") is None
                            else float(metric_gt.get("id_mm"))
                        ),
                        "id_max_mm": (
                            None
                            if metric_gt.get("id_max_mm") is None
                            else float(metric_gt.get("id_max_mm"))
                        ),
                        "args": vis_args,
                    }
                )
    sample_pbar.close()
    if (visualize_eval or visualize_metric_rankings) and pred_rank_visualizations:
        _log_pred_metric_rankings(
            pred_rank_visualizations,
            split_name=_split_tag_from_file_name(os.path.basename(path)),
            topk=int(max(1, visualize_metric_topk)),
            hand_style=visualize_hand_style,
        )
    if (
        (visualize_eval or visualize_metric_rankings)
        and compare_gt
        and gt_rank_visualizations
        and int(visualize_gt_topk) > 0
    ):
        topk = int(max(1, visualize_gt_topk))
        _log_gt_metric_rankings(
            gt_rank_visualizations,
            split_name=_split_tag_from_file_name(os.path.basename(path)),
            topk=topk,
            hand_style=visualize_hand_style,
        )

    result = {
        "file_name": os.path.basename(path),
        "per_sample_rows": per_sample_rows,
        "overall": _aggregate(per_sample_rows, min_cr_for_valid),
        "per_object_diversity": _per_object_diversity(per_sample_rows),
        "bimart_file_metrics": {"apd_multi": _compute_bimart_apd(bimart_apd_groups)},
        "bimart_file_metrics_gt": {
            "apd_multi": _compute_bimart_apd(bimart_apd_groups_gt)
        },
    }
    if visualize_eval:
        _log_eval_file_summary(
            result["file_name"],
            _split_tag_from_file_name(result["file_name"]),
            result["overall"],
        )
    return result


def write_per_sample_csv(path: str, all_results: list[dict]) -> None:
    header = [
        "file_name",
        "object",
        "text",
        "sample_idx",
        "IV_cm3",
        "ID_mm",
        "ID_Max_mm",
        "CR",
        "valid_contact",
        "VR_left_joint_distances_mm",
        "VR_right_joint_distances_mm",
        "VR_joint_distances_mm",
        "success",
        "CR_gt",
        "valid_contact_gt",
        "success_gt",
        "GT_IoU",
        "GT_Precision",
        "GT_Recall",
        "GT_F1",
        "CR_error_to_GT",
        "ID_gt_mm",
        "ID_error_to_GT_mm",
        "ID_Max_gt_mm",
        "ID_Max_error_to_GT_mm",
        "IV_gt_cm3",
        "IV_error_to_GT_cm3",
        "BimArt_APD_multi",
        "BimArt_Accel",
        "BimArt_Penetration_1cm",
        "BimArt_Contact_pct",
        "BimArt_GT_Accel",
        "BimArt_GT_Penetration_1cm",
        "BimArt_GT_Contact_pct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in all_results:
            for row in result["per_sample_rows"]:
                writer.writerow(
                    [
                        result["file_name"],
                        row["object"],
                        row["text"],
                        row["sample_idx"],
                        _format_csv_float(row.get("iv_cm3")),
                        _format_csv_float(row.get("id_mm")),
                        _format_csv_float(row.get("id_max_mm")),
                        f"{row['cr']:.6f}",
                        int(row.get("valid_contact", False)),
                        _format_vr_joint_distances_mm(row, "left"),
                        _format_vr_joint_distances_mm(row, "right"),
                        ",".join(
                            x
                            for x in (
                                _format_vr_joint_distances_mm(row, "left"),
                                _format_vr_joint_distances_mm(row, "right"),
                            )
                            if x
                        ),
                        int(row.get("success", False)),
                        "" if row.get("cr_gt") is None else f"{row['cr_gt']:.6f}",
                        (
                            ""
                            if row.get("cr_gt") is None
                            else int(row.get("valid_contact_gt", False))
                        ),
                        (
                            ""
                            if row.get("cr_gt") is None
                            else int(row.get("success_gt", False))
                        ),
                        "" if row.get("gt_iou") is None else f"{row['gt_iou']:.6f}",
                        (
                            ""
                            if row.get("gt_precision") is None
                            else f"{row['gt_precision']:.6f}"
                        ),
                        (
                            ""
                            if row.get("gt_recall") is None
                            else f"{row['gt_recall']:.6f}"
                        ),
                        "" if row.get("gt_f1") is None else f"{row['gt_f1']:.6f}",
                        (
                            ""
                            if row.get("cr_gt") is None
                            else f"{abs(row['cr'] - row['cr_gt']):.6f}"
                        ),
                        _format_csv_float(row.get("id_gt")),
                        (
                            ""
                            if row.get("id_gt") is None or row.get("id_mm") is None
                            else f"{abs(row['id_mm'] - row['id_gt']):.6f}"
                        ),
                        _format_csv_float(row.get("id_max_gt")),
                        (
                            ""
                            if row.get("id_max_gt") is None
                            or row.get("id_max_mm") is None
                            else f"{abs(row['id_max_mm'] - row['id_max_gt']):.6f}"
                        ),
                        _format_csv_float(row.get("iv_gt")),
                        (
                            ""
                            if row.get("iv_gt") is None or row.get("iv_cm3") is None
                            else f"{abs(row['iv_cm3'] - row['iv_gt']):.6f}"
                        ),
                        _format_csv_float(
                            (result.get("bimart_file_metrics", {}) or {}).get(
                                "apd_multi"
                            )
                        ),
                        _format_csv_float((row.get("bimart") or {}).get("jitter")),
                        _format_csv_float(
                            (row.get("bimart") or {}).get("penetration_1cm")
                        ),
                        _format_csv_float((row.get("bimart") or {}).get("contact")),
                        _format_csv_float((row.get("bimart_gt") or {}).get("jitter")),
                        _format_csv_float(
                            (row.get("bimart_gt") or {}).get("penetration_1cm")
                        ),
                        _format_csv_float((row.get("bimart_gt") or {}).get("contact")),
                    ]
                )


def write_file_avg_csv(path: str, all_results: list[dict]) -> None:
    def _f(row: dict, key: str, default: float = 0.0) -> str:
        try:
            value = row.get(key, default)
            if value is None:
                return ""
            return f"{float(value):.6f}"
        except Exception:
            if default is None:
                return ""
            return f"{float(default):.6f}"

    def _i(row: dict, key: str, default: int = 0) -> int:
        try:
            return int(row.get(key, default))
        except Exception:
            return int(default)

    header = [
        "file_name",
        "samples",
        "IV_cm3",
        "ID_mm",
        "ID_Max_mm",
        "CR",
        "success_rate",
        "success_samples",
        "valid_contact_rate",
        "valid_samples",
        "IV_cm3_valid",
        "ID_mm_valid",
        "ID_Max_mm_valid",
        "gt_samples",
        "CR_gt",
        "GT_IoU",
        "GT_Precision",
        "GT_Recall",
        "GT_F1",
        "CR_error_to_GT",
        "ID_gt_mm",
        "ID_error_to_GT_mm",
        "ID_Max_gt_mm",
        "ID_Max_error_to_GT_mm",
        "ID_gt_samples",
        "IV_gt_valid_cm3",
        "ID_gt_valid_mm",
        "ID_Max_gt_valid_mm",
        "IV_error_valid_to_GT_cm3",
        "ID_error_valid_to_GT_mm",
        "ID_Max_error_valid_to_GT_mm",
        "valid_gt_samples",
        "SampleDiversity_m",
        "SampleDiversity_prompts",
        "ObjectAvgSampleDiversity_m",
        "ObjectAvgSampleDiversity_objects",
        "OverallDiversityLocal_m",
        "OverallDiversityLocal_samples",
        "OverallDiversity_m",
        "OverallDiversity_samples",
        "ObjectAvgOverallDiversity_m",
        "ObjectAvgOverallDiversity_objects",
        "RelRotDiversity_deg",
        "RelRotDiversity_objects",
        "RelRotDiversityGT_deg",
        "RelRotDiversityGT_objects",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in all_results:
            row = result["overall"]
            writer.writerow(
                [
                    result["file_name"],
                    _i(row, "samples"),
                    _f(row, "iv_cm3"),
                    _f(row, "id_mm"),
                    _f(row, "id_max_mm"),
                    _f(row, "cr"),
                    _f(row, "success_rate"),
                    _i(row, "success_samples"),
                    _f(row, "valid_contact_rate"),
                    _i(row, "valid_samples"),
                    _f(row, "iv_cm3_valid"),
                    _f(row, "id_mm_valid"),
                    _f(row, "id_max_mm_valid"),
                    _i(row, "gt_samples"),
                    _f(row, "cr_gt"),
                    _f(row, "gt_iou"),
                    _f(row, "gt_precision"),
                    _f(row, "gt_recall"),
                    _f(row, "gt_f1"),
                    _f(row, "cr_err_to_gt"),
                    _f(row, "id_gt"),
                    _f(row, "id_err_to_gt"),
                    _f(row, "id_max_gt"),
                    _f(row, "id_max_err_to_gt"),
                    _i(row, "id_gt_samples"),
                    _f(row, "iv_gt_valid"),
                    _f(row, "id_gt_valid"),
                    _f(row, "id_max_gt_valid"),
                    _f(row, "iv_err_to_gt_valid"),
                    _f(row, "id_err_to_gt_valid"),
                    _f(row, "id_max_err_to_gt_valid"),
                    _i(row, "valid_gt_samples"),
                    _f(row, "sample_diversity_m"),
                    _i(row, "sample_diversity_prompts"),
                    _f(row, "object_avg_sample_diversity_m"),
                    _i(row, "object_avg_sample_diversity_objects"),
                    _f(row, "overall_diversity_local_m"),
                    _i(row, "overall_diversity_local_samples"),
                    _f(row, "overall_diversity_m"),
                    _i(row, "overall_diversity_samples"),
                    _f(row, "object_avg_overall_diversity_m"),
                    _i(row, "object_avg_overall_diversity_objects"),
                    _f(row, "relrot_diversity_deg"),
                    _i(row, "relrot_object_count"),
                    _f(row, "relrot_diversity_gt_deg"),
                    _i(row, "relrot_gt_object_count"),
                ]
            )


def write_object_diversity_csv(path: str, all_results: list[dict]) -> None:
    header = [
        "file_name",
        "split",
        "object",
        "samples",
        "SampleDiversity_m",
        "SampleDiversity_prompts",
        "OverallDiversity_m",
        "OverallDiversity_samples",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in all_results:
            file_name = result.get("file_name", "")
            split = _split_tag_from_file_name(file_name)
            per_object = result.get("per_object_diversity", {})
            for object_key in sorted(per_object.keys()):
                row = per_object[object_key]
                writer.writerow(
                    [
                        file_name,
                        split,
                        object_key,
                        int(row.get("samples", 0)),
                        _format_csv_float(row.get("sample_diversity_m")),
                        int(row.get("sample_diversity_prompts", 0)),
                        _format_csv_float(row.get("overall_diversity_m")),
                        int(row.get("overall_diversity_samples", 0)),
                    ]
                )


def _aggregate_results_by_split(
    all_results: list[dict],
) -> list[tuple[str, str, list[dict]]]:
    buckets: list[tuple[str, str, list[dict]]] = []
    for result in all_results:
        rows = result.get("per_sample_rows", [])
        file_name = result.get("file_name", "")
        split = _split_tag_from_file_name(file_name)
        if rows:
            buckets.append((split, file_name, rows))
    split_order = {"seen": 0, "unseen": 1, "other": 2}
    return sorted(
        buckets,
        key=lambda x: (split_order.get(x[0], 99), x[1]),
    )


def _summary_metric_rows(
    all_results: list[dict], min_cr_for_valid: float
) -> list[dict]:
    rows = []
    for split_name, file_name, split_rows in _aggregate_results_by_split(all_results):
        agg = _aggregate(split_rows, min_cr_for_valid)
        agg["split"] = split_name
        agg["file_name"] = file_name
        rows.append(agg)
    return rows


def _filtered_gt_reference_rows(rows: list[dict]) -> list[dict]:
    gt_rows = [row for row in rows if row.get("cr_gt") is not None]
    if not gt_rows:
        return rows
    gt_valid_rows = [row for row in gt_rows if bool(row.get("valid_contact_gt", False))]
    return gt_valid_rows if gt_valid_rows else gt_rows


def _gt_reference_row(
    split_name: str, rows: list[dict], source_file_name: Optional[str] = None
) -> Optional[dict]:
    total_gt_samples = int(len(rows))
    all_gt_rows = [row for row in rows if row.get("cr_gt") is not None]
    rows = _filtered_gt_reference_rows(rows)
    agg = _aggregate(rows, min_cr_for_valid=0.0)
    sample_diversity_m, sample_diversity_prompts = _sample_diversity_m(
        rows, traj_key="wrist_traj_gt"
    )
    overall_diversity_local_m, overall_diversity_local_samples = _overall_diversity_m(
        rows, traj_key="wrist_traj_gt"
    )
    overall_diversity_m, overall_diversity_samples = _overall_diversity_m(
        rows, traj_key="wrist_traj_gt_aligned"
    )
    per_object_diversity = _per_object_diversity(rows, traj_key="wrist_traj_gt")
    gt_label = f"{split_name.capitalize()}_G.T"
    if source_file_name:
        gt_label = f"{gt_label} (from {source_file_name})"
    if agg["samples"] <= 0:
        return {
            "split": split_name,
            "file_name": gt_label,
            "samples": 0,
            "valid_contact_rate": 0.0,
            "success_rate": 0.0,
            "cr": 0.0,
            "iv_cm3": None,
            "id_mm": None,
            "id_max_mm": None,
            "sample_diversity_m": sample_diversity_m,
            "object_avg_sample_diversity_m": _mean_optional(
                item.get("sample_diversity_m") for item in per_object_diversity.values()
            )
            or 0.0,
            "object_avg_sample_diversity_objects": int(len(per_object_diversity)),
            "overall_diversity_local_m": overall_diversity_local_m,
            "overall_diversity_local_samples": overall_diversity_local_samples,
            "overall_diversity_m": overall_diversity_m,
            "valid_samples": 0,
            "sample_diversity_prompts": sample_diversity_prompts,
            "overall_diversity_samples": overall_diversity_samples,
            "object_avg_overall_diversity_m": _mean_optional(
                item.get("overall_diversity_m")
                for item in per_object_diversity.values()
            )
            or 0.0,
            "object_avg_overall_diversity_objects": int(len(per_object_diversity)),
            "cr_err_to_gt": 0.0,
            "iv_err_to_gt": None,
            "id_err_to_gt": None,
            "id_max_err_to_gt": None,
            "is_gt_row": True,
        }
    if agg["gt_samples"] <= 0:
        return {
            "split": split_name,
            "file_name": gt_label,
            "samples": 0,
            "valid_contact_rate": 0.0,
            "success_rate": 0.0,
            "cr": 0.0,
            "iv_cm3": None,
            "id_mm": None,
            "id_max_mm": None,
            "sample_diversity_m": sample_diversity_m,
            "object_avg_sample_diversity_m": _mean_optional(
                item.get("sample_diversity_m") for item in per_object_diversity.values()
            )
            or 0.0,
            "object_avg_sample_diversity_objects": int(len(per_object_diversity)),
            "overall_diversity_local_m": overall_diversity_local_m,
            "overall_diversity_local_samples": overall_diversity_local_samples,
            "overall_diversity_m": overall_diversity_m,
            "valid_samples": 0,
            "sample_diversity_prompts": sample_diversity_prompts,
            "overall_diversity_samples": overall_diversity_samples,
            "object_avg_overall_diversity_m": _mean_optional(
                item.get("overall_diversity_m")
                for item in per_object_diversity.values()
            )
            or 0.0,
            "object_avg_overall_diversity_objects": int(len(per_object_diversity)),
            "cr_err_to_gt": 0.0,
            "iv_err_to_gt": None,
            "id_err_to_gt": None,
            "id_max_err_to_gt": None,
            "is_gt_row": True,
        }
    return {
        "split": split_name,
        "file_name": gt_label,
        "samples": total_gt_samples,
        "valid_contact_rate": (
            float(agg["valid_gt_samples"] / total_gt_samples)
            if total_gt_samples > 0
            else 0.0
        ),
        "success_rate": agg.get("success_rate_gt", 0.0),
        "cr": agg["cr_gt"],
        "iv_cm3": agg["iv_gt_valid"],
        "id_mm": agg["id_gt_valid"],
        "id_max_mm": agg["id_max_gt_valid"],
        "sample_diversity_m": sample_diversity_m,
        "object_avg_sample_diversity_m": _mean_optional(
            item.get("sample_diversity_m") for item in per_object_diversity.values()
        )
        or 0.0,
        "object_avg_sample_diversity_objects": int(len(per_object_diversity)),
        "overall_diversity_local_m": overall_diversity_local_m,
        "overall_diversity_local_samples": overall_diversity_local_samples,
        "overall_diversity_m": overall_diversity_m,
        "valid_samples": agg["valid_gt_samples"],
        "sample_diversity_prompts": sample_diversity_prompts,
        "overall_diversity_samples": overall_diversity_samples,
        "object_avg_overall_diversity_m": _mean_optional(
            item.get("overall_diversity_m") for item in per_object_diversity.values()
        )
        or 0.0,
        "object_avg_overall_diversity_objects": int(len(per_object_diversity)),
        "cr_err_to_gt": 0.0,
        "iv_err_to_gt": None,
        "id_err_to_gt": None,
        "id_max_err_to_gt": None,
        "is_gt_row": True,
    }


def _summary_rows_with_gt(
    all_results: list[dict], min_cr_for_valid: float
) -> list[dict]:
    base_rows = _summary_metric_rows(all_results, min_cr_for_valid)
    grouped_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    fallback_rows: dict[str, list[dict]] = {"seen": [], "unseen": []}
    gt_source_file: dict[str, Optional[str]] = {"seen": None, "unseen": None}
    for result in all_results:
        file_name = result.get("file_name", "")
        split = _split_tag_from_file_name(result.get("file_name", ""))
        rows = result.get("per_sample_rows", [])
        if split not in grouped_rows or not rows:
            continue
        if not _allow_gt_source_file(file_name):
            continue
        if not fallback_rows[split]:
            fallback_rows[split] = list(rows)
            if gt_source_file[split] is None:
                gt_source_file[split] = file_name
        has_gt_hand = any(bool(r.get("gt_hand_available")) for r in rows)
        if has_gt_hand and not grouped_rows[split]:
            grouped_rows[split] = list(rows)
            gt_source_file[split] = file_name

    for split in grouped_rows:
        if not grouped_rows[split]:
            grouped_rows[split] = fallback_rows[split]

    rows_by_split: dict[str, list[dict]] = {"seen": [], "unseen": [], "other": []}
    for row in base_rows:
        row["is_gt_row"] = False
        row["cr_err_to_gt"] = row.get("cr_err_to_gt", 0.0)
        row["iv_err_to_gt"] = (
            abs(row["iv_cm3"] - row["iv_gt"])
            if row.get("iv_gt_samples", 0) > 0
            and row.get("iv_cm3") is not None
            and row.get("iv_gt") is not None
            else None
        )
        row["id_err_to_gt"] = row.get("id_err_to_gt", None)
        row["id_max_err_to_gt"] = row.get("id_max_err_to_gt", None)
        rows_by_split.setdefault(str(row.get("split", "other")), []).append(row)

    out = []
    split_order = ("seen", "unseen", "other")
    for split_name in split_order:
        split_rows = rows_by_split.get(split_name, [])
        if not split_rows:
            continue
        if split_name in {"seen", "unseen"} and grouped_rows.get(split_name):
            gt_row = _gt_reference_row(
                split_name,
                grouped_rows.get(split_name, []),
                source_file_name=gt_source_file.get(split_name),
            )
            if gt_row is not None and int(gt_row.get("samples", 0)) > 0:
                out.append(gt_row)
        out.extend(split_rows)
    return out


def _format_float(value: float, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _format_csv_float(value, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _format_percent_from_ratio(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def _format_percent_with_count(
    value: float,
    count: Optional[int],
    total: Optional[int],
    digits: int = 2,
) -> str:
    percent = _format_percent_from_ratio(value, digits=digits)
    if count is None or total is None:
        return percent
    return f"{percent} ({int(count)}/{int(total)})"


def _print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("[WARN] no rows")
        return
    widths = [len(col) for col in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    sep = "-+-".join("-" * widths[i] for i in range(len(columns)))
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _tail_text(text: str, max_lines: int = 20) -> str:
    lines = str(text).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _flatten_numeric_scalars(value, prefix: str = "") -> dict[str, float]:
    out = {}
    if isinstance(value, dict):
        for key, sub_value in value.items():
            key_str = str(key)
            sub_prefix = f"{prefix}.{key_str}" if prefix else key_str
            out.update(_flatten_numeric_scalars(sub_value, prefix=sub_prefix))
        return out
    if isinstance(value, (list, tuple)):
        return out
    if isinstance(value, bool):
        out[prefix] = float(value)
        return out
    if isinstance(value, (int, float, np.floating, np.integer)):
        if np.isfinite(value):
            out[prefix] = float(value)
        return out
    return out


def _read_key_value_text_file(path: str) -> dict[str, float]:
    metrics = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def _parse_latethoi_native_result(path: str) -> dict[str, float]:
    with open(path, "r") as f:
        payload = json.load(f)
    metrics = {
        "sample_diversity": payload.get("sample_diversity"),
        "overall_diversity": payload.get("overall_diversity"),
        "off_ground_contact_ratio": payload.get("off_ground_contact_ratio"),
        "off_ground_contact_rate": payload.get("off_ground_contact_rate"),
    }
    for hand_key in ("rhand", "lhand"):
        hand_payload = payload.get(hand_key, {})
        if not isinstance(hand_payload, dict):
            continue
        for metric_key in (
            "inter_volume_mean",
            "inter_depth_mean",
            "inter_depth_max",
            "contact_ratio_mean",
            "contact_ratio_off_ground",
            "jerk",
        ):
            value = hand_payload.get(metric_key)
            if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(
                value
            ):
                metrics[f"{hand_key}.{metric_key}"] = float(value)
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating, np.integer))
        and np.isfinite(value)
    }


def _parse_generic_native_result(path: str) -> dict[str, float]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            payload = json.load(f)
        return _flatten_numeric_scalars(payload)
    return _read_key_value_text_file(path)


def _format_external_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _resolve_native_result_path(
    folder: str, result_path: Optional[str], default_name: str
) -> str:
    if result_path:
        return os.path.abspath(os.path.expanduser(result_path))
    return os.path.join(os.path.abspath(os.path.expanduser(folder)), default_name)


def _missing_native_summary(
    method_name: str,
    source_label: str,
    result_path: str,
) -> dict:
    return {
        "name": method_name,
        "folder": source_label,
        "result_path": result_path,
        "metrics": {},
        "status": "missing",
    }


def _run_native_summary_command(
    method_name: str,
    repo_dir: str,
    command_template: str,
    folder: str,
    python_bin: str,
) -> bool:
    context = {
        "folder": os.path.abspath(os.path.expanduser(folder)),
        "repo": os.path.abspath(os.path.expanduser(repo_dir)),
        "python": python_bin,
    }
    command_str = command_template.format(**context)
    cmd = shlex.split(command_str)
    print(f"[INFO] running {method_name} native summary command: {command_str}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=context["repo"],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as ex:
        print(f"[WARN] failed to launch {method_name} native summary command: {ex}")
        return False
    if completed.returncode != 0:
        print(
            f"[WARN] {method_name} native summary command failed "
            f"(exit={completed.returncode})."
        )
        stdout_tail = _tail_text(completed.stdout)
        stderr_tail = _tail_text(completed.stderr)
        if stdout_tail:
            print(f"[WARN] {method_name} stdout tail:\n{stdout_tail}")
        if stderr_tail:
            print(f"[WARN] {method_name} stderr tail:\n{stderr_tail}")
        return False
    return True


def _build_external_method_table(
    method_name: str,
    source_label: str,
    metrics: dict[str, float],
) -> tuple[list[str], list[list[str]]]:
    if method_name == "LatetHOI":
        if not metrics:
            return ["Source", "Status"], [[source_label, "result not found"]]
        columns = [
            "Source",
            "SD",
            "OD",
            "R IV mean",
            "L IV mean",
            "R ID mean",
            "L ID mean",
            "Off-ground CR",
            "Off-ground Rate",
            "R Jerk",
            "L Jerk",
        ]
        rows = [
            [
                source_label,
                _format_external_metric(metrics.get("sample_diversity")),
                _format_external_metric(metrics.get("overall_diversity")),
                _format_external_metric(metrics.get("rhand.inter_volume_mean")),
                _format_external_metric(metrics.get("lhand.inter_volume_mean")),
                _format_external_metric(metrics.get("rhand.inter_depth_mean")),
                _format_external_metric(metrics.get("lhand.inter_depth_mean")),
                _format_external_metric(metrics.get("off_ground_contact_ratio")),
                _format_external_metric(metrics.get("off_ground_contact_rate")),
                _format_external_metric(metrics.get("rhand.jerk")),
                _format_external_metric(metrics.get("lhand.jerk")),
            ]
        ]
        return columns, rows
    if method_name == "Text2HOI":
        if not metrics:
            return ["Source", "Status"], [[source_label, "result not found"]]

    numeric_items = [
        (key, metrics[key])
        for key in sorted(metrics.keys())
        if isinstance(metrics[key], (int, float, np.floating, np.integer))
    ]
    if not numeric_items:
        return ["Source", "Status"], [[source_label, "result not found"]]
    columns = ["Source"] + [key for key, _ in numeric_items]
    rows = [
        [source_label] + [_format_external_metric(value) for _, value in numeric_items]
    ]
    return columns, rows


def _render_external_method_markdown(
    title: str, columns: list[str], rows: list[list[str]]
) -> str:
    if not rows:
        return f"## {title}\n\n_No rows._\n"
    header = "| " + " | ".join(columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |\n"
    body = "".join(
        "| " + " | ".join(str(cell) for cell in row) + " |\n" for row in rows
    )
    return f"## {title}\n\n{header}{sep}{body}\n"


def _collect_native_method_summaries(args) -> list[dict]:
    python_bin = sys.executable
    method_specs = [
        {
            "name": "LatetHOI",
            "folder": args.latethoi_native_folder,
            "repo_dir": args.latethoi_repo_dir,
            "command": args.latethoi_native_command,
            "result_path": args.latethoi_native_result,
            "default_result_name": ".result.json",
            "parser": _parse_latethoi_native_result,
        },
        {
            "name": "Text2HOI",
            "folder": args.text2hoi_native_folder,
            "repo_dir": args.text2hoi_repo_dir,
            "command": args.text2hoi_native_command,
            "result_path": args.text2hoi_native_result,
            "default_result_name": ".result.json",
            "parser": _parse_generic_native_result,
        },
    ]
    summaries = []
    for spec in method_specs:
        configured_folder = spec["folder"]
        folder = (
            os.path.abspath(os.path.expanduser(configured_folder))
            if configured_folder
            else ""
        )
        if spec["result_path"]:
            result_path = os.path.abspath(os.path.expanduser(spec["result_path"]))
        elif folder:
            result_path = _resolve_native_result_path(
                folder, None, spec["default_result_name"]
            )
        else:
            result_path = ""
        command = spec["command"]
        if args.run_native_method_summaries and command and configured_folder:
            _run_native_summary_command(
                spec["name"],
                spec["repo_dir"],
                command,
                folder,
                python_bin,
            )
        if not result_path or not os.path.exists(result_path):
            result_label = result_path if result_path else "(not configured)"
            print(
                f"[WARN] {spec['name']} native summary result not found: {result_label}"
            )
            summaries.append(
                _missing_native_summary(
                    spec["name"],
                    folder if folder else "(not configured)",
                    result_label,
                )
            )
            continue
        try:
            metrics = spec["parser"](result_path)
        except Exception as ex:
            print(
                f"[WARN] failed to parse {spec['name']} native summary result "
                f"'{result_path}': {ex}"
            )
            continue
        summaries.append(
            {
                "name": spec["name"],
                "folder": folder,
                "result_path": result_path,
                "metrics": metrics,
            }
        )
    return summaries


def _build_category_tables(
    summary_rows: list[dict],
) -> tuple[list[list[str]], list[list[str]]]:
    physics_rows = []
    motion_rows = []
    split_order = {"seen": 0, "unseen": 1, "other": 2}
    for row in sorted(
        summary_rows,
        key=lambda x: (
            split_order.get(x["split"], 99),
            not bool(x.get("is_gt_row", False)),
            x.get("file_name", "") != "ALL_FILES",
            x.get("file_name", ""),
        ),
    ):
        split = row["split"].capitalize()
        physics_rows.append(
            [
                split,
                row.get("file_name", ""),
                str(row["samples"]),
                _format_percent_with_count(
                    row["valid_contact_rate"],
                    row.get("valid_samples"),
                    row.get("samples"),
                ),
                _format_percent_from_ratio(row["cr"]),
                _format_percent_from_ratio(row.get("success_rate", 0.0)),
                _format_float(row["iv_cm3"]),
                _format_float(row["id_mm"]),
                _format_float(row["id_max_mm"]),
            ]
        )
        motion_rows.append(
            [
                split,
                row.get("file_name", ""),
                str(row["samples"]),
                _format_float(row["sample_diversity_m"]),
                _format_float(row.get("object_avg_sample_diversity_m", 0.0)),
                _format_float(row.get("overall_diversity_local_m", 0.0)),
                _format_float(row["overall_diversity_m"]),
                _format_float(row.get("object_avg_overall_diversity_m", 0.0)),
            ]
        )
    return physics_rows, motion_rows


def write_summary_csv(path: str, summary_rows: list[dict]) -> None:
    header = [
        "split",
        "file_name",
        "samples",
        "VR (%) ↑",
        "CR (%) ↑",
        "SR (%) ↑",
        "IV (cm^3) ↓",
        "ID (mm) ↓",
        "ID_max (mm) ↓",
        "SD_m ↑",
        "ObjectAvgSD_m ↑",
        "OD_local_m ↑",
        "OD_m ↑",
        "ObjectAvgOD_m ↑",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in summary_rows:
            writer.writerow(
                [
                    row["split"],
                    row.get("file_name", ""),
                    row["samples"],
                    _format_percent_with_count(
                        row["valid_contact_rate"],
                        row.get("valid_samples"),
                        row.get("samples"),
                    ),
                    _format_percent_from_ratio(row["cr"]),
                    _format_percent_from_ratio(row.get("success_rate", 0.0)),
                    _format_float(row["iv_cm3"], digits=2),
                    _format_float(row["id_mm"], digits=2),
                    _format_float(row["id_max_mm"], digits=2),
                    (_format_float(row["sample_diversity_m"], digits=2)),
                    (_format_float(row.get("object_avg_sample_diversity_m"), digits=2)),
                    (_format_float(row.get("overall_diversity_local_m"), digits=2)),
                    (_format_float(row["overall_diversity_m"], digits=2)),
                    (
                        _format_float(
                            row.get("object_avg_overall_diversity_m"), digits=2
                        )
                    ),
                ]
            )


def write_summary_markdown(
    path: str, summary_rows: list[dict], all_results: Optional[list[dict]] = None
) -> None:
    physics_rows, motion_rows = _build_category_tables(summary_rows)
    bimart_rows = _build_bimart_table_rows(_bimart_summary_rows(all_results or []))

    def _bold_best_values(
        rows: list[list[str]],
        directions: dict[int, str],
    ) -> list[list[str]]:
        if not rows:
            return rows
        out = [list(row) for row in rows]
        for col_idx, direction in directions.items():
            parsed_vals = []
            for row_idx, row in enumerate(out):
                file_name = str(row[1])
                if "G.T" in file_name:
                    continue
                try:
                    cell = str(row[col_idx]).strip()
                    numeric_token = cell.split()[0]
                    parsed_vals.append((row_idx, float(numeric_token)))
                except Exception:
                    continue
            if not parsed_vals:
                continue
            best_val = (
                max(v for _, v in parsed_vals)
                if direction == "max"
                else min(v for _, v in parsed_vals)
            )
            for row_idx, value in parsed_vals:
                if value == best_val:
                    out[row_idx][col_idx] = f"**{out[row_idx][col_idx]}**"
        return out

    physics_rows_md = _bold_best_values(
        physics_rows,
        directions={
            3: "max",  # VR
            4: "max",  # CR
            5: "max",  # SR
            6: "min",  # IV
            7: "min",  # ID
            8: "min",  # ID_max
        },
    )
    motion_rows_md = _bold_best_values(
        motion_rows,
        directions={
            3: "max",  # SD
            4: "max",  # ObjectAvgSD
            5: "max",  # OD_local
            6: "max",  # OD
            7: "max",  # ObjectAvgOD
        },
    )

    with open(path, "w") as f:
        f.write("# Interaction Metrics Summary\n\n")
        f.write("## Physics\n\n")
        f.write(
            "| Split | File | Samples | VR (%) ↑ | CR (%) ↑ | SR (%) ↑ | IV (cm^3) ↓ | ID (mm) ↓ | ID_max (mm) ↓ |\n"
        )
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in physics_rows_md:
            f.write(
                f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |\n"
            )
        f.write("\n## Motion\n\n")
        f.write(
            "| Split | File | Samples | SD (m) ↑ | Object Avg SD (m) ↑ | OD Local (m) ↑ | OD Aligned (m) ↑ | Object Avg OD (m) ↑ |\n"
        )
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in motion_rows_md:
            f.write(
                f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} |\n"
            )
        if bimart_rows:
            f.write("\n## BimArt-Compatible Metrics\n\n")
            f.write(
                "| Split | File | Computed | APD multi ↓ | Accel ↓ | Penetration 1cm ↓ | Contact % ↑ |\n"
            )
            f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
            for row in bimart_rows:
                f.write(
                    f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |\n"
                )
        if all_results:
            f.write(_write_physics_distribution_markdown(path, all_results))


def _build_latethoi_lastframe_table_rows(
    summary_rows: list[dict],
) -> list[list[str]]:
    rows = []
    split_order = {"seen": 0, "unseen": 1, "other": 2}
    for row in sorted(
        summary_rows,
        key=lambda x: (
            split_order.get(x["split"], 99),
            x.get("file_name", "") != "ALL_FILES",
            x.get("file_name", ""),
        ),
    ):
        rows.append(
            [
                row["split"].capitalize(),
                row.get("file_name", ""),
                str(row.get("samples", 0)),
                _format_float(row.get("sample_diversity_last_frame"), digits=4),
                _format_float(row.get("overall_diversity_last_frame"), digits=4),
                _format_float(row.get("iv_cm3"), digits=2),
                _format_float(row.get("id_mm"), digits=2),
                _format_float(row.get("id_max_mm"), digits=2),
                _format_float(row.get("inter_volume_mean"), digits=6),
                _format_float(row.get("inter_depth_mean"), digits=6),
                _format_float(row.get("inter_depth_max"), digits=6),
                _format_float(row.get("contact_ratio_mean"), digits=4),
                _format_float(row.get("contact_ratio_off_ground"), digits=4),
                _format_float(row.get("off_ground_contact_ratio"), digits=4),
                _format_float(row.get("off_ground_contact_rate"), digits=4),
                _format_float(row.get("jerk"), digits=4),
            ]
        )
    return rows


def _build_text2hoi_table_rows(summary_rows: list[dict]) -> list[list[str]]:
    rows = []
    split_order = {"seen": 0, "unseen": 1, "other": 2}
    for row in sorted(
        summary_rows,
        key=lambda x: (
            split_order.get(x.get("split", "other"), 99),
            x.get("file_name", "") != "ALL_FILES",
            x.get("file_name", ""),
        ),
    ):
        rows.append(
            [
                str(row.get("split", "other")).capitalize(),
                row.get("file_name", ""),
                str(row.get("samples", 0)),
                _format_float(row.get("penetration_loss_m"), digits=6),
                _format_float(row.get("penetration_max_m"), digits=6),
                _format_float(row.get("interior_object_ratio"), digits=4),
                _format_float(row.get("contact_object_ratio"), digits=4),
                _format_float(row.get("contact_joint_ratio"), digits=4),
            ]
        )
    return rows


def _build_latethoi_native_table_rows(summary_rows: list[dict]) -> list[list[str]]:
    rows = []
    split_order = {"seen": 0, "unseen": 1, "other": 2}
    for row in sorted(
        summary_rows,
        key=lambda x: (
            split_order.get(x.get("split", "other"), 99),
            bool(x.get("is_gt_row", False)),
            x.get("file_name", "") != "ALL_FILES",
            x.get("file_name", ""),
        ),
    ):
        if bool(row.get("is_gt_row", False)):
            continue
        rows.append(
            [
                str(row.get("split", "other")).capitalize(),
                row.get("file_name", ""),
                str(row.get("samples", 0)),
                _format_float(row.get("sample_diversity_last_frame"), digits=4),
                _format_float(row.get("overall_diversity_last_frame"), digits=4),
                _format_float(row.get("inter_volume_mean"), digits=6),
                _format_float(row.get("inter_depth_mean"), digits=6),
                _format_float(row.get("inter_depth_max"), digits=6),
                _format_float(row.get("contact_ratio_mean"), digits=4),
                _format_float(row.get("contact_ratio_contact"), digits=4),
                _format_float(row.get("contact_ratio_max"), digits=4),
            ]
        )
    return rows


def _build_diffh2o_native_table_rows(summary_rows: list[dict]) -> list[list[str]]:
    rows = []
    split_order = {"seen": 0, "unseen": 1, "other": 2}
    for row in sorted(
        summary_rows,
        key=lambda x: (
            split_order.get(x.get("split", "other"), 99),
            x.get("file_name", "") != "ALL_FILES",
            x.get("file_name", ""),
        ),
    ):
        rows.append(
            [
                str(row.get("split", "other")).capitalize(),
                row.get("file_name", ""),
                str(row.get("samples", 0)),
                _format_float(row.get("sample_diversity_m"), digits=4),
                _format_float(row.get("overall_diversity_m"), digits=4),
                _format_float(row.get("iv_count_mean"), digits=4),
                _format_float(row.get("inter_depth_mean"), digits=6),
                _format_float(row.get("contact_ratio_mean"), digits=4),
            ]
        )
    return rows


def write_latethoi_lastframe_summary_csv(path: str, summary_rows: list[dict]) -> None:
    header = [
        "split",
        "file_name",
        "samples",
        "LF_SD_m",
        "LF_OD_m",
        "IV_cm3",
        "ID_mm",
        "ID_max_mm",
        "inter_volume_mean_m3",
        "inter_depth_mean_m",
        "inter_depth_max_m",
        "contact_ratio_mean",
        "contact_ratio_off_ground",
        "off_ground_contact_ratio",
        "off_ground_contact_rate",
        "jerk",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in summary_rows:
            writer.writerow(
                [
                    row["split"],
                    row.get("file_name", ""),
                    row.get("samples", 0),
                    _format_csv_float(row.get("sample_diversity_last_frame")),
                    _format_csv_float(row.get("overall_diversity_last_frame")),
                    _format_csv_float(row.get("iv_cm3")),
                    _format_csv_float(row.get("id_mm")),
                    _format_csv_float(row.get("id_max_mm")),
                    _format_csv_float(row.get("inter_volume_mean")),
                    _format_csv_float(row.get("inter_depth_mean")),
                    _format_csv_float(row.get("inter_depth_max")),
                    _format_csv_float(row.get("contact_ratio_mean")),
                    _format_csv_float(row.get("contact_ratio_off_ground")),
                    _format_csv_float(row.get("off_ground_contact_ratio")),
                    _format_csv_float(row.get("off_ground_contact_rate")),
                    _format_csv_float(row.get("jerk")),
                ]
            )


def write_latethoi_lastframe_summary_markdown(
    path: str, summary_rows: list[dict]
) -> None:
    rows = _build_latethoi_lastframe_table_rows(summary_rows)
    with open(path, "w") as f:
        f.write("# LatetHOI Last-Frame Metrics Summary\n\n")
        f.write(
            "LatetHOI metrics were adapted to use only the last frame of each sample. "
            "Temporal-only fields (`off_ground_contact_*`, `jerk`) are left as `NA`.\n\n"
        )
        f.write(
            "| Split | File | Samples | LF SD (m) ↑ | LF OD (m) ↑ | IV (cm^3) ↓ | ID (mm) ↓ | ID max (mm) ↓ | LatetHOI IV mean (m^3) ↓ | LatetHOI ID mean (m) ↓ | LatetHOI ID max (m) ↓ | CR mean ↑ | CR off-ground ↑ | Off-ground CR ↑ | Off-ground Rate ↑ | Jerk ↓ |\n"
        )
        f.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            f.write(
                f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} | {row[10]} | {row[11]} | {row[12]} | {row[13]} | {row[14]} | {row[15]} |\n"
            )


def write_native_method_summaries_markdown(path: str, summaries: list[dict]) -> None:
    with open(path, "w") as f:
        f.write("# Native Method Summaries\n\n")
        if not summaries:
            f.write("_No native method summaries were collected._\n")
            return
        for summary in summaries:
            columns, rows = _build_external_method_table(
                summary["name"], summary["folder"], summary["metrics"]
            )
            f.write(
                _render_external_method_markdown(
                    f"{summary['name']} Native Summary", columns, rows
                )
            )
            f.write(f"Result file: `{summary['result_path']}`\n\n")


def write_computed_method_summaries_markdown(
    path: str,
    latethoi_native_rows: list[dict],
    diffh2o_native_rows: list[dict],
    latethoi_project_rows: list[dict],
    diffh2o_project_rows: list[dict],
) -> None:
    def _format_cm3(value, digits: int = 2) -> str:
        if value is None:
            return "NA"
        return _format_float(float(value) * 1e6, digits=digits)

    def _format_mm(value, digits: int = 2) -> str:
        if value is None:
            return "NA"
        return _format_float(float(value) * 1e3, digits=digits)

    def _format_pct(value, digits: int = 2) -> str:
        if value is None:
            return "NA"
        return _format_float(float(value) * 100.0, digits=digits)

    def _rows_to_keyed_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
        out = {}
        for row in rows:
            out[(str(row.get("split", "")), str(row.get("file_name", "")))] = row
        return out

    def _collect_ordered_keys(*row_groups: list[dict]) -> list[tuple[str, str]]:
        split_order = {"seen": 0, "unseen": 1, "other": 2}
        seen = set()
        ordered = []
        for rows in row_groups:
            for row in rows:
                key = (str(row.get("split", "")), str(row.get("file_name", "")))
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(key)
        ordered.sort(key=lambda x: (split_order.get(x[0], 99), x[1]))
        return ordered

    def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
        if not rows:
            return "_No rows._\n"
        header = "| " + " | ".join(columns) + " |\n"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |\n"
        body = "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)
        return header + sep + body

    lat_native = _rows_to_keyed_map(latethoi_native_rows)
    diff_native = _rows_to_keyed_map(diffh2o_native_rows)
    lat_project = _rows_to_keyed_map(latethoi_project_rows)
    diff_project = _rows_to_keyed_map(diffh2o_project_rows)
    keys = _collect_ordered_keys(
        latethoi_native_rows,
        diffh2o_native_rows,
        latethoi_project_rows,
        diffh2o_project_rows,
    )

    physics_common_rows = []
    physics_latethoi_rows = []
    physics_diffh2o_rows = []
    motion_common_rows = []

    for split_name, file_name in keys:
        lat_n = lat_native.get((split_name, file_name), {})
        diff_n = diff_native.get((split_name, file_name), {})
        lat_p = lat_project.get((split_name, file_name), {})
        diff_p = diff_project.get((split_name, file_name), {})
        is_gt_row = "G.T" in str(file_name)
        split_label = str(split_name).capitalize()
        samples = (
            lat_n.get("samples")
            or diff_n.get("samples")
            or lat_p.get("samples")
            or diff_p.get("samples")
            or 0
        )

        lat_iv = (
            lat_p.get("inter_volume_mean")
            if lat_p
            else lat_n.get("inter_volume_mean")
            if is_gt_row
            else None
        )
        lat_id = (
            lat_p.get("inter_depth_mean")
            if lat_p
            else lat_n.get("inter_depth_mean")
            if is_gt_row
            else None
        )
        lat_id_max = (
            lat_p.get("inter_depth_max")
            if lat_p
            else lat_n.get("inter_depth_max")
            if is_gt_row
            else None
        )
        lat_cr_mean = (
            lat_p.get("contact_ratio_mean")
            if lat_p
            else lat_n.get("contact_ratio_mean")
            if is_gt_row
            else None
        )
        lat_cr_contact = (
            lat_p.get("contact_ratio_contact")
            if lat_p
            else lat_n.get("contact_ratio_contact")
            if is_gt_row
            else None
        )

        if lat_p or diff_p or (is_gt_row and (lat_n or diff_n)):
            physics_common_rows.append(
                [
                    split_label,
                    file_name,
                    str(int(samples)),
                    _format_cm3(lat_iv),
                    _format_cm3(diff_p.get("inter_volume_mean")),
                    _format_mm(lat_id),
                    _format_mm(diff_p.get("inter_depth_mean")),
                    _format_mm(lat_id_max),
                    _format_mm(diff_p.get("inter_depth_max")),
                    _format_pct(lat_cr_mean),
                    _format_pct(diff_p.get("contact_ratio_mean")),
                    _format_pct(lat_cr_contact),
                    _format_pct(diff_p.get("contact_ratio_contact")),
                ]
            )
        if lat_p or (is_gt_row and lat_n):
            physics_latethoi_rows.append(
                [
                    split_label,
                    file_name,
                    str(int(samples)),
                    _format_pct(
                        lat_p.get("contact_ratio_off_ground")
                        if lat_p
                        else lat_n.get("contact_ratio_off_ground")
                    ),
                    _format_pct(
                        lat_p.get("off_ground_contact_ratio")
                        if lat_p
                        else lat_n.get("off_ground_contact_ratio")
                    ),
                    _format_pct(
                        lat_p.get("off_ground_contact_rate")
                        if lat_p
                        else lat_n.get("off_ground_contact_rate")
                    ),
                    _format_float(
                        lat_p.get("jerk") if lat_p else lat_n.get("jerk"), digits=6
                    ),
                ]
            )
        if diff_p:
            physics_diffh2o_rows.append(
                [
                    split_label,
                    file_name,
                    str(int(samples)),
                    _format_float(diff_p.get("jerk_pos"), digits=6),
                    _format_float(diff_p.get("jerk_ang"), digits=6),
                ]
            )
        if lat_n or diff_n:
            motion_common_rows.append(
                [
                    split_label,
                    file_name,
                    str(int(samples)),
                    _format_float(lat_n.get("sample_diversity_last_frame"), digits=4),
                    _format_float(diff_n.get("sample_diversity_m"), digits=4),
                    _format_float(lat_n.get("overall_diversity_last_frame"), digits=4),
                    _format_float(diff_n.get("overall_diversity_m"), digits=4),
                ]
            )

    with open(path, "w") as f:
        f.write("# Computed Method Summaries\n\n")
        f.write("## Physics\n\n")
        f.write("### Common\n\n")
        f.write(
            _markdown_table(
                [
                    "Split",
                    "File",
                    "Samples",
                    "LatetHOI IV mean (cm^3)",
                    "DiffH2O IV mean (cm^3)",
                    "LatetHOI ID mean (mm)",
                    "DiffH2O ID mean (mm)",
                    "LatetHOI ID max (mm)",
                    "DiffH2O ID max (mm)",
                    "LatetHOI CR mean (%)",
                    "DiffH2O CR mean (%)",
                    "LatetHOI CR contact (%)",
                    "DiffH2O CR contact (%)",
                ],
                physics_common_rows,
            )
        )
        f.write("\n### LatetHOI\n\n")
        f.write(
            _markdown_table(
                [
                    "Split",
                    "File",
                    "Samples",
                    "CR off-ground (%)",
                    "Off-ground CR (%)",
                    "Off-ground Rate (%)",
                    "Jerk",
                ],
                physics_latethoi_rows,
            )
        )
        f.write("\n### DiffH2O\n\n")
        f.write(
            _markdown_table(
                ["Split", "File", "Samples", "Jerk pos", "Jerk ang"],
                physics_diffh2o_rows,
            )
        )
        f.write("\n## Motion\n\n")
        f.write("### Common\n\n")
        f.write(
            _markdown_table(
                [
                    "Split",
                    "File",
                    "Samples",
                    "LatetHOI SD",
                    "DiffH2O SD",
                    "LatetHOI OD",
                    "DiffH2O OD",
                ],
                motion_common_rows,
            )
        )


def _split_tag_from_file_name(file_name: str) -> str:
    name = os.path.basename(str(file_name)).lower()
    if name.startswith("us_") or "_us_" in name:
        return "unseen"
    if name.startswith("s_") or "_s_" in name:
        return "seen"
    return "other"


def _allow_gt_source_file(file_name: str) -> bool:
    name = os.path.basename(str(file_name)).lower()
    banned_tokens = ("cov_map", "diffh2o")
    return not any(token in name for token in banned_tokens)


def _print_summary_block(
    rows: list[tuple[str, dict]],
    min_cr_for_valid: float,
    title: str,
    gt_row: Optional[dict] = None,
) -> None:
    if not rows:
        print(f"\n=== {title} ===")
        print("[WARN] no valid samples found in this split.")
        return

    print(f"\n=== {title} ===")
    print("=== File-wise Averages (Overall: all samples) ===")
    print(
        "file | samples | IV[cm^3] | ID[mm] | IDmax[mm] | CR | success rate | SD[m] | OD[m] | valid rate"
    )
    if gt_row is not None:
        print(
            f"G.T | {gt_row.get('samples', 0)} | {_format_float(gt_row.get('iv_cm3'), digits=4)} | {_format_float(gt_row.get('id_mm'), digits=4)} | "
            f"{_format_float(gt_row.get('id_max_mm'), digits=4)} | {gt_row.get('cr', 0.0):.4f} | {gt_row.get('success_rate', 0.0):.4f} | "
            f"{gt_row.get('sample_diversity_m', 0.0):.4f} | {gt_row.get('overall_diversity_m', 0.0):.4f} | "
            f"{_format_percent_with_count(gt_row.get('valid_contact_rate', 0.0), gt_row.get('valid_samples', 0), gt_row.get('samples', 0), digits=2)}"
        )
    else:
        print("G.T | 0 | NA | NA | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000")
    for file_name, row in rows:
        print(
            f"{file_name} | {row['samples']} | {_format_float(row['iv_cm3'], digits=4)} | {_format_float(row['id_mm'], digits=4)} | "
            f"{_format_float(row['id_max_mm'], digits=4)} | {row['cr']:.4f} | {row.get('success_rate', 0.0):.4f} | {row['sample_diversity_m']:.4f} | "
            f"{row['overall_diversity_m']:.4f} | {_format_percent_with_count(row['valid_contact_rate'], row.get('valid_samples', 0), row.get('samples', 0), digits=2)}"
        )

    print("=== Ranking (Overall): success rate higher is better ===")
    for i, (file_name, row) in enumerate(
        sorted(rows, key=lambda x: -x[1].get("success_rate", 0.0)), start=0
    ):
        print(
            f"{i}. {file_name} | success_rate={row.get('success_rate', 0.0):.4f} "
            f"({row.get('success_samples', 0)}/{row['samples']})"
        )

    print("=== Ranking (Overall): valid rate higher is better ===")
    for i, (file_name, row) in enumerate(
        sorted(rows, key=lambda x: -x[1]["valid_contact_rate"]), start=0
    ):
        print(
            f"{i}. {file_name} | valid_rate={row['valid_contact_rate']:.4f} "
            f"({row['valid_samples']}/{row['samples']})"
        )

    print("=== Ranking (Overall): CR higher is better ===")
    cr_rank_rows = [x for x in rows if float(x[1].get("cr", 0.0)) != 0.0]
    for i, (file_name, row) in enumerate(
        sorted(cr_rank_rows, key=lambda x: -x[1]["cr"]), start=0
    ):
        print(
            f"{i}. {file_name} | CR={row['cr']:.4f} "
            f"| valid_rate={row['valid_contact_rate']:.4f}"
        )

    print(
        f"=== Ranking (Valid-only): IV/ID lower is better ({MIN_CONTACT_KEY_JOINTS}+ joints contact at the last frame) ==="
    )
    iv_rank_rows = [
        x
        for x in rows
        if x[1].get("iv_cm3_valid") is not None and float(x[1]["iv_cm3_valid"]) != 0.0
    ]
    if not iv_rank_rows:
        print("IV disabled")
    else:
        by_valid_iv = sorted(
            iv_rank_rows,
            key=lambda x: (
                x[1]["iv_cm3_valid"] if x[1]["valid_samples"] > 0 else float("inf")
            ),
        )
        for i, (file_name, row) in enumerate(by_valid_iv, start=0):
            if row["valid_samples"] <= 0:
                print(f"{i}. {file_name} | no valid-contact samples")
            else:
                print(
                    f"{i}. {file_name} | IV_valid={row['iv_cm3_valid']:.4f} cm^3 | "
                    f"n={row['valid_samples']}"
                )

    print(
        f"=== Ranking (Valid-only): ID lower is better ({MIN_CONTACT_KEY_JOINTS}+ joints contact at the last frame) ==="
    )
    id_rank_rows = [
        x
        for x in rows
        if x[1].get("id_mm_valid") is not None and float(x[1]["id_mm_valid"]) != 0.0
    ]
    if not id_rank_rows:
        print("ID disabled")
    else:
        by_valid_id = sorted(
            id_rank_rows,
            key=lambda x: (
                x[1]["id_mm_valid"] if x[1]["valid_samples"] > 0 else float("inf")
            ),
        )
        for i, (file_name, row) in enumerate(by_valid_id, start=0):
            if row["valid_samples"] <= 0:
                print(f"{i}. {file_name} | no valid-contact samples")
            else:
                print(
                    f"{i}. {file_name} | ID_valid={row['id_mm_valid']:.4f} mm | "
                    f"IDmax_valid={row['id_max_mm_valid']:.4f} mm | n={row['valid_samples']}"
                )

    # Relative-rotation debug print sections removed by request.


def print_summary(all_results: list[dict], min_cr_for_valid: float) -> None:
    rows = []
    for result in all_results:
        row = result["overall"]
        if row["samples"] <= 0:
            continue
        file_name = result["file_name"]
        rows.append((file_name, row))

    if not rows:
        print("[WARN] no valid samples found.")
        return

    seen_rows = [(f, r) for f, r in rows if _split_tag_from_file_name(f) == "seen"]
    unseen_rows = [(f, r) for f, r in rows if _split_tag_from_file_name(f) == "unseen"]
    other_rows = [(f, r) for f, r in rows if _split_tag_from_file_name(f) == "other"]
    summary_rows = _summary_rows_with_gt(all_results, min_cr_for_valid)
    seen_gt_row = next(
        (
            r
            for r in summary_rows
            if r.get("split") == "seen" and bool(r.get("is_gt_row"))
        ),
        None,
    )
    unseen_gt_row = next(
        (
            r
            for r in summary_rows
            if r.get("split") == "unseen" and bool(r.get("is_gt_row"))
        ),
        None,
    )

    _print_summary_block(
        seen_rows, min_cr_for_valid, "Seen Objects (_s_)", gt_row=seen_gt_row
    )
    _print_summary_block(
        unseen_rows, min_cr_for_valid, "Unseen Objects (_us_)", gt_row=unseen_gt_row
    )
    if other_rows:
        _print_summary_block(other_rows, min_cr_for_valid, "Other Files (no _s_/_us_)")
    physics_rows, motion_rows = _build_category_tables(summary_rows)
    _print_table(
        "Physics Summary",
        [
            "Split",
            "File",
            "Samples",
            "VR (%) ↑",
            "CR (%) ↑",
            "SR (%) ↑",
            "IV (cm^3) ↓",
            "ID (mm) ↓",
            "ID_max (mm) ↓",
        ],
        physics_rows,
    )
    _print_table(
        "Motion Summary",
        [
            "Split",
            "File",
            "Samples",
            "SD (m) ↑",
            "Object Avg SD (m) ↑",
            "OD Local (m) ↑",
            "OD Aligned (m) ↑",
            "Object Avg OD (m) ↑",
        ],
        motion_rows,
    )

    latethoi_project_rows = _build_latethoi_project_table_rows(
        _project_metric_summary_rows(all_results, "latethoi_project")
    )
    _print_table(
        "LatetHOI Project-Code Summary",
        [
            "Split",
            "File",
            "Computed",
            "IV mean (m^3) ↓",
            "ID mean (m) ↓",
            "ID max (m) ↓",
            "CR mean ↑",
            "CR contact ↑",
            "CR off-ground ↑",
            "Jerk ↓",
        ],
        latethoi_project_rows,
    )

    diffh2o_project_rows = _build_diffh2o_project_table_rows(
        _project_metric_summary_rows(all_results, "diffh2o_project")
    )
    _print_table(
        "DiffH2O Project-Code Summary",
        [
            "Split",
            "File",
            "Computed",
            "IV mean (m^3) ↓",
            "ID mean (m) ↓",
            "ID max (m) ↓",
            "CR mean ↑",
            "CR contact ↑",
            "Jerk pos ↓",
            "Jerk ang ↓",
        ],
        diffh2o_project_rows,
    )

    bimart_rows = _build_bimart_table_rows(_bimart_summary_rows(all_results))
    _print_table(
        "BimArt Metrics Summary",
        [
            "Split",
            "File",
            "Computed",
            "APD multi ↓",
            "Accel ↓",
            "Penetration 1cm ↓",
            "Contact % ↑",
        ],
        bimart_rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "Desktop/hot3d_vis"),
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[
            "s_cov_map.pkl",
            # "s_bps_1stage.pkl",
            # "us_cov_map.pkl",
            # "us_bps_1stage.pkl",
            "diffh2o.pkl",
            "s_bps_bim_cano.pkl",
            "s_bps_bim_cano_dist.pkl",
            # "bps_hand.pkl",
        ],
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.005,
        help="meters; threshold for contact decision",
    )
    parser.add_argument(
        "--min-cr-for-valid",
        type=float,
        default=0.02,
        help="legacy option; valid-contact now uses last-frame joint count criterion",
    )
    parser.add_argument(
        "--text-filter",
        nargs="*",
        default=None,
        help="optional exact text filter list",
    )
    parser.add_argument(
        "--gt-file-name",
        type=str,
        default="us_bps_1stage.pkl",
        help="file used for GT reference extraction",
    )
    parser.add_argument(
        "--compare-gt",
        action="store_true",
        help="enable GT comparison/visualization for the file named by --gt-file-name (default: enabled)",
    )
    parser.add_argument(
        "--skip-compare-gt",
        dest="compare_gt",
        action="store_false",
        help="skip GT comparison/visualization",
    )
    parser.add_argument(
        "--visualize-eval",
        action="store_true",
        help="visualize each sample's last frame and evaluated metrics in Rerun",
    )
    parser.add_argument(
        "--visualize-metric-rankings",
        action="store_true",
        help="visualize per-file top/bottom metric-ranked samples in Rerun as eval/<file>/<metric>/<Top|Bottom>/<sample>",
    )
    parser.add_argument(
        "--visualize-metric-topk",
        type=int,
        default=5,
        help="number of top/bottom samples per metric to log when metric ranking visualization is enabled",
    )
    parser.add_argument(
        "--visualize-hand-style",
        type=str,
        choices=("mesh", "vertices"),
        default="mesh",
        help="when visualizing hands in Rerun, render either the MANO mesh or all 778 vertices as points",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="*",
        type=int,
        default=None,
        help="optional sample indices to evaluate/visualize",
    )
    parser.add_argument(
        "--visualize-gt-vr-fail-only",
        action="store_true",
        help="deprecated; GT visualization now shows only SR-failed samples",
    )
    parser.add_argument(
        "--visualize-gt-topk",
        type=int,
        default=5,
        help="when --visualize-eval and GT comparison are enabled, visualize top/bottom-K Seen.G.T samples for CR/IV/ID/ID_max",
    )
    parser.add_argument(
        "--no-compute-iv",
        dest="compute_iv",
        action="store_false",
        help="disable IV computation",
    )
    parser.add_argument(
        "--no-compute-id",
        dest="compute_id",
        action="store_false",
        help="disable ID computation",
    )
    parser.add_argument(
        "--diversity-vis-dir",
        type=str,
        default=None,
        help="optional directory for SD/OD diversity plots",
    )
    parser.add_argument(
        "--resample-object-pc-from-mesh",
        action="store_true",
        help="replace obj.pkl object point clouds with 1024 surface samples drawn from obj_path meshes",
    )
    parser.add_argument(
        "--resampled-object-png-dir",
        type=str,
        default=None,
        help="optional directory for PNG snapshots of object point clouds used after mesh resampling; defaults to ./resampled_object_vis when resampling is enabled",
    )
    parser.add_argument(
        "--run-native-method-summaries",
        action="store_true",
        help="run external project-native evaluation commands before reading their summary result files",
    )
    parser.add_argument(
        "--latethoi-repo-dir",
        type=str,
        default=os.path.abspath(
            os.path.join(HOT3D_ROOT, "..", "..", "LatetHOI", "projects", "mdm_hand")
        ),
        help="LatetHOI project directory used for native summary execution",
    )
    parser.add_argument(
        "--latethoi-native-folder",
        type=str,
        default=None,
        help="folder containing LatetHOI .pth predictions for native evaluation/result loading",
    )
    parser.add_argument(
        "--latethoi-native-command",
        type=str,
        default="{python} tools/eval_motion.py --eval -f {folder}",
        help="command template run inside --latethoi-repo-dir; placeholders: {python}, {folder}, {repo}",
    )
    parser.add_argument(
        "--latethoi-native-result",
        type=str,
        default=None,
        help="path to LatetHOI native summary result; defaults to <folder>/.result.json",
    )
    parser.add_argument(
        "--diffh2o-repo-dir",
        type=str,
        default=os.path.abspath(os.path.join(HOT3D_ROOT, "..", "..", "diffh2o")),
        help="DiffH2O project directory used for project-code metric execution",
    )
    parser.add_argument(
        "--run-project-native-metrics",
        dest="run_project_native_metrics",
        action="store_true",
        default=True,
        help="compute LatetHOI/DiffH2O summaries by calling their project metric modules unchanged on converted HOT3D geometry (default: enabled)",
    )
    parser.add_argument(
        "--skip-project-native-metrics",
        dest="run_project_native_metrics",
        action="store_false",
        help="skip LatetHOI/DiffH2O project-code metric execution",
    )
    parser.add_argument(
        "--project-native-eval-frames",
        type=int,
        default=50,
        help="maximum evenly sampled frames per sequence for project-code metric adapters",
    )
    parser.add_argument(
        "--run-bimart-metrics",
        dest="run_bimart_metrics",
        action="store_true",
        default=True,
        help="compute BimArt-compatible metrics (default: enabled)",
    )
    parser.add_argument(
        "--skip-bimart-metrics",
        dest="run_bimart_metrics",
        action="store_false",
        help="skip BimArt-compatible metrics to reduce runtime",
    )
    parser.add_argument(
        "--text2hoi-repo-dir",
        type=str,
        default=os.path.abspath(os.path.join(HOT3D_ROOT, "..", "..", "Text2HOI")),
        help="Text2HOI project directory used for native summary execution",
    )
    parser.add_argument(
        "--text2hoi-native-folder",
        type=str,
        default=None,
        help="folder containing Text2HOI predictions or native summary outputs",
    )
    parser.add_argument(
        "--text2hoi-native-command",
        type=str,
        default=None,
        help="optional command template run inside --text2hoi-repo-dir; placeholders: {python}, {folder}, {repo}",
    )
    parser.add_argument(
        "--text2hoi-native-result",
        type=str,
        default=None,
        help="path to Text2HOI native summary result; defaults to <folder>/.result.json",
    )
    parser.set_defaults(compute_id=True, compute_iv=True, compare_gt=True)
    args = parser.parse_args()

    def _resolve_input_path(file_name: str, input_dir: str) -> str:
        expanded = os.path.expanduser(file_name)
        if os.path.isabs(expanded):
            return expanded
        candidate = os.path.join(os.path.expanduser(input_dir), expanded)
        if os.path.exists(candidate):
            return candidate
        return os.path.abspath(expanded)

    home = os.path.expanduser("~")
    if args.visualize_eval or args.visualize_metric_rankings:
        rerun_started = _start_rerun_visualization()
        if not rerun_started:
            args.visualize_eval = False
            args.visualize_metric_rankings = False

    obj_pkl_path = os.path.join(home, "Desktop/hot3d_vis/obj.pkl")
    object_model = ObjectModel(obj_pkl_path)

    # Use lowercased keys for robust text matching.
    obj_pc_by_key = {}
    obj_pc_normals_by_key = {}
    obj_mesh_by_key = {}
    object_axis_alignment_by_key = {}
    proxy_mesh_cache = {}
    resampled_object_vis_rows = []
    for k, pc in object_model.obj_pcs.items():
        key = str(k).lower()
        original_pc = np.asarray(pc, dtype=np.float32)
        pc_to_use = original_pc
        normals = None
        if object_model.obj_pc_normals is not None:
            normals = object_model.obj_pc_normals.get(k)
        normals_to_use = (
            np.asarray(normals, dtype=np.float32) if normals is not None else None
        )
        if normals is not None:
            obj_pc_normals_by_key[key] = normals_to_use
        obj_path_value = object_model.obj_path.get(k)
        resolved_mesh = None
        if obj_path_value is None:
            print(f"[WARN] missing obj_path entry for '{key}' in obj.pkl")
        else:
            resolved_mesh = _load_object_mesh(
                obj_pkl_path, obj_path_value, pc_to_use, object_key=key
            )
        original_metric_mesh = _get_or_build_proxy_mesh_from_object_pc(
            original_pc, object_key=key, proxy_cache=proxy_mesh_cache
        )
        if resolved_mesh is not None:
            if args.resample_object_pc_from_mesh:
                sampled_pc, sampled_normals = _sample_object_pc_from_mesh(
                    resolved_mesh, object_key=key, count=int(pc_to_use.shape[0])
                )
                if sampled_pc is not None:
                    pc_to_use = sampled_pc
                    normals_to_use = sampled_normals
                    resampled_object_vis_rows.append(
                        {
                            "object_key": key,
                            "original_pc": original_pc.copy(),
                            "used_pc": np.asarray(sampled_pc, dtype=np.float32).copy(),
                            "mesh_vertices": np.asarray(
                                resolved_mesh.vertices, dtype=np.float32
                            ).copy(),
                            "mesh_faces": np.asarray(
                                resolved_mesh.faces, dtype=np.int64
                            ).copy(),
                            "orig_proxy_vertices": (
                                np.asarray(
                                    original_metric_mesh.vertices, dtype=np.float32
                                ).copy()
                                if original_metric_mesh is not None
                                else np.zeros((0, 3), dtype=np.float32)
                            ),
                            "orig_proxy_faces": (
                                np.asarray(
                                    original_metric_mesh.faces, dtype=np.int64
                                ).copy()
                                if original_metric_mesh is not None
                                else np.zeros((0, 3), dtype=np.int64)
                            ),
                        }
                    )
        metric_mesh = _get_or_build_proxy_mesh_from_object_pc(
            pc_to_use, object_key=key, proxy_cache=proxy_mesh_cache
        )
        if (
            args.resample_object_pc_from_mesh
            and metric_mesh is not None
            and resampled_object_vis_rows
            and resampled_object_vis_rows[-1].get("object_key") == key
        ):
            resampled_object_vis_rows[-1]["used_proxy_vertices"] = np.asarray(
                metric_mesh.vertices, dtype=np.float32
            ).copy()
            resampled_object_vis_rows[-1]["used_proxy_faces"] = np.asarray(
                metric_mesh.faces, dtype=np.int64
            ).copy()
        if metric_mesh is not None:
            obj_mesh_by_key[key] = metric_mesh
        elif resolved_mesh is not None:
            print(
                f"[WARN] using resolved mesh for metrics on '{key}' because "
                "point-cloud proxy mesh reconstruction failed."
            )
            obj_mesh_by_key[key] = resolved_mesh
        obj_pc_by_key[key] = torch.as_tensor(pc_to_use, dtype=torch.float32)
        object_axis_alignment_by_key[key] = _object_axis_alignment_from_points(
            pc_to_use
        )
        if normals_to_use is not None:
            obj_pc_normals_by_key[key] = np.asarray(normals_to_use, dtype=np.float32)
        elif key in obj_pc_normals_by_key:
            del obj_pc_normals_by_key[key]

    if args.resample_object_pc_from_mesh:
        object_vis_dir = (
            args.resampled_object_png_dir
            if args.resampled_object_png_dir
            else os.path.join(os.getcwd(), "resampled_object_vis")
        )
        _save_resampled_object_visualizations(resampled_object_vis_rows, object_vis_dir)

    print("[INFO] initializing MANO hand layers...")
    l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
    r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)
    print("[INFO] MANO hand layers ready.")

    text_filter = None
    if args.text_filter:
        text_filter = {str(t).strip().lower() for t in args.text_filter}
    sample_idx_filter = set(args.sample_indices) if args.sample_indices else None

    all_results = []
    print(f"[INFO] evaluating {len(args.input_files)} input file(s)...")
    for file_name in tqdm.tqdm(args.input_files, desc="files"):
        path = _resolve_input_path(file_name, args.input_dir)
        if not os.path.exists(path):
            print(f"[WARN] missing file: {path}")
            continue
        all_results.append(
            evaluate_file(
                path,
                obj_pc_by_key,
                obj_pc_normals_by_key,
                obj_mesh_by_key,
                object_axis_alignment_by_key,
                l_hand_layer,
                r_hand_layer,
                args.contact_threshold,
                text_filter,
                args.min_cr_for_valid,
                args.compute_id,
                args.compute_iv,
                args.compare_gt,
                visualize_eval=args.visualize_eval,
                visualize_hand_style=args.visualize_hand_style,
                sample_idx_filter=sample_idx_filter,
                visualize_gt_vr_fail_only=args.visualize_gt_vr_fail_only,
                visualize_gt_topk=args.visualize_gt_topk,
                visualize_metric_rankings=args.visualize_metric_rankings,
                visualize_metric_topk=args.visualize_metric_topk,
                run_project_native_metrics=args.run_project_native_metrics,
                latethoi_repo_dir=args.latethoi_repo_dir,
                diffh2o_repo_dir=args.diffh2o_repo_dir,
                project_native_eval_frames=args.project_native_eval_frames,
                run_bimart_metrics=args.run_bimart_metrics,
            )
        )

    if not all_results:
        print("[WARN] no valid files processed.")
        return

    output_root = os.getcwd()
    out_sample = os.path.join(output_root, "interaction_metrics_per_sample.csv")
    out_file = os.path.join(output_root, "interaction_metrics_file_avg.csv")
    out_object_div = os.path.join(
        output_root, "interaction_metrics_diversity_by_object.csv"
    )
    out_summary_csv = os.path.join(output_root, "interaction_metrics_summary.csv")
    out_summary_md = os.path.join(output_root, "interaction_metrics_summary.md")
    out_native_md = os.path.join(output_root, "native_method_summaries.md")
    write_per_sample_csv(out_sample, all_results)
    write_file_avg_csv(out_file, all_results)
    write_object_diversity_csv(out_object_div, all_results)
    summary_rows = _summary_rows_with_gt(all_results, args.min_cr_for_valid)
    write_summary_csv(out_summary_csv, summary_rows)
    write_summary_markdown(out_summary_md, summary_rows, all_results=all_results)
    if args.diversity_vis_dir:
        _plot_diversity_visualizations(
            all_results, summary_rows, args.diversity_vis_dir
        )
    latethoi_native_summary_rows = _latethoi_lastframe_summary_rows(all_results)
    diffh2o_native_summary_rows = _diffh2o_native_summary_rows(all_results)
    latethoi_project_summary_rows = _project_metric_summary_rows(
        all_results, "latethoi_project"
    )
    diffh2o_project_summary_rows = _project_metric_summary_rows(
        all_results, "diffh2o_project"
    )
    write_computed_method_summaries_markdown(
        out_native_md,
        latethoi_native_summary_rows,
        diffh2o_native_summary_rows,
        latethoi_project_summary_rows,
        diffh2o_project_summary_rows,
    )
    print_summary(all_results, args.min_cr_for_valid)
    print(f"\nSaved per-sample CSV: {out_sample}")
    print(f"Saved file-average CSV: {out_file}")
    print(f"Saved object-diversity CSV: {out_object_div}")
    print(f"Saved split-summary CSV: {out_summary_csv}")
    print(f"Saved split-summary Markdown: {out_summary_md}")
    print(f"Saved native-method Markdown: {out_native_md}")
    if args.visualize_eval or args.visualize_metric_rankings:
        _log_rerun_status("Evaluation complete.")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
