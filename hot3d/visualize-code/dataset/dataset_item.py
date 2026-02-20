import argparse

import os
import torch
import numpy as np

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import json
import inspect
import re
import hashlib
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
from sklearn.manifold import TSNE

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)


def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)


def _set_frame_time(frame_idx: int) -> None:
    # Rerun API differs by version.
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_idx)
    else:
        rr.set_time("frame", sequence=frame_idx)




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


def _color_for_text(text: str) -> list[int]:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return [int(digest[0]), int(digest[1]), int(digest[2])]


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


def _extract_of_with_target(text: str) -> Optional[str]:
    """Return the target between 'of' and 'with' for grouping."""
    match = re.search(r"\bof\s+(.+?)\s+with\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    target = re.sub(r"\s+", " ", match.group(1)).strip()
    return target.lower() if target else None


def _base_text_before_with(text: str) -> str:
    return text.strip()


def _build_grouping_keys(text: str) -> tuple[str, str]:
    """Group by object first, then by full text (include hand details)."""
    object_key = _extract_of_with_target(text) or text
    full_text = _base_text_before_with(text)
    action_key = f"{object_key}::{full_text}"
    return object_key, action_key


with open(os.path.join(home, "Desktop/hot3d_vis/instance.json"), "r") as f:
    instance_ = json.load(f)
label_merged_path = os.path.join(home, "label_merged.json")
if not os.path.exists(label_merged_path):
    local_path = os.path.join(os.path.dirname(__file__), "..", "..", "label_merged.json")
    local_path = os.path.abspath(local_path)
    if os.path.exists(local_path):
        label_merged_path = local_path
    else:
        local_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "label_merged.json")
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

def _norm_key(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value

_label_key_map = {}
_label_part_map = {}
if isinstance(label_merged, dict):
    for k, v in label_merged.items():
        _label_key_map[_norm_key(k)] = k
        if isinstance(v, dict):
            _label_part_map[k] = { _norm_key(pk): pk for pk in v.keys() }

object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
obj_pc = dict()

for obj_name in object_model.obj_pcs.keys():
    _, pc, _, _ = object_model(obj_name)
    obj_pc[obj_name] = torch.tensor(pc)

l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

def _to_tensor(data):
    if torch.is_tensor(data):
        return data
    if isinstance(data, (list, tuple)) and len(data) > 0 and torch.is_tensor(data[0]):
        return torch.stack(list(data), dim=0)
    return torch.as_tensor(data)


def _coerce_index(idx):
    if isinstance(idx, int):
        return idx
    if isinstance(idx, str) and idx.isdigit():
        return int(idx)
    return 0


def _slice_dict_of_arrays(source: dict, idx: int) -> dict:
    sliced = {}
    for key, value in source.items():
        if isinstance(value, (list, tuple)) and len(value) > idx:
            sliced[key] = value[idx]
        elif isinstance(value, np.ndarray) and value.shape[0] > idx:
            sliced[key] = value[idx]
        elif torch.is_tensor(value) and value.shape[0] > idx:
            sliced[key] = value[idx]
        else:
            sliced[key] = value
    return sliced


def _build_item_from_fields(dataset, idx: int) -> dict:
    # Reconstruct a sample by slicing any per-sample fields in __dict__.
    item = {}
    for key, value in getattr(dataset, "__dict__", {}).items():
        if isinstance(value, (list, tuple)) and len(value) > idx:
            item[key] = value[idx]
            continue
        if isinstance(value, np.ndarray) and value.shape[0] > idx:
            item[key] = value[idx]
            continue
        if torch.is_tensor(value) and value.shape[0] > idx:
            item[key] = value[idx]
            continue

    # Enrich common derived fields if present.
    if "object_idx" in item and hasattr(dataset, "obj_json"):
        object_idx = int(np.asarray(item["object_idx"]))
        object_name = dataset.obj_json[str(object_idx)]["instance_name"]
        item["obj_name"] = object_name
        if hasattr(dataset, "norm_obj_pc_list"):
            item["normalized_obj_pc"] = dataset.norm_obj_pc_list[object_name]
    if "action" in item and "text" not in item:
        item["text"] = item["action"]
    return item


def _extract_item(source, idx):
    if isinstance(source, DataLoader):
        source = source.dataset

    if isinstance(source, dict):
        return _slice_dict_of_arrays(source, idx)

    if isinstance(source, (list, tuple, np.ndarray, torch.Tensor)):
        return source[idx]

    if isinstance(source, Dataset) and hasattr(source, "__getitem__"):
        try:
            item = source[idx]
            if isinstance(item, dict):
                return item
            return {"item": item}
        except Exception:
            pass

    if hasattr(source, "__dict__"):
        item = _build_item_from_fields(source, idx)
        if item:
            return item

    raise TypeError(f"Unsupported pickle payload type: {type(source)}")


def _load_payload(path: str):
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Retry with weights_only=False for trusted local debug snapshots.
    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)


def _infer_sample_count(payload) -> int:
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, (list, tuple)):
                return len(value)
            if isinstance(value, np.ndarray) and value.ndim > 0:
                return int(value.shape[0])
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    if isinstance(payload, (list, tuple, np.ndarray)):
        return len(payload)
    if torch.is_tensor(payload) and payload.ndim > 0:
        return int(payload.shape[0])
    return 1


def visualize_rr(recoding_name, idx=None, max_items: int = 32):
    # Ensure rerun is initialized even when visualize_rr is called directly.
    try:
        rr.init("Input Data", spawn=True)
    except Exception:
        pass

    payload = _load_payload(recoding_name)
    if idx is None:
        total = _infer_sample_count(payload)
        idx_list = list(range(min(total, max_items)))
    elif isinstance(idx, (list, tuple, range)):
        idx_list = [_coerce_index(v) for v in idx]
    else:
        idx_list = [_coerce_index(idx)]

    def _as_numpy(v):
        if torch.is_tensor(v):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    def _to_mask(v):
        if v is None:
            return None
        arr = _as_numpy(v).reshape(-1)
        if arr.dtype == bool:
            return arr
        return arr > 0

    def _to_scores(v):
        if v is None:
            return None
        arr = _as_numpy(v).reshape(-1)
        if arr.dtype == bool:
            return arr.astype(np.float32)
        arr = arr.astype(np.float32)
        if arr.size == 0:
            return arr
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax <= vmin:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - vmin) / (vmax - vmin)

    def _safe_text(entry):
        if isinstance(entry, (list, tuple)):
            entry = entry[0] if len(entry) > 0 else "item"
        if not isinstance(entry, str):
            entry = str(entry)
        entry = re.sub(r"\s+", "_", entry.strip())
        return entry or "item"

    for sample_idx in idx_list:
        item = _extract_item(payload, sample_idx)
        sample_root = f"sample_{sample_idx:02d}"
        rr.log(
            f"debug/loaded/{sample_root}",
            rr.Points3D(positions=[[0.0, 0.0, 0.0]], radii=0.02, colors=[255, 255, 255]),
            static=True,
        )
        print(f"[INFO] visualize_rr loaded: {recoding_name}, idx={sample_idx}")

        text = item["text"]
        n_obj = item.get("normalized_obj_pc", item.get("normalize_obj_pc"))
        n_obj_secondary = item.get(
            "normalized_obj_pc_secondary",
            item.get("normalize_obj_pc_secondary"),
        )
        afford_raw = item.get("afford_map")
        if afford_raw is None:
            afford_raw = item.get("affordance") or item.get("affordnace")
        affordance = (1 - afford_raw) if afford_raw is not None else None
        cov_map = item.get("cov_map")

        text_entries = text if isinstance(text, (list, tuple)) else [text]

        def _select_entry(value, entry_idx: int):
            arr = _as_numpy(value)
            if arr.ndim == 0:
                return arr
            # After _extract_item(..., idx), most tensors are already per-sample.
            # Only index by entry_idx when first dim actually matches text entry count.
            if len(text_entries) > 1 and arr.shape[0] == len(text_entries):
                return arr[entry_idx]
            return arr

        for batch_idx, text_entry in enumerate(text_entries):
            object_key = "object"
            part_key = "part"
            if isinstance(text_entry, str):
                match = re.search(
                    r"grab\s+(?P<part>.+?)\s+of\s+(?P<object>.+)$",
                    text_entry,
                    re.IGNORECASE,
                )
                if match:
                    part_key = _safe_text(match.group("part"))
                    object_raw = match.group("object").strip()
                    object_raw = re.split(r"\s+with\s+", object_raw, flags=re.IGNORECASE)[0]
                    object_raw = object_raw.rstrip(" .,:;")
                    object_key = _safe_text(object_raw)
                else:
                    object_key = _safe_text(text_entry)
                    part_key = "part"
            object_norm = _norm_key(object_key)
            object_lookup = _label_key_map.get(object_norm)
            part_lookup = None
            if object_lookup is not None:
                part_norm = _norm_key(part_key)
                part_lookup = _label_part_map.get(object_lookup, {}).get(part_norm)

            log_prefix = f"{sample_root}/{object_key}/{part_key}/{batch_idx}"
            if isinstance(n_obj, (list, tuple)) and len(text_entries) > 1:
                obj_points = _as_numpy(n_obj[batch_idx])
            else:
                obj_points = _as_numpy(n_obj)

            obj_points_secondary = None
            if n_obj_secondary is not None:
                if isinstance(n_obj_secondary, (list, tuple)) and len(text_entries) > 1:
                    obj_points_secondary = _as_numpy(n_obj_secondary[batch_idx])
                else:
                    obj_points_secondary = _as_numpy(n_obj_secondary)

            if obj_points.ndim == 3:
                frames = obj_points.shape[0]
            else:
                frames = 1
                obj_points = obj_points[None, ...]
            if obj_points_secondary is not None and obj_points_secondary.ndim != 3:
                obj_points_secondary = obj_points_secondary[None, ...]

            aff_mask = None
            cov_mask = None
            aff_scores = None
            cov_scores = None
            if affordance is not None:
                aff_value = _select_entry(affordance, batch_idx)
                aff_mask = _to_mask(aff_value)
                aff_scores = _to_scores(aff_value)
            if cov_map is not None:
                cov_value = _select_entry(cov_map, batch_idx)
                cov_mask = _to_mask(cov_value)
                cov_scores = _to_scores(cov_value)

            offset_step = 3
            for frame_idx in range(frames):
                _set_frame_time(frame_idx)

                points = obj_points[frame_idx]

                # Visualize normalized primary/secondary object point clouds side-by-side.
                primary_points = points + np.array([-2 * offset_step, 0.0, 0.0], dtype=np.float32)
                primary_colors = np.zeros_like(points, dtype=np.uint8)
                primary_colors[:] = [0, 180, 255]
                rr.log(
                    f"{log_prefix}/normalize_obj_pc",
                    rr.Points3D(
                        positions=primary_points,
                        radii=0.05,
                        colors=primary_colors,
                    ),
                )

                if obj_points_secondary is not None:
                    secondary_frame_idx = frame_idx if frame_idx < obj_points_secondary.shape[0] else 0
                    points_secondary = obj_points_secondary[secondary_frame_idx]
                    secondary_points = points_secondary + np.array([-offset_step, 0.0, 0.0], dtype=np.float32)
                    secondary_colors = np.zeros_like(points_secondary, dtype=np.uint8)
                    secondary_colors[:] = [0, 255, 120]
                    rr.log(
                        f"{log_prefix}/normalize_obj_pc_secondary",
                        rr.Points3D(
                            positions=secondary_points,
                            radii=0.05,
                            colors=secondary_colors,
                        ),
                    )

                base_colors = np.zeros_like(points, dtype=np.uint8)
                base_colors[:] = [0, 120, 255]

                if (
                    aff_mask is not None
                    and aff_scores is not None
                    and aff_mask.shape[0] == points.shape[0]
                    and aff_scores.shape[0] == points.shape[0]
                ):
                    base = np.array([0, 120, 255], dtype=np.float32)
                    hot = np.array([255, 255, 0], dtype=np.float32)
                    scores = np.clip(aff_scores, 0.0, 1.0)[:, None]
                    aff_colors = (base + (hot - base) * scores).astype(np.uint8)
                    aff_points = points + np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    rr.log(
                        f"{log_prefix}/afford_map",
                        rr.Points3D(
                            positions=aff_points,
                            radii=0.05,
                            colors=aff_colors,
                        ),
                    )

                # if cov_mask is not None and cov_mask.shape[0] == points.shape[0]:
                #     cov_colors = np.zeros_like(points, dtype=np.uint8)
                #     cov_colors[:] = [0, 120, 255]
                #     cov_colors[cov_mask] = [255, 255, 0]
                #     cov_points = points + np.array([offset_step, 0.0, 0.0], dtype=np.float32)
                #     rr.log(
                #         f"{log_prefix}/cov_map",
                #         rr.Points3D(
                #             positions=cov_points,
                #             radii=0.05,
                #             colors=cov_colors,
                #         ),
                #     )

                if (
                    aff_scores is not None
                    and aff_scores.shape[0] == points.shape[0]
                    and label_merged is not None
                    and object_lookup is not None
                    and part_lookup is not None
                ):
                    part_indices = np.asarray(label_merged[object_lookup][part_lookup], dtype=np.int64)
                    part_mask = np.zeros(points.shape[0], dtype=np.float32)
                    valid = (part_indices >= 0) & (part_indices < part_mask.shape[0])
                    part_mask[part_indices[valid]] = 1.0
                    part_colors = np.zeros_like(points, dtype=np.uint8)
                    part_colors[:] = [0, 120, 255]
                    part_colors[part_mask.astype(bool)] = [255, 255, 0]
                    part_points = points + np.array([2 * offset_step, 0.0, 0.0], dtype=np.float32)
                    rr.log(
                        f"{log_prefix}/part_mask",
                        rr.Points3D(
                            positions=part_points,
                            radii=0.05,
                            colors=part_colors,
                        ),
                    )
                    if cov_scores is None or cov_scores.shape[0] != points.shape[0]:
                        cov_scores = np.zeros_like(aff_scores, dtype=np.float32)
                    new_scores = 0.4 * aff_scores + 0.1 * part_mask + 0.5 * cov_scores
                    base = np.array([0, 120, 255], dtype=np.float32)
                    hot = np.array([255, 255, 0], dtype=np.float32)
                    scores = np.clip(new_scores, 0.0, 1.0)[:, None]
                    new_colors = (base + (hot - base) * scores).astype(np.uint8)
                    new_points = points + np.array([3 * offset_step, 0.0, 0.0], dtype=np.float32)
                    rr.log(
                        f"{log_prefix}/New_map",
                        rr.Points3D(
                            positions=new_points,
                            radii=0.05,
                            colors=new_colors,
                        ),
                    )


def main():
    rr.init("Input Data", spawn=True)

    recoding_name = f"{home}/Desktop/hot3d_vis/data_debug.pt"
    visualize_rr(recoding_name, idx=None, max_items=32)



if __name__ == "__main__":
    main()
