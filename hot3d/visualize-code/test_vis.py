

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import itertools
import json
import ast
import inspect
import re
from typing import Optional

from projectaria_tools.core.sophus import SE3
try:
    from projectaria_tools.utils.rerun_helpers import ToTransform3D
except Exception:
    ToTransform3D = None
import trimesh

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

from data_loaders.mano_layer import MANOHandModel
from hot3d.mano import build_mano_aa

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

import trimesh

from collections import defaultdict

import os
import torch
import numpy as np


from rot import *

import rerun as rr
import pickle


def rotate_pc_y(points, degrees):
    radians = np.deg2rad(degrees)
    cos_t = np.cos(radians)
    sin_t = np.sin(radians)
    rot = torch.tensor(
        [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]],
        dtype=points.dtype,
        device=points.device,
    )
    return points @ rot.T


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)


def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
    if ToTransform3D is None:
        return
    rr.log(label, ToTransform3D(pose, False), static=static)


def _sanitize_entity_path(text: str) -> str:
    sanitized = re.sub(r"\s+", "_", text.strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.\\-]", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "entry"


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

def _maybe_parse_list_string(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return value

def _flatten_texts(value):
    value = _maybe_parse_list_string(value)
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten_texts(v))
        return out
    return [value]


def _as_numpy_points(points):
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return np.asarray(points)

rr.init("test_vis", spawn=True)
def main(file_name, offset_step: float = 0.12) -> None:
    with open(os.path.join(home, file_name), "rb") as f:
        item_list = pickle.load(f)
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for _, (rotated_pc, cov, text) in enumerate(item_list):
        raw_texts = _flatten_texts(text)
        text_keys = [str(t) for t in raw_texts]

        if isinstance(rotated_pc, torch.Tensor):
            rotated_pc_np = rotated_pc.detach().cpu().numpy()
        else:
            rotated_pc_np = np.asarray(rotated_pc)

        cov_np = None
        if cov is not None:
            cov_np = np.asarray(cov)

        for text_idx, text_key in enumerate(text_keys):
            points = rotated_pc_np
            if rotated_pc_np.ndim == 3 and rotated_pc_np.shape[0] == len(text_keys):
                points = rotated_pc_np[text_idx]
            elif isinstance(rotated_pc, (list, tuple)) and len(rotated_pc) == len(text_keys):
                points = _as_numpy_points(rotated_pc[text_idx])

            if points.ndim != 2 or points.shape[1] != 3:
                continue
            cov_row = None
            if cov_np is not None:
                if cov_np.ndim == 2 and cov_np.shape[0] == len(text_keys):
                    cov_row = cov_np[text_idx]
                elif cov_np.ndim == 1 and cov_np.shape[0] == points.shape[0]:
                    cov_row = cov_np
            if cov_row is None or cov_row.shape[0] != points.shape[0]:
                cov_row = np.zeros(points.shape[0], dtype=np.int32)
            grouped[text_key].append((points, cov_row))

    text_counters: dict[str, int] = defaultdict(int)
    for text_idx, (text_key, pcs) in enumerate(grouped.items()):
        base_path = _sanitize_entity_path(text_key)
        offset = np.array([offset_step * text_idx, 0.0, 0.0], dtype=np.float32)
        for points, cov_row in pcs:
            points = points + offset
            cov_row = np.asarray(cov_row).astype(np.int32)
            if cov is None:
                colors = np.tile(np.array([[0, 121, 121]], dtype=np.uint8), (points.shape[0], 1))
            else:
                colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
                colors[:] = [0, 0, 255]
                colors[cov_row > 0] = [255, 255, 0]
            rr.log(
                f"{base_path}/{text_counters[text_key]}",
                rr.Points3D(
                    positions=points,
                    radii=0.05,
                    colors=colors,
                ),
                static=True,
            )
            text_counters[text_key] += 1

    #     raw_texts = _flatten_texts(text)
    #     text_keys = [str(t) for t in raw_texts]

    #     if isinstance(rotated_pc, torch.Tensor):
    #         rotated_pc_np = rotated_pc.detach().cpu().numpy()
    #     else:
    #         rotated_pc_np = np.asarray(rotated_pc)

    #     cov_np = None
    #     if cov is not None:
    #         cov_np = np.asarray(cov)

    #     for text_idx, text_key in enumerate(text_keys):
    #         points = rotated_pc_np
    #         if rotated_pc_np.ndim == 3 and rotated_pc_np.shape[0] == len(text_keys):
    #             points = rotated_pc_np[text_idx]
    #         elif isinstance(rotated_pc, (list, tuple)) and len(rotated_pc) == len(text_keys):
    #             points = _as_numpy_points(rotated_pc[text_idx])

    #         if points.ndim != 2 or points.shape[1] != 3:
    #             continue
    #         cov_row = None
    #         if cov_np is not None:
    #             if cov_np.ndim == 2 and cov_np.shape[0] == len(text_keys):
    #                 cov_row = cov_np[text_idx]
    #             elif cov_np.ndim == 1 and cov_np.shape[0] == points.shape[0]:
    #                 cov_row = cov_np
    #         if cov_row is None or cov_row.shape[0] != points.shape[0]:
    #             cov_row = np.zeros(points.shape[0], dtype=np.int32)
    #         grouped[text_key].append((points, cov_row))

    # def _log_axes(base_path: str, origin: np.ndarray, length: float = 0.3) -> None:
    #     origins = np.repeat(origin[None, :], 3, axis=0)
    #     vectors = np.array(
    #         [
    #             [length, 0.0, 0.0],
    #             [0.0, length, 0.0],
    #             [0.0, 0.0, length],
    #         ],
    #         dtype=np.float32,
    #     )
    #     colors = np.array(
    #         [
    #             [255, 0, 0],
    #             [0, 255, 0],
    #             [0, 0, 255],
    #         ],
    #         dtype=np.uint8,
    #     )
    #     rr.log(
    #         f"{base_path}/axes",
    #         rr.Arrows3D(origins=origins, vectors=vectors, colors=colors, radii=0.002),
    #         static=True,
    #     )
    #     rr.log(
    #         f"{base_path}/axis_labels",
    #         rr.Points3D(
    #             positions=[
    #                 origin + np.array([length, 0.0, 0.0], dtype=np.float32),
    #                 origin + np.array([0.0, length, 0.0], dtype=np.float32),
    #                 origin + np.array([0.0, 0.0, length], dtype=np.float32),
    #             ],
    #             radii=0.01,
    #             colors=colors,
    #             labels=["X", "Y", "Z"],
    #         ),
    #         static=True,
    #     )

    # for text_idx, (text_key, pcs) in enumerate(grouped.items()):
    #     base_path = _sanitize_entity_path(text_key)
    #     offset = np.array([offset_step * text_idx, 0.0, 0.0], dtype=np.float32)
    #     _log_axes(base_path, offset, length=0.3)
    #     for i, (points, cov_row) in enumerate(pcs):
    #         points = points + offset
    #         cov_row = np.asarray(cov_row).astype(np.int32)
    #         colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    #         colors[:] = [0, 0, 255]
    #         colors[cov_row > 0] = [255, 255, 0]
    #         rr.log(
    #             f"{base_path}/items/{i:06d}",
    #             rr.Points3D(
    #                 positions=points,
    #                 radii=0.05,
    #                 colors=colors,
    #             ),
    #             static=True,
    #         )


if __name__ == "__main__":
    main(file_name="Desktop/hot3d_vis/contact_dataset.pkl", offset_step=0.12)
    # main(file_name="Desktop/hot3d_vis/contact_grab_init_2.pkl", offset_step=-0.12)
