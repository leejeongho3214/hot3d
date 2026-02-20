from rot import *
from data_loaders.mano_layer import MANOHandModel
from mano import build_mano_aa
import numpy as np
import pickle
import rerun as rr
import torch
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import TSNE
import trimesh
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from projectaria_tools.core.sophus import SE3
import argparse
import json
import inspect
import re
import hashlib
from typing import Optional
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT3D_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if HOT3D_ROOT not in sys.path:
    sys.path.insert(0, HOT3D_ROOT)


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


def gaussian_smooth(vertices, sigma=1):
    return gaussian_filter1d(vertices, sigma=sigma, axis=0)


def log_image(image: np.array, label: str, static=False) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(pose: SE3, label: str, static=False) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)


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
    text = _coerce_text(text).lower()
    for keyword in _PART_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return keyword
    return None


def _sanitize_entity_path(text: str) -> str:
    """Make a rerun-friendly path segment while keeping the text semantics."""
    text = _coerce_text(text)
    sanitized = re.sub(r"\s+", "_", text.strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.\\-]", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "entry"


def _coerce_text(text) -> str:
    if isinstance(text, (list, tuple)):
        if len(text) == 0:
            return ""
        return _coerce_text(text[0])
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)
    return text


def _extract_of_with_target(text: str) -> Optional[str]:
    """Return the target between 'of' and 'with' for grouping."""
    text = _coerce_text(text)
    match = re.search(r"\bof\s+(.+?)\s+with\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    target = re.sub(r"\s+", " ", match.group(1)).strip()
    return target.lower() if target else None


def _base_text_before_with(text: str) -> str:
    return _coerce_text(text).strip()


def _build_grouping_keys(text: str) -> tuple[str, str]:
    """Group by object first, then by full text (include hand details)."""
    text = _coerce_text(text)
    object_key = _extract_of_with_target(text) or text
    full_text = _base_text_before_with(text)
    action_key = f"{object_key}::{full_text}"
    return object_key, action_key


def _get_batch_text(text, batch_idx: int) -> str:
    if isinstance(text, (list, tuple)):
        if len(text) > batch_idx:
            return _coerce_text(text[batch_idx])
    return _coerce_text(text)


with open(os.path.join(home, "Desktop/hot3d_vis/instance.json"), "r") as f:
    instance_ = json.load(f)

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


def _set_frame_time(frame_idx: int) -> None:
    # Rerun API differs by version: some expose set_time_sequence, others set_time.
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_idx)
    else:
        rr.set_time("frame", sequence=frame_idx)


def _apply_offset(points, offset_xyz):
    if torch.is_tensor(points):
        offset = torch.tensor(
            offset_xyz, dtype=points.dtype, device=points.device)
        return points + offset
    return np.asarray(points) + np.asarray(offset_xyz, dtype=np.float32)


def _vertex_colors_like(points, rgb_color):
    if torch.is_tensor(points):
        n = int(points.shape[0])
    else:
        n = int(np.asarray(points).shape[0])
    colors = np.tile(np.asarray(rgb_color, dtype=np.uint8), (n, 1))
    return colors


def _log_offset_anchor(label: str, offset_xyz, color, anchor_id: str) -> None:
    anchor_pos = np.asarray(
        [[offset_xyz[0], offset_xyz[1] + 0.25, offset_xyz[2]]], dtype=np.float32
    )
    rr.log(
        f"anchors/{anchor_id}",
        rr.Points3D(
            positions=anchor_pos,
            radii=[0.006],
            colors=[list(color)],
            labels=[label],
        ),
        static=True,
    )


def visualize_rr(
    recoding_name,
    idx,
    offset_xyz=(0.0, 0.0, 0.0),
    run_color=(121, 121, 121),
    log_gt=False,
):
    with open(recoding_name, "rb") as f:
        item = pickle.load(f)

        action_to_sample_counts = defaultdict(int)
        ref_obj = {}
        method = "course"
        tsne_vectors = []
        tsne_labels = []
        tsne_colors = []
        tsne_names = []
        text_colors = {}

        for (
            fine_lhand,
            fine_rhand,
            x_obj,
            text,
            course_lhand,
            course_rhand,
            _,
            cond_enc,
            est_cov_map,
            _,
            gt_x_obj,
        ) in item:
            for batch_idx in range(len(course_lhand)):
                text_entry_full = _get_batch_text(text, batch_idx)
                text_entry_base = text_entry_full
                object_key, action_key = _build_grouping_keys(text_entry_full)
                base_path = _sanitize_entity_path(object_key)
                text_path = _sanitize_entity_path(text_entry_base)
                hand_path = (
                    "hands"
                    if "both" in text_entry_full.lower()
                    else "r_hand" if "right" in text_entry_full.lower() else "l_hand"
                )

                sample_idx = action_to_sample_counts[action_key]
                action_to_sample_counts[action_key] += 1

                cond_tensor = _to_tensor(cond_enc)
                cond_vec = (
                    cond_tensor[batch_idx] if cond_tensor.ndim > 1 else cond_tensor
                )
                cond_vec = cond_vec.detach().cpu().numpy().reshape(-1)
                tsne_vectors.append(cond_vec)
                part_label = _extract_part_keyword(text_entry_base)
                part_label = (
                    part_label if part_label in {
                        "body", "rim", "handle"} else "x"
                )
                tsne_labels.append(part_label)
                tsne_names.append(part_label)
                if text_entry_base not in text_colors:
                    text_colors[text_entry_base] = _color_for_text(
                        text_entry_base)
                tsne_colors.append(text_colors[text_entry_base])

                text_root_path = f"original/{base_path}/{text_path}"
                sample_path = f"{text_root_path}/sample_{sample_idx:03d}"
                gt_prompt_path = f"{text_root_path}/_gt_obj/sample_{sample_idx:03d}"

                # if text_entry_full.split("of ")[-1].split(" with")[0].lower() not in [
                #     "mug_white"
                # ]:
                #     continue

                # obj_vertices = process_obj_result(
                #     obj_pc[text_entry_full.split("of ")[-1].split(" with")[0].lower()],
                #     x_obj[batch_idx],
                # )

                if object_key not in obj_pc:
                    print(
                        f"[WARN] unresolved object key: '{object_key}' from text '{text_entry_full}'"
                    )
                    continue
                x_obj_vertices = process_obj_result(
                    obj_pc[object_key], x_obj[batch_idx])
                gt_x_obj_vertices = None
                if (
                    log_gt
                    and gt_x_obj is not None
                    and torch.is_tensor(gt_x_obj)
                    and gt_x_obj.ndim == 3
                    and gt_x_obj.shape[0] > batch_idx
                ):
                    gt_x_obj_vertices = process_obj_result(
                        obj_pc[object_key], gt_x_obj[batch_idx]
                    )

                r_hand_vertices, r_hand_faces = (
                    process_hand_result(
                        r_hand_layer, _to_tensor(course_rhand[batch_idx])
                    )
                    if method == "course"
                    else process_hand_result(
                        r_hand_layer, _to_tensor(fine_rhand[batch_idx])
                    )
                )
                l_hand_vertices, l_hand_faces = (
                    process_hand_result(
                        l_hand_layer, _to_tensor(course_lhand[batch_idx])
                    )
                    if method == "course"
                    else process_hand_result(
                        l_hand_layer, _to_tensor(fine_lhand[batch_idx])
                    )
                )

                if action_key not in ref_obj:
                    ref_obj[action_key] = x_obj_vertices[0].detach().cpu().numpy()

                r_mesh = trimesh.Trimesh(
                    vertices=r_hand_vertices[0], faces=r_hand_faces, process=False
                )
                l_mesh = trimesh.Trimesh(
                    vertices=l_hand_vertices[0], faces=l_hand_faces, process=False
                )

                for frame_idx in range(r_hand_vertices.shape[0]):
                    _set_frame_time(frame_idx)
                    r_pos = _apply_offset(
                        r_hand_vertices[frame_idx], offset_xyz)
                    l_pos = _apply_offset(
                        l_hand_vertices[frame_idx], offset_xyz)
                    x_obj_pos = _apply_offset(
                        x_obj_vertices[frame_idx], offset_xyz)
                    gt_x_obj_pos = None
                    if gt_x_obj_vertices is not None:
                        gt_x_obj_pos = _apply_offset(
                            gt_x_obj_vertices[frame_idx], offset_xyz
                        )

                    if "right" in text_entry_full.lower():
                        rr.log(
                            f"{sample_path}/{hand_path}/{idx}",
                            rr.Mesh3D(
                                vertex_positions=r_pos,
                                triangle_indices=r_hand_faces,
                                vertex_normals=r_mesh.vertex_normals,
                                vertex_colors=_vertex_colors_like(
                                    r_pos, run_color),
                            ),
                        )

                    elif "left" in text_entry_full.lower():
                        rr.log(
                            f"{sample_path}/{hand_path}/{idx}",
                            rr.Mesh3D(
                                vertex_positions=l_pos,
                                triangle_indices=l_hand_faces,
                                vertex_normals=l_mesh.vertex_normals,
                                vertex_colors=_vertex_colors_like(
                                    l_pos, run_color),
                            ),
                        )
                    else:
                        rr.log(
                            f"{sample_path}/{hand_path}_r/{idx}",
                            rr.Mesh3D(
                                vertex_positions=r_pos,
                                triangle_indices=r_hand_faces,
                                vertex_normals=r_mesh.vertex_normals,
                                vertex_colors=_vertex_colors_like(
                                    r_pos, run_color),
                            ),
                        )
                        rr.log(
                            f"{sample_path}/{hand_path}_l/{idx}",
                            rr.Mesh3D(
                                vertex_positions=l_pos,
                                triangle_indices=l_hand_faces,
                                vertex_normals=l_mesh.vertex_normals,
                                vertex_colors=_vertex_colors_like(
                                    l_pos, run_color),
                            ),
                        )

                    rr.log(
                        f"{sample_path}/object_x_obj/{idx}",
                        rr.Points3D(
                            positions=x_obj_pos,
                            radii=0.005,
                            colors=list(run_color),
                        ),
                    )
                    if gt_x_obj_pos is not None:
                        rr.log(
                            f"{gt_prompt_path}/{idx}",
                            rr.Points3D(
                                positions=gt_x_obj_pos,
                                radii=0.005,
                                colors=[255, 255, 255],
                            ),
                        )

                    # colors_cov = np.zeros_like(obj_pc[obj_name], dtype=np.uint8)
                    # colors_cov[est_cov_map[batch_idx] == 1] = [255, 255, 0]
                    # colors_cov[est_cov_map[batch_idx] == 0] = [0, 0, 255]

                    # rr.log(
                    #     f"{sample_path}/est_cov_map",
                    #     rr.Points3D(
                    #         positions=obj_vertices[frame_idx],
                    #         radii=0.005,
                    #         colors=colors_cov,
                    #     ),
                    # )
        # _log_cond_tsne(tsne_vectors, tsne_labels, tsne_colors, tsne_names)


def main():
    rr.init("Input Data", spawn=True)

    # Horizontal spacing between runs so different files are shown side-by-side.
    offset_step = 1.0
    run_colors = [
        (235, 87, 87),   # red
        (47, 128, 237),  # blue
        (39, 174, 96),   # green
    ]
    input_files = [
        "grab_exc_rot_aug.pkl",
        "grab_exc_rot_aug_gaze_emb_token_vec.pkl",
        "grab_exc_rot_aug_gaze_emb_token_vec_ro.pkl",
        # "grab_exc_rot_aug_gaze_emb_token_vec_ro_afford_mix.pkl",
    ]
    for run_idx, file_name in enumerate(input_files):
        offset_xyz = (run_idx * offset_step, 0.0, 0.0)
        run_color = run_colors[run_idx % len(run_colors)]
        _log_offset_anchor(
            file_name,
            offset_xyz,
            run_color,
            f"run_{run_idx}",
        )
        recoding_name = f"{home}/Desktop/hot3d_vis/{file_name}"
        visualize_rr(
            recoding_name,
            file_name,
            offset_xyz=offset_xyz,
            run_color=run_color,
            log_gt=True,
        )


if __name__ == "__main__":
    main()
