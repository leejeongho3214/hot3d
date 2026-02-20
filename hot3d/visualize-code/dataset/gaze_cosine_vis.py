import argparse
import os
import pickle
import re
import sys
from typing import Optional

import numpy as np
import rerun as rr
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if VIS_ROOT not in sys.path:
    sys.path.insert(0, VIS_ROOT)

from rot import rot6d_to_rotmat  # noqa: E402


class ObjectModel:
    def __init__(self, pkl_file: str):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        self.object_name = data["object_name"]
        self.obj_pcs = data["obj_pcs"]
        self.obj_pc_normals = data["obj_pc_normals"]
        self.point_sets = data["point_sets"]
        self.obj_path = data["obj_path"]

    def __call__(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_name[object_name]
        point_set = self.point_sets[object_name].copy()
        obj_pc = self.obj_pcs[object_name].copy()
        obj_pc_normal = self.obj_pc_normals[object_name].copy()
        obj_path = self.obj_path[object_name]
        return point_set, obj_pc, obj_pc_normal, obj_path


def _set_frame_time(frame_idx: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_idx)
    else:
        rr.set_time("frame", sequence=frame_idx)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_payload(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # PyTorch 2.6+ defaults weights_only=True. Retry with False for trusted local files.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch versions without weights_only arg.
            return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _infer_sample_count(payload) -> int:
    if isinstance(payload, dict):
        for v in payload.values():
            arr = _to_numpy(v)
            if arr is not None and arr.ndim > 0:
                return int(arr.shape[0])
            if isinstance(v, (list, tuple)):
                return len(v)
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return 1


def _select_sample(value, sample_idx: int, sample_count: int):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == sample_count:
            return value[sample_idx]
        return value
    arr = _to_numpy(value)
    if arr is not None and arr.ndim > 0 and arr.shape[0] == sample_count:
        return arr[sample_idx]
    return value


def _extract_object_name(text_entry: Optional[str], obj_name_entry) -> Optional[str]:
    if obj_name_entry is not None:
        s = str(obj_name_entry).strip().lower()
        if s:
            return s
    if text_entry is None:
        return None
    text = str(text_entry)
    match = re.search(r"\bof\s+(.+?)\s+with\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return None


def _parse_gaze_frame(gaze_entry, frame_idx: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    arr = _to_numpy(gaze_entry)
    if arr is None:
        return None
    if arr.ndim >= 4 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim == 3 and arr.shape[1:] == (2, 3):
        if frame_idx >= arr.shape[0]:
            return None
        return arr[frame_idx, 0], arr[frame_idx, 1]
    if arr.ndim == 2 and arr.shape == (2, 3):
        return arr[0], arr[1]
    return None


def _parse_phase_boundary_idx(boundary_entry, num_frames: int) -> Optional[int]:
    arr = _to_numpy(boundary_entry)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        idx = int(arr.item())
        return max(0, min(idx, max(0, num_frames - 1)))

    arr = arr.reshape(-1)
    if arr.size == 0:
        return None

    # Case 1) explicit index array-like: [idx] / [idx, ...]
    if arr.size <= 4 and np.issubdtype(arr.dtype, np.number):
        idx = int(arr[0])
        # Support normalized boundary in [0, 1].
        if isinstance(arr[0].item(), float) and 0.0 <= float(arr[0]) <= 1.0:
            idx = int(round(float(arr[0]) * max(0, num_frames - 1)))
        return max(0, min(idx, max(0, num_frames - 1)))

    # Case 2) per-frame mask/indicator, pick first non-zero position.
    nz = np.flatnonzero(arr.astype(np.float32))
    if nz.size > 0:
        return int(nz[0])

    return None


def process_obj_result(obj_verts: torch.Tensor, obj_params: torch.Tensor):
    if obj_params.dim() == 2:
        # obj_params: (T, 9)
        obj_trans = obj_params[:, :3]
        obj_rot6d = obj_params[:, 3:9]
        obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
        if obj_verts.dim() == 2:
            # obj_verts: (K, 3)
            obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(1)
            return obj_verts_transformed, obj_pc_rotated
        if obj_verts.dim() == 3:
            # obj_verts: (B, K, 3), broadcast over batch
            obj_pc_rotated = torch.einsum("tij,bkj->btki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(0).unsqueeze(2)
            return obj_verts_transformed, obj_pc_rotated
    elif obj_params.dim() == 3:
        # obj_params: (B, T, 9)
        obj_trans = obj_params[..., :3]
        obj_rot6d = obj_params[..., 3:9]
        bsz, nframes = obj_params.shape[:2]
        obj_rotmat = rot6d_to_rotmat(obj_rot6d.reshape(-1, 6)).reshape(
            bsz, nframes, 3, 3
        )
        if obj_verts.dim() == 2:
            # obj_verts: (K, 3), broadcast over batch
            obj_pc_rotated = torch.einsum("btij,kj->btki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
            return obj_verts_transformed, obj_pc_rotated
        if obj_verts.dim() == 3:
            # obj_verts: (B, K, 3)
            obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
            return obj_verts_transformed, obj_pc_rotated
    raise ValueError(
        f"Unsupported shapes: obj_params {tuple(obj_params.shape)}, obj_verts {tuple(obj_verts.shape)}"
    )


def visualize(data_path: str, sample_indices: Optional[list[int]], spawn: bool = True) -> None:
    payload = _load_payload(data_path)
    if not isinstance(payload, dict):
        raise TypeError("Expected dict payload.")

    home = os.path.expanduser("~")
    object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
    obj_pc = {}
    for obj_name in object_model.obj_pcs.keys():
        _, pc, _, _ = object_model(obj_name)
        obj_pc[obj_name] = torch.tensor(pc, dtype=torch.float32)

    rr.init("Gaze Raw + Object x_obj", spawn=spawn)

    sample_count = _infer_sample_count(payload)
    if sample_indices is None:
        sample_indices = list(range(sample_count))

    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= sample_count:
            continue

        text_entry = _select_sample(payload.get("text"), sample_idx, sample_count)
        obj_name_entry = _select_sample(payload.get("obj_name"), sample_idx, sample_count)
        object_name = _extract_object_name(text_entry, obj_name_entry)
        if object_name is None or object_name not in obj_pc:
            print(f"[WARN] skip sample {sample_idx:03d}: unresolved object '{object_name}'")
            continue

        x_obj_entry = _select_sample(payload.get("x_obj"), sample_idx, sample_count)
        gaze_entry = _select_sample(payload.get("gaze_vec"), sample_idx, sample_count)
        phase_boundary_source = payload.get("phase_boundary_idx")
        if phase_boundary_source is None:
            phase_boundary_source = payload.get("phase_boundary_index")
        if phase_boundary_source is None:
            phase_boundary_source = payload.get("phase_boundary")
        if phase_boundary_source is None:
            phase_boundary_source = payload.get("phase_idx")
        phase_boundary_entry = _select_sample(
            phase_boundary_source, sample_idx, sample_count
        )
        if x_obj_entry is None or gaze_entry is None:
            print(f"[WARN] skip sample {sample_idx:03d}: missing x_obj or gaze_vec")
            continue

        x_obj_arr = _to_numpy(x_obj_entry)
        if x_obj_arr is None:
            continue
        if x_obj_arr.ndim == 1:
            x_obj_arr = x_obj_arr[None, :]
        if x_obj_arr.ndim != 2 or x_obj_arr.shape[1] < 9:
            print(f"[WARN] skip sample {sample_idx:03d}: bad x_obj shape {x_obj_arr.shape}")
            continue

        obj_seq, _ = process_obj_result(
            obj_pc[object_name], torch.as_tensor(x_obj_arr[:, :9], dtype=torch.float32)
        )
        obj_seq = obj_seq.detach().cpu().numpy()

        nframes_entry = _select_sample(payload.get("nframes"), sample_idx, sample_count)
        num_frames = obj_seq.shape[0]
        if nframes_entry is not None:
            nframes_arr = _to_numpy(nframes_entry).reshape(-1)
            if nframes_arr.size > 0:
                num_frames = min(num_frames, int(max(1, nframes_arr[0])))

        phase_boundary_idx = _parse_phase_boundary_idx(phase_boundary_entry, num_frames)
        pre_phase_color = [255, 255, 0]
        post_phase_color = [255, 80, 80]

        base = f"sample_{sample_idx:03d}"
        if text_entry is not None:
            rr.log(
                f"{base}/text",
                rr.TextLog(str(text_entry), level=rr.TextLogLevel.INFO),
                static=True,
            )

        for frame_idx in range(num_frames):
            _set_frame_time(frame_idx)
            points = obj_seq[frame_idx]
            gaze_frame = _parse_gaze_frame(gaze_entry, frame_idx)
            point_color = (
                pre_phase_color
                if phase_boundary_idx is None or frame_idx < phase_boundary_idx
                else post_phase_color
            )
            point_colors = np.tile(
                np.asarray(point_color, dtype=np.uint8), (points.shape[0], 1)
            )

            rr.log(
                f"{base}/object_pc",
                rr.Points3D(positions=points, radii=0.005, colors=point_colors),
            )

            if gaze_frame is None:
                continue
            origin, gaze_vec = gaze_frame
            rr.log(
                f"{base}/gaze_origin",
                rr.Points3D(positions=[origin], radii=0.01, colors=[255, 255, 255]),
            )
            rr.log(
                f"{base}/gaze_vector",
                rr.Arrows3D(
                    origins=[origin],
                    vectors=[gaze_vec],
                    colors=[[255, 255, 0]],
                    labels=["gaze_raw"],
                ),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.expanduser("~"), "Desktop/hot3d_vis/dataset_debug.pt"),
    )
    parser.add_argument("--sample-idx", type=int, action="append", default=None)
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--no-spawn", dest="spawn", action="store_false")
    parser.set_defaults(spawn=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize(args.data, args.sample_idx, spawn=args.spawn)


if __name__ == "__main__":
    main()
