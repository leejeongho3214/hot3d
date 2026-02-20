import argparse
import json
import os
import pickle
import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import rerun as rr
import trimesh
from gradio_rerun import Rerun
from gradio_rerun.events import SelectionChange


def _load_obj_points(pkl_path: str, object_key: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    obj_pcs = data.get("obj_pcs", {})
    point_sets = data.get("point_sets", {})
    if object_key not in obj_pcs:
        raise KeyError(f"object_key not found: {object_key}")
    points = np.asarray(obj_pcs[object_key], dtype=np.float32)
    point_set = np.asarray(point_sets[object_key], dtype=np.int64)
    return points, point_set

67
def _load_labels(part_dir: str, object_key: str, point_set: np.ndarray) -> np.ndarray:
    csv_path = os.path.join(part_dir, object_key, "face_labeled_rgb_mapping.csv")
    mesh_path = os.path.join(part_dir, object_key, f"{object_key}.ply")
    if not os.path.exists(csv_path) or not os.path.exists(mesh_path):
        return np.zeros(point_set.shape[0], dtype=np.int32)

    df = pd.read_csv(csv_path)
    mesh = trimesh.load(mesh_path, process=False)
    num_vertices = len(mesh.vertices)
    vertex_labels = np.full(num_vertices, -1, dtype=np.int32)
    for _, row in df.iterrows():
        v1, v2, v3, label = int(row["v1"]), int(row["v2"]), int(row["v3"]), int(row["label"])
        for v in (v1, v2, v3):
            if 0 <= v < num_vertices:
                vertex_labels[v] = label
    vertex_labels[vertex_labels == -1] = 0

    labels = np.zeros(point_set.shape[0], dtype=np.int32)
    valid = (point_set >= 0) & (point_set < num_vertices)
    labels[valid] = vertex_labels[point_set[valid]]
    return labels


def _labels_to_map(labels: np.ndarray) -> Dict[str, List[int]]:
    label_map: Dict[str, List[int]] = {}
    for idx, lab in enumerate(labels.tolist()):
        key = str(int(lab))
        label_map.setdefault(key, []).append(idx)
    return label_map


def _map_to_labels(label_map: Dict[str, List[int]], count: int) -> np.ndarray:
    labels = np.zeros(count, dtype=np.int32)
    for k, indices in label_map.items():
        try:
            lab = int(k)
        except ValueError:
            continue
        for idx in indices:
            if 0 <= idx < count:
                labels[idx] = lab
    return labels


def _color_for_label(label: int) -> np.ndarray:
    fixed = {
        0: [255, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 255, 0],
        4: [255, 0, 255],
        5: [0, 255, 255],
        6: [128, 0, 0],
        7: [0, 128, 0],
        8: [0, 0, 128],
        9: [128, 128, 128],
    }
    if label in fixed:
        return np.array(fixed[label], dtype=np.uint8)
    rng = np.random.default_rng(label)
    return rng.integers(0, 256, size=3, dtype=np.uint8)


def _labels_to_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for lab in np.unique(labels):
        colors[labels == lab] = _color_for_label(int(lab))
    return colors


def _format_label_choice(label_id: str, label_names: Dict[str, str]) -> str:
    name = label_names.get(label_id, "")
    return f"{label_id}: {name}" if name else label_id


def _parse_label_choice(choice: str) -> str:
    if ":" in choice:
        return choice.split(":", 1)[0].strip()
    return choice.strip()


def _build_label_choices(labels: List[str], label_names: Dict[str, str]) -> List[str]:
    return [_format_label_choice(str(l), label_names) for l in sorted(labels, key=lambda x: int(x))]


def _legend_text(labels: List[str], label_names: Dict[str, str]) -> str:
    lines = ["Legend:"]
    for lab in sorted(labels, key=lambda x: int(x)):
        name = label_names.get(str(lab), "")
        color = _color_for_label(int(lab)).tolist()
        color_name = _color_name_for_label(int(lab), color)
        label = f"{lab} ({name})" if name else str(lab)
        lines.append(f"- {label}: {color_name}")
    return "\n".join(lines)


def _color_name_for_label(label: int, color: List[int]) -> str:
    fixed = {
        0: "빨강",
        1: "초록",
        2: "파랑",
        3: "노랑",
        4: "마젠타",
        5: "시안",
        6: "진한 빨강",
        7: "진한 초록",
        8: "진한 파랑",
        9: "회색",
    }
    if label in fixed:
        return fixed[label]
    return "임의색"


def _find_nearest_index(points: np.ndarray, target: np.ndarray) -> int:
    dists = np.linalg.norm(points - target[None, :], axis=1)
    return int(np.argmin(dists))


_STATE: dict[str, Any] = {
    "recording_id": "",
    "points": None,
    "label_map": {},
    "labels": None,
    "object_key": "",
    "out_path": "",
    "history": [],
    "current_label": "0",
    "k": 1,
    "label_names": {},
}


def _get_recording(recording_id: str) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="hot3d_label_points", recording_id=recording_id)


def _log_points() -> bytes:
    recording_id = _STATE["recording_id"]
    points = _STATE["points"]
    labels = _STATE["labels"]
    if recording_id == "" or points is None or labels is None:
        return b""
    rec = _get_recording(recording_id)
    stream = rec.binary_stream()  # type: ignore
    rec.set_time("frame", sequence=0)
    colors = _labels_to_colors(labels)
    rec.log("object/points", rr.Points3D(points, colors=colors, radii=0.003))
    return stream.read()


def _save_label_map() -> None:
    out_path = _STATE["out_path"]
    object_key = _STATE["object_key"]
    label_map = _STATE["label_map"]
    label_names = _STATE.get("label_names", {})
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    payload = {object_key: label_map, "_label_names": label_names}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _on_select(change: SelectionChange):
    points = _STATE["points"]
    labels = _STATE["labels"]
    if points is None or labels is None:
        return

    evt = change.payload
    if not evt.items or len(evt.items) != 1:
        return
    item = evt.items[0]
    if item.type != "entity" or item.position is None:
        return

    pos = np.array(item.position, dtype=np.float32)
    k = int(_STATE.get("k", 1))
    dists = np.linalg.norm(points - pos[None, :], axis=1)
    nearest = np.argsort(dists)[:k]

    label_map = _STATE["label_map"]
    changed: List[Tuple[int, int, int]] = []
    new_label = int(_STATE["current_label"])
    for idx in nearest.tolist():
        prev_label = int(labels[idx])
        if prev_label == new_label:
            continue
        labels[idx] = new_label
        label_map.setdefault(str(prev_label), [])
        label_map.setdefault(str(new_label), [])
        if idx in label_map[str(prev_label)]:
            label_map[str(prev_label)].remove(idx)
        if idx not in label_map[str(new_label)]:
            label_map[str(new_label)].append(idx)
        changed.append((idx, prev_label, new_label))

    if not changed:
        return

    _STATE["labels"] = labels
    _STATE["label_map"] = label_map
    _STATE["history"].append(changed)

    _save_label_map()
    stream_bytes = _log_points()
    status = f"Set {len(changed)} points to label {new_label}"
    yield stream_bytes, status


def _on_undo():
    history = _STATE["history"]
    labels = _STATE["labels"]
    if not history or labels is None:
        return
    changed = history.pop()
    label_map = _STATE["label_map"]
    for idx, prev_label, new_label in changed:
        labels[idx] = prev_label
        label_map.setdefault(str(prev_label), [])
        label_map.setdefault(str(new_label), [])
        if idx in label_map[str(new_label)]:
            label_map[str(new_label)].remove(idx)
        if idx not in label_map[str(prev_label)]:
            label_map[str(prev_label)].append(idx)
    _STATE["labels"] = labels
    _STATE["label_map"] = label_map

    _save_label_map()
    stream_bytes = _log_points()
    status = f"Undo: restored {len(changed)} points"
    yield stream_bytes, status


def _on_save():
    _save_label_map()
    stream_bytes = _log_points()
    status = f"Saved to {_STATE['out_path']}"
    yield stream_bytes, status


def _list_objects(part_dir: str) -> List[str]:
    if not os.path.isdir(part_dir):
        return []
    objs = []
    for name in os.listdir(part_dir):
        if name.startswith("."):
            continue
        csv_path = os.path.join(part_dir, name, "face_labeled_rgb_mapping.csv")
        ply_path = os.path.join(part_dir, name, f"{name}.ply")
        if os.path.exists(csv_path) and os.path.exists(ply_path):
            objs.append(name)
    return sorted(objs)


def _load_object(object_key: str, obj_pkl: str, part_dir: str, out_path: str, load_existing: bool):
    points, point_set = _load_obj_points(obj_pkl, object_key)
    labels = _load_labels(part_dir, object_key, point_set)
    label_map = _labels_to_map(labels)

    label_names: Dict[str, str] = {}
    if load_existing and os.path.exists(out_path):
        with open(out_path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "_label_names" in payload and isinstance(payload["_label_names"], dict):
                label_names = payload["_label_names"]
            if object_key in payload:
                label_map = payload[object_key]
                labels = _map_to_labels(label_map, points.shape[0])

    _STATE["recording_id"] = str(uuid.uuid4())
    _STATE["points"] = points
    _STATE["labels"] = labels
    _STATE["label_map"] = label_map
    _STATE["object_key"] = object_key
    _STATE["out_path"] = out_path
    _STATE["history"] = []
    _STATE["current_label"] = sorted(label_map.keys(), key=lambda x: int(x))[0] if label_map else "0"
    _STATE["label_names"] = label_names

    stream_bytes = _log_points()
    choices = _build_label_choices(list(label_map.keys()), label_names) if label_map else ["0"]
    legend = _legend_text(list(label_map.keys()), label_names)
    count = int(points.shape[0])
    if count != 1024:
        status = f"[WARN] {object_key}: {count} pts (expected 1024)"
    else:
        status = f"Loaded {object_key} ({count} pts)"
    return stream_bytes, gr.Dropdown(choices=choices, value=_format_label_choice(_STATE["current_label"], label_names)), status, legend


def _on_add_label(new_label: str):
    new_label = new_label.strip()
    if new_label == "":
        return gr.Dropdown(), "Label empty"
    try:
        int(new_label)
    except ValueError:
        return gr.Dropdown(), "Label must be numeric"

    label_map = _STATE["label_map"]
    if new_label not in label_map:
        label_map[new_label] = []
        _STATE["label_map"] = label_map
    label_names = _STATE.get("label_names", {})
    choices = _build_label_choices(list(label_map.keys()), label_names)
    return gr.Dropdown(choices=choices, value=_format_label_choice(new_label, label_names)), f"Added label {new_label}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label object points with UI.")
    parser.add_argument("--object", default="mug_white", help="object key")
    parser.add_argument(
        "--obj-pkl",
        default=os.path.join(os.path.expanduser("~"), "Desktop/hot3d_vis/obj.pkl"),
        help="path to obj.pkl",
    )
    parser.add_argument(
        "--part-dir",
        default=os.path.join(os.path.expanduser("~"), "Desktop/hot3d_vis/part"),
        help="part directory",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output json path",
    )
    parser.add_argument("--load-existing", action="store_true", help="load existing json if present")
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.expanduser("~"), f"{args.object}_labels.json")
    object_list = _list_objects(args.part_dir)

    with gr.Blocks() as demo:
        gr.Markdown("## Point Labeling (click points to assign label)")
        with gr.Row():
            viewer = Rerun(
                streaming=True,
                panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "expanded"},
                height=640,
            )
        with gr.Row():
            object_dropdown = gr.Dropdown(
                choices=object_list if object_list else [args.object],
                value=args.object if args.object in object_list else (object_list[0] if object_list else args.object),
                label="Object",
            )
            load_btn = gr.Button("Load object")
        with gr.Row():
            label_dropdown = gr.Dropdown(
                choices=["0"],
                value="0",
                label="Current label",
            )
            new_label = gr.Textbox(label="Add label", placeholder="e.g. 5")
            add_label_btn = gr.Button("Add label")
            k_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=1,
                step=1,
                label="Nearest count per click",
            )
        with gr.Row():
            label_name = gr.Textbox(label="Label name", placeholder="e.g. handle")
            set_label_btn = gr.Button("Set label name")
        with gr.Row():
            undo_btn = gr.Button("Undo")
            save_btn = gr.Button("Save")
        status = gr.Textbox(label="Status", value="")
        legend_box = gr.Textbox(label="Legend", value="", lines=8)

        def _on_load(obj_key: str):
            out = out_path if args.out else os.path.join(os.path.expanduser("~"), f"{obj_key}_labels.json")
            load_existing = args.load_existing or os.path.exists(out)
            result = _load_object(obj_key, args.obj_pkl, args.part_dir, out, load_existing)
            yield result

        def _on_label_change(val: str):
            label_id = _parse_label_choice(val)
            _STATE["current_label"] = label_id
            return f"Current label: {label_id}"

        def _on_set_label_name(label_choice: str, name: str):
            label_id = _parse_label_choice(label_choice)
            name = name.strip()
            label_names = _STATE.get("label_names", {})
            if name == "":
                label_names.pop(label_id, None)
            else:
                label_names[label_id] = name
            _STATE["label_names"] = label_names
            choices = _build_label_choices(list(_STATE["label_map"].keys()), label_names)
            legend = _legend_text(list(_STATE["label_map"].keys()), label_names)
            _save_label_map()
            return gr.Dropdown(choices=choices, value=_format_label_choice(label_id, label_names)), legend, "Updated label name"

        load_btn.click(_on_load, inputs=[object_dropdown], outputs=[viewer, label_dropdown, status, legend_box])
        label_dropdown.change(_on_label_change, inputs=[label_dropdown], outputs=[status])
        k_slider.change(lambda k: _STATE.update({"k": int(k)}) or f"Nearest count set to {int(k)}",
                        inputs=[k_slider], outputs=[status])
        add_label_btn.click(_on_add_label, inputs=[new_label], outputs=[label_dropdown, status])
        viewer.selection_change(_on_select, inputs=[], outputs=[viewer, status])
        undo_btn.click(_on_undo, inputs=[], outputs=[viewer, status])
        save_btn.click(_on_save, inputs=[], outputs=[viewer, status])
        set_label_btn.click(_on_set_label_name, inputs=[label_dropdown, label_name], outputs=[label_dropdown, legend_box, status])

    demo.launch(ssr_mode=False)


if __name__ == "__main__":
    main()
