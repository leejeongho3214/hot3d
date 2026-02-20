import argparse
import json
import os
import pickle
import uuid
from typing import Any, List

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from gradio_rerun.events import SelectionChange


def _load_obj_points(pkl_path: str, object_key: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    obj_pcs = data.get("obj_pcs", {})
    if object_key not in obj_pcs:
        raise KeyError(f"object_key not found: {object_key}")
    return np.asarray(obj_pcs[object_key], dtype=np.float32)


def _write_indices(out_path: str, object_key: str, indices: List[int]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    payload = {"object_key": object_key, "indices": indices}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _find_nearest_index(points: np.ndarray, target: np.ndarray) -> int:
    dists = np.linalg.norm(points - target[None, :], axis=1)
    return int(np.argmin(dists))


def _find_nearest_indices(points: np.ndarray, target: np.ndarray, k: int) -> List[int]:
    dists = np.linalg.norm(points - target[None, :], axis=1)
    return np.argsort(dists)[:k].tolist()


_STATE: dict[str, Any] = {
    "recording_id": "",
    "points": None,
    "out_path": "",
    "object_key": "",
    "selected": [],
    "k": 10,
    "history": [],
}


def _get_recording(recording_id: str) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="hot3d_pick_points", recording_id=recording_id)


def _stream_base_points():
    recording_id = _STATE["recording_id"]
    points = _STATE["points"]
    if recording_id == "" or points is None:
        return
    rec = _get_recording(recording_id)
    stream = rec.binary_stream()  # type: ignore
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="object"),
        collapse_panels=True,
    )
    rec.send_blueprint(blueprint)
    rec.set_time("frame", sequence=0)
    base_colors = np.tile(np.array([[160, 160, 160, 60]], dtype=np.uint8), (points.shape[0], 1))
    rec.log("object/points", rr.Points3D(points, radii=0.003, colors=base_colors))
    _STATE["selected"] = []
    _write_indices(_STATE["out_path"], _STATE["object_key"], [])
    yield stream.read()


def _log_selected_points(rec: rr.RecordingStream, points: np.ndarray, indices: List[int]) -> bytes:
    stream = rec.binary_stream()  # type: ignore
    rec.set_time("frame", sequence=0)
    if indices:
        selected = points[np.array(indices, dtype=np.int64)]
        colors = np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (selected.shape[0], 1))
        rec.log("object/selected", rr.Points3D(selected, colors=colors, radii=0.003))
    else:
        rec.log("object/selected", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
    return stream.read()


def _on_select(change: SelectionChange):
    recording_id = _STATE["recording_id"]
    points = _STATE["points"]
    out_path = _STATE["out_path"]
    object_key = _STATE["object_key"]
    selected_indices: List[int] = _STATE["selected"]
    if recording_id == "" or points is None:
        return

    evt = change.payload
    items = getattr(evt, "items", None)
    if not items or len(items) != 1:
        return
    item = items[0]
    if getattr(item, "type", None) != "entity" or getattr(item, "position", None) is None:
        return

    pos = np.array(item.position, dtype=np.float32)
    nearest = _find_nearest_indices(points, pos, int(_STATE.get("k", 10)))
    if nearest:
        existing = set(selected_indices)
        added = [idx for idx in nearest if idx not in existing]
        if added:
            selected_indices.extend(added)
            _STATE["selected"] = selected_indices
            _STATE["history"].append(added)

    _write_indices(out_path, object_key, selected_indices)

    rec = _get_recording(recording_id)
    stream_bytes = _log_selected_points(rec, points, selected_indices)
    status = f"Added {len(nearest)} nearest (total {len(selected_indices)})"
    yield stream_bytes, status


def _on_clear():
    recording_id = _STATE["recording_id"]
    points = _STATE["points"]
    out_path = _STATE["out_path"]
    object_key = _STATE["object_key"]
    if recording_id == "" or points is None:
        return
    _write_indices(out_path, object_key, [])
    _STATE["selected"] = []
    rec = _get_recording(recording_id)
    stream_bytes = _log_selected_points(rec, points, [])
    yield stream_bytes, "Cleared selections"

def _on_save():
    recording_id = _STATE["recording_id"]
    points = _STATE["points"]
    out_path = _STATE["out_path"]
    object_key = _STATE["object_key"]
    selected_indices: List[int] = _STATE["selected"]
    if recording_id == "" or points is None:
        return
    _write_indices(out_path, object_key, selected_indices)
    rec = _get_recording(recording_id)
    stream_bytes = _log_selected_points(rec, points, selected_indices)
    status = f"Saved {len(selected_indices)} indices to {out_path}"
    yield stream_bytes, status

def _on_undo():
    recording_id = _STATE["recording_id"]
    points = _STATE["points"]
    out_path = _STATE["out_path"]
    object_key = _STATE["object_key"]
    selected_indices: List[int] = _STATE["selected"]
    if recording_id == "" or points is None:
        return
    if not selected_indices:
        return
    history: List[List[int]] = _STATE.get("history", [])
    removed_chunk: List[int] = []
    if history:
        removed_chunk = history.pop()
        removed_set = set(removed_chunk)
        remaining = [idx for idx in selected_indices if idx not in removed_set]
    else:
        k = int(_STATE.get("k", 10))
        removed_chunk = selected_indices[-k:]
        remaining = selected_indices[:-k] if len(selected_indices) > k else []
    _STATE["history"] = history
    _STATE["selected"] = remaining
    _write_indices(out_path, object_key, remaining)
    rec = _get_recording(recording_id)
    stream_bytes = _log_selected_points(rec, points, remaining)
    status = f"Undo: removed {len(removed_chunk)} (total {len(remaining)})"
    yield stream_bytes, status


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick object point indices via Rerun UI.")
    parser.add_argument(
        "--pkl",
        default=os.path.join(os.path.expanduser("~"), "Desktop/hot3d_vis/obj.pkl"),
        help="path to obj.pkl",
    )
    parser.add_argument(
        "--object",
        default="mug_white",
        help="object key to use (default: mug_white)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output json path for indices",
    )
    args = parser.parse_args()

    points = _load_obj_points(args.pkl, args.object)
    out_path = args.out or os.path.join(os.path.expanduser("~"), f"{args.object}_indices.json")

    with gr.Blocks() as demo:
        gr.Markdown("## Pick points (click in the viewer)")
        with gr.Row():
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "expanded",
                },
                height=640,
            )
        with gr.Row():
            load_btn = gr.Button("Load object")
            undo_btn = gr.Button("Undo last")
            clear_btn = gr.Button("Clear selections")
            save_btn = gr.Button("Save")
        with gr.Row():
            k_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=10,
                step=1,
                label="Nearest count per click",
            )
        status = gr.Textbox(label="Status", value="")

        _STATE["recording_id"] = str(uuid.uuid4())
        _STATE["points"] = points
        _STATE["out_path"] = out_path
        _STATE["object_key"] = args.object
        _STATE["selected"] = []
        _STATE["history"] = []
        _STATE["k"] = 10

        load_btn.click(
            _stream_base_points,
            inputs=[],
            outputs=[viewer],
        )

        viewer.selection_change(
            _on_select,
            inputs=[],
            outputs=[viewer, status],
        )

        undo_btn.click(
            _on_undo,
            inputs=[],
            outputs=[viewer, status],
        )

        clear_btn.click(
            _on_clear,
            inputs=[],
            outputs=[viewer, status],
        )

        save_btn.click(
            _on_save,
            inputs=[],
            outputs=[viewer, status],
        )

        k_slider.change(
            lambda k: _STATE.update({"k": int(k)}) or f"Nearest count set to {int(k)}",
            inputs=[k_slider],
            outputs=[status],
        )

    demo.launch(ssr_mode=False)


if __name__ == "__main__":
    main()
