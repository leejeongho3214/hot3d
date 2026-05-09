from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import rerun as rr
from IPython.display import display
from rerun.event import EntitySelectionItem, SelectionChangeEvent
from rerun.notebook import Viewer


DEFAULT_OBJ_PKL = Path("/Users/jeongho/Desktop/hot3d_vis/obj.pkl")
DEFAULT_INPUT_LABELS = Path(
    "/Users/jeongho/Library/CloudStorage/SynologyDrive-home/Vscode/hot3d/label_merged.json"
)
DEFAULT_OUTPUT_LABELS = Path("/Users/jeongho/Desktop/hot3d_vis/label_merged_3parts.json")
PART_NAMES = ("part_0", "part_1", "part_2")
PART_COLORS = np.asarray(
    [
        [231, 76, 60],
        [46, 204, 113],
        [52, 152, 219],
    ],
    dtype=np.uint8,
)
SELECTED_COLOR = np.asarray([[255, 215, 0]], dtype=np.uint8)


def _load_obj_pcs(obj_pkl_path: Path) -> dict[str, np.ndarray]:
    with obj_pkl_path.open("rb") as f:
        obj_data = pickle.load(f)
    return {name: np.asarray(points) for name, points in obj_data["obj_pcs"].items()}


def _load_source_labels(labels_path: Path) -> dict[str, dict[str, list[int]]]:
    with labels_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _initial_assignments(
    obj_pcs: dict[str, np.ndarray], source_labels: dict[str, dict[str, list[int]]]
) -> dict[str, np.ndarray]:
    assignments: dict[str, np.ndarray] = {}

    for object_name, points in obj_pcs.items():
        point_count = len(points)
        part_ids = np.zeros(point_count, dtype=np.int32)
        labels_for_object = source_labels.get(object_name)

        if labels_for_object:
            for new_part_id, old_part_name in enumerate(list(labels_for_object.keys())[:2]):
                indices = np.asarray(labels_for_object[old_part_name], dtype=np.int64)
                part_ids[indices] = new_part_id

        assignments[object_name] = part_ids

    return assignments


def _shared_object_names(
    obj_pcs: dict[str, np.ndarray], source_labels: dict[str, dict[str, list[int]]]
) -> list[str]:
    shared = sorted(set(obj_pcs.keys()) & set(source_labels.keys()))
    if not shared:
        raise ValueError("No overlapping object names were found between obj.pkl and the input label JSON.")
    return shared


class RerunPartLabeler:
    def __init__(
        self,
        obj_pkl_path: Path = DEFAULT_OBJ_PKL,
        input_labels_path: Path = DEFAULT_INPUT_LABELS,
        output_labels_path: Path = DEFAULT_OUTPUT_LABELS,
    ) -> None:
        self.obj_pkl_path = Path(obj_pkl_path)
        self.input_labels_path = Path(input_labels_path)
        self.output_labels_path = Path(output_labels_path)

        self.obj_pcs = _load_obj_pcs(self.obj_pkl_path)
        self.source_labels = _load_source_labels(self.input_labels_path)
        self.object_names = _shared_object_names(self.obj_pcs, self.source_labels)
        self.assignments = _initial_assignments(self.obj_pcs, self.source_labels)
        self.selected_indices: set[int] = set()
        self.current_object = self.object_names[0]

        rr.init("hot3d_part_labeler", default_enabled=True)
        self.viewer = Viewer(width=1000, height=720)
        self.viewer.on_event(self._on_viewer_event)

        self.object_dropdown = widgets.Dropdown(
            options=self.object_names,
            value=self.current_object,
            description="Object",
            layout=widgets.Layout(width="360px"),
        )
        self.active_part = widgets.ToggleButtons(
            options=[(name, idx) for idx, name in enumerate(PART_NAMES)],
            value=0,
            description="Target",
            tooltips=["Assign selection to part_0", "Assign selection to part_1", "Assign selection to part_2"],
        )
        self.prev_button = widgets.Button(description="Prev", button_style="")
        self.next_button = widgets.Button(description="Next", button_style="")
        self.assign_button = widgets.Button(description="Assign Selected", button_style="primary")
        self.assign_on_click = widgets.Checkbox(
            value=False,
            description="Assign on click",
            indent=False,
        )
        self.clear_selection_button = widgets.Button(description="Clear Selected")
        self.reset_object_button = widgets.Button(description="Reset Object")
        self.save_button = widgets.Button(description="Save JSON", button_style="success")
        self.status_html = widgets.HTML()
        self.help_html = widgets.HTML(
            value=(
                "<b>Usage</b>: drag-select or click points in the Rerun viewer, "
                "choose <code>part_0/1/2</code>, then press <code>Assign Selected</code>. "
                "If you want one-by-one editing, enable <code>Assign on click</code>. "
                "Use <code>Save JSON</code> to write the current labels."
            )
        )

        self.object_dropdown.observe(self._on_object_changed, names="value")
        self.prev_button.on_click(self._on_prev_clicked)
        self.next_button.on_click(self._on_next_clicked)
        self.assign_button.on_click(self._on_assign_clicked)
        self.clear_selection_button.on_click(self._on_clear_selection_clicked)
        self.reset_object_button.on_click(self._on_reset_object_clicked)
        self.save_button.on_click(self._on_save_clicked)

        self._render_current_object()

    def _point_radius(self, points: np.ndarray) -> float:
        span = points.max(axis=0) - points.min(axis=0)
        diagonal = float(np.linalg.norm(span))
        return max(diagonal / 220.0, 1e-4)

    def _entity_path(self, object_name: str) -> str:
        return f"world/{object_name}"

    def _selected_entity_path(self, object_name: str) -> str:
        return f"world/{object_name}/selected"

    def _part_counts(self, object_name: str) -> list[int]:
        part_ids = self.assignments[object_name]
        return [int(np.sum(part_ids == idx)) for idx in range(len(PART_NAMES))]

    def _nearest_index(self, object_name: str, position: Iterable[float]) -> int:
        point = np.asarray(list(position), dtype=np.float64)
        points = self.obj_pcs[object_name]
        distances = np.sum((points - point[None, :]) ** 2, axis=1)
        return int(np.argmin(distances))

    def _selected_indices_from_event(self, event: SelectionChangeEvent) -> set[int]:
        expected_path = self._entity_path(self.current_object)
        indices: set[int] = set()

        for item in event.items:
            if not isinstance(item, EntitySelectionItem):
                continue
            if item.entity_path != expected_path:
                continue

            if item.instance_id is not None:
                indices.add(int(item.instance_id))
            elif item.position is not None:
                indices.add(self._nearest_index(self.current_object, item.position))

        return indices

    def _on_viewer_event(self, event: object) -> None:
        if not isinstance(event, SelectionChangeEvent):
            return

        self.selected_indices = self._selected_indices_from_event(event)
        if self.assign_on_click.value and self.selected_indices:
            self._assign_selected_to_active_part()
            return
        self._render_selected_overlay()
        self._update_status(
            f"Selected {len(self.selected_indices)} points on <code>{self.current_object}</code>."
        )

    def _render_current_object(self) -> None:
        object_name = self.current_object
        points = self.obj_pcs[object_name]
        part_ids = self.assignments[object_name]
        colors = PART_COLORS[part_ids]

        rr.log(self._entity_path(object_name), rr.Clear.recursive())
        rr.log(
            self._entity_path(object_name),
            rr.Points3D(
                points,
                colors=colors,
                radii=[self._point_radius(points)] * len(points),
            ),
        )
        self._render_selected_overlay()

        counts = self._part_counts(object_name)
        self._update_status(
            "Loaded "
            f"<code>{object_name}</code>. "
            f"Counts: part_0={counts[0]}, part_1={counts[1]}, part_2={counts[2]}. "
            "Select points in the viewer to edit."
        )

    def _render_selected_overlay(self) -> None:
        object_name = self.current_object
        rr.log(self._selected_entity_path(object_name), rr.Clear.recursive())

        if not self.selected_indices:
            return

        points = self.obj_pcs[object_name]
        selected_idx = np.asarray(sorted(self.selected_indices), dtype=np.int64)
        selected_points = points[selected_idx]
        rr.log(
            self._selected_entity_path(object_name),
            rr.Points3D(
                selected_points,
                colors=np.repeat(SELECTED_COLOR, len(selected_points), axis=0),
                radii=[self._point_radius(points) * 2.2] * len(selected_points),
            ),
        )

    def _update_status(self, message: str) -> None:
        selected = len(self.selected_indices)
        self.status_html.value = (
            f"<div style='font-family: monospace; white-space: normal;'>{message}<br>"
            f"Current target: <b>{PART_NAMES[self.active_part.value]}</b> | "
            f"Selected: <b>{selected}</b> | "
            f"Output: <code>{self.output_labels_path}</code></div>"
        )

    def _on_object_changed(self, change: dict) -> None:
        self.current_object = str(change["new"])
        self.selected_indices.clear()
        self._render_current_object()

    def _on_prev_clicked(self, _: widgets.Button) -> None:
        current_index = self.object_names.index(self.current_object)
        self.object_dropdown.value = self.object_names[(current_index - 1) % len(self.object_names)]

    def _on_next_clicked(self, _: widgets.Button) -> None:
        current_index = self.object_names.index(self.current_object)
        self.object_dropdown.value = self.object_names[(current_index + 1) % len(self.object_names)]

    def _on_assign_clicked(self, _: widgets.Button) -> None:
        if not self.selected_indices:
            self._update_status("No selected points to assign.")
            return

        self._assign_selected_to_active_part()

    def _assign_selected_to_active_part(self) -> None:
        target_part = int(self.active_part.value)
        indices = np.asarray(sorted(self.selected_indices), dtype=np.int64)
        part_ids = self.assignments[self.current_object].copy()
        part_ids[indices] = target_part
        self.assignments[self.current_object] = part_ids
        assigned_count = len(indices)
        self.selected_indices.clear()
        self._render_current_object()
        self._update_status(
            f"Assigned {assigned_count} points on <code>{self.current_object}</code> "
            f"to <b>{PART_NAMES[target_part]}</b>."
        )

    def _on_clear_selection_clicked(self, _: widgets.Button) -> None:
        self.selected_indices.clear()
        self._render_selected_overlay()
        self._update_status(f"Cleared pending selection for <code>{self.current_object}</code>.")

    def _on_reset_object_clicked(self, _: widgets.Button) -> None:
        restored = _initial_assignments(
            {self.current_object: self.obj_pcs[self.current_object]},
            {self.current_object: self.source_labels.get(self.current_object, {})},
        )
        self.assignments[self.current_object] = restored[self.current_object]
        self.selected_indices.clear()
        self._render_current_object()

    def export_labels(self) -> dict[str, dict[str, list[int]]]:
        exported: dict[str, dict[str, list[int]]] = {}
        for object_name in self.object_names:
            part_ids = self.assignments[object_name]
            exported[object_name] = {
                part_name: np.where(part_ids == part_index)[0].astype(int).tolist()
                for part_index, part_name in enumerate(PART_NAMES)
            }
        return exported

    def _on_save_clicked(self, _: widgets.Button) -> None:
        self.output_labels_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_labels_path.open("w", encoding="utf-8") as f:
            json.dump(self.export_labels(), f, indent=2)
        self._update_status(f"Saved labels to <code>{self.output_labels_path}</code>.")

    def display(self) -> None:
        controls = widgets.VBox(
            [
                self.help_html,
                widgets.HBox([self.object_dropdown, self.prev_button, self.next_button]),
                widgets.HBox([self.active_part, self.assign_on_click]),
                widgets.HBox([self.assign_button, self.clear_selection_button]),
                widgets.HBox([self.reset_object_button, self.save_button]),
                self.status_html,
            ]
        )
        display(controls)
        self.viewer.display(block_until_ready=True)
        self._render_current_object()


def launch_labeler(
    obj_pkl_path: Path = DEFAULT_OBJ_PKL,
    input_labels_path: Path = DEFAULT_INPUT_LABELS,
    output_labels_path: Path = DEFAULT_OUTPUT_LABELS,
) -> RerunPartLabeler:
    labeler = RerunPartLabeler(
        obj_pkl_path=obj_pkl_path,
        input_labels_path=input_labels_path,
        output_labels_path=output_labels_path,
    )
    labeler.display()
    return labeler
