#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d import proj3d


PART_NAMES = ("part_0", "part_1", "part_2")
PART_COLORS = np.asarray(
    [
        [0.25, 0.47, 0.95],
        [0.93, 0.33, 0.23],
        [0.18, 0.73, 0.38],
    ],
    dtype=np.float32,
)
SELECTED_EDGE_COLOR = np.asarray([1.0, 0.84, 0.0], dtype=np.float32)
DRAG_THRESHOLD_PX = 6.0
MODE_SELECT = "select"
MODE_ROTATE = "rotate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local point-cloud part labeler.")
    parser.add_argument(
        "--obj-pkl",
        type=Path,
        default=Path("/Users/jeongho/Desktop/hot3d_vis/obj.pkl"),
    )
    parser.add_argument(
        "--input-labels",
        type=Path,
        default=Path("/Users/jeongho/Library/CloudStorage/SynologyDrive-home/Vscode/hot3d/label_merged.json"),
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=Path("/Users/jeongho/Desktop/hot3d_vis/label_merged_3parts.json"),
    )
    parser.add_argument(
        "--pick-radius",
        type=float,
        default=18.0,
        help="Max screen-space distance in pixels for vertex picking.",
    )
    return parser.parse_args()


def load_obj_pcs(obj_pkl_path: Path) -> dict[str, np.ndarray]:
    with obj_pkl_path.open("rb") as f:
        obj_data = pickle.load(f)
    return {name: np.asarray(points) for name, points in obj_data["obj_pcs"].items()}


def load_source_labels(labels_path: Path) -> dict[str, dict[str, list[int]]]:
    with labels_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def initial_assignments(
    obj_pcs: dict[str, np.ndarray], source_labels: dict[str, dict[str, list[int]]]
) -> dict[str, np.ndarray]:
    assignments: dict[str, np.ndarray] = {}
    for object_name, points in obj_pcs.items():
        part_ids = np.zeros(len(points), dtype=np.int32)
        part_labels = source_labels.get(object_name)
        if part_labels:
            for part_id, src_name in enumerate(list(part_labels.keys())[:2]):
                indices = np.asarray(part_labels[src_name], dtype=np.int64)
                part_ids[indices] = part_id
        assignments[object_name] = part_ids
    return assignments


def shared_object_names(
    obj_pcs: dict[str, np.ndarray], source_labels: dict[str, dict[str, list[int]]]
) -> list[str]:
    shared = sorted(set(obj_pcs.keys()) & set(source_labels.keys()))
    if not shared:
        raise ValueError("No overlapping object names were found between obj.pkl and the input label JSON.")
    return shared


class LocalPartLabeler:
    def __init__(
        self,
        obj_pcs: dict[str, np.ndarray],
        source_labels: dict[str, dict[str, list[int]]],
        output_path: Path,
        pick_radius: float,
    ) -> None:
        self.obj_pcs = obj_pcs
        self.source_labels = source_labels
        self.output_path = output_path
        self.pick_radius = float(pick_radius)

        self.object_names = shared_object_names(obj_pcs, source_labels)
        self.assignments = initial_assignments(obj_pcs, source_labels)
        self.object_index = 0
        self.active_part = 0
        self.interaction_mode = MODE_SELECT
        self.last_picked_idx: int | None = None
        self.selected_indices: set[int] = set()
        self._press_xy: tuple[float, float] | None = None
        self._suppress_release_pick = False
        self._drag_additive = True

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.06, 0.14, 0.68, 0.8], projection="3d")
        self.status_ax = self.fig.add_axes([0.06, 0.04, 0.68, 0.06])
        self.status_ax.axis("off")

        self.radio_ax = self.fig.add_axes([0.79, 0.68, 0.18, 0.20])
        self.part_radio = RadioButtons(
            self.radio_ax,
            labels=list(PART_NAMES),
            active=self.active_part,
        )

        self.prev_ax = self.fig.add_axes([0.79, 0.58, 0.08, 0.06])
        self.next_ax = self.fig.add_axes([0.89, 0.58, 0.08, 0.06])
        self.reset_ax = self.fig.add_axes([0.79, 0.49, 0.18, 0.06])
        self.save_ax = self.fig.add_axes([0.79, 0.40, 0.18, 0.06])
        self.assign_ax = self.fig.add_axes([0.79, 0.31, 0.18, 0.06])
        self.clear_ax = self.fig.add_axes([0.79, 0.22, 0.18, 0.06])
        self.mode_ax = self.fig.add_axes([0.79, 0.13, 0.18, 0.06])

        self.prev_button = Button(self.prev_ax, "Prev")
        self.next_button = Button(self.next_ax, "Next")
        self.reset_button = Button(self.reset_ax, "Reset Object")
        self.save_button = Button(self.save_ax, "Save JSON")
        self.assign_button = Button(self.assign_ax, "Assign Selected")
        self.clear_button = Button(self.clear_ax, "Clear Selected")
        self.mode_button = Button(self.mode_ax, "Mode: Select")

        self.help_ax = self.fig.add_axes([0.79, 0.02, 0.18, 0.09])
        self.help_ax.axis("off")
        self.help_ax.text(
            0.0,
            1.0,
            "\n".join(
                [
                    "Controls",
                    "Mode Select: drag box-select",
                    "Mode Rotate: drag to orbit",
                    "click point: assign 1 vertex",
                    "Enter: assign selected",
                    "Esc/c: clear selection",
                    "a: toggle additive drag",
                    "u: undo last drag-add",
                    "m: toggle mode",
                    "1/2/3: select part",
                    "n/p: next or prev object",
                    "r: reset object",
                    "s: save json",
                ]
            ),
            va="top",
            fontsize=10,
            family="monospace",
        )

        self.part_radio.on_clicked(self._on_part_changed)
        self.prev_button.on_clicked(self._on_prev)
        self.next_button.on_clicked(self._on_next)
        self.reset_button.on_clicked(self._on_reset)
        self.save_button.on_clicked(self._on_save)
        self.assign_button.on_clicked(self._on_assign_selected)
        self.clear_button.on_clicked(self._on_clear_selected)
        self.mode_button.on_clicked(self._on_toggle_mode)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)
        self.selection_history: list[set[int]] = []
        self.drag_rect = Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            facecolor=(1.0, 0.84, 0.0, 0.22),
            edgecolor=(1.0, 0.6, 0.0, 0.95),
            linewidth=1.8,
            visible=False,
            transform=self.fig.transFigure,
        )
        self.fig.add_artist(self.drag_rect)

        self.scatter = None
        self._draw()

    @property
    def current_object(self) -> str:
        return self.object_names[self.object_index]

    def _point_size(self, points: np.ndarray) -> float:
        span = points.max(axis=0) - points.min(axis=0)
        diagonal = float(np.linalg.norm(span))
        return max(12.0, min(80.0, diagonal * 300.0))

    def _set_equal_axes(self, points: np.ndarray) -> None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = float(np.max(maxs - mins) / 2.0)
        radius = max(radius, 1e-6)
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)

    def _counts_text(self) -> str:
        part_ids = self.assignments[self.current_object]
        counts = [int(np.sum(part_ids == idx)) for idx in range(3)]
        picked = "None" if self.last_picked_idx is None else str(self.last_picked_idx)
        selected = len(self.selected_indices)
        return (
            f"object={self.current_object}  "
            f"part_0={counts[0]}  part_1={counts[1]}  part_2={counts[2]}  "
            f"active={PART_NAMES[self.active_part]}  mode={self.interaction_mode}  "
            f"picked={picked}  selected={selected}  additive={self._drag_additive}"
        )

    def _set_interaction_mode(self, mode: str) -> None:
        self.interaction_mode = mode
        if mode != MODE_SELECT:
            self._hide_drag_rect()
        self.mode_button.label.set_text(
            "Mode: Select" if mode == MODE_SELECT else "Mode: Rotate"
        )
        self.fig.canvas.draw_idle()

    def _hide_drag_rect(self) -> None:
        self.drag_rect.set_visible(False)
        self.fig.canvas.draw_idle()

    def _update_drag_rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        fig_w, fig_h = self.fig.bbox.width, self.fig.bbox.height
        if fig_w <= 0 or fig_h <= 0:
            return
        rx0 = min(x0, x1) / fig_w
        rx1 = max(x0, x1) / fig_w
        ry0 = min(y0, y1) / fig_h
        ry1 = max(y0, y1) / fig_h
        self.drag_rect.set_xy((rx0, ry0))
        self.drag_rect.set_width(rx1 - rx0)
        self.drag_rect.set_height(ry1 - ry0)
        self.drag_rect.set_visible(True)
        self.fig.canvas.draw_idle()

    def _draw(self) -> None:
        self.ax.cla()
        points = self.obj_pcs[self.current_object]
        part_ids = self.assignments[self.current_object]
        colors = PART_COLORS[part_ids]

        edgecolors = np.tile(colors, (1, 1))
        linewidths = np.zeros(len(points), dtype=np.float32)
        if self.selected_indices:
            selected = np.asarray(sorted(self.selected_indices), dtype=np.int64)
            edgecolors[selected] = SELECTED_EDGE_COLOR
            linewidths[selected] = 1.2
        if self.last_picked_idx is not None and 0 <= self.last_picked_idx < len(points):
            edgecolors[self.last_picked_idx] = SELECTED_EDGE_COLOR
            linewidths[self.last_picked_idx] = 1.8

        self.scatter = self.ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=self._point_size(points),
            c=colors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            depthshade=False,
            picker=False,
        )
        self._set_equal_axes(points)
        self.ax.set_title(self.current_object)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.status_ax.cla()
        self.status_ax.axis("off")
        self.status_ax.text(
            0.0,
            0.5,
            self._counts_text(),
            va="center",
            fontsize=11,
            family="monospace",
        )
        self.fig.canvas.draw_idle()

    def _project_points(self) -> tuple[np.ndarray, np.ndarray]:
        points = self.obj_pcs[self.current_object]
        xs, ys, zs = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], self.ax.get_proj())
        projected = np.column_stack([xs, ys])
        display_xy = self.ax.transData.transform(projected)
        return points, display_xy

    def _pick_vertex(self, event: MouseEvent) -> int | None:
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return None
        _, display_xy = self._project_points()
        deltas = display_xy - np.asarray([[event.x, event.y]], dtype=np.float64)
        distances = np.linalg.norm(deltas, axis=1)
        idx = int(np.argmin(distances))
        if float(distances[idx]) > self.pick_radius:
            return None
        return idx

    def _assign_vertex(self, vertex_idx: int) -> None:
        self.assignments[self.current_object][vertex_idx] = self.active_part
        self.last_picked_idx = vertex_idx
        self.selected_indices.clear()
        self._draw()

    def _on_press(self, event: MouseEvent) -> None:
        if self.interaction_mode != MODE_SELECT:
            return
        if event.inaxes != self.ax or event.button != 1 or event.x is None or event.y is None:
            return
        self._press_xy = (float(event.x), float(event.y))
        self._suppress_release_pick = False
        self._update_drag_rect(event.x, event.y, event.x, event.y)

    def _on_motion(self, event: MouseEvent) -> None:
        if self.interaction_mode != MODE_SELECT:
            return
        if self._press_xy is None or event.x is None or event.y is None:
            return
        self._update_drag_rect(self._press_xy[0], self._press_xy[1], float(event.x), float(event.y))

    def _on_release(self, event: MouseEvent) -> None:
        if self.interaction_mode != MODE_SELECT:
            return
        if event.inaxes != self.ax or event.button != 1 or event.x is None or event.y is None:
            return
        self._hide_drag_rect()
        if self._suppress_release_pick:
            self._suppress_release_pick = False
            self._press_xy = None
            return
        if self._press_xy is None:
            return

        dx = float(event.x) - self._press_xy[0]
        dy = float(event.y) - self._press_xy[1]
        if np.hypot(dx, dy) > DRAG_THRESHOLD_PX:
            self._apply_drag_select(self._press_xy[0], self._press_xy[1], float(event.x), float(event.y))
            self._press_xy = None
            return
        self._press_xy = None

        vertex_idx = self._pick_vertex(event)
        if vertex_idx is None:
            return
        self._assign_vertex(vertex_idx)

    def _apply_drag_select(self, x0: float, y0: float, x1: float, y1: float) -> None:
        _, display_xy = self._project_points()
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        mask = (
            (display_xy[:, 0] >= x0)
            & (display_xy[:, 0] <= x1)
            & (display_xy[:, 1] >= y0)
            & (display_xy[:, 1] <= y1)
        )
        new_selection = set(np.where(mask)[0].astype(int).tolist())
        if self._drag_additive:
            if new_selection:
                self.selection_history.append(set(new_selection))
            self.selected_indices |= new_selection
        else:
            self.selection_history = [set(new_selection)] if new_selection else []
            self.selected_indices = new_selection
        self.last_picked_idx = None
        self._suppress_release_pick = True
        self._draw()

    def _on_assign_selected(self, _event) -> None:
        if not self.selected_indices:
            return
        indices = np.asarray(sorted(self.selected_indices), dtype=np.int64)
        self.assignments[self.current_object][indices] = self.active_part
        self.selected_indices.clear()
        self.selection_history.clear()
        self.last_picked_idx = None
        self._draw()

    def _on_clear_selected(self, _event) -> None:
        self.selected_indices.clear()
        self.selection_history.clear()
        self.last_picked_idx = None
        self._draw()

    def _on_toggle_mode(self, _event) -> None:
        next_mode = MODE_ROTATE if self.interaction_mode == MODE_SELECT else MODE_SELECT
        self._set_interaction_mode(next_mode)

    def _on_toggle_additive(self) -> None:
        self._drag_additive = not self._drag_additive
        self._draw()

    def _on_undo_selection_add(self) -> None:
        if not self.selection_history:
            return
        self.selection_history.pop()
        rebuilt: set[int] = set()
        for chunk in self.selection_history:
            rebuilt |= chunk
        self.selected_indices = rebuilt
        self.last_picked_idx = None
        self._draw()

    def _on_part_changed(self, label: str) -> None:
        self.active_part = PART_NAMES.index(label)
        self._draw()

    def _on_prev(self, _event) -> None:
        self.object_index = (self.object_index - 1) % len(self.object_names)
        self.last_picked_idx = None
        self.selected_indices.clear()
        self.selection_history.clear()
        self._draw()

    def _on_next(self, _event) -> None:
        self.object_index = (self.object_index + 1) % len(self.object_names)
        self.last_picked_idx = None
        self.selected_indices.clear()
        self.selection_history.clear()
        self._draw()

    def _on_reset(self, _event) -> None:
        restored = initial_assignments(
            {self.current_object: self.obj_pcs[self.current_object]},
            {self.current_object: self.source_labels.get(self.current_object, {})},
        )
        self.assignments[self.current_object] = restored[self.current_object]
        self.last_picked_idx = None
        self.selected_indices.clear()
        self.selection_history.clear()
        self._draw()

    def export_labels(self) -> dict[str, dict[str, list[int]]]:
        exported: dict[str, dict[str, list[int]]] = {}
        for object_name in self.object_names:
            part_ids = self.assignments[object_name]
            exported[object_name] = {
                part_name: np.where(part_ids == part_idx)[0].astype(int).tolist()
                for part_idx, part_name in enumerate(PART_NAMES)
            }
        return exported

    def _on_save(self, _event) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(self.export_labels(), f, indent=2)
        self.status_ax.cla()
        self.status_ax.axis("off")
        self.status_ax.text(
            0.0,
            0.5,
            f"{self._counts_text()}  saved={self.output_path}",
            va="center",
            fontsize=11,
            family="monospace",
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "1":
            self.active_part = 0
            self.part_radio.set_active(0)
        elif event.key == "2":
            self.active_part = 1
            self.part_radio.set_active(1)
        elif event.key == "3":
            self.active_part = 2
            self.part_radio.set_active(2)
        elif event.key == "n":
            self._on_next(None)
        elif event.key == "p":
            self._on_prev(None)
        elif event.key == "r":
            self._on_reset(None)
        elif event.key == "s":
            self._on_save(None)
        elif event.key == "enter":
            self._on_assign_selected(None)
        elif event.key in {"escape", "c"}:
            self._on_clear_selected(None)
        elif event.key == "m":
            self._on_toggle_mode(None)
        elif event.key == "a":
            self._on_toggle_additive()
        elif event.key == "u":
            self._on_undo_selection_add()

    def show(self) -> None:
        plt.show()


def main() -> None:
    args = parse_args()
    obj_pcs = load_obj_pcs(args.obj_pkl)
    source_labels = load_source_labels(args.input_labels)
    app = LocalPartLabeler(
        obj_pcs=obj_pcs,
        source_labels=source_labels,
        output_path=args.output_labels,
        pick_radius=args.pick_radius,
    )
    app.show()


if __name__ == "__main__":
    main()
