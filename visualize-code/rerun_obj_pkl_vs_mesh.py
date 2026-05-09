#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh


DEFAULT_OBJ_PKL = Path("/Users/jeongho/Desktop/hot3d_vis/obj.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize obj.pkl 1024-point objects and their PLY meshes in Rerun."
    )
    parser.add_argument("--obj-pkl", type=Path, default=DEFAULT_OBJ_PKL, help="Path to obj.pkl")
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=None,
        help="Root directory for relative obj_path entries. Defaults to obj.pkl parent.",
    )
    parser.add_argument(
        "--object",
        dest="object_names",
        action="append",
        default=None,
        help="Object name to visualize. Can be passed multiple times. Defaults to all objects.",
    )
    parser.add_argument(
        "--rrd-output",
        type=Path,
        default=None,
        help="If set, save a .rrd file instead of spawning the Rerun viewer.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn the viewer. Useful when only testing logging.",
    )
    parser.add_argument(
        "--overview-cols",
        type=int,
        default=4,
        help="Number of object columns per side in the comparison layout.",
    )
    parser.add_argument(
        "--cell-gap",
        type=float,
        default=2.8,
        help="Spacing between normalized objects in the grid.",
    )
    parser.add_argument(
        "--cell-scale",
        type=float,
        default=1.7,
        help="Scale of each normalized object in the grid.",
    )
    parser.add_argument(
        "--side-gap",
        type=float,
        default=14.0,
        help="X-axis gap between obj.pkl point clouds and PLY meshes.",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.018,
        help="Rerun point radius for obj.pkl point clouds.",
    )
    parser.add_argument(
        "--mesh-point-radius",
        type=float,
        default=0.010,
        help="Rerun point radius for PLY mesh vertices.",
    )
    return parser.parse_args()


def load_obj_data(obj_pkl_path: Path) -> dict:
    with obj_pkl_path.open("rb") as f:
        data = pickle.load(f)
    missing = {"obj_pcs", "obj_path"} - set(data)
    if missing:
        raise KeyError(f"{obj_pkl_path} is missing required keys: {sorted(missing)}")
    return data


def selected_object_names(data: dict, requested: list[str] | None) -> list[str]:
    available = sorted(data["obj_pcs"].keys())
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise KeyError(f"Unknown object name(s): {missing}. Available examples: {available[:8]}")
    return requested


def mesh_path_for_object(data: dict, mesh_root: Path, object_name: str) -> Path:
    obj_path = Path(data["obj_path"][object_name])
    return obj_path if obj_path.is_absolute() else mesh_root / obj_path


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {mesh_path}, got {type(loaded)!r}")
    return loaded


def normalized_to_cell(points: np.ndarray, cell_center: np.ndarray, cell_scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = max(float(np.max(maxs - mins)), 1e-6)
    return ((points - center) / extent) * cell_scale + cell_center


def color_for_index(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 76, 60],
            [52, 152, 219],
            [46, 204, 113],
            [241, 196, 15],
            [155, 89, 182],
            [26, 188, 156],
            [230, 126, 34],
            [149, 165, 166],
            [231, 76, 120],
            [127, 140, 141],
        ],
        dtype=np.uint8,
    )
    return palette[index % len(palette)]


def log_title() -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "description",
        rr.TextDocument(
            "Left group: obj.pkl 1024-point samples. Right group: all vertices from PLY meshes loaded from obj_path.",
            media_type="text/markdown",
        ),
        static=True,
    )


def log_object_pair(
    object_name: str,
    object_index: int,
    cols: int,
    data: dict,
    mesh_root: Path,
    *,
    cell_gap: float,
    cell_scale: float,
    side_gap: float,
    point_radius: float,
    mesh_point_radius: float,
) -> None:
    row = object_index // cols
    col = object_index % cols
    base_center = np.asarray([col * cell_gap, -row * cell_gap, 0.0], dtype=np.float64)
    pc_center = base_center.copy()
    mesh_center = base_center + np.asarray([side_gap, 0.0, 0.0], dtype=np.float64)

    color = color_for_index(object_index)
    pc = np.asarray(data["obj_pcs"][object_name], dtype=np.float64)
    mesh_path = mesh_path_for_object(data, mesh_root, object_name)
    mesh = load_mesh(mesh_path)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)

    pc_vis = normalized_to_cell(pc, pc_center, cell_scale).astype(np.float32)
    mesh_vis = normalized_to_cell(mesh_vertices, mesh_center, cell_scale).astype(np.float32)

    rr.log(
        f"world/obj_pkl/{object_index:02d}_{object_name}/points",
        rr.Points3D(
            pc_vis,
            radii=np.full(len(pc_vis), point_radius, dtype=np.float32),
            colors=np.repeat(color[None, :], len(pc_vis), axis=0),
        ),
        static=True,
    )
    rr.log(
        f"world/ply_mesh_vertices/{object_index:02d}_{object_name}/points",
        rr.Points3D(
            mesh_vis,
            radii=np.full(len(mesh_vis), mesh_point_radius, dtype=np.float32),
            colors=np.repeat(
                np.asarray([[color[0], color[1], color[2]]], dtype=np.uint8),
                len(mesh_vis),
                axis=0,
            ),
        ),
        static=True,
    )
    rr.log(
        f"metadata/{object_index:02d}_{object_name}",
        rr.TextDocument(
            "\n".join(
                [
                    f"# {object_name}",
                    f"- obj.pkl points: {len(pc):,}",
                    f"- PLY mesh vertices shown as points: {len(mesh.vertices):,}",
                    f"- PLY mesh faces loaded but not visualized: {len(mesh.faces):,}",
                    f"- mesh path: `{mesh_path}`",
                ]
            ),
            media_type="text/markdown",
        ),
        static=True,
    )
    print(
        f"{object_name}: obj.pkl points={len(pc):,}, "
        f"mesh vertex points={len(mesh.vertices):,}, faces not visualized={len(mesh.faces):,}"
    )


def main() -> None:
    args = parse_args()
    obj_pkl = args.obj_pkl.expanduser().resolve()
    mesh_root = (
        args.mesh_root.expanduser().resolve()
        if args.mesh_root is not None
        else obj_pkl.parent
    )

    data = load_obj_data(obj_pkl)
    object_names = selected_object_names(data, args.object_names)

    spawn = not args.no_spawn and args.rrd_output is None
    rr.init("hot3d_obj_pkl_vs_ply_mesh", spawn=spawn)
    if args.rrd_output is not None:
        rrd_output = args.rrd_output.expanduser().resolve()
        rrd_output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(rrd_output)
        print(f"saving rerun recording: {rrd_output}")

    log_title()
    cols = max(1, args.overview_cols)
    for index, object_name in enumerate(object_names):
        log_object_pair(
            object_name,
            index,
            cols,
            data,
            mesh_root,
            cell_gap=args.cell_gap,
            cell_scale=args.cell_scale,
            side_gap=args.side_gap,
            point_radius=args.point_radius,
            mesh_point_radius=args.mesh_point_radius,
        )

    rows = int(math.ceil(len(object_names) / cols))
    print(f"logged {len(object_names)} objects in a {rows}x{cols} grid per side")


if __name__ == "__main__":
    main()
