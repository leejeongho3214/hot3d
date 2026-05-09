#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh


DEFAULT_OBJ_PKL = Path("/Users/jeongho/Desktop/hot3d_vis/obj.pkl")
DEFAULT_OUTPUT_RRD = Path(
    "/Users/jeongho/Library/CloudStorage/SynologyDrive-home/Vscode/hot3d/"
    "hot3d/visualize-code/mesh_sampling_preview/mesh_sampling_preview.rrd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize original object meshes and 20k face-sampled meshes in Rerun."
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
        help="Object name to visualize. Can be passed multiple times. Defaults to heavy meshes only.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=20000,
        help="Target face budget for sampled meshes.",
    )
    parser.add_argument(
        "--min-vertices",
        type=int,
        default=0,
        help="Only include meshes with at least this many vertices.",
    )
    parser.add_argument(
        "--rrd-output",
        type=Path,
        default=DEFAULT_OUTPUT_RRD,
        help="Save to this .rrd file.",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Also spawn the Rerun viewer after logging.",
    )
    parser.add_argument(
        "--overview-cols",
        type=int,
        default=3,
        help="Number of object columns in the comparison layout.",
    )
    parser.add_argument(
        "--cell-gap",
        type=float,
        default=3.4,
        help="Spacing between normalized objects in the grid.",
    )
    parser.add_argument(
        "--cell-scale",
        type=float,
        default=1.9,
        help="Scale of each normalized object in the grid.",
    )
    parser.add_argument(
        "--side-gap",
        type=float,
        default=10.5,
        help="X gap between original and sampled mesh groups.",
    )
    return parser.parse_args()


def load_obj_data(obj_pkl_path: Path) -> dict:
    with obj_pkl_path.open("rb") as f:
        data = pickle.load(f)
    missing = {"obj_pcs", "obj_path"} - set(data)
    if missing:
        raise KeyError(f"{obj_pkl_path} is missing required keys: {sorted(missing)}")
    return data


def mesh_path_for_object(data: dict, mesh_root: Path, object_name: str) -> Path:
    obj_path = Path(data["obj_path"][object_name])
    return obj_path if obj_path.is_absolute() else mesh_root / obj_path


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {mesh_path}, got {type(loaded)!r}")
    return trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )


def sample_mesh_faces(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    face_count = int(mesh.faces.shape[0])
    if face_count <= max_faces:
        return mesh.copy()
    face_idx = np.linspace(0, face_count - 1, max_faces, dtype=np.int64)
    faces_sampled = np.asarray(mesh.faces[face_idx], dtype=np.int64)
    unique_vertices, inverse = np.unique(faces_sampled.reshape(-1), return_inverse=True)
    vertices_sampled = np.asarray(mesh.vertices[unique_vertices], dtype=np.float64)
    faces_remapped = inverse.reshape(-1, 3).astype(np.int64)
    return trimesh.Trimesh(
        vertices=vertices_sampled,
        faces=faces_remapped,
        process=False,
    )


def normalized_to_cell(points: np.ndarray, cell_center: np.ndarray, cell_scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = max(float(np.max(maxs - mins)), 1e-6)
    return ((points - center) / extent) * cell_scale + cell_center


def log_title(max_faces: int) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "description",
        rr.TextDocument(
            "\n".join(
                [
                    "# Mesh Sampling Preview",
                    "",
                    f"- left group: original meshes from `obj_path`",
                    f"- right group: face-sampled meshes capped at `{max_faces}` faces",
                    "- meshes are independently normalized per object for shape comparison",
                ]
            ),
            media_type="text/markdown",
        ),
        static=True,
    )


def log_mesh_entity(path: str, mesh: trimesh.Trimesh, color: np.ndarray) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    vertex_colors = np.repeat(color[None, :], len(vertices), axis=0)
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
            vertex_colors=vertex_colors,
        ),
        static=True,
    )
    rr.log(
        f"{path}/vertices",
        rr.Points3D(
            vertices,
            radii=np.full((len(vertices),), 0.0035, dtype=np.float32),
            colors=np.repeat(color[None, :3], len(vertices), axis=0),
        ),
        static=True,
    )

    unique_edges = np.unique(
        np.sort(
            np.concatenate(
                [
                    faces[:, [0, 1]],
                    faces[:, [1, 2]],
                    faces[:, [2, 0]],
                ],
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    edge_positions = vertices[unique_edges]
    rr.log(
        f"{path}/wireframe",
        rr.LineStrips3D(
            edge_positions,
            colors=np.repeat(color[None, :3], len(edge_positions), axis=0),
            radii=np.full((len(edge_positions),), 0.0008, dtype=np.float32),
        ),
        static=True,
    )


def select_object_names(
    data: dict,
    requested: list[str] | None,
    mesh_root: Path,
    max_faces: int,
    min_vertices: int,
) -> list[str]:
    available = sorted(data["obj_pcs"].keys())
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise KeyError(f"Unknown object name(s): {missing}. Available examples: {available[:8]}")
        return requested

    selected = []
    for object_name in available:
        mesh = load_mesh(mesh_path_for_object(data, mesh_root, object_name))
        vertex_count = int(mesh.vertices.shape[0])
        face_count = int(mesh.faces.shape[0])
        if vertex_count >= int(min_vertices):
            selected.append((object_name, vertex_count, face_count))
    selected.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [name for name, _, _ in selected]


def main() -> None:
    args = parse_args()
    obj_pkl = args.obj_pkl.expanduser().resolve()
    mesh_root = (
        args.mesh_root.expanduser().resolve()
        if args.mesh_root is not None
        else obj_pkl.parent
    )
    rrd_output = args.rrd_output.expanduser().resolve()
    rrd_output.parent.mkdir(parents=True, exist_ok=True)

    data = load_obj_data(obj_pkl)
    object_names = select_object_names(
        data,
        args.object_names,
        mesh_root,
        max_faces=args.max_faces,
        min_vertices=args.min_vertices,
    )

    if args.spawn:
        rr.init("mesh_sampling_preview", spawn=True)
    else:
        rr.init("mesh_sampling_preview", spawn=False)
        rr.save(str(rrd_output))

    log_title(args.max_faces)

    cols = max(int(args.overview_cols), 1)
    original_color = np.asarray([74, 131, 245, 220], dtype=np.uint8)
    sampled_color = np.asarray([242, 145, 41, 230], dtype=np.uint8)

    for object_index, object_name in enumerate(object_names):
        row = object_index // cols
        col = object_index % cols
        base_center = np.asarray(
            [col * args.cell_gap, -row * args.cell_gap, 0.0], dtype=np.float64
        )
        original_center = base_center.copy()
        sampled_center = base_center + np.asarray(
            [args.side_gap, 0.0, 0.0], dtype=np.float64
        )

        mesh_path = mesh_path_for_object(data, mesh_root, object_name)
        original_mesh = load_mesh(mesh_path)
        sampled_mesh = sample_mesh_faces(original_mesh, args.max_faces)

        original_vis = trimesh.Trimesh(
            vertices=normalized_to_cell(
                np.asarray(original_mesh.vertices, dtype=np.float64),
                original_center,
                args.cell_scale,
            ),
            faces=np.asarray(original_mesh.faces, dtype=np.int64),
            process=False,
        )
        sampled_vis = trimesh.Trimesh(
            vertices=normalized_to_cell(
                np.asarray(sampled_mesh.vertices, dtype=np.float64),
                sampled_center,
                args.cell_scale,
            ),
            faces=np.asarray(sampled_mesh.faces, dtype=np.int64),
            process=False,
        )

        base_name = f"{object_index:02d}_{object_name}"
        log_mesh_entity(f"world/original/{base_name}", original_vis, original_color)
        log_mesh_entity(f"world/sampled_20k/{base_name}", sampled_vis, sampled_color)
        rr.log(
            f"metadata/{base_name}",
            rr.TextDocument(
                "\n".join(
                    [
                        f"# {object_name}",
                        f"- original vertices: {len(original_mesh.vertices):,}",
                        f"- original faces: {len(original_mesh.faces):,}",
                        f"- sampled vertices: {len(sampled_mesh.vertices):,}",
                        f"- sampled faces: {len(sampled_mesh.faces):,}",
                        f"- mesh path: `{mesh_path}`",
                    ]
                ),
                media_type="text/markdown",
            ),
            static=True,
        )
        print(
            f"{object_name}: original F={len(original_mesh.faces):,}, "
            f"sampled F={len(sampled_mesh.faces):,}, "
            f"saved to group index {object_index}"
        )

    print(f"Saved Rerun data: {rrd_output}")


if __name__ == "__main__":
    main()
