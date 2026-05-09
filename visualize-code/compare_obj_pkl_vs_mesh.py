#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import pickle
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_OBJ_PKL = Path("/Users/jeongho/Desktop/hot3d_vis/obj.pkl")
DEFAULT_OUTPUT_DIR = Path("/Users/jeongho/Desktop/hot3d_vis/obj_vs_mesh_vis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare obj.pkl 1024-point object samples against the original PLY meshes."
    )
    parser.add_argument("--obj-pkl", type=Path, default=DEFAULT_OBJ_PKL, help="Path to obj.pkl")
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=None,
        help="Root directory for relative obj_path entries. Defaults to obj.pkl parent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where visualization images are written.",
    )
    parser.add_argument(
        "--object",
        dest="object_names",
        action="append",
        default=None,
        help="Object name to visualize. Can be passed multiple times. Defaults to all objects.",
    )
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter size for obj.pkl points.")
    parser.add_argument(
        "--max-faces",
        type=int,
        default=6000,
        help="Maximum mesh faces drawn per object for speed/readability.",
    )
    parser.add_argument(
        "--overview-cols",
        type=int,
        default=4,
        help="Number of object columns in the overview layout.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    return parser.parse_args()


def load_obj_data(obj_pkl_path: Path) -> dict:
    with obj_pkl_path.open("rb") as f:
        data = pickle.load(f)
    required = {"obj_pcs", "obj_path"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"{obj_pkl_path} is missing required keys: {sorted(missing)}")
    return data


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {mesh_path}, got {type(loaded)!r}")
    return loaded


def object_names_from_data(data: dict, requested: list[str] | None) -> list[str]:
    available = sorted(data["obj_pcs"].keys())
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise KeyError(f"Unknown object name(s): {missing}. Available examples: {available[:8]}")
    return requested


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def format_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=42)
    ax.grid(False)


def draw_point_cloud(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    color: tuple[float, float, float],
    point_size: float,
    label: str | None = None,
) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=point_size,
        c=[color],
        alpha=0.95,
        linewidths=0,
        label=label,
    )


def sampled_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces
    idx = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
    return faces[idx]


def draw_mesh(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_faces: int,
    face_color: tuple[float, float, float, float] = (0.35, 0.62, 0.95, 0.34),
    edge_color: tuple[float, float, float, float] = (0.12, 0.18, 0.24, 0.16),
) -> None:
    faces_to_draw = sampled_faces(faces, max_faces)
    triangles = vertices[faces_to_draw]
    collection = Poly3DCollection(
        triangles,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=0.12,
    )
    ax.add_collection3d(collection)


def convex_hull_reconstruction(points: np.ndarray) -> trimesh.Trimesh:
    cloud = trimesh.points.PointCloud(np.asarray(points, dtype=np.float64))
    hull = cloud.convex_hull
    return trimesh.Trimesh(
        vertices=np.asarray(hull.vertices, dtype=np.float64),
        faces=np.asarray(hull.faces, dtype=np.int64),
        process=False,
    )


def _nearest_point_distances(query: np.ndarray, points: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    out = np.full((query.shape[0],), np.inf, dtype=np.float64)
    if query.shape[0] == 0 or points.shape[0] == 0:
        return out
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        diff = query[start:stop, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        out[start:stop] = np.min(dist, axis=1)
    return out


def original_guided_reconstruction(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    neighbor_rings: int = 1,
) -> trimesh.Trimesh:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"expected points with shape (N, 3), got {points.shape}")

    _closest, _dist, tri_ids = trimesh.proximity.closest_point(mesh, points)
    tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)
    tri_ids = tri_ids[(tri_ids >= 0) & (tri_ids < len(mesh.faces))]
    if tri_ids.size == 0:
        raise ValueError("closest-point projection did not return valid triangle ids")

    kept_face_indices = set(int(idx) for idx in np.unique(tri_ids))
    if neighbor_rings > 0 and hasattr(mesh, "face_adjacency"):
        face_adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
        adjacency: dict[int, set[int]] = {}
        for a, b in face_adjacency:
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
        frontier = set(kept_face_indices)
        for _ in range(int(neighbor_rings)):
            expanded = set(frontier)
            for face_idx in frontier:
                expanded.update(adjacency.get(int(face_idx), set()))
            frontier = expanded - kept_face_indices
            kept_face_indices.update(expanded)

    kept_face_indices = np.asarray(sorted(kept_face_indices), dtype=np.int64)
    guided = mesh.submesh([kept_face_indices], append=True, only_watertight=False)
    if isinstance(guided, trimesh.Scene):
        guided = guided.dump(concatenate=True)
    if not isinstance(guided, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh from guided reconstruction, got {type(guided)!r}")
    return trimesh.Trimesh(
        vertices=np.asarray(guided.vertices, dtype=np.float64),
        faces=np.asarray(guided.faces, dtype=np.int64),
        process=False,
    )


def mesh_path_for_object(data: dict, mesh_root: Path, object_name: str) -> Path:
    obj_path = Path(data["obj_path"][object_name])
    return obj_path if obj_path.is_absolute() else mesh_root / obj_path


def render_object_comparison(
    object_name: str,
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    convex_hull: trimesh.Trimesh,
    guided_mesh: trimesh.Trimesh,
    output_path: Path,
    *,
    point_size: float,
    max_faces: int,
    mesh_path: Path,
    dpi: int,
) -> None:
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    hull_vertices = np.asarray(convex_hull.vertices, dtype=np.float64)
    hull_faces = np.asarray(convex_hull.faces, dtype=np.int64)
    guided_vertices = np.asarray(guided_mesh.vertices, dtype=np.float64)
    guided_faces = np.asarray(guided_mesh.faces, dtype=np.int64)
    all_points = np.vstack([points, mesh_vertices, hull_vertices, guided_vertices])

    fig = plt.figure(figsize=(18, 5.4))
    ax_pc = fig.add_subplot(141, projection="3d")
    ax_mesh = fig.add_subplot(142, projection="3d")
    ax_hull = fig.add_subplot(143, projection="3d")
    ax_guided = fig.add_subplot(144, projection="3d")

    draw_point_cloud(ax_pc, points, color=(0.94, 0.34, 0.22), point_size=point_size)
    set_equal_axes(ax_pc, all_points)
    format_axis(ax_pc, f"obj.pkl sampled vertices\n{len(points):,} points")

    draw_mesh(ax_mesh, mesh_vertices, mesh_faces, max_faces=max_faces)
    set_equal_axes(ax_mesh, all_points)
    format_axis(ax_mesh, f"PLY mesh\n{len(mesh_vertices):,} vertices / {len(mesh_faces):,} faces")

    draw_mesh(
        ax_hull,
        hull_vertices,
        hull_faces,
        max_faces=max_faces,
        face_color=(0.95, 0.57, 0.18, 0.36),
        edge_color=(0.36, 0.18, 0.04, 0.18),
    )
    set_equal_axes(ax_hull, all_points)
    format_axis(
        ax_hull,
        f"convex hull\n{len(hull_vertices):,} vertices / {len(hull_faces):,} faces",
    )

    draw_mesh(
        ax_guided,
        guided_vertices,
        guided_faces,
        max_faces=max_faces,
        face_color=(0.22, 0.72, 0.44, 0.34),
        edge_color=(0.08, 0.24, 0.14, 0.16),
    )
    set_equal_axes(ax_guided, all_points)
    format_axis(
        ax_guided,
        f"original-guided proxy\n{len(guided_vertices):,} vertices / {len(guided_faces):,} faces",
    )

    fig.suptitle(f"{object_name} | {mesh_path.name}", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def normalized_to_cell(points: np.ndarray, cell_center: np.ndarray, cell_scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = max(float(np.max(maxs - mins)), 1e-6)
    return ((points - center) / extent) * cell_scale + cell_center


def render_overview(
    object_names: list[str],
    point_clouds: dict[str, np.ndarray],
    meshes: dict[str, trimesh.Trimesh],
    hull_meshes: dict[str, trimesh.Trimesh],
    guided_meshes: dict[str, trimesh.Trimesh],
    output_path: Path,
    *,
    overview_cols: int,
    point_size: float,
    max_faces: int,
    dpi: int,
) -> None:
    cols = max(1, overview_cols)
    rows = int(math.ceil(len(object_names) / cols))
    cell_gap = 2.7
    cell_scale = 1.7
    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(cols * 6.2, rows * 2.2 + 1.2))
    ax_pc = fig.add_subplot(131, projection="3d")
    ax_mesh = fig.add_subplot(132, projection="3d")
    ax_proxy = fig.add_subplot(133, projection="3d")

    overview_pc_points = []
    overview_mesh_points = []
    overview_proxy_points = []
    for idx, object_name in enumerate(object_names):
        row = idx // cols
        col = idx % cols
        cell_center = np.asarray([col * cell_gap, -row * cell_gap, 0.0], dtype=np.float64)
        color = cmap(idx % cmap.N)[:3]

        pc = normalized_to_cell(point_clouds[object_name], cell_center, cell_scale)
        draw_point_cloud(ax_pc, pc, color=color, point_size=max(point_size * 0.35, 1.0))
        overview_pc_points.append(pc)
        ax_pc.text(cell_center[0], cell_center[1], -1.35, object_name, fontsize=6, ha="center")

        mesh = meshes[object_name]
        vertices = normalized_to_cell(np.asarray(mesh.vertices), cell_center, cell_scale)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        draw_mesh(
            ax_mesh,
            vertices,
            faces,
            max_faces=max_faces,
            face_color=(color[0], color[1], color[2], 0.32),
            edge_color=(0.1, 0.1, 0.1, 0.10),
        )
        overview_mesh_points.append(vertices)
        ax_mesh.text(cell_center[0], cell_center[1], -1.35, object_name, fontsize=6, ha="center")

        guided = guided_meshes[object_name]
        guided_vertices = normalized_to_cell(np.asarray(guided.vertices), cell_center, cell_scale)
        guided_faces = np.asarray(guided.faces, dtype=np.int64)
        draw_mesh(
            ax_proxy,
            guided_vertices,
            guided_faces,
            max_faces=max_faces,
            face_color=(color[0], color[1], color[2], 0.34),
            edge_color=(0.1, 0.1, 0.1, 0.10),
        )
        hull = hull_meshes[object_name]
        hull_vertices = normalized_to_cell(np.asarray(hull.vertices), cell_center, cell_scale)
        hull_faces = np.asarray(hull.faces, dtype=np.int64)
        draw_mesh(
            ax_proxy,
            hull_vertices,
            hull_faces,
            max_faces=max_faces,
            face_color=(0.95, 0.57, 0.18, 0.12),
            edge_color=(0.35, 0.18, 0.04, 0.10),
        )
        overview_proxy_points.append(guided_vertices)
        ax_proxy.text(cell_center[0], cell_center[1], -1.35, object_name, fontsize=6, ha="center")

    pc_stack = np.vstack(overview_pc_points)
    mesh_stack = np.vstack(overview_mesh_points)
    proxy_stack = np.vstack(overview_proxy_points)
    set_equal_axes(ax_pc, pc_stack)
    set_equal_axes(ax_mesh, mesh_stack)
    set_equal_axes(ax_proxy, proxy_stack)
    format_axis(ax_pc, f"obj.pkl point clouds\n{len(object_names)} objects x 1,024 points")
    format_axis(ax_mesh, "PLY meshes")
    format_axis(ax_proxy, "original-guided proxy meshes\nconvex hull overlay in orange")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    obj_pkl = args.obj_pkl.expanduser().resolve()
    mesh_root = (
        args.mesh_root.expanduser().resolve()
        if args.mesh_root is not None
        else obj_pkl.parent
    )
    output_dir = args.output_dir.expanduser().resolve()
    per_object_dir = output_dir / "per_object"
    per_object_dir.mkdir(parents=True, exist_ok=True)

    data = load_obj_data(obj_pkl)
    object_names = object_names_from_data(data, args.object_names)

    point_clouds: dict[str, np.ndarray] = {}
    meshes: dict[str, trimesh.Trimesh] = {}
    hull_meshes: dict[str, trimesh.Trimesh] = {}
    guided_meshes: dict[str, trimesh.Trimesh] = {}
    for object_name in object_names:
        points = np.asarray(data["obj_pcs"][object_name], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{object_name}: expected obj.pkl points with shape (N, 3), got {points.shape}")
        mesh_path = mesh_path_for_object(data, mesh_root, object_name)
        mesh = load_mesh(mesh_path)
        hull_mesh = convex_hull_reconstruction(points)
        guided_mesh = original_guided_reconstruction(mesh, points)

        point_clouds[object_name] = points
        meshes[object_name] = mesh
        hull_meshes[object_name] = hull_mesh
        guided_meshes[object_name] = guided_mesh
        render_object_comparison(
            object_name,
            points,
            mesh,
            hull_mesh,
            guided_mesh,
            per_object_dir / f"{object_name}.png",
            point_size=args.point_size,
            max_faces=args.max_faces,
            mesh_path=mesh_path,
            dpi=args.dpi,
        )
        print(
            f"{object_name}: obj.pkl points={len(points):,}, "
            f"mesh vertices={len(mesh.vertices):,}, faces={len(mesh.faces):,}, "
            f"hull faces={len(hull_mesh.faces):,}, guided faces={len(guided_mesh.faces):,}"
        )

    render_overview(
        object_names,
        point_clouds,
        meshes,
        hull_meshes,
        guided_meshes,
        output_dir / "overview_obj_pkl_vs_ply_mesh.png",
        overview_cols=args.overview_cols,
        point_size=args.point_size,
        max_faces=args.max_faces,
        dpi=args.dpi,
    )
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
