import argparse
import os
import sys

import numpy as np
import torch
import trimesh

import rerun as rr

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from rot import axis_angle_to_rotmat
from mano import build_mano_aa


def _resolve_mesh_path(npz_path: str, mesh_path: str) -> str:
    if os.path.isabs(mesh_path):
        return mesh_path
    return os.path.normpath(os.path.join(os.path.dirname(npz_path), mesh_path))


def _load_mesh_vertices(path: str) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    if hasattr(mesh, "vertices"):
        return np.asarray(mesh.vertices, dtype=np.float32)
    raise ValueError(f"Unsupported mesh type: {type(mesh)}")


def _sample_vertices(vertices: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or vertices.shape[0] <= max_points:
        return vertices
    idx = np.linspace(0, vertices.shape[0] - 1, max_points).astype(np.int64)
    return vertices[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a grab npz (object only).")
    parser.add_argument(
        "--npz",
        default="/Users/jeongho/Desktop/hot3d_vis/mug_drink_4.npz",
        help="path to grab npz file",
    )
    parser.add_argument("--step", type=int, default=10, help="frame step")
    parser.add_argument("--max-points", type=int, default=5000, help="max points to render")
    parser.add_argument("--radius", type=float, default=0.002, help="point radius")
    parser.add_argument("--show-hands", action="store_false", help="render MANO hands")
    args = parser.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    obj_info = npz["object"].item()
    params = obj_info["params"]
    mesh_path = _resolve_mesh_path(args.npz, obj_info["object_mesh"])

    vertices = _load_mesh_vertices(mesh_path)
    vertices = _sample_vertices(vertices, args.max_points)

    transl = np.asarray(params["transl"], dtype=np.float32)
    orient = np.asarray(params["global_orient"], dtype=np.float32)

    v = torch.from_numpy(vertices).float()
    t = torch.from_numpy(transl).float()
    r = torch.from_numpy(orient).float()
    rotmat = axis_angle_to_rotmat(r)  # (T,3,3)

    lhand = npz["lhand"].item()
    rhand = npz["rhand"].item()
    lparams = lhand["params"]
    rparams = rhand["params"]

    l_layer = r_layer = None
    if args.show_hands:
        l_layer = build_mano_aa(is_rhand=False, flat_hand=False)
        r_layer = build_mano_aa(is_rhand=True, flat_hand=False)
        l_faces = l_layer.faces.copy().astype(np.int32)
        r_faces = r_layer.faces.copy().astype(np.int32)

    rr.init("grab_npz", spawn=True)
    for frame_idx in range(0, t.shape[0], args.step):
        rr.set_time("frame", sequence=frame_idx)
        rot = torch.einsum("ij,vj->vi", rotmat[frame_idx], v)
        pts = (rot + t[frame_idx]).detach().cpu().numpy()
        rr.log(
            "object",
            rr.Points3D(
                positions=pts,
                radii=args.radius,
                colors=[0, 121, 121],
            ),
        )

        if args.show_hands and l_layer is not None and r_layer is not None:
            betas = torch.zeros((1, 10), dtype=torch.float32)

            l_global = torch.from_numpy(lparams["global_orient"][frame_idx : frame_idx + 1]).float()
            l_pose = torch.from_numpy(lparams["fullpose"][frame_idx : frame_idx + 1]).float()
            l_out = l_layer(global_orient=l_global, hand_pose=l_pose, betas=betas)
            l_verts = l_out.vertices[0].detach().cpu().numpy()
            l_verts = l_verts + lparams["transl"][frame_idx]

            r_global = torch.from_numpy(rparams["global_orient"][frame_idx : frame_idx + 1]).float()
            r_pose = torch.from_numpy(rparams["fullpose"][frame_idx : frame_idx + 1]).float()
            r_out = r_layer(global_orient=r_global, hand_pose=r_pose, betas=betas)
            r_verts = r_out.vertices[0].detach().cpu().numpy()
            r_verts = r_verts + rparams["transl"][frame_idx]

            rr.log(
                "l_hand",
                rr.Mesh3D(
                    vertex_positions=l_verts,
                    triangle_indices=l_faces,
                ),
            )
            rr.log(
                "r_hand",
                rr.Mesh3D(
                    vertex_positions=r_verts,
                    triangle_indices=r_faces,
                ),
            )


if __name__ == "__main__":
    main()
