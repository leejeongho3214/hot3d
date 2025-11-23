import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import rerun as rr
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Hot3DVisualizer import Hot3DVisualizer  # noqa: E402
from data_loaders.headsets import Headset  # noqa: E402
from data_loaders.loader_hand_poses import HandType  # noqa: E402
from data_loaders.loader_object_library import (  # noqa: E402
    ObjectLibrary,
    load_object_library,
)
from data_loaders.mano_layer import loadManoHandModel  # noqa: E402
from dataset_api import Hot3dDataProvider  # noqa: E402
from projectaria_tools.core.mps import get_eyegaze_point_at_depth  # noqa: E402
from projectaria_tools.core.sensor_data import (  # noqa: E402
    TimeDomain,
    TimeQueryOptions,
)
from projectaria_tools.core.stream_id import StreamId  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HOT3D scene point cloud together with hand-object interactions."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.expanduser("~/Desktop/hot3d_vis"),
        help="Root directory containing the HOT3D assets and sequences.",
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        default="P0001_10a27bf7",
        help="Identifier of the sequence folder located under dataset-root.",
    )
    parser.add_argument(
        "--sequence-folder",
        type=str,
        default=None,
        help="Direct path to a sequence folder. Overrides --dataset-root/--sequence-id.",
    )
    parser.add_argument(
        "--object-library",
        type=str,
        default=None,
        help="Path to the HOT3D object library. Defaults to <dataset-root>/assets.",
    )
    parser.add_argument(
        "--mano-model",
        type=str,
        default=None,
        help="Path to the MANO model directory. Required if --hand-type=MANO.",
    )
    parser.add_argument(
        "--hand-type",
        type=str,
        choices=["UMETRACK", "MANO"],
        default="UMETRACK",
        help="Hand data source used for visualization and interaction detection.",
    )
    parser.add_argument(
        "--streams",
        type=str,
        nargs="*",
        default=None,
        help="Subset of stream ids to visualize. Defaults to all available streams.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Temporal stride when iterating over timestamps.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of frames processed after applying stride.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.02,
        help="Distance threshold in meters to flag hand-object contacts.",
    )
    parser.add_argument(
        "--contact-radius",
        type=float,
        default=0.01,
        help="Radius used to render contact points.",
    )
    parser.add_argument(
        "--max-contact-pairs",
        type=int,
        default=64,
        help="Maximum number of contact pairs logged per hand-object combination.",
    )
    parser.add_argument(
        "--object-point-samples",
        type=int,
        default=5000,
        help="Number of surface samples per object used for proximity queries.",
    )
    parser.add_argument(
        "--rrd-output",
        type=str,
        default=None,
        help="Optional path to store the Rerun recording (.rrd).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable spawning the Rerun viewer window.",
    )
    parser.add_argument(
        "--gaze-max-distance",
        type=float,
        default=3.0,
        help="Maximum distance in meters to extend eye gaze rays when searching for intersections.",
    )
    parser.add_argument(
        "--gaze-hit-threshold",
        type=float,
        default=0.05,
        help="Radial threshold in meters to consider the gaze ray hitting an object sample.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Optional[Path]]:
    dataset_root = Path(args.dataset_root).expanduser()
    sequence_folder = (
        Path(args.sequence_folder).expanduser()
        if args.sequence_folder is not None
        else dataset_root / args.sequence_id
    )
    object_library_folder = (
        Path(args.object_library).expanduser()
        if args.object_library is not None
        else dataset_root / "assets"
    )
    mano_model_folder: Optional[Path] = None
    if args.hand_type == "MANO":
        mano_model_folder = (
            Path(args.mano_model).expanduser()
            if args.mano_model is not None
            else dataset_root / "mano_v1_2" / "models"
        )

    if not sequence_folder.exists():
        raise FileNotFoundError(f"Sequence folder not found: {sequence_folder}")
    if not object_library_folder.exists():
        raise FileNotFoundError(f"Object library folder not found: {object_library_folder}")
    if mano_model_folder is not None and not mano_model_folder.exists():
        raise FileNotFoundError(f"MANO model folder not found: {mano_model_folder}")

    return sequence_folder, object_library_folder, mano_model_folder


def load_object_surface_samples(
    object_library: ObjectLibrary,
    object_uids: Iterable[str],
    sample_count: int,
) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    for object_uid in object_uids:
        cad_filepath = ObjectLibrary.get_cad_asset_path(
            object_library_folderpath=object_library.asset_folder_name,
            object_id=object_uid,
        )
        if not os.path.exists(cad_filepath):
            print(f"[WARN] Missing CAD file for object {object_uid}: {cad_filepath}")
            continue
        try:
            loaded_mesh = trimesh.load(
                cad_filepath,
                process=True,
            )   
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load mesh for {object_uid}: {exc}")
            continue

        if isinstance(loaded_mesh, trimesh.Scene):
            if not loaded_mesh.geometry:
                print(f"[WARN] Empty scene for object {object_uid}: {cad_filepath}")
                continue
            component_meshes = [
                geom.copy()
                for geom in loaded_mesh.geometry.values()
                if isinstance(geom, trimesh.Trimesh)
            ]
            if not component_meshes:
                print(
                    f"[WARN] Scene geometry not convertible to meshes for {object_uid}: {cad_filepath}"
                )
                continue
            mesh = (
                component_meshes[0]
                if len(component_meshes) == 1
                else trimesh.util.concatenate(component_meshes)
            )
        elif isinstance(loaded_mesh, trimesh.Trimesh):
            mesh = loaded_mesh
        else:
            print(f"[WARN] Unsupported mesh type for object {object_uid}: {type(loaded_mesh)}")
            continue

        if mesh.is_empty:
            print(f"[WARN] Mesh with no vertices for object {object_uid}: {cad_filepath}")
            continue

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if sample_count > 0 and len(vertices) > sample_count:
            # Use surface sampling for a uniform point distribution over the mesh.
            samples, _ = trimesh.sample.sample_surface(mesh, sample_count)
            cache[object_uid] = np.asarray(samples, dtype=np.float32)
        else:
            cache[object_uid] = vertices
    return cache


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def log_eye_gaze_intersections(
    timestamp_ns: int,
    data_provider: Hot3dDataProvider,
    world_object_samples: Dict[str, np.ndarray],
    object_library: ObjectLibrary,
    max_distance: float,
    hit_threshold: float,
) -> None:
    device_provider = data_provider.device_data_provider
    device_pose_provider = data_provider.device_pose_data_provider

    if (
        data_provider.get_device_type() is not Headset.Aria
        or device_provider is None
        or device_pose_provider is None
    ):
        rr.log("world/eye_gaze", rr.Clear.recursive())
        return

    eye_gaze = device_provider.get_eye_gaze(timestamp_ns)
    if eye_gaze is None:
        rr.log("world/eye_gaze", rr.Clear.recursive())
        return

    headset_pose_with_dt = device_pose_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
        acceptable_time_delta=0,
    )
    if headset_pose_with_dt is None:
        rr.log("world/eye_gaze", rr.Clear.recursive())
        return

    T_world_device = headset_pose_with_dt.pose3d.T_world_device.to_matrix()
    T_device_cpf = (
        device_provider.get_device_calibration().get_transform_device_cpf().to_matrix()
    )
    T_world_cpf = T_world_device @ T_device_cpf

    origin_world = (T_world_cpf @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    gaze_point_local = get_eyegaze_point_at_depth(
        eye_gaze.yaw, eye_gaze.pitch, depth_m=max_distance
    )
    gaze_point_world = (T_world_cpf @ np.array([*gaze_point_local, 1.0]))[:3]

    direction = gaze_point_world - origin_world
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-6:
        rr.log("world/eye_gaze", rr.Clear.recursive())
        return

    direction_unit = direction / direction_norm
    best_distance = max_distance
    best_object_uid: Optional[str] = None
    best_point = origin_world + direction_unit * best_distance

    for object_uid, samples_world in world_object_samples.items():
        if samples_world.size == 0:
            continue
        diff = samples_world - origin_world
        param_t = diff @ direction_unit
        valid_mask = param_t >= 0.0
        if not np.any(valid_mask):
            continue
        diff_valid = diff[valid_mask]
        t_valid = param_t[valid_mask]
        perpendicular = diff_valid - np.outer(t_valid, direction_unit)
        radial_distance = np.linalg.norm(perpendicular, axis=1)
        candidate_mask = radial_distance <= hit_threshold
        if not np.any(candidate_mask):
            continue
        candidate_indices = np.where(candidate_mask)[0]
        candidate_t = t_valid[candidate_indices]
        min_idx = candidate_indices[np.argmin(candidate_t)]
        distance_along_ray = t_valid[min_idx]
        if distance_along_ray < best_distance:
            best_distance = float(distance_along_ray)
            best_object_uid = object_uid
            best_point = origin_world + direction_unit * best_distance

    arrow_vector = direction_unit * best_distance
    rr.log(
        "world/eye_gaze/ray",
        rr.Arrows3D(
            origins=[origin_world.tolist()],
            vectors=[arrow_vector.tolist()],
            colors=[[255, 255, 0]],
        ),
    )

    if best_object_uid is not None:
        object_name = object_library.object_id_to_name_dict.get(
            best_object_uid, best_object_uid
        )
        rr.log(
            "world/eye_gaze/hit",
            rr.Points3D(
                positions=[best_point.tolist()],
                colors=[[255, 0, 255]],
                radii=max(hit_threshold, 1e-3),
                labels=[f"hit: {object_name}"],
            ),
        )
    else:
        rr.log("world/eye_gaze/hit", rr.Clear.recursive())


def log_contacts(
    timestamp_ns: int,
    hand_provider,
    object_provider,
    object_library: ObjectLibrary,
    object_samples: Dict[str, np.ndarray],
    contact_threshold: float,
    contact_radius: float,
    max_pairs: int,
) -> Dict[str, np.ndarray]:
    if object_provider is None:
        rr.log("world/interactions", rr.Clear.recursive())
        return {}

    hand_poses = (
        hand_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=0,
        )
        if hand_provider is not None
        else None
    )
    object_poses = object_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
        acceptable_time_delta=0,
    )

    if object_poses is None:
        rr.log("world/interactions", rr.Clear.recursive())
        return {}

    world_samples_map: Dict[str, np.ndarray] = {}
    handled_object_uids = set()
    for object_uid, object_pose in object_poses.pose3d_collection.poses.items():
        handled_object_uids.add(object_uid)
        object_name = object_library.object_id_to_name_dict.get(object_uid, object_uid)
        samples = object_samples.get(object_uid)
        interaction_root = f"world/interactions/{object_name}_{object_uid}"
        if samples is None:
            rr.log(interaction_root, rr.Clear.recursive())
            continue

        transform = object_pose.T_world_object.to_matrix()
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        world_samples = transform_points(samples, rotation, translation)
        world_samples_map[object_uid] = world_samples

        if hand_poses is None:
            rr.log(interaction_root, rr.Clear.recursive())
            continue

        tree = cKDTree(world_samples)
        for _, hand_pose in hand_poses.pose3d_collection.poses.items():
            hand_vertices = hand_provider.get_hand_mesh_vertices(hand_pose)
            if hand_vertices is None:
                continue
            hand_vertices_np = np.asarray(hand_vertices, dtype=np.float32)

            distances, indices = tree.query(hand_vertices_np, k=1)
            mask = distances < contact_threshold
            contact_path = f"{interaction_root}/{hand_pose.handedness_label()}"
            if not np.any(mask):
                rr.log(contact_path, rr.Clear.recursive())
                continue
            mask_indices = np.where(mask)[0]
            if len(mask_indices) > max_pairs:
                closest_order = np.argsort(distances[mask_indices])
                mask_indices = mask_indices[closest_order[:max_pairs]]
            contact_points = hand_vertices_np[mask_indices]
            nearest_object_points = world_samples[np.asarray(indices)[mask_indices]]

            rr.log(
                f"{contact_path}/vectors",
                rr.Arrows3D(
                    origins=contact_points,
                    vectors=nearest_object_points - contact_points,
                    colors=[255, 200, 0],
                ),
            )
            rr.log(
                f"{contact_path}/hand_points",
                rr.Points3D(contact_points, colors=[255, 0, 0], radii=contact_radius),
            )
            rr.log(
                f"{contact_path}/object_points",
                rr.Points3D(
                    nearest_object_points,
                    colors=[0, 255, 0],
                    radii=max(contact_radius / 2, 1e-3),
                ),
            )

    missing_objects = set(object_samples.keys()) - handled_object_uids
    for object_uid in missing_objects:
        object_name = object_library.object_id_to_name_dict.get(object_uid, object_uid)
        rr.log(f"world/interactions/{object_name}_{object_uid}", rr.Clear.recursive())

    return world_samples_map


def main() -> None:
    args = parse_args()
    sequence_folder, object_library_folder, mano_model_folder = resolve_paths(args)

    object_library = load_object_library(
        object_library_folderpath=str(object_library_folder)
    )
    mano_model = loadManoHandModel(str(mano_model_folder)) if mano_model_folder else None

    data_provider = Hot3dDataProvider(
        sequence_folder=str(sequence_folder),
        object_library=object_library,
        mano_hand_model=mano_model,
        fail_on_missing_data=False,
    )

    hand_enum = HandType.Umetrack if args.hand_type == "UMETRACK" else HandType.Mano
    if hand_enum is HandType.Mano and data_provider.mano_hand_data_provider is None:
        raise RuntimeError("MANO hand provider unavailable for this sequence.")
    if hand_enum is HandType.Umetrack and data_provider.umetrack_hand_data_provider is None:
        raise RuntimeError("UMETRACK hand provider unavailable for this sequence.")

    if args.rrd_output is not None:
        rr.init("HOT3D Interaction Viewer", spawn=False)
        rr.save(args.rrd_output)
    else:
        rr.init("HOT3D Interaction Viewer", spawn=not args.headless)

    visualizer = Hot3DVisualizer(data_provider, hand_enum)

    device_provider = data_provider.device_data_provider
    stream_ids: Sequence[StreamId]
    if args.streams:
        stream_ids = [StreamId(stream) for stream in args.streams]
    else:
        stream_ids = device_provider.get_image_stream_ids()

    visualizer.log_static_assets(stream_ids)

    object_samples = load_object_surface_samples(
        object_library=object_library,
        object_uids=data_provider.object_pose_data_provider.object_uids_with_poses
        if data_provider.object_pose_data_provider is not None
        else [],
        sample_count=args.object_point_samples,
    )

    timestamps = device_provider.get_sequence_timestamps()
    active_hand_provider = (
        data_provider.umetrack_hand_data_provider
        if hand_enum is HandType.Umetrack
        else data_provider.mano_hand_data_provider
    )
    object_provider = data_provider.object_pose_data_provider

    for frame_idx, timestamp_ns in enumerate(tqdm(timestamps[:: args.stride])):
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        rr.set_time_sequence("frame", frame_idx)
        rr.set_time_nanos("timestamp_ns", int(timestamp_ns))

        visualizer.log_dynamic_assets(stream_ids, timestamp_ns)
        world_object_samples = log_contacts(
            timestamp_ns=timestamp_ns,
            hand_provider=active_hand_provider,
            object_provider=object_provider,
            object_library=object_library,
            object_samples=object_samples,
            contact_threshold=args.contact_threshold,
            contact_radius=args.contact_radius,
            max_pairs=args.max_contact_pairs,
        )
        log_eye_gaze_intersections(
            timestamp_ns=timestamp_ns,
            data_provider=data_provider,
            world_object_samples=world_object_samples,
            object_library=object_library,
            max_distance=args.gaze_max_distance,
            hit_threshold=args.gaze_hit_threshold,
        )


if __name__ == "__main__":
    main()
