#
# Section 0: DataProvider initialization
#
# Take home message:
# - Device data, such as Image data stream is indexed with a stream_id
# - Intrinsics and Extrinsics calibration relative to the device coordinates is available for each CAMERA/stream_id
#
# Data Requirements:
# - a sequence
# - the object library
# Optional:
# - To use the Mano hand you need to have the LEFT/RIGHT *.pkl hand models (available)

import os
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel

home = os.path.expanduser("~")
hot3d_dataset_path = sequence_path = home + "/Desktop/dataset/P0001_9b6feab7"
object_library_path = home +"/Desktop/assets"
mano_hand_model_path = home + "/Desktop/mano_v1_2/models"

if not os.path.exists(sequence_path) or not os.path.exists(object_library_path):
    print("Invalid input sequence or library path.")
    print("Please do update the path to VALID values for your system.")
    raise
#
# Init the object library
#
object_library = load_object_library(object_library_folderpath=object_library_path)

#
# Init the HANDs model
# If None, the UmeTrack HANDs model will be used
#
mano_hand_model = None
if mano_hand_model_path is not None:
    mano_hand_model = MANOHandModel(mano_hand_model_path)

#
# Initialize hot3d data provider
#
hot3d_data_provider = Hot3dDataProvider(
    sequence_folder=sequence_path,
    object_library=object_library,
    mano_hand_model=mano_hand_model,
)
print(f"data_provider statistics: {hot3d_data_provider.get_data_statistics()}")

#
# Utility functions
# Used for interactive display in the following sections
#
import rerun as rr
import numpy as np

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D


def log_image(
    image: np.array,
    label: str,
    static=False
) -> None:
    rr.log(label, rr.Image(image), static=static)


def log_pose(
    pose: SE3,
    label: str,
    static=False
) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)
    

# Section 1: Device calibration and Image data

from tqdm import tqdm

#
# Retrieve some statistics about the "IMAGE" VRS recording
#

# Getting the device data provider (alias)
device_data_provider = hot3d_data_provider.device_data_provider

# Retrieve the list of image stream supported by this sequence
# It will return the RGB and SLAM Left/Right image streams
image_stream_ids = device_data_provider.get_image_stream_ids()
# Retrieve a list of timestamps for the sequence (in nanoseconds)
timestamps = device_data_provider.get_sequence_timestamps()

print(f"Sequence: {os.path.basename(os.path.normpath(sequence_path))}")
print(f"Device type is {hot3d_data_provider.get_device_type()}")
print(f"Image stream ids: {image_stream_ids}")
print(f"Number of timestamp for this sequence: {len(timestamps)}")
print(
    f"Duration of the sequence: {(timestamps[-1] - timestamps[0]) / 1e9} (seconds)"
)  # Timestamps are in nanoseconds



from collections import defaultdict
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.calibration import FISHEYE624
from projectaria_tools.core.mps import get_eyegaze_point_at_depth
import trimesh
from tqdm import tqdm

# Alias over the HEADSET/Device pose data provider
device_pose_provider = hot3d_data_provider.device_pose_data_provider

rr.init("3D Gaze Trajectory", spawn=True)


gaze_3d_points = []
timestamps = device_data_provider.get_sequence_timestamps()
timestamps = timestamps[::3]

for idx, timestamp_ns in tqdm(enumerate(timestamps[:300])):

    # rr.set_time_nanos("synchronization_time", int(timestamp_ns))
    # rr.set_time_sequence("timestamp", timestamp_ns)

    # Headset pose
    pose3d_dt = device_pose_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )
    if pose3d_dt is None:
        continue

    T_world_device = pose3d_dt.pose3d.T_world_device

    # Eye gaze data
    eye_gaze = hot3d_data_provider.device_data_provider.get_eye_gaze(timestamp_ns)
    if eye_gaze is None:
        continue

    # Gaze vector in CPF coordinates, assume 1m depth if not provided
    gaze_cpf = get_eyegaze_point_at_depth(
        eye_gaze.yaw,
        eye_gaze.pitch,
        depth_m=eye_gaze.depth or 1.0
    )

    T_device_cpf = hot3d_data_provider.device_data_provider.get_device_calibration().get_transform_device_cpf()
    gaze_in_world = (T_world_device @ T_device_cpf @ gaze_cpf)  # 이미 좌표값

    gaze_3d_points.append(gaze_in_world)#
# Section 2: Pose data
#
# Take home message:
# - the device_pose_provider enables you to retrieve the Headset pose as (T_world_device)
# - moving to the device to a given camera can be done by using calibration data and combining SE3 poses
#   - such as T_world_camera = T_world_device @ T_device_camera
#

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
# Alias over the HEADSET/Device pose data provider
device_pose_provider = hot3d_data_provider.device_pose_data_provider
from Hot3DVisualizer import Hot3DVisualizer



hot3d_visualizer = Hot3DVisualizer(hot3d_data_provider)
pose_translations = []
object_cache_status = {}

hand_data_provider = hot3d_data_provider.mano_hand_data_provider if hot3d_data_provider.mano_hand_data_provider is not None else hot3d_data_provider.umetrack_hand_data_provider

# Accumulate HAND poses translations as list, to show a LINE strip HAND trajectory
left_hand_pose_translations = []
right_hand_pose_translations = []
vertex_hit_counter = defaultdict(lambda: defaultdict(int))

traj_list = []
# Retrieve the position of the device in the world frame at a given timestamp
for idx, timestamp_ns in tqdm(enumerate(timestamps)):

    rr.set_time_nanos("synchronization_time", int(timestamp_ns))
    # rr.set_time_sequence("timestamp", timestamp_ns)
    rr.set_time_sequence("idx", idx)
    
    
    # 3D Object 시각화 추가
    # 3D object pose 불러오기
    object_poses_with_dt = hot3d_data_provider.object_pose_data_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )

    # # 3D object 시각화
    # Hot3DVisualizer.log_object_poses(
    #     label="world/objects",
    #     object_poses_with_dt=object_poses_with_dt,
    #     object_pose_data_provider=hot3d_data_provider.object_pose_data_provider,
    #     object_library=hot3d_data_provider.object_library,
    #     object_cache_status=object_cache_status
    # )

    headset_pose3d_with_dt = None
    if device_pose_provider is None:
        continue
    headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )

    if headset_pose3d_with_dt is None:
        continue

    headset_pose3d = headset_pose3d_with_dt.pose3d
    T_world_device = headset_pose3d.T_world_device
        
    eye_gaze = hot3d_data_provider.device_data_provider.get_eye_gaze(timestamp_ns)
            
    gaze_vector_in_cpf = get_eyegaze_point_at_depth(
        eye_gaze.yaw, eye_gaze.pitch, depth_m=1.0  # 벡터는 방향만 필요하니 depth 임의 고정
    )

    gaze_origin = (T_world_device @ T_device_cpf @ np.array([0, 0, 0]))
    gaze_target = (T_world_device @ T_device_cpf @ gaze_vector_in_cpf)

    gaze_direction = gaze_target - gaze_origin
    
    # 물체 Pose 불러오기
    object_poses_with_dt = hot3d_data_provider.object_pose_data_provider.get_pose_at_timestamp(
        timestamp_ns, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE
    )
    
    hand_poses_with_dt = None
    if hand_data_provider is None:
        continue
    
    hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )

    if hand_poses_with_dt is None:
        continue
        
    hand_pose_collection = hand_poses_with_dt.pose3d_collection

    for hand_pose_data in hand_pose_collection.poses.values():
        # Retrieve the handedness of the hand (i.e Left or Right)
        handedness_label = hand_pose_data.handedness_label()

        T_world_wrist = hand_pose_data.wrist_pose
        log_pose(pose=T_world_wrist, label=f"world/hand/{handedness_label}")

        # Accumulate HAND poses translations as list, to show a LINE strip HAND trajectory
        if hand_pose_data.is_left_hand():
            left_hand_pose_translations.append(T_world_wrist.translation()[0])
        elif hand_pose_data.is_right_hand():
            right_hand_pose_translations.append(T_world_wrist.translation()[0])
            
    color_map = defaultdict()
    # 랜덤 색상 생성
    for i in object_poses_with_dt.pose3d_collection.poses.keys():
        np.random.seed(hash(i) % 2**32)  # 물체별 고정된 랜덤 색상
        random_color = np.random.randint(0, 256, size=3).tolist()
        color_map[i] = random_color

    all_nearest = []
    obj_mesh = defaultdict()
    # 누적 카운트: {object_id: {vertex_index: count}}
    
    for hand_pose_data in hand_pose_collection.poses.values():
        # Retrieve the handedness of the hand (i.e Left or Right)
        handedness_label = hand_pose_data.handedness_label()

        
        # 양손 모두 Mesh 처리
        hand_mesh_vertices = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
        hand_triangles, hand_vertex_normals = hand_data_provider.get_hand_mesh_faces_and_normals(hand_pose_data)

        rr.log(
            f"world/{handedness_label}/mesh_faces",
            rr.Mesh3D(
                vertex_positions=hand_mesh_vertices,
                triangle_indices=hand_triangles,
                vertex_colors=[255, 255, 255]
            ),
        )
    
    
    for obj_id, obj_pose in object_poses_with_dt.pose3d_collection.poses.items():
        object_cad_asset_filepath = hot3d_data_provider.object_library.get_cad_asset_path(
                    object_library_folderpath=object_library.asset_folder_name,
                    object_id=obj_id,
                )
        
        rr.log(
            f"world/mesh/{obj_id}",
            rr.Asset3D(
                path=object_cad_asset_filepath,
            ),
        )
        mesh = trimesh.load(object_cad_asset_filepath).to_geometry() 
        
        T = object_poses_with_dt.pose3d_collection.poses[obj_id].T_world_object.to_matrix()
        vertices_local = mesh.vertices
        vertices_homo = np.hstack([vertices_local, np.ones((vertices_local.shape[0], 1))])
        vertices_world = (T @ vertices_homo.T).T[:, :3]
        
        obj_mesh[obj_id] = vertices_world
        
        scene_mesh = trimesh.Trimesh(vertices=vertices_world, faces=mesh.faces)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(scene_mesh)        
        ray_origins = np.atleast_2d(np.asarray(gaze_origin).reshape(-1, 3))
        ray_directions = np.atleast_2d(np.asarray(gaze_direction).reshape(-1, 3))
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins, ray_directions
        )
        
        log_pose(pose=obj_pose.T_world_object, label=f"world/mesh/{obj_id}",)
        
        rr.log(
            "world/gaze_vector",
            rr.Arrows3D(
                origins=[gaze_origin],
                vectors=[gaze_direction],
            ), )
        
        stream_id = StreamId("214-1")
        eye_gaze = device_data_provider.get_eye_gaze(timestamp_ns=timestamp_ns)
        
        headset_pose3d = headset_pose3d_with_dt.pose3d
        T_world_device = headset_pose3d.T_world_device
        
        # Aria 카메라의 pose 구하기
        camera_label = device_data_provider.get_image_stream_label(stream_id)
        camera_calib = device_data_provider.get_device_calibration().get_camera_calib(camera_label)
        T_device_camera = camera_calib.get_transform_device_camera()
        T_world_camera = T_world_device @ T_device_camera

        log_pose(T_world_camera, "world/camera_pose")

        rr.log(
            "world/viewpoint/camera",
            rr.Transform3D(
                    translation=T_world_camera.translation().tolist()[0],
                    rotation=T_world_camera.rotation().to_quat().tolist()[0],
                ),
            static=True)
        
        T_device_cpf = hot3d_data_provider.device_data_provider.get_device_calibration().get_transform_device_cpf()
        
        gaze_vector_in_cpf = get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, depth_m=1.0  # 벡터는 방향만 필요하니 depth 임의 고정
        )
        gaze_origin = (T_world_device @ T_device_cpf @ np.array([0, 0, 0]))
        gaze_target = (T_world_device @ T_device_cpf @ gaze_vector_in_cpf)
        gaze_direction = gaze_target - gaze_origin
        
        camera_model = FISHEYE624
    
        eye_gaze_reprojection_data = (
            device_data_provider.get_eye_gaze_in_camera(
                stream_id, timestamp_ns, camera_model=camera_model
            )
        )
        if (
            eye_gaze_reprojection_data is None
            or not eye_gaze_reprojection_data.any()
        ):
            continue
        
        # 4. 교차 지점 시각화
        if len(locations) > 0:
            distances = np.linalg.norm(locations - ray_origins[0], axis=1)
            nearest_index = np.argmin(distances)
            nearest_location = locations[nearest_index]
            nearest_distance = distances[nearest_index]
            
            vertex_hit_counter[obj_id][nearest_index] += 1

            all_nearest.append({
                "obj_id": obj_id,
                "location": nearest_location,
                "distance": nearest_distance
            })

            traj_list.append(nearest_location)
            
            sphere = trimesh.creation.icosphere(radius=0.01, subdivisions=2)
            sphere.apply_translation(nearest_location)

    # 모든 물체에 대한 후보 중 가장 가까운 것 선택
    if len(all_nearest) > 0:
        best = min(all_nearest, key=lambda x: x["distance"])
        nearest_location = best["location"]

        traj_list.append(nearest_location)

        sphere = trimesh.creation.icosphere(radius=0.01, subdivisions=2)
        sphere.apply_translation(nearest_location)

        rr.log(
            "world/intersection_point_sphere",
            rr.Mesh3D(
                vertex_positions=sphere.vertices,
                triangle_indices=sphere.faces,
                vertex_colors=[[255, 0, 0]]
            )
        )
            
    else:
        rr.log(
                f"world/intersection_point_sphere",
                rr.Clear.flat(),
            )
    
    image_stream_label = device_data_provider.get_image_stream_label(stream_id)
    # Retrieve the image data for a given timestamp
    image_data = device_data_provider.get_image(timestamp_ns, stream_id)
    # Visualize the image data (it's a numpy array)
    # rr.log(f"world/img", rr.Image(image_data))
    
    rr.log(
    f"world/gaze",
    rr.Points2D(eye_gaze_reprojection_data, radii=20),
    )
            

# for obj_id, mesh_vertices in obj_mesh.items():
#     counts = vertex_hit_counter[obj_id]

#     # 기본 색상 설정
#     vertex_colors = np.zeros((mesh_vertices.shape[0], 3), dtype=np.uint8) + 255  # 기본 흰색

#     for idx, count in counts.items():
#         intensity = min(255, count * 20)  # 카운트 기반 색상 (적절히 조정)
#         vertex_colors[idx] = [255, 255 - intensity, 255 - intensity]  # 빨간색 계열

#     rr.log(
#         f"world/heatmap/{obj_id}",
#         rr.Mesh3D(
#             vertex_positions=mesh_vertices,
#             triangle_indices=mesh.faces,
#             vertex_colors=vertex_colors.tolist()
#         )
#     )
            
rr.log(
    "world/interact_point_traj", rr.LineStrips3D(traj_list), static=True)
            
            
# rr.notebook_show()
rr.save(f" ~/Library/CloudStorage/SynologyDrive-14inch_mac/record/P0001_8d136980.rrd")


