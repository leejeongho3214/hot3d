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

from collections import defaultdict
import json
import os
from scipy.spatial import cKDTree
import numpy as np
import trimesh
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel

seq_name = "P0003_e3a74169"

home = os.path.expanduser("~")
hot3d_dataset_path = home + "/Desktop/"
sequence_path = os.path.join(hot3d_dataset_path, seq_name)
object_library_path = os.path.join(hot3d_dataset_path, "assets")
mano_hand_model_path = os.path.join(home, "Desktop/mano_v1_2/models")

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


from tqdm import tqdm

device_data_provider = hot3d_data_provider.device_data_provider

image_stream_ids = device_data_provider.get_image_stream_ids()
# Retrieve a list of timestamps for the sequence (in nanoseconds)
timestamps = device_data_provider.get_sequence_timestamps()
# Used for interactive display in the following sections
#
import rerun as rr
import numpy as np

import trimesh

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from data_loaders.headsets import Headset
from projectaria_tools.core.calibration import FISHEYE624
from data_loaders.loader_object_library import ObjectLibrary
if hot3d_data_provider.get_device_type() is not Headset.Aria:
    pass


device_data_provider = hot3d_data_provider.device_data_provider
# Use RGB image


from projectaria_tools.core.mps import get_eyegaze_point_at_depth
from projectaria_tools.core.stream_id import StreamId  # @manual
stream_id = StreamId("214-1")

object_box2d_data_provider = hot3d_data_provider.object_box2d_data_provider
device_data_provider = hot3d_data_provider.device_data_provider
# Alias over the Object pose data provider
object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
device_pose_provider = hot3d_data_provider.device_pose_data_provider


# Retrieve a distinct color mapping for object bounding box
# by using a colormap (i.e associate a object_uid to a specific color)
object_uids = list(object_box2d_data_provider.object_uids) # list of available object_uid used to map them to [0, 1, 2, ...] indices
object_mesh = defaultdict()
vertex_hit_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
contact_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

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

os.chdir("hot3d/")

# for name_json in tqdm(os.listdir("analysis"), desc = "json_list"):
name_json = home + f"/Desktop/analysis/{seq_name}.json"
with open(f"{name_json}", "r") as f:
    content = json.load(f)
    
    
rr.init("GazeHeatmapAnalysis", spawn=True)
# for idx, timestamp_ns in tqdm(enumerate(timestamps)):
#     rr.set_time_sequence("timestamp", idx)
#     for stream_id in image_stream_ids:
#         if stream_id == StreamId("214-1"):
#             # Retrieve the image stream label as string
#             image_stream_label = device_data_provider.get_image_stream_label(stream_id)
#             # Retrieve the image data for a given timestamp
#             image_data = device_data_provider.get_image(timestamp_ns, stream_id)
#             # Visualize the image data (it's a numpy array)
#             log_image(label=f"img/{image_stream_label}", image=image_data)


for object_name, value_list in tqdm(content['double'].items()):
    vertex_color_total = defaultdict()
    if object_name != "coffee_pot&bottle_mustard":
        continue
    
    name1, name2 = object_name.split('&')
    for a_id, value in tqdm(enumerate(value_list)):
        value[1][0]  = 3000
        for idx, frame_idx in enumerate(tqdm(range(value[1][0], value[1][1]))):
            timestamp_ns = timestamps[frame_idx]
            rr.set_time_sequence("timestamp", idx)

            # We are showing EyeGaze reprojection only on the RGB image stream
            if stream_id != StreamId("214-1"):
                continue
            
            object_poses_with_dt = hot3d_data_provider.object_pose_data_provider.get_pose_at_timestamp(
                timestamp_ns, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE
                )
            
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
            
            eye_gaze = device_data_provider.get_eye_gaze(timestamp_ns=timestamp_ns)
            
            headset_pose3d = headset_pose3d_with_dt.pose3d
            T_world_device = headset_pose3d.T_world_device
            
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
            
            flag = True
            for obj_id, _ in object_poses_with_dt.pose3d_collection.poses.items():
                if object_library.object_id_to_name_dict[obj_id] not in [name1, name2]:
                    continue
                
                if obj_id not in object_mesh.keys():
                    object_cad_asset_filepath = hot3d_data_provider.object_library.get_cad_asset_path(
                                object_library_folderpath=object_library.asset_folder_name,
                                object_id=obj_id,
                            )
                    mesh = trimesh.load(object_cad_asset_filepath).to_geometry() 
                    object_mesh[obj_id] = mesh
                    
                    T = object_poses_with_dt.pose3d_collection.poses[obj_id].T_world_object.to_matrix()
                    vertices_local = object_mesh[obj_id].vertices
                    vertices_homo = np.hstack([vertices_local, np.ones((vertices_local.shape[0], 1))])
                    vertices_world = (T @ vertices_homo.T).T[:, :3]
                    
                else:
                    T = object_poses_with_dt.pose3d_collection.poses[obj_id].T_world_object.to_matrix()
                    vertices_local = object_mesh[obj_id].vertices
                    vertices_homo = np.hstack([vertices_local, np.ones((vertices_local.shape[0], 1))])
                    vertices_world = (T @ vertices_homo.T).T[:, :3]
                
                tree = cKDTree(vertices_world)
                
                hand_poses_with_dt = hot3d_data_provider.umetrack_hand_data_provider.get_pose_at_timestamp(
                    timestamp_ns=timestamp_ns,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                    acceptable_time_delta=0,
                )
                l_hand, r_hand = None, None
                
                hand_pose_collection = hand_poses_with_dt.pose3d_collection
                for hand_pose_data in hand_pose_collection.poses.values():
                    if hand_pose_data.is_left_hand():
                        l_hand = hot3d_data_provider.umetrack_hand_data_provider.get_hand_mesh_vertices(hand_pose_data)

                    elif hand_pose_data.is_right_hand():
                        r_hand = hot3d_data_provider.umetrack_hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
                
                
                if l_hand is not None:
                    distances, indices = tree.query(l_hand)
                    min_index = indices[distances < 0.002]
                    for i in min_index:
                        contact_counter[obj_id][a_id][i] += 1
                
                if r_hand is not None:
                    distances, indices = tree.query(r_hand)
                    min_index = indices[distances < 0.002]
                    for i in min_index:
                        contact_counter[obj_id][a_id][i] += 1
                    
                
                scene_mesh = trimesh.Trimesh(vertices=vertices_world, faces=object_mesh[obj_id].faces)
                intersector = trimesh.ray.ray_triangle.RayMeshIntersector(scene_mesh)        
                ray_origins = np.atleast_2d(np.asarray(gaze_origin).reshape(-1, 3))
                ray_directions = np.atleast_2d(np.asarray(gaze_direction).reshape(-1, 3))
                locations, index_ray, index_tri = intersector.intersects_location(
                    ray_origins, ray_directions
                )
                        
                if len(locations) > 0:
                    distances = np.linalg.norm(locations - ray_origins[0], axis=1)
                    nearest_index = np.argmin(distances)
                    nearest_location = locations[nearest_index]
                    nearest_distance = distances[nearest_index]

                    # 모든 교차 face 처리
                    for tri_idx in index_tri:
                        face_vertices = scene_mesh.faces[tri_idx]

                        for v_idx in face_vertices:
                            if v_idx < scene_mesh.vertices.shape[0]:
                                vertex_hit_counter[obj_id][a_id][v_idx] += 1
                            else:
                                print(f"Skipping invalid vertex index {v_idx} for object {obj_id}")
                                
                mesh_vertices = object_mesh[obj_id]
                counts = vertex_hit_counter[obj_id][a_id]
                contacts = contact_counter[obj_id][a_id]
                
                mesh_vertices = mesh_vertices.vertices
                # 기본 색상 설정
                vertex_colors = np.zeros((mesh_vertices.shape[0], 3), dtype=np.uint8) + 255  # 기본 흰색
                
                total_value = sum(list(counts.values()))

                for idx, count in counts.items():
                    intensity = min(255, 40 * count)
                    vertex_colors[idx] = [255, 255 - intensity, 255 - intensity]  # 빨간색 계열
                    
                for idx, count in contacts.items():
                    intensity = min(255, 40 * count)
                    if vertex_colors[idx][1] != 255:
                        vertex_colors[idx] = [0, 255, 0]    
                    else:
                        vertex_colors[idx] = [255 - intensity, 255 - intensity, 255]
                    
                    
                base_label = f"/{object_library.object_id_to_name_dict[obj_id]}/{value[1][0]}~{value[1][1]}"
                rr.log(
                    f"/heatmap/{base_label}",
                    rr.Mesh3D(
                        vertex_positions=vertices_world,
                        triangle_indices=object_mesh[obj_id].faces,
                        vertex_colors=vertex_colors.tolist()
                    )
                )

                
                if flag:
                    rr.log(f"/gaze/{base_label}", rr.Clear.flat())
                    rr.log(
                    f"/gaze/{base_label}",
                    rr.Points2D(eye_gaze_reprojection_data, radii=20),
                    )
                    flag = False
                    
                    image_stream_label = device_data_provider.get_image_stream_label(stream_id)
                    # Retrieve the image data for a given timestamp
                    image_data = device_data_provider.get_image(timestamp_ns, stream_id)
                    # Visualize the image data (it's a numpy array)
                    rr.log(f"/img/{base_label}", rr.Clear.flat())
                    rr.log(f"/img/{base_label}", rr.Image(image_data))

    obj_id = object_library.object_name_to_id_dict[object_name]
    mesh_vertices = object_mesh[obj_id]
    total_counter = defaultdict(int)
    total_contact = defaultdict(int)

    # 모든 ac_id에 대해 합산
    for ac_id in vertex_hit_counter[obj_id]:
        for v_idx, count in vertex_hit_counter[obj_id][ac_id].items():
            total_counter[v_idx] += count
        for v_idx, count in contact_counter[obj_id][ac_id].items():
            total_contact[v_idx] += count       
            
    
    mesh_vertices = mesh_vertices.vertices
    # 기본 색상 설정
    vertex_colors = np.zeros((mesh_vertices.shape[0], 3), dtype=np.uint8) + 255  # 기본 흰색
    
    for idx, count in total_counter.items():
        intensity = min(255, 40 * count)
        vertex_colors[idx] = [255, 255 - intensity, 255 - intensity]  # 빨간색 계열
        
    for idx, count in total_contact.items():
        intensity = min(255, 40 * count)
        if vertex_colors[idx][1] != 255:
            vertex_colors[idx] = [0, 255, 0]    
        else:
            vertex_colors[idx] = [255 - intensity, 255 - intensity, 255]
            
    rr.log(
        f"/heatmap_t/{base_label}",
        rr.Mesh3D(
            vertex_positions=mesh_vertices,
            triangle_indices=mesh.faces,
            vertex_colors=vertex_colors.tolist()
        )
    )