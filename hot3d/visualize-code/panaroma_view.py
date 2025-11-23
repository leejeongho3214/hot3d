import os

import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
import rerun as rr
from tqdm import tqdm
from projectaria_tools.core.stream_id import StreamId
from scipy.spatial.transform import Rotation as R
import rerun as rr
import numpy as np
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
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
    
def pixel_to_ray(u, v, intrinsics):
    fx, fy = intrinsics.get_focal_lengths()
    cx, cy = intrinsics.get_principal_point()
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return ray / np.linalg.norm(ray)

def direction_to_equirectangular_uv(direction, pano_width, pano_height):
    x, y, z = direction
    theta = np.arctan2(x, z)
    phi = np.arcsin(y / np.linalg.norm(direction))
    u = int((theta + np.pi) / (2 * np.pi) * pano_width)
    v = int((phi + (np.pi / 2)) / np.pi * pano_height)
    return u % pano_width, np.clip(v, 0, pano_height - 1)
    
home = os.path.expanduser("~")
hot3d_dataset_path = home + "/Desktop/hot3d_vis"
sequence_path = os.path.join(hot3d_dataset_path, "P0001_4bf4e21a")
object_library_path = os.path.join(hot3d_dataset_path, "assets")
mano_hand_model_path = os.path.join(home, "Desktop/hot3d_vis/mano_v1_2/models")

if not os.path.exists(sequence_path) or not os.path.exists(object_library_path):
    print("Invalid input sequence or library path.")
    print("Please do update the path to VALID values for your system.")
    raise

object_library = load_object_library(object_library_folderpath=object_library_path)

mano_hand_model = None
if mano_hand_model_path is not None:
    mano_hand_model = MANOHandModel(mano_hand_model_path)

hot3d_data_provider = Hot3dDataProvider(
    sequence_folder=sequence_path,
    object_library=object_library,
    mano_hand_model=mano_hand_model,
)

device_data_provider = hot3d_data_provider.device_data_provider

# Retrieve the list of image stream supported by this sequence
# It will return the RGB and SLAM Left/Right image streams
image_stream_ids = device_data_provider.get_image_stream_ids()
# Retrieve a list of timestamps for the sequence (in nanoseconds)
timestamps = device_data_provider.get_sequence_timestamps()

rr.init("Device images")
rec = rr.memory_recording()

def pose_from_quat_trans(quat_wxyz, trans_xyz):
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # wxyz → xyzw
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = trans_xyz
    return T

def matrix_to_rerun_pose(T):
    r = R.from_matrix(T[:3, :3])
    q = r.as_quat()  # xyzw
    return rr.Transform3D(
        translation=T[:3, 3].tolist(),
        rotation=rr.Quaternion(xyzw=q.tolist()),
        from_parent=False
    )

pano_width = 2048
pano_height = 1024
pano_image = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
pano_mask = np.zeros((pano_height, pano_width), dtype=np.uint8)

# How to iterate over timestamps using a slice to show one timestamp every 200
timestamps_slice = slice(None, None, 200)
device_pose_provider = hot3d_data_provider.device_pose_data_provider
stream_id = StreamId("214-1")

images = []
# Loop over the timestamps of the sequence and visualize corresponding data
for timestamp_ns in tqdm(timestamps[timestamps_slice]):
    headset_pose3d_with_dt = None
    if device_pose_provider is None:
        continue
    headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )
    [extrinsics, intrinsics] = device_data_provider.get_camera_calibration(stream_id)
    if headset_pose3d_with_dt is None:
        continue

    headset_pose3d = headset_pose3d_with_dt.pose3d
    T_world_device = headset_pose3d.T_world_device
    
    [extrinsics, intrinsics] = device_data_provider.get_camera_calibration(stream_id)
    
    T_world_camera = T_world_device @ extrinsics
    T_wc_mat = T_world_camera.to_matrix3x4()
    image_np = device_data_provider.get_image(timestamp_ns, stream_id)
    if image_np is None:
        continue
    image_np = image_np.astype(np.uint8)
    images.append(image_np)
    
    
# OpenCV 스티처로 파노라마 생성
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, pano = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    cv2.imshow("Panorama", pano)
    cv2.waitKey(0)        # 창 닫기까지 대기
    cv2.destroyAllWindows()
    print("파노라마 저장 완료!")
else:
    print("파노라마 생성 실패. 상태 코드:", status)

