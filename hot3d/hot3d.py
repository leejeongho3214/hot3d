# Section 0: DataProvider initialization
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
from projectaria_tools.core.stream_id import StreamId
from data_loaders.mano_layer import MANOHandModel
home = os.path.expanduser("~")
hot3d_dataset_path = sequence_path = home + "/Desktop/dataset/P0009_e71e2f24"
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

# Utility functions
# Used for interactive display in the following sections
#
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


# Init a rerun context to visualize the sequence file images
rr.init("Device images", spawn=True)
stream_id = StreamId("214-1")
# How to iterate over timestamps using a slice to show one timestamp every 200
# Loop over the timestamps of the sequence and visualize corresponding data
for idx, timestamp_ns in tqdm(enumerate(timestamps[::3])):
    rr.set_time_sequence("idx", idx * 3)
    image_stream_label = device_data_provider.get_image_stream_label(stream_id)
    # Retrieve the image data for a given timestamp
    image_data = device_data_provider.get_image(timestamp_ns, stream_id)
    # Visualize the image data (it's a numpy array)
    log_image(label=f"img/{image_stream_label}", image=image_data)


#
# Retrieve Camera calibration (intrinsics and extrinsics) for a given stream_id
#
for stream_id in image_stream_ids:
    # Retrieve the camera calibration (intrinsics and extrinsics) for a given stream_id
    [extrinsics, intrinsics] = device_data_provider.get_camera_calibration(stream_id)
    print(intrinsics)
    # We will show in next section how to visualize the position of the camera in the world frame

# Showing the rerun window
rr.notebook_show()