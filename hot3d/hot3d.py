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
from data_loaders.loader_hand_poses import Handedness
from collections import defaultdict
import os
import trimesh
from rerun.archetypes import Boxes2D
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
home = os.path.expanduser("~")
hot3d_dataset_path = sequence_path = home + "/Desktop/P0003_ebdc6ff7"
object_library_path = home +"/Desktop/assets"
mano_hand_model_path = home + "/Desktop/mano_v1_2/models"
from scipy.spatial import cKDTree
signal = True
from scipy.spatial import cKDTree
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import matplotlib.pyplot as plt

import numpy as np
import rerun as rr  # @manual

from data_loaders.hand_common import LANDMARK_CONNECTIVITY
from data_loaders.headsets import Headset
from data_loaders.loader_hand_poses import HandType
from data_loaders.loader_object_library import ObjectLibrary
from projectaria_tools.core.stream_id import StreamId  # @manual

try:
    from dataset_api import Hot3dDataProvider  # @manual
except ImportError:
    from hot3d.dataset_api import Hot3dDataProvider

from data_loaders.HandDataProviderBase import (  # @manual
    HandDataProviderBase,
    HandPose3dCollectionWithDt,
)
from data_loaders.ObjectBox2dDataProvider import (  # @manual
    ObjectBox2dCollectionWithDt,
    ObjectBox2dProvider,
)

from data_loaders.ObjectPose3dProvider import (  # @manual
    ObjectPose3dCollectionWithDt,
    ObjectPose3dProvider,
)

from projectaria_tools.core.calibration import (
    CameraCalibration,
    DeviceCalibration,
    FISHEYE624,
    LINEAR,
)
from projectaria_tools.core.mps import get_eyegaze_point_at_depth  # @manual

from projectaria_tools.core.mps.utils import (  # @manual
    filter_points_from_confidence,
    filter_points_from_count,
)

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.utils.rerun_helpers import (  # @manual
    AriaGlassesOutline,
    ToTransform3D,
)

def get_bbox_center(box):
    cx = box.left + box.width / 2
    cy = box.top + box.height / 2
    return (cx, cy)

def distance(a, b_center):
    return ((a[0] - b_center[0]) ** 2 + (a[1] - b_center[1]) ** 2) ** 0.5

def is_overlap(box1, box2):
    x1_min, x1_max = box1[0], box1[1]
    y1_min, y1_max = box1[2], box1[3]
    
    x2_min, x2_max = box2[0], box2[1]
    y2_min, y2_max = box2[2], box2[3]
    
    # 겹치는 경우: min-max 사이에 교차점이 있어야 함
    return not (x1_max < x2_min or x2_max < x1_min or
                y1_max < y2_min or y2_max < y1_min)     

class Hot3DVisualizer:
    def __init__(
        self,
        hot3d_data_provider,
        hand_type: HandType = HandType.Umetrack,
        **kargs
    ) -> None:
        
        ## Load the item to be saved
        for key, value in kargs.items():
            setattr(self, key, value)
        self.gaze_list = defaultdict(lambda: None)
        self.obj = {'r_contact_bbox': defaultdict(lambda: None), 'vertex': defaultdict(), 'l_contact_bbox': defaultdict(lambda: None)}
        self.seq = []
        self.check_no_gaze = []
        self.hand = {'right': defaultdict(lambda: None), 'left': defaultdict(lambda: None), 'r_vertex': defaultdict(), 'l_vertex': defaultdict()}
        ############################################################
        
        self._hot3d_data_provider = hot3d_data_provider
        # Device calibration and Image stream data
        self._device_data_provider = hot3d_data_provider.device_data_provider
                # Data provider at time T (for device & objects & hand poses)
        self._device_pose_provider = hot3d_data_provider.device_pose_data_provider
        self._hand_data_provider = (
            hot3d_data_provider.umetrack_hand_data_provider
            if hand_type == HandType.Umetrack
            else hot3d_data_provider.mano_hand_data_provider
        )
        if hand_type is HandType.Umetrack:
            print("Hot3DVisualizer is using UMETRACK hand model")
        elif hand_type is HandType.Mano:
            print("Hot3DVisualizer is using MANO hand model")
        self._object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
        self._object_box2d_data_provider = (
            hot3d_data_provider.object_box2d_data_provider
        )
        # Object library
        self._object_library = hot3d_data_provider.object_library

        # If required
        # Retrieve a distinct color mapping for object bounding box to show consistent color across stream_ids
        # - Use a Colormap for visualizing object bounding box
        self._object_box2d_colors = None
        if self._object_box2d_data_provider is not None:
            color_map = plt.get_cmap("viridis")
            self._object_box2d_colors = color_map(
                np.linspace(0, 1, len(self._object_box2d_data_provider.object_uids))
            )

        # Keep track of what 3D assets has been loaded/unloaded so we will load them only when needed
        self._object_cache_status = {}

        # To be parametrized later
        self._jpeg_quality = 75

    def log_dynamic_assets(
        self,
        stream_ids: List[StreamId],
        timestamp_ns: int,
        idx: int,
        flag: bool,
    ) -> None:
        """
        Log dynamic assets:
        I.e assets that are moving, such as:
        - 3D assets
        - Device pose
        - Hands
        - Object poses
        - Image related specifics assets
        - images (stream_ids)
        - Object Bounding boxes
        - Aria Eye Gaze
        """

        #
        ## Retrieve and log data that is not stream_id dependent (pure 3D data)
        #
        acceptable_time_delta = 0
        
        ## Add frame info
        self.time = idx


        hand_poses_with_dt = None
        if self._hand_data_provider is not None:
            hand_poses_with_dt = self._hand_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
                acceptable_time_delta=acceptable_time_delta,
            )

        #
        ## Log Hand poses
        #
        self.log_hands(
            "world/hands",  # /{handedness_label}/... will be added as necessary
            self._hand_data_provider,
            hand_poses_with_dt,
            timestamp_ns,
            show_hand_mesh=True,
            show_hand_vertices=False,
            show_hand_landmarks=False,
        )
        
        object_poses_with_dt = None
        if self._object_pose_data_provider is not None:
            object_poses_with_dt = (
                self._object_pose_data_provider.get_pose_at_timestamp(
                    timestamp_ns=timestamp_ns,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                    acceptable_time_delta=acceptable_time_delta,
                )
            )

        #
        ## Log stream dependent data
        #
        for stream_id in stream_ids:
            #
            ## Log Image data
            #
            
            #
            ## Eye Gaze image reprojection
            #
            if self._hot3d_data_provider.get_device_type() is Headset.Aria:
                # We are showing EyeGaze reprojection only on the RGB image stream
                if stream_id != StreamId("214-1"):
                    continue

                # Reproject EyeGaze for raw and pinhole images
                camera_configurations = FISHEYE624
                eye_gaze_reprojection_data = (
                    self._device_data_provider.get_eye_gaze_in_camera(
                        stream_id, timestamp_ns, camera_model=camera_configurations
                    )
                )
                if (
                    eye_gaze_reprojection_data is None
                    or not eye_gaze_reprojection_data.any()
                ):
                    continue

                label = (
                    f"world/device/{stream_id}/eye-gaze_projection"
                    if camera_configurations == LINEAR
                    else f"world/device/{stream_id}_raw/eye-gaze_projection_raw"
                )
                
                if camera_configurations != LINEAR: self.gaze_list[self.time] = [label, eye_gaze_reprojection_data]
                # rr.log(
                #     label,
                #     rr.Points2D(eye_gaze_reprojection_data, radii=20),
                #     # TODO consistent color and size depending of camera resolution
                # )

            # Undistorted image (required if you want see reprojected 3D mesh on the images)
            image_data = self._device_data_provider.get_undistorted_image(
                timestamp_ns, stream_id
            )
            if image_data is not None:
                rr.log(
                    f"world/device/{stream_id}",
                    rr.Image(image_data).compress(jpeg_quality=self._jpeg_quality),
                )

            # Raw device images (required for object bounding box visualization)
            image_data = self._device_data_provider.get_image(timestamp_ns, stream_id)
            if image_data is not None:
                rr.log(
                    f"world/device/{stream_id}_raw",
                    rr.Image(image_data).compress(jpeg_quality=self._jpeg_quality),
                )

            if (
                self._object_box2d_data_provider is not None
                and stream_id in self._object_box2d_data_provider.stream_ids
            ):
                box2d_collection_with_dt = (
                    self._object_box2d_data_provider.get_bbox_at_timestamp(
                        stream_id=stream_id,
                        timestamp_ns=timestamp_ns,
                        time_query_options=TimeQueryOptions.CLOSEST,
                        time_domain=TimeDomain.TIME_CODE,
                    )
                )
                
                
                if (
                    eye_gaze_reprojection_data is None
                    or not eye_gaze_reprojection_data.any()
                ): 
                    self.check_no_gaze.append(self.time)
                
                self.log_object_bounding_boxes(
                    object_poses_with_dt,
                    stream_id,
                    box2d_collection_with_dt,
                    self._object_box2d_data_provider,
                    self._object_library,
                    self._object_box2d_colors,
                    eye_gaze_reprojection_data
                )

        return self
            

    @staticmethod
    def log_aria_glasses(
        label: str,
        device_calibration: DeviceCalibration,
        use_cad_calibration: bool = True,
    ) -> None:
        ## Plot Project Aria Glasses outline (as lines)
        aria_glasses_point_outline = AriaGlassesOutline(
            device_calibration, use_cad_calibration
        )
        rr.log(label, rr.LineStrips3D([aria_glasses_point_outline]), static=True)

    @staticmethod
    def log_calibration(
        label: str,
        camera_calibration: CameraCalibration,
    ) -> None:
        rr.log(
            label,
            rr.Pinhole(
                resolution=[
                    camera_calibration.get_image_size()[0],
                    camera_calibration.get_image_size()[1],
                ],
                focal_length=float(camera_calibration.get_focal_lengths()[0]),
            ),
            static=True,
        )

    @staticmethod
    def log_pose(label: str, pose: SE3, static=False) -> None:
        rr.log(label, ToTransform3D(pose, False), static=static)

    def log_hands(
        self,
        label: str,
        hand_data_provider: HandDataProviderBase,
        hand_poses_with_dt: HandPose3dCollectionWithDt,
        timestamp_ns,
        show_hand_mesh=True,
        show_hand_vertices=True,
        show_hand_landmarks=True,
    ):
        logged_right_hand_data = False
        logged_left_hand_data = False
        if hand_poses_with_dt is None:
            return

        hand_pose_collection = hand_poses_with_dt.pose3d_collection
        hand_box2d_data_provider = hot3d_data_provider.hand_box2d_data_provider
        
        box2d_collection_with_dt = (
            hand_box2d_data_provider.get_bbox_at_timestamp(
                stream_id=StreamId("214-1"),        ## Fix the ID of the Aria glasses
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
        )
        
        
        from data_loaders.loader_hand_poses import LEFT_HAND_INDEX, RIGHT_HAND_INDEX
        # We have valid data, returned as a collection
        # i.e for each hand_uid, we retrieve its BBOX and visibility
        
        for hand_uid in [LEFT_HAND_INDEX, RIGHT_HAND_INDEX]:
            hand_name = "left" if hand_uid == LEFT_HAND_INDEX else "right"
            
            if hand_uid not in box2d_collection_with_dt.box2d_collection.box2ds.keys():
                continue
            
            axis_aligned_box2d = box2d_collection_with_dt.box2d_collection.box2ds[hand_uid]
            bbox = axis_aligned_box2d.box2d
            
            if bbox is None:
                continue
            
            self.hand[hand_name][self.time] = bbox    ## Accumulate the presence of hand bounding boxes in each frame
            
            rr.log(
                f"world/device/{stream_id}_raw/bbox/{hand_name}",
                rr.Boxes2D(
                    mins=[bbox.left, bbox.top],
                    sizes=[bbox.width, bbox.height],
                ))
        
        ## Save both hand's mesh vertex
        for hand_pose_data in hand_pose_collection.poses.values():
            if hand_pose_data.is_left_hand():
                logged_left_hand_data = True
                self.hand['l_vertex'][self.time] = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
            elif hand_pose_data.is_right_hand():
                logged_right_hand_data = True
                self.hand['r_vertex'][self.time] = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)

        # If some hand data has not been logged, do not show it in the visualizer
        if logged_left_hand_data is False:
            rr.log(f"{label}/left", rr.Clear.recursive())
        if logged_right_hand_data is False:
            rr.log(f"{label}/right", rr.Clear.recursive())

    ## Check if the hand and the object are overlapping
    @staticmethod
    def is_point_in_box(point, box):
        """“
        point: (x, y) 좌표
        box: bounding box 객체 또는 dict
            box.left, box.top, box.width, box.height가 존재해야 함
        """
        x, y = point
        x_min = box.left
        x_max = box.left + box.width
        y_min = box.top
        y_max = box.top + box.height
        
        return 1 if (x_min <= x <= x_max) and (y_min <= y <= y_max) else 0

    @staticmethod
    def get_overlap_area(box1, box2):
        """
        box1, box2는 각각 .left, .top, .width, .height를 가진 객체
        """
        x1_min = box1.left
        x1_max = box1.left + box1.width
        y1_min = box1.top
        y1_max = box1.top + box1.height

        x2_min = box2.left
        x2_max = box2.left + box2.width
        y2_min = box2.top
        y2_max = box2.top + box2.height

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        return x_overlap * y_overlap

    def log_object_bounding_boxes(
        self,
        object_poses_with_dt,
        stream_id: StreamId,
        box2d_collection_with_dt: Optional[ObjectBox2dCollectionWithDt],
        object_box2d_data_provider: ObjectBox2dProvider,
        object_library: ObjectLibrary,
        bbox_colors: np.ndarray,
        eye_gaze_reprojection_data
    ):
        """
        Object bounding boxes (valid for native raw images).
        - We assume that the image corresponding to the stream_id has been logged beforehand as 'world/device/{stream_id}_raw/'
        """

        # Keep a mapping to know what object has been seen, and which one has not
        object_uids = list(object_box2d_data_provider.object_uids)
        logging_status = {x: False for x in object_uids}

        if (
            box2d_collection_with_dt is None
            or box2d_collection_with_dt.box2d_collection is None
        ):
            # No bounding box are retrieved, we clear all the bounding box visualization existing so far
            rr.log(f"world/device/{stream_id}_raw/bbox", rr.Clear.recursive())
            return

        object_uids_at_query_timestamp = (
            box2d_collection_with_dt.box2d_collection.object_uid_list
        )
        
        ## Load object vertices once for the target sequence
        global signal
        if signal:
            signal = False
            for object_uid in hot3d_data_provider.object_box2d_data_provider.object_uids:
                object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
                    object_library_folderpath=object_library.asset_folder_name,
                    object_id=object_uid,
                )
                mesh = trimesh.load(object_cad_asset_filepath).to_geometry() 
                vertices_local = mesh.vertices  
                self.obj['vertex'][object_uid] = vertices_local
            self.obj['vertex'] = dict(self.obj['vertex'])
        
        
        for object_uid in object_uids_at_query_timestamp:
            ## Retrieve the bounding boxes of the objects in order
            object_name = object_library.object_id_to_name_dict[object_uid]
            
            axis_aligned_box2d = box2d_collection_with_dt.box2d_collection.box2ds[
                object_uid
            ]
            box = axis_aligned_box2d.box2d
            if box is None:
                continue
            
            ## Visibility ratio of the object
            visibility_ratio = axis_aligned_box2d.visibility_ratio  
            
            logging_status[object_uid] = True
            
            ## Save the visibility ratio of the object within the image.
            if visibility_ratio > 0.9: 
                self.vis_num[object_uid] += 1 
                
            ## Acquire the duration for which the gaze remains within each object’s bounding box
            inside = Hot3DVisualizer.is_point_in_box(eye_gaze_reprojection_data, box)
            if inside and not self.in_box[object_uid]:
                self.enter_time[object_uid] = self.time
                self.in_box[object_uid] = True

            elif not inside and self.in_box[object_uid]:
                self.durations[object_uid].append([self.enter_time[object_uid], self.time - 1, visibility_ratio])
                self.in_box[object_uid] = False
                
            ## Fix the issue where the duration was skipped on the last frame.
            if flag and inside:
                self.durations[object_uid].append([self.enter_time[object_uid], self.time, visibility_ratio])
            
            rr.log(
                f"world/device/{stream_id}_raw/bbox/{object_name}",
                rr.Boxes2D(
                    mins=[box.left, box.top],
                    sizes=[box.width, box.height],
                    colors=bbox_colors[object_uids.index(object_uid)],
                    labels=str(round(visibility_ratio, 2))
                ),
            )
            
        ## Define bounding box positions for each hand
        if self.hand['right'][self.time] is not None:
            r_hand = [self.hand['right'][self.time].left, self.hand['right'][self.time].left + self.hand['right'][self.time].width,
                    self.hand['right'][self.time].top, self.hand['right'][self.time].top + self.hand['right'][self.time].height]
            right_ = True
        else:
            right_ = False
            
        if self.hand['left'][self.time] is not None:
            l_hand = [self.hand['left'][self.time].left, self.hand['left'][self.time].left + self.hand['left'][self.time].width,
                    self.hand['left'][self.time].top, self.hand['left'][self.time].top + self.hand['left'][self.time].height]
            left_ = True
        else:
            left_ = False

        
        c_box = box2d_collection_with_dt.box2d_collection.box2ds
        for k, v in c_box.items():
            if v.box2d == None:
                continue
            obj = [v.box2d.left, v.box2d.left + v.box2d.width,
                v.box2d.top, v.box2d.top + v.box2d.height]
            
            overlap_left = left_ and is_overlap(l_hand, obj)
            overlap_right = right_ and is_overlap(r_hand, obj)

            if overlap_left or overlap_right:
                # 물체를 월드 좌표로 변환
                T = object_poses_with_dt.pose3d_collection.poses[k].T_world_object.to_matrix()
                vertices_local = self.obj['vertex'][k]
                vertices_homo = np.hstack([vertices_local, np.ones((vertices_local.shape[0], 1))])
                vertices_world = (T @ vertices_homo.T).T[:, :3]
                tree = cKDTree(vertices_world)

            # 왼손 접촉 처리
            if overlap_left:
                distances, _ = tree.query(self.hand['l_vertex'][self.time])
                min_dist = distances.min()
                if min_dist < 0.05:
                    existing = self.obj['l_contact_bbox'].get(self.time)
                    if existing is None or existing[-1] > min_dist:
                        self.obj['l_contact_bbox'][self.time] = [object_library.object_id_to_name_dict[k], c_box[k].box2d, min_dist]

            # 오른손 접촉 처리
            if overlap_right:
                distances, _ = tree.query(self.hand['r_vertex'][self.time])
                min_dist = distances.min()
                if min_dist < 0.05:
                    existing = self.obj['r_contact_bbox'].get(self.time)
                    if existing is None or existing[-1] > min_dist:
                        self.obj['r_contact_bbox'][self.time] = [object_library.object_id_to_name_dict[k], c_box[k].box2d, min_dist]



                    
            
        # If some object are not visible, we clear the bounding box visualization
        for key, value in logging_status.items():
            if not value:
                object_name = object_library.object_id_to_name_dict[key]
                rr.log(
                    f"world/device/{stream_id}_raw/bbox/{object_name}",
                    rr.Clear.flat(),
                )
rr.init("Hot3D", spawn=True)

stream_id = StreamId("214-1")
device_data_provider = hot3d_data_provider.device_data_provider
timestamps = device_data_provider.get_sequence_timestamps()
object_box2d_data_provider = hot3d_data_provider.object_box2d_data_provider
object_uids = list(object_box2d_data_provider.object_uids)

func_dict = {
    'vis_num':      {uid: 0      for uid in object_uids},
    'in_box':       {uid: False  for uid in object_uids},
    'enter_time':   {uid: None   for uid in object_uids},
    'durations':    {uid: []     for uid in object_uids},
}
hot3d = Hot3DVisualizer(hot3d_data_provider, HandType.Mano, **func_dict)


timestamps = timestamps[:]


flag = False
for idx, time in enumerate(timestamps):
    if idx == len(timestamps) - 1: flag = True
    rr.set_time_sequence("frame", idx)
    output = hot3d.log_dynamic_assets([stream_id], time, idx, flag)
output.vis_num = {output._object_library.object_id_to_name_dict.get(k, k): v for k, v in output.vis_num.items()}
output.durations = {output._object_library.object_id_to_name_dict.get(k, k): v for k, v in output.durations.items()}

## Define the label for the gaze point indicating the target being gazed at
gaze_logs = []
gazing_obj = []
for idx, _ in enumerate(timestamps):
    gazing_msgs = []
    for obj_name, intervals in output.durations.items():
        for start, end, vis_ratio in intervals:
            if start <= idx <= end:
                duration_frames = end - start
                gazing_msgs.append([f"Gazing at {obj_name} for {idx-start} of {duration_frames} frames", vis_ratio, obj_name])
                break  
            
    if gazing_msgs:
        if len(gazing_msgs) > 1:
            gaze_logs.append([2, max(gazing_msgs, key=lambda x: x[1])[0], max(gazing_msgs, key=lambda x: x[1])[-1], "Interacting now"])
        else:
            gaze_logs.append([2, gazing_msgs[0][0], gazing_msgs[0][-1], 0])
                
        gazing_obj.append(gazing_msgs[0][-1])
    else:
        gaze_logs.append([1, "Not Gazing", 0,0])
        gazing_obj.append(None)
        

if len(output.gaze_list) != len(gaze_logs): 
    raise ValueError(f"{len(output.gaze_list)} != {len(gaze_logs)}")

# 고유값 추출 및 인덱스 부여
unique_labels = list(set(gazing_obj))
label_to_class = {label: idx + 1 for idx, label in enumerate(unique_labels)}

# 매핑
class_id = [label_to_class[label] for label in gazing_obj]

## Need to convert defaultdict to dict where it is too volatile
output.hand.update({k: dict(v) for k, v in output.hand.items() if isinstance(v, defaultdict)})
output.obj.update({k: dict(v) for k, v in output.obj.items() if isinstance(v, defaultdict)})

# 1. hand와 object의 bbox가 겹치는 HOI 상황일 때, Gaze point가 그 안에 속해 있는 비율
#     ==>  [(Gaze_point가 안에 속해있는 frame) / (HOI 상황의 frame)]을 모두 더한 뒤, 평균을 냄 

for idx, (time, txt) in enumerate(output.gaze_list.items()):
    if idx != time:
        ValueError, f"{idx} != {time}"
    rr.set_time_sequence("frame", idx)
    rr.log(
        txt[0],
        rr.Points2D(txt[1], radii=20, labels=gaze_logs[idx][1], class_ids=class_id),
    )
    
    if idx in output.obj['l_contact_bbox'].keys():
        contact_hand = f"left_hand is grasping with {output.obj['l_contact_bbox'][idx][0]}"
        if idx in output.obj['r_contact_bbox'].keys():
            contact_hand = f"both_hand is grasping with {output.obj['r_contact_bbox'][idx][0]}"
    elif idx in output.obj['r_contact_bbox'].keys():
        contact_hand = f"right_hand is grasping with {output.obj['r_contact_bbox'][idx][0]}"
    else:
        contact_hand = None    
        
    if contact_hand:
        rr.log(
            txt[0] + "/box",
            Boxes2D(
                mins=[0, 0],  # 좌상단
                sizes=[1480, 1480],  # 해상도 맞게 조정
                colors=[255, 0, 0],
                draw_order=100,
                labels=contact_hand,
            ),
        )
    else:
        rr.log(
            txt[0] + "/box",
            rr.Clear.flat())
        
            
    
