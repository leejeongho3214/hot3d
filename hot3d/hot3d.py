from collections import defaultdict
import json
import os
import trimesh
from rerun.archetypes import Boxes2D
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from scipy.spatial import cKDTree


from typing import List, Optional
from tqdm import tqdm
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

from projectaria_tools.core.calibration import (
    CameraCalibration,
    DeviceCalibration,
    FISHEYE624,
    LINEAR,
)

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.sophus import SE3  # @manual
from projectaria_tools.utils.rerun_helpers import (  # @manual
    AriaGlassesOutline,
    ToTransform3D,
)

home = os.path.expanduser("~")
f_root = home + "/Dataset/ljh/dataset/hot3d/full-hot3d"
folder_list = os.listdir(f_root)
stream_id = StreamId("214-1")

def find_name_obj(idx, start_idx, c_hand_list, output, txt):
    valid_keys = [k for k in range(start_idx, idx) if c_hand_list.get(k) is not None]
    object = []
    
    for valid_idx in valid_keys:
        if c_hand_list[valid_idx].split('_')[0] == "left":
            object.append(output.obj['l_contact_bbox'][valid_idx][0])
            
        elif c_hand_list[valid_idx].split('_')[0] in ["both-S", "right"]:
            object.append(output.obj['r_contact_bbox'][valid_idx][0])
            
        elif c_hand_list[valid_idx].split('_')[0] == "both-D":
            object.append(output.obj['l_contact_bbox'][valid_idx][0])
            object.append(output.obj['r_contact_bbox'][valid_idx][0])
            
            # bbox_l, bbox_r = output.obj['l_contact_bbox'][valid_idx][1], output.obj['r_contact_bbox'][valid_idx][1]
            # bbox_l_center, bbox_r_center = [int((bbox_l[0] + bbox_l[1])/2), int((bbox_l[2] + bbox_l[3])/2)], [int((bbox_r[0] + bbox_r[1])/2), int((bbox_r[2] + bbox_r[3])/2)]
            
            # if np.linalg.norm(txt[1] - bbox_l_center) < np.linalg.norm(txt[1] - bbox_r_center):
            #     c_hand = ["both-Dl", output.obj['l_contact_bbox'][valid_idx][0]]
            # else:
            #     c_hand = ["both-Dr", output.obj['r_contact_bbox'][valid_idx][0]]
    unique_obj = defaultdict(int)

    for key in object:
        unique_obj[key] += 1
        
    for key in object:
        if unique_obj[key] < 60:
            del unique_obj[key]
    
    return list(unique_obj.keys())


def is_point_in_bbox(point, bbox):
    x, y = point
    x_min = bbox.left
    x_max = bbox.left + bbox.width
    y_min = bbox.top
    y_max = bbox.top + bbox.height
    return x_min <= x <= x_max and y_min <= y <= y_max
        
def compute_section_ratio(section, output):
    start, end = section[:2]
    inside_count = 0
    total_count = 0

    for idx in range(start, end + 1):
        if idx not in output.gaze_list:
            continue  # 데이터 없는 프레임 무시

        points = output.gaze_list[idx][1]  # Points2D 좌표 배열
        l_bbox_info = output.obj['l_bbox'].get(idx)
        r_bbox_info = output.obj['r_bbox'].get(idx)

        if all(key == None for key in [l_bbox_info, r_bbox_info]):
            continue  # bbox 없는 프레임 무시

        total_count += 1
        if any(is_point_in_bbox(points, bbox_pos[0]) for bbox_pos in [l_bbox_info, r_bbox_info] if bbox_pos is not None):
            inside_count += 1
            

    return [inside_count, total_count] if total_count > 0 else 0

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
        self.obj = {'r_contact_bbox': defaultdict(lambda: None), 'vertex': defaultdict(), 'l_contact_bbox': defaultdict(lambda: None), 'l_bbox': defaultdict(lambda: None), 'r_bbox': defaultdict(lambda: None)}
        self.seq = []
        self.image = defaultdict(lambda: None)
        self.check_no_gaze = []
        self.hand = {'right': defaultdict(lambda: None), 'left': defaultdict(lambda: None), 'r_vertex': defaultdict(lambda: None), 'l_vertex': defaultdict(lambda: None)}
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
        self.flag = flag
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
                
                if camera_configurations != LINEAR: 
                    self.gaze_list[self.time] = [label, list(eye_gaze_reprojection_data)]
                # rr.log(
                #     label,
                #     rr.Points2D(eye_gaze_reprojection_data, radii=20),
                #     # TODO consistent color and size depending of camera resolution
                # )

            # # Undistorted image (required if you want see reprojected 3D mesh on the images)
            # image_data = self._device_data_provider.get_undistorted_image(
            #     timestamp_ns, stream_id
            # )
            # if image_data is not None:
            #     rr.log(
            #         f"world/device/{stream_id}",
            #         rr.Image(image_data).compress(jpeg_quality=self._jpeg_quality),
            #     )

            # # Raw device images (required for object bounding box visualization)
            image_data = self._device_data_provider.get_image(timestamp_ns, stream_id)
            if image_data is not None:
                self.image[self.time] = timestamp_ns
                
                # rr.log(
                #     f"world/device/{stream_id}_raw",
                #     rr.Image(image_data).compress(jpeg_quality=self._jpeg_quality),
                # )

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
        hand_box2d_data_provider = self._hot3d_data_provider.hand_box2d_data_provider
        
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
            for object_uid in self._hot3d_data_provider.object_box2d_data_provider.object_uids:
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
            
            ## Acquire the duration for which the gaze remains within each object’s bounding box
            inside = Hot3DVisualizer.is_point_in_box(eye_gaze_reprojection_data, box)
            if inside and not self.in_box[object_uid]:
                self.enter_time[object_uid] = self.time
                self.in_box[object_uid] = True

            elif not inside and self.in_box[object_uid]:
                self.durations[object_uid].append([self.enter_time[object_uid], self.time - 1, visibility_ratio])
                self.in_box[object_uid] = False
                
            ## Fix the issue where the duration was skipped on the last frame.
            if self.flag and inside:
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
                
                axis_aligned_box2d = box2d_collection_with_dt.box2d_collection.box2ds[k]
                box = axis_aligned_box2d.box2d
                

            # 왼손 접촉 처리
            if overlap_left and self.hand['l_vertex'][self.time] is not None:
                distances, _ = tree.query(self.hand['l_vertex'][self.time])
                min_dist = distances.min()
                if min_dist < 0.05:
                    existing = self.obj['l_contact_bbox'].get(self.time)
                    if existing is None or existing[-1] > min_dist:
                        self.obj['l_contact_bbox'][self.time] = [object_library.object_id_to_name_dict[k], obj, min_dist]
                        self.obj['l_bbox'][self.time] = [box, object_library.object_id_to_name_dict[k]]

            # 오른손 접촉 처리
            if overlap_right and self.hand['r_vertex'][self.time] is not None:
                distances, _ = tree.query(self.hand['r_vertex'][self.time])
                min_dist = distances.min()
                if min_dist < 0.05:
                    existing = self.obj['r_contact_bbox'].get(self.time)
                    if existing is None or existing[-1] > min_dist:
                        self.obj['r_contact_bbox'][self.time] = [object_library.object_id_to_name_dict[k], obj, min_dist]
                        self.obj['r_bbox'][self.time] = [box, object_library.object_id_to_name_dict[k]]
                        

        # If some object are not visible, we clear the bounding box visualization
        for key, value in logging_status.items():
            if not value:
                object_name = object_library.object_id_to_name_dict[key]
                rr.log(
                    f"world/device/{stream_id}_raw/bbox/{object_name}",
                    rr.Clear.flat(),
                )
                
                
                
def main(f_name):
    global signal
    signal = True

    save_total = defaultdict(lambda: defaultdict(list))
    sequence_path = os.path.join(f_root, f_name) 
    object_library_path = home +"/Dataset/ljh/dataset/hot3d/assets"
    mano_hand_model_path = home + "/dir/mano_v1_2/models"

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
  
    # rr.init("Hot3D", spawn=True)

    stream_id = StreamId("214-1")
    device_data_provider = hot3d_data_provider.device_data_provider
    timestamps = device_data_provider.get_sequence_timestamps()
    
    object_box2d_data_provider = hot3d_data_provider.object_box2d_data_provider
    object_uids = list(object_box2d_data_provider.object_uids)

    func_dict = {
        'in_box':       {uid: False  for uid in object_uids},
        'enter_time':   {uid: None   for uid in object_uids},
        'durations':    {uid: []     for uid in object_uids},
    }
    hot3d = Hot3DVisualizer(hot3d_data_provider, HandType.Mano, **func_dict)

    flag = False
    for idx, time in tqdm(enumerate(timestamps), desc="Step 1"):
        if idx == len(timestamps) - 1: 
            flag = True
        rr.set_time_sequence("frame", idx)
        output = hot3d.log_dynamic_assets([stream_id], time, idx, flag)
        
    ## Change the uid of object to name
    output.durations = {output._object_library.object_id_to_name_dict.get(k): v for k, v in output.durations.items()}
    
    ## Need to convert defaultdict to dict where it is too volatile
    output.gaze_list = dict(output.gaze_list)   
    output.hand.update({k: dict(v) for k, v in output.hand.items() if isinstance(v, defaultdict)})
    output.obj.update({k: dict(v) for k, v in output.obj.items() if isinstance(v, defaultdict)})

    ## Define the label for the gaze point indicating the target being gazed at
    gaze_logs = []
    gazing_obj = []
    for idx, _ in tqdm(enumerate(timestamps), desc="Step 2"):
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
            


    # 고유값 추출 및 인덱스 부여
    unique_labels = list(set(gazing_obj))
    label_to_class = {label: idx + 1 for idx, label in enumerate(unique_labels)}

    # 매핑
    class_id = [label_to_class[label] for label in gazing_obj]

    # 1. hand와 object의 bbox가 겹치는 HOI 상황일 때, Gaze point가 그 안에 속해 있는 비율
    #     ==>  [(Gaze_point가 안에 속해있는 frame) / (HOI 상황의 frame)]을 모두 더한 뒤, 평균을 냄 
    total_list = []
    c_hand_list = defaultdict(lambda: None)
    contact_flag = False
    start_idx = None
    
    for idx in tqdm(range(len(timestamps)), desc = "Step 3"):
        if idx not in output.gaze_list.keys():
            continue
        txt = output.gaze_list[idx]
        rr.set_time_sequence("frame", idx)
        rr.log(
            txt[0],
            rr.Points2D(txt[1], radii=20, labels=gaze_logs[idx][1], class_ids=[class_id[idx]]),
        )
        
        if idx in output.obj['l_contact_bbox'].keys():
            contact_hand = f"left_hand is grasping with {output.obj['l_contact_bbox'][idx][0]}"
            if idx in output.obj['r_contact_bbox'].keys():
                if output.obj['r_contact_bbox'][idx][0] == output.obj['l_contact_bbox'][idx][0]:
                    contact_hand = f"both-S_hand is grasping with same {output.obj['r_contact_bbox'][idx][0]}"
                else:
                    contact_hand = f"both-D_hand is grasping with each object, {output.obj['l_contact_bbox'][idx][0]} and {output.obj['r_contact_bbox'][idx][0]}"
        elif idx in output.obj['r_contact_bbox'].keys():
            contact_hand = f"right_hand is grasping with {output.obj['r_contact_bbox'][idx][0]}"
        else:
            contact_hand = None  
            
        c_hand_list[idx] = contact_hand
            
        if contact_hand and not contact_flag:
            start_idx = idx
            contact_flag = True
                        
        elif not contact_hand and contact_flag:
            contact_flag = False
            if (idx - start_idx) < 60:
                continue
            
            c_hand = find_name_obj(idx, start_idx, c_hand_list, output, txt)
            total_list.append([start_idx, idx - 1, c_hand])
            
        else:
            if contact_hand and idx == len(timestamps) - 1 and contact_flag and (idx - start_idx) > 60:
                c_hand = find_name_obj(idx, start_idx, c_hand_list, output, txt)
                total_list.append([start_idx, idx, c_hand])
            
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
            
    ratios = defaultdict(lambda: defaultdict(list))
    import cv2
    number_action = 0
    for section in total_list:
        key = section[-1]
        if len(key) > 1: 
            key = "&".join(key)
            o_num = "double"
        elif len(key) == 1:
            key = key[0]    
            o_num = "single"
        else:   
            key = "error"
            o_num = "error"
        
        value = compute_section_ratio(section, output)
        ratios[o_num][key].append([value, section[:2]])
        save_point = list(range(section[0], section[1], 30))
        number_action += 1
        
        for point in save_point:
            if output.image[point] is None: 
                os.makedirs(f"save_img/{f_name}", exist_ok=True)
                cv2.imwrite(os.path.join(f"save_img/{f_name}", f"{key}_{number_action}_{point}_Skip.jpg"), np.zeros((224, 224, 1)))
            
            image = output._device_data_provider.get_image(output.image[point], stream_id)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            os.makedirs(f"save_img/{o_num}/{f_name}", exist_ok=True)
            cv2.imwrite(os.path.join(f"save_img/{o_num}/{f_name}", f"{key}_{number_action}_{point}.jpg"), image)

    save_total[f_name] = ratios
    save_total = {
        key: sorted(value, key = lambda x: x[1][0]) for key, value in dict(save_total[f_name]).items()
    }    
    os.makedirs("analysis", exist_ok=True)
    with open(f"analysis/{f_name}.json", "w") as f:
        json.dump(save_total, f)
        f.close()
    
    os.makedirs("Full-analysis", exist_ok=True)
    
    output.obj['vertex'] = {output._object_library._object_id_to_name_dict[key]: [list(v) for v in value] for key, value in output.obj['vertex'].items()}
    output.obj['l_bbox'] = {key: [[value[0].left, value[0].right, value[0].top, value[0].bottom], value[1]] for key, value in output.obj['l_bbox'].items() }
    output.obj['r_bbox'] = {key: [[value[0].left, value[0].right, value[0].top, value[0].bottom], value[1]] for key, value in output.obj['r_bbox'].items() }
    
    total_dict = {
        "meta": {"total_frame": len(timestamps), "seq_name": f_name, "object": [output._object_library._object_id_to_name_dict[uid] for uid in object_uids]},
        "gaze": output.gaze_list,
        "contact": output.obj
    }

    with open(f"Full-analysis/{f_name}.json", "w") as f:
        json.dump(total_dict, f)
        f.close()
                
if __name__ == "__main__":
    home = os.path.expanduser("~")
    with open(f"{home}/dir/hot3d/obj.json", "r") as f:
        a = json.load(f)
        for f_name in a[1]:
            main(f_name.split('.')[0])