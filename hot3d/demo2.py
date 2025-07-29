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
#
# Utility functions
# Used for interactive display in the following sections
#
from matplotlib import pyplot as plt
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
    
import os
import torch
import numpy as np
from rot import *

import rerun as rr
import pickle


from data_loaders.mano_layer import MANOHandModel
import os

from mano import build_mano_aa
import trimesh

home = os.path.expanduser("~")
mano_hand_model_path = os.path.join(home, "Desktop/mano_v1_2/models")
mano_hand_model = None
if mano_hand_model_path is not None:
    mano_hand_model = MANOHandModel(mano_hand_model_path)

def process_hand_result(hand_layer, hand_params):
    hand_pose = hand_params[:, 3:]
    hand_pose = rot6d_to_axis_angle(hand_pose).reshape(-1, 48)
    hand_trans = hand_params[:, :3]
    duration = hand_trans.shape[0]
    out = hand_layer(
        global_orient=hand_pose[:, :3],
        hand_pose=hand_pose[:, 3:48],
        betas=torch.zeros((duration, 10))
    )
    hand_trans = hand_trans.unsqueeze(1)
    hand_vertices = out.vertices + hand_trans
    hand_faces = hand_layer.faces.copy().astype(np.int16)
    hand_faces = torch.LongTensor(hand_faces)
    return hand_vertices, hand_faces

def process_obj_result(obj_verts, obj_params):
    obj_trans = obj_params[:, :3]
    obj_rot6d = obj_params[:, 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
    obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(1)
    return obj_verts_transformed
    
rr.init("Input Data", spawn= True)   

def rigid_transform(A, B):
    """두 포인트 클라우드 A, B (N,3) numpy array에 대해, B로 정렬하는 R, t 반환"""
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


with open(f"{home}/Desktop/qq.pkl", "rb") as f:
    items = pickle.load(f)
    for idx, item in enumerate(items):
        hand_vertz, hand_faces, obj_vertz, pred_contact_map, ori_obj = item
        pred_contact_map = pred_contact_map[0]
        colors = np.zeros_like(obj_vertz[0], dtype=np.uint8) # (N, 3)
        colors[:] = [0, 0, 255]         
        colors[pred_contact_map == 1] = [255, 0, 0] 
        normals = trimesh.Trimesh(vertices=hand_vertz[0], faces=hand_faces, process=False)
        
        for frame_idx in range(len(obj_vertz)):
            rr.set_time_sequence("frame", frame_idx)
            
            rr.log(
                f"world/{idx}/pc_pred",
                rr.Points3D(
                    positions=obj_vertz[frame_idx],
                    radii=0.005,
                    # colors=colors,
                )
            )        
            
            rr.log(
                f"world/{idx}/hand",
                rr.Mesh3D(
                    vertex_positions=hand_vertz[frame_idx],
                    triangle_indices=hand_faces,
                    vertex_normals=normals.vertex_normals
                ),
            )
            
    #     rr.set_time_sequence("frame", 0)
    #     # rr.log(
    #     #     f"world/{idx}/ori_obj",
    #     #     rr.Points3D(
    #     #         positions=ori_obj,
    #     #         radii=0.005,
    #     #         colors=[0, 255, 0],
    #     #     ))
    # #     cmap = plt.get_cmap('jet')

    # # # 0~1 범위로 normalize 되어 있다고 가정
    # #     colors = (cmap(pred_contact_map)[:, :3] * 255).astype(np.uint8)  #
        
    #     rr.log(
    #         f"world/{idx}/contact_map",
    #         rr.Points3D(
    #             positions=obj_vertz[0] + torch.tensor([0, 0, 0.1 * idx]),
    #             radii=0.005,
    #             colors=colors,
    #         ))
rr.notebook_show()