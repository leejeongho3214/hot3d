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
import glob
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

import trimesh
a = []
mesh_list = glob.glob(f"{home}/Desktop/previous/*.glb")
for idx, mesh_path in enumerate(mesh_list):
    mesh = trimesh.load(mesh_path, maintain_order=True)
    
    if isinstance(mesh, trimesh.Scene):
        # 여러 geometry가 있는 경우: Scene 형태
        vertices = []
        for name, geom in mesh.geometry.items():
            print(f"{name}: {geom}")
            vertices.append(geom.vertices)  # (N, 3)
        # 필요시 하나로 합치기
        vertices = np.vstack(vertices)
    else:
        # 단일 mesh인 경우
        vertices = mesh.vertices  # (N, 3)
        
    
    rr.log(f"world/grab/{idx}",             
           rr.Points3D(
                positions=vertices,
                radii=0.05,
            ))
    
    a.append(vertices.shape[0])
print(sorted(enumerate(a), key = lambda x: x[1]))
# mesh = trimesh.load("/home/ijeongho/Desktop/223371871635142.ply", maintain_order = True)
# rr.log(f"world/grab/{idx}",             
#         rr.Points3D(
#             positions=mesh.vertices,
#             radii=0.005,
#         ))
    
rr.notebook_show()