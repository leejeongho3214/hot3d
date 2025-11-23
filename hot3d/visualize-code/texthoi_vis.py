
import rerun as rr
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
    
import torch
import numpy as np
from rot import *

import pickle

from data_loaders.mano_layer import MANOHandModel

from mano import build_mano_aa

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


home = os.path.expanduser("~")
mano_hand_model_path = os.path.join(home, "Desktop/hot3d_vis/mano_v1_2/models")
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


    
rr.init("Input Data", spawn= True)   

with open(f"{home}/Desktop/hot3d_vis/a.pkl", "rb") as f:
    item = pickle.load(f)
    obj_pc = item['obj_pc']
    obj_param = item['x_obj']
    l_hand, r_hand = item['x_lhand'], item['x_rhand']
    
    text = item['text']
    conv_map = item['cov_map']
    

for batch_idx in range(len(obj_pc)):
    l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
    r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)
    
    r_hand_vertices, r_hand_faces = process_hand_result(r_hand_layer, r_hand[batch_idx])
    l_hand_vertices, l_hand_faces = process_hand_result(l_hand_layer, l_hand[batch_idx])
    
    # if "right" in item['text'][batch_idx] or "Right" in item['text'][batch_idx]:
    #     hand_vertices, hand_faces = process_hand_result(r_hand_layer, r_hand[batch_idx])
    # else:
    #     hand_vertices, hand_faces = process_hand_result(l_hand_layer, l_hand[batch_idx])
        
    obj_vertices = process_obj_result(obj_pc[batch_idx], obj_param[batch_idx])
    
    # # 기준 물체: 첫 번째 시퀀스의 frame 0 (기준 위치)
    # ref_obj_vertices = process_obj_result(obj_pc[0], obj_param[0])[0].cpu().numpy()

    # # 현재 시퀀스의 frame 0 → 기준 위치로 정렬할 변환행렬
    # obj0_np = obj_vertices[0].cpu().numpy()
    # R, t = rigid_transform(obj0_np, ref_obj_vertices)

    for frame_idx in range(len(obj_param[batch_idx])):
        rr.set_time_sequence("frame", frame_idx)

        # # 물체 정렬
        # cur_obj = obj_vertices[frame_idx].cpu().numpy()
        # aligned_obj = (R @ cur_obj.T).T + t

        # # 손 정렬
        # cur_hand = hand_vertices[frame_idx].cpu().numpy()
        # aligned_hand = (R @ cur_hand.T).T + t

        # # rerun 시각화
        # rr.log(
        #     f"world/hand/{batch_idx}",
        #     rr.Mesh3D(
        #         vertex_positions=aligned_hand,
        #         triangle_indices=hand_faces,
        #     ),
        # )
        # rr.log(
        #     f"world/object_pc/{batch_idx}",
        #     rr.Points3D(
        #         positions= aligned_obj,
        #         radii=0.005,
        #         colors=[0, 255, 0],
        #     )
        # )
        rr.log(
            f"world/hand/{batch_idx}/l",
            rr.Mesh3D(
                vertex_positions=l_hand_vertices[frame_idx],
                triangle_indices=l_hand_faces,
            ),
        )
        
        rr.log(
            f"world/hand/{batch_idx}/r",
            rr.Mesh3D(
                vertex_positions=r_hand_vertices[frame_idx],
                triangle_indices=r_hand_faces,
            ),
        )
        
        rr.log(
            f"world/object_pc/{batch_idx}",
            rr.Points3D(
                positions= obj_vertices[frame_idx],
                radii=0.005,
                colors=[0, 255, 0],
            )
        )
        
        
    colors = np.zeros_like(obj_vertices[0], dtype=np.uint8) # (N, 3)
    colors[conv_map[batch_idx] == 1] = [255, 0, 0]
    colors[conv_map[batch_idx] == 0] = [0, 0, 255]
    
    labels = [text[batch_idx] ] * len(colors)

    rr.log(
        f"world/contact_map/{batch_idx}",
        rr.Points3D(
            positions=obj_vertices[0],
            radii=0.005,
            colors=colors,
            labels = labels
        )
    )