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
import json
import rerun as rr
import numpy as np

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
import tqdm


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

class ObjectModel:
    def __init__(self, pkl_file):
        self.pkl_file = pkl_file
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            self.object_name = data["object_name"]
            self.obj_pcs = data["obj_pcs"]
            self.obj_pc_normals = data["obj_pc_normals"]
            self.point_sets = data["point_sets"]
            self.obj_path = data["obj_path"]
            if "obj_pc_top" in data:
                self.obj_pc_top = data["obj_pc_top"]
            else:
                self.obj_pc_top = None

    def __call__(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_name[object_name]
        point_set = self.point_sets[object_name].copy()
        obj_pc = self.obj_pcs[object_name].copy()
        obj_pc_normal = self.obj_pc_normals[object_name].copy()
        obj_path = self.obj_path[object_name]
        if self.obj_pc_top is not None:
            obj_pc_top = self.obj_pc_top[object_name].copy()
            return point_set, obj_pc, obj_pc_normal, obj_path, obj_pc_top
        else:
            return point_set, obj_pc, obj_pc_normal, obj_path


home = os.path.expanduser("~")
mano_hand_model_path = os.path.join(home, "Desktop/mano_v1_2/models")
mano_hand_model = None
if mano_hand_model_path is not None:
    mano_hand_model = MANOHandModel(mano_hand_model_path)

def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d

def proc_torch_frame(l):
    if isinstance(l, list) or isinstance(l, np.ndarray):
        l = [torch.FloatTensor(_l).unsqueeze(0) for _l in l]
        l = torch.cat(l)
    return l

def transform_obj_to_xdata(obj_matrix):
    orl = proc_torch_frame(obj_matrix) # object rotation list
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]
    nframes = obj_rotmat.shape[0]
    rot6d_torch = rotmat_to_rot6d(obj_rotmat).reshape(nframes, 6)
    xdata = torch.cat([obj_trans, rot6d_torch], dim=1)
    xdata = proc_numpy(xdata)
    return xdata


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


import torch
import torch.nn.functional as F

def get_points_near_ray(point_cloud, ray_origin, ray_direction, max_distance=0.01):
    """
    point_cloud: (N, 3)
    ray_origin: (3,)
    ray_direction: (3,) - normalized
    max_distance: float - threshold distance from ray
    Returns:
        matched_points: (M, 3) points within threshold distance
        matched_indices: (M,) indices of those points
    """
    vec_to_points = point_cloud - ray_origin.unsqueeze(0)  # (N, 3)

    # 투영된 t값 (Ray 상의 거리)
    t = (vec_to_points * ray_direction).sum(dim=1)  # (N,)
    t = torch.clamp(t, min=0.0)

    # Ray 상 투영점
    proj_points = ray_origin.unsqueeze(0) + t.unsqueeze(1) * ray_direction.unsqueeze(0)  # (N, 3)

    # 각 point와 투영점 사이 거리
    dists = F.pairwise_distance(point_cloud, proj_points)  # (N,)

    # 일정 거리 이하 필터링
    mask = dists < max_distance
    matched_points = point_cloud[mask]
    matched_indices = torch.where(mask)[0]

    return matched_points, matched_indices


with open(os.path.join(home, "Desktop/instance.json"), "r") as f:
    instance_ = json.load(f)

object_model = ObjectModel(os.path.join(home, "Desktop/obj.pkl"))
with open(f"{home}/Desktop/object_based3.pkl", "rb") as f:
    item = pickle.load(f)

for idx, value_list in item['mug_white'].items():
    all_matched_points = []
    
    for i in range(len(value_list['obj_pose_rts'])):
        rr.set_time_sequence("frame", i)
        obj_pose_rt_data = value_list['obj_pose_rts'][i]
        gaze_data = value_list['gaze'][i]
        
        ray_origin, ray_direction = gaze_data
        
        obj_idx = obj_pose_rt_data[0]
        object_ext = obj_pose_rt_data[1:].reshape(1, 4, 4)
        
        post_obj_pose = transform_obj_to_xdata(object_ext)
        _, obj_pc, _, _ = object_model(instance_[str(int(obj_idx))]['instance_name'])
        
        post_obj_pose = torch.tensor(post_obj_pose)
        obj_pc = torch.tensor(obj_pc)
        result_obj_verts = process_obj_result(obj_pc, post_obj_pose)
        
        colors = np.zeros_like(result_obj_verts[0], dtype=np.uint8) # (N, 3)
        matched_points, index = get_points_near_ray(
            result_obj_verts[0], torch.tensor(ray_origin).squeeze(1), torch.tensor(ray_direction).squeeze(1), max_distance=0.01
        )
        all_matched_points.append(index)

        rr.log(
            f"world/pc_pred",
            rr.Points3D(
                positions=result_obj_verts[0],
                radii=0.005,
                # colors=colors,
            )
        )    
    
    # 하나로 합치기
    all_values = torch.cat(all_matched_points)

    # unique 값과 그 인덱스 얻기
    unique_values = torch.unique(all_values, return_inverse=False, return_counts=False)
    colors[:] = [0, 0, 255]
    colors[unique_values] = [255, 0, 0]
    rr.log(
        f"world/gaze_map",
        rr.Points3D(
            positions=result_obj_verts[0],
            radii=0.005,
            colors=colors,
        )
    )   
    break
    # rr.log(f"world/gaze", rr.Arrows3D(origins=[ray_origin], vectors=[ray_direction]))    
    

# # ref_obj_vertices = item[0][-3][0]
# for batch_idx in range(len(item)):
#     rhand_vertice, rhand_face, obj, contact, pc = item[batch_idx]
    
#     mesh = trimesh.Trimesh(vertices=rhand_vertice[0], faces=rhand_face)
#     normals = mesh.vertex_normals  # (778, 3)
#             # 예: 1 → 빨간색 [255, 0, 0], 0 → 파란색 [0, 0, 255]
#     colors = np.zeros_like(pc[0], dtype=np.uint8) # (N, 3)

#     cc = [   0,    4,   12,   14,   16,   28,   30,   35,   37,   46,   49,   54,
#           55,   59,   71,   74,   75,   78,   80,   85,   89,   92,   94,   95,
#           99,  100,  102,  103,  106,  107,  119,  128,  134,  139,  147,  150,
#          155,  158,  162,  166,  170,  172,  173,  177,  180,  183,  198,  204,
#          214,  218,  220,  224,  228,  230,  235,  248,  250,  251,  257,  260,
#          263,  265,  269,  270,  273,  281,  284,  290,  296,  297,  299,  303,
#          305,  307,  311,  317,  320,  322,  323,  331,  332,  334,  342,  345,
#          346,  350,  353,  366,  373,  375,  378,  379,  382,  390,  393,  403,
#          407,  408,  410,  416,  418,  423,  435,  441,  444,  446,  463,  479,
#          481,  487,  488,  492,  500,  502,  506,  509,  513,  514,  515,  518,
#          519,  522,  528,  530,  533,  540,  544,  561,  566,  571,  572,  575,
#          577,  580,  585,  588,  589,  591,  605,  614,  617,  623,  632,  638,
#          640,  645,  650,  654,  655,  659,  662,  665,  668,  669,  672,  674,
#          680,  690,  691,  693,  698,  699,  715,  717,  722,  724,  730,  733,
#          734,  735,  736,  738,  743,  746,  747,  750,  755,  765,  769,  772,
#          773,  775,  776,  777,  785,  786,  791,  796,  803,  804,  809,  817,
#          819,  823,  828,  830,  831,  832,  833,  835,  837,  838,  844,  864,
#          865,  868,  870,  872,  883,  886,  889,  893,  894,  896,  899,  901,
#          916,  918,  922,  927,  939,  940,  941,  946,  950,  952,  964,  968,
#          974,  975,  990,  993,  994,  998, 1005, 1006, 1012, 1021, 1022, 1023]
    
#     # colors[contact[0] == 1] = [255, 0, 0]
#     # colors[contact[0] == 0] = [0, 0, 255]
    
#     colors[:] = [0, 0, 255]         
#     colors[cc] = [255, 0, 0] 
    
#     # colors2 = np.zeros_like(pc[0], dtype=np.uint8) # (N, 3)

#     # colors2[cov_map[0] == 1] = [255, 255, 0]
#     # colors2[cov_map[0] == 0] = [0, 0, 255]
    
    
#     # R, t = rigid_transform(np.array(obj[0]), np.array(ref_obj_vertices))
    
#     for frame_idx in range(len(rhand_vertice)):
#         rr.set_time_sequence("frame", frame_idx)
        
#         # # 물체 정렬
#         # cur_obj = obj[frame_idx].cpu().numpy()
#         # aligned_obj = (R @ cur_obj.T).T + t

#         # # 손 정렬
#         # cur_hand = rhand_vertice[frame_idx].cpu().numpy()
#         # aligned_hand = (R @ cur_hand.T).T + t
        
#         # rr.log(
#         #     f"world/{batch_idx}/rhand",
#         #     rr.Mesh3D(
#         #         vertex_positions=rhand_vertice[frame_idx],
#         #         triangle_indices=rhand_face,
#         #         vertex_normals=normals
#         #     ),
#         # )

#         rr.log(
#             f"world/{batch_idx}/pc_pred",
#             rr.Points3D(
#                 positions=pc[0] + torch.tensor([0.3, 0, 0]),
#                 radii=0.005,
#                 colors=colors,
#             )
#         )        
        
#         # rr.log(
#         #     f"world/{batch_idx}/pc_gt",
#         #     rr.Points3D(
#         #         positions=pc[0],
#         #         radii=0.005,
#         #         colors=colors2,
#         #     )
#         # )
        
#         # rr.log(
#         #     f"world/{batch_idx}/object_pc",
#         #     rr.Points3D(
#         #         positions=obj[frame_idx],
#         #         radii=0.005,
#         #         colors=[0, 255, 0],
#         #     )
#         # )
        
#         # rr.log(
#         #     f"world/{batch_idx}/course_hand",
#         #     rr.Mesh3D(
#         #         vertex_positions=rhand_vertice2[frame_idx],
#         #         triangle_indices=rhand_face,
#         #         vertex_normals=normals
#         #     ),
#         # )

#         # rr.log(
#         #     f"world/{batch_idx}/course_object_pc",
#         #     rr.Points3D(
#         #         positions=obj2[frame_idx],
#         #         radii=0.005,
#         #         colors=[0, 255, 0],
#         #     ))
rr.notebook_show()