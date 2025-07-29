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
from matplotlib import cm, pyplot as plt
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


def get_contact_map(idx, v_num, is_hand):
    contact_map = np.zeros(v_num)
    if is_hand:
        contact_map[idx] = 1
    return contact_map


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


def process_dist_map(
    max_nframes, init_frame, 
    cf_idx, cov_idx, chj_idx, 
    dist_value, is_hand
):
    dist_map = np.zeros((max_nframes, 1024, 21), dtype=np.float32)
    if is_hand:
        f_idx_filtered = np.where((init_frame<=cf_idx) & (cf_idx<init_frame+max_nframes))[0]
        cf_idx_selected = cf_idx[f_idx_filtered]
        cf_idx_moved = cf_idx_selected-init_frame
        cov_idx_selected = cov_idx[f_idx_filtered]
        chj_idx_selected = chj_idx[f_idx_filtered]
        dist_value_selected = dist_value[f_idx_filtered]
        dist_map[cf_idx_moved, cov_idx_selected, chj_idx_selected] = dist_value_selected
    return dist_map

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
with np.load(home + "/Desktop/data.npz", allow_pickle=True) as data:
    object_idx = data["object_idx"]
    x_lhand = data["x_lhand"]
    x_rhand = data["x_rhand"]
    x_obj = data["x_obj"]
    lhand_org = data["lhand_org"]
    rhand_org = data["rhand_org"]
    lcf_idx = data["lcf_idx"] # left hand contact frame idx
    lcov_idx = data["lcov_idx"] # left contact object verts idx
    lchj_idx = data["lchj_idx"] # left contact hand joints idx
    ldist_value = data["ldist_value"]
    rcf_idx = data["rcf_idx"] # right hand contact frame idx
    rcov_idx = data["rcov_idx"] # right contact object verts idx
    rchj_idx = data["rchj_idx"] # right contact hand joints idx
    rdist_value = data["rdist_value"]
    is_lhand = data["is_lhand"]
    is_rhand = data["is_rhand"]
    action_id = data["action"]
    action_name = data["action_name"]
    nframes = data["nframes"]
    gaze_map = data["gaze_map"]
    gaze = data["gaze"]
    cam_pose = data["cam_pose"]
    score_map = data["score_map"]
    # post_obj_pc = data['post_obj']
    
score_map = torch.stack(list(score_map))
cov = []
_, obj_pc, _, _ = object_model(instance_[str(int(object_idx[0]))]['instance_name'])

for idx in range(len(rcf_idx)):
    rdist_map = process_dist_map(
        60, # max_frames
        0, rcf_idx[idx], # init_frames
        rcov_idx[idx], rchj_idx[idx], 
        rdist_value[idx], is_rhand[idx])

    lcov_map = get_contact_map(lcov_idx[idx], 1024, is_lhand[idx])
    rcov_map = get_contact_map(rcov_idx[idx], 1024, is_rhand[idx])
    cov_map = (lcov_map+rcov_map)>0
    cov_map = cov_map.astype(np.float32)
    cov.append(torch.tensor(cov_map))

cov = torch.stack(cov)
gaz = torch.zeros([len(gaze_map), 1024])

for i, idx in enumerate(gaze_map):
    gaz[i][idx] = 1 
    
vertz = []
[vertz.append(process_obj_result(torch.tensor(obj_pc), torch.tensor(x_obj[i]))) for i in range(len(x_obj))]

post_obj_pc = torch.stack(vertz)

gaze = torch.stack(list(gaze)).squeeze(-1)
gaze = torch.cat([gaze, torch.ones([gaze.shape[0], gaze.shape[1], gaze.shape[2],  1])], dim = -1)
# cam_pose_inv = torch.linalg.inv(torch.tensor(cam_pose))
cam_pose = torch.tensor(cam_pose)

gaze_origin_cam = torch.einsum('bfij,bfpj->bfpi', cam_pose, gaze[:, :, 0:1, :])  # (B, F, P, 4)
gaze_origin_cam = gaze_origin_cam[..., :3]  # (B, F, P, 3)

R = cam_pose[..., :3, :3]  # 회전 행렬만 추출
gaze_dir_cam = torch.einsum('bfij,bfpj->bfpi', R, gaze[:, :, 1:2, :-1])  # (B, F, P, 3)

l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)


for i in range(len(cov)):
    r_hand_vertices, r_hand_faces = process_hand_result(r_hand_layer, torch.tensor(x_rhand[i]))
    l_hand_vertices, l_hand_faces = process_hand_result(l_hand_layer, torch.tensor(x_lhand[i]))
    lmesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)
    rmesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
    
    for f in range(len(gaze[0])):
        rr.set_time_sequence("frame", f)
        rr.log(f"world/{i}/gaze", rr.Arrows3D(origins=[gaze_origin_cam[i][f][0][:3]], vectors=[gaze_dir_cam[i][f][0][:3]]))    
        rr.log(
        f"world/{i}/obj",
        rr.Points3D(
            positions=post_obj_pc[i][f],
            radii=0.005,
            colors= [0, 0, 255]
        ))
        rr.log(
        f"world/{i}/rhand",
            rr.Mesh3D(
                vertex_positions=r_hand_vertices[f],
                triangle_indices=r_hand_faces,
                vertex_normals=rmesh.vertex_normals
            ),)
        
        rr.log(
        f"world/{i}/lhand",
            rr.Mesh3D(
                vertex_positions=l_hand_vertices[f],
                triangle_indices=l_hand_faces,
                vertex_normals=lmesh.vertex_normals
            ),)
        
    attention = score_map[i]
    ## 0, 29, 45
    with open("normal_sm", "wb") as f:
        pickle.dump(attention, f)

    # colormap 가져오기 (jet: 파랑~초록~노랑~빨강)
    cmap = plt.get_cmap('jet')

    # 0~1 범위로 normalize 되어 있다고 가정
    colors = (cmap(attention)[:, :3] * 255).astype(np.uint8)  #
    
    # colors[score_map[i] == 1] = [255, 0, 0]
    # colors[score_map[i] == 0] = [0, 0, 255]
    rr.set_time_sequence("frame", 0)
    rr.log(
        f"world/{i}/score",
        rr.Points3D(
            positions=[-1.0, -1.0, +1.0] * obj_pc,
            radii=0.005,
            colors=colors,
        ))
    
    colors = np.zeros_like(obj_pc, dtype=np.uint8) # (N, 3)
    colors[cov[i] == 1] = [255, 255, 0]
    colors[cov[i] == 0] = [0, 0, 255]

    rr.log(
        f"world/{i}/cm",
        rr.Points3D(
            positions=[-1.0, -1.0, +1.0] * obj_pc + [0.2, 0, 0],
            radii=0.005,
            colors=colors,
        ))

    
rr.notebook_show()