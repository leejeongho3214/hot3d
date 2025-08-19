import json
import math
import os

import numpy as np
import torch


from utils import *
home = os.path.expanduser("~")

def object_vis(rr, instance_, object_model):
    rr.init("aa", spawn=True)
    for instance_id in instance_.keys():
        if instance_[instance_id]['instance_name'] not in object_model.object_name or instance_[instance_id]['instance_name'] in ["cellphone", "potato_masher"]:
            continue
        
        _, obj_pc, _, _ = object_model(instance_[instance_id]['instance_name'])
        
        rr.log(
                f"world/objects/{instance_[instance_id]['instance_name']}",
                rr.Asset3D(
                    path=os.path.join(home, f"Desktop/assets/{instance_id}.glb"),
                ),
            )
        
        rr.log(
            f"world/{instance_[instance_id]['instance_name']}",
            rr.Points3D(
                positions=obj_pc ,
                radii=0.005,
            ))
        
def data_load():
    with open(os.path.join(home, "Desktop/hot3d_vis/instance.json"), "r") as f:
        instance_ = json.load(f)

    with np.load(home + "/Desktop/hot3d_vis/mug_white.npz", allow_pickle=True) as data:
        object_idx = data["object_idx"]
        x_lhand = data["x_lhand"]
        x_rhand = data["x_rhand"]
        x_obj = data["x_obj"]
        lcov_idx = data["lcov_idx"] # left contact object verts idx
        rcf_idx = data["rcf_idx"] # right hand contact frame idx
        rcov_idx = data["rcov_idx"] # right contact object verts idx
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        gaze_map = data["gaze_map"]
        gaze = data["gaze"]
        cam_pose = data["cam_pose"]
        
    return instance_, object_idx, x_lhand, x_rhand, x_obj, lcov_idx, rcf_idx, rcov_idx, is_lhand, is_rhand, action_name, gaze_map, gaze, cam_pose

def contact_gaze(rcf_idx, rcov_idx, lcov_idx, is_lhand, is_rhand, gaze_map, gaze, cam_pose):
    cov = []
    for idx in range(len(rcf_idx)):
        lcov_map = get_contact_map(lcov_idx[idx], 1024, is_lhand[idx])
        rcov_map = get_contact_map(rcov_idx[idx], 1024, is_rhand[idx])
        cov_map = (lcov_map+rcov_map)>0
        cov_map = cov_map.astype(np.float32)
        cov.append(torch.tensor(cov_map))
        
    '''
    기존 Gaze를 camera coordinate -> world coordinate로 변환
    '''
    gaze = torch.stack(list(gaze)).squeeze(-1)
    gaze = torch.cat([gaze, torch.ones([gaze.shape[0], gaze.shape[1], gaze.shape[2],  1])], dim = -1)
    cam_pose = torch.tensor(cam_pose)

    gaze_origin_cam = torch.einsum('bfij,bfpj->bfpi', cam_pose, gaze[:, :, 0:1, :])  # (B, F, P, 4)
    gaze_origin_cam = gaze_origin_cam[..., :3]  # (B, F, P, 3)

    R = cam_pose[..., :3, :3]  # 회전 행렬만 추출
    gaze_dir_cam = torch.einsum('bfij,bfpj->bfpi', R, gaze[:, :, 1:2, :-1])  # (B, F, P, 3)
            
    gaze_index = torch.zeros([gaze_map.shape[0], gaze_map.shape[1], 1024])
    for i, g_map in enumerate(gaze_map):
        for j, num in enumerate(g_map):
            if len(num) == 0: 
                continue
            gaze_index[i][j][num] = 1 
            
    return cov, gaze_index, gaze_origin_cam, gaze_dir_cam

def angle_calcu(wanted_grip, x_obj, gaze_map, gaze_origin_cam, post_obj_pc):
    angle = []
    for idx, i in enumerate(wanted_grip):
        obj_rotmat = rot6d_to_rotmat(torch.tensor(x_obj[i][:, 3:])).reshape(-1, 3, 3)
        
        if all(x.numel() == 0 for x in gaze_map[i]): 
            continue
        
        obj_rotmat = obj_rotmat[0]
        x_axis_rot = obj_rotmat[:, 0]  # (3,)
        z_axis_rot = obj_rotmat[:, 2]  # (3,)

        origin = (torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0))).to(torch.float64)
        
        plane_normal = torch.cross(x_axis_rot, z_axis_rot)  # 또는 y_axis_rot
        plane_normal = (plane_normal / plane_normal.norm()).to(torch.float64)
        
        p0 = origin  # shape: (3,)
        n = (plane_normal / plane_normal.norm()).to(torch.float64)  # 회전된 xz 평면의 법선

        gaze_target = torch.mean(post_obj_pc[i, :30][:, gaze_map[i][-1].to(torch.bool)], dim=(0, 1))

        v = gaze_target.to(torch.float64)

        # projection
        v_proj = v - torch.dot(v - p0, n) * n
        gaze_proj_vec = v_proj - p0
        gaze_proj_vec = gaze_proj_vec / gaze_proj_vec.norm()

        gaze_vec = gaze_target - origin

        gaze_proj = gaze_vec - torch.dot(gaze_vec, plane_normal.to(torch.float64)) * plane_normal
        gaze_proj = gaze_proj / gaze_proj.norm()
        
        # 3. 각도 계산 (Gaze vector와 평면 사이 각도)
        cos_theta = torch.abs(torch.dot(gaze_vec, plane_normal))  # 절댓값: 0~1
        theta_rad = torch.arccos(cos_theta.clamp(-1.0, 1.0))       # 안전하게 clamp
        theta_deg2 = torch.rad2deg(theta_rad).item()
        
        origin_proj = origin - torch.dot(origin - origin, plane_normal.to(torch.float64)) * plane_normal.to(torch.float64)

        x_axis_unit = x_axis_rot / x_axis_rot.norm()
        cos_theta = torch.clamp(torch.dot(gaze_proj, x_axis_unit.to(torch.float64)), -1.0, 1.0)
        theta_rad = torch.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)

        angle.append([idx, theta_deg, v_proj, gaze_proj_vec, (90.0 - theta_deg2)])
    
    return angle

def hand_gaze_calcu(rr, r_hand_vertices, r_hand_faces, rmesh, r_hand_joints, gaze_origin_cam, post_obj_pc, gaze_map, i):
    # 1. 손 mesh 시각화
    rr.log("hand/vertices", rr.Mesh3D(
                vertex_positions=r_hand_vertices[0],
                triangle_indices=r_hand_faces,
                vertex_normals=rmesh.vertex_normals
    ))

    # 2. 손 관절 시각화 (joint)
    rr.log("hand/joints", rr.Points3D(
        positions=r_hand_joints[0],
        colors=[[255, 0, 0]] * r_hand_joints.shape[1],
        radii=0.005,
        labels=[f"j{i}" for i in range(r_hand_joints.shape[1])]
    ))
    # 손목 좌표
    wrist = r_hand_joints[0, 0]  # (3,)

    # 손끝 좌표 평균
    fingertip_indices = [3, 6, 9, 12, 15]
    fingertips = r_hand_joints[0, fingertip_indices]  # (5, 3)
    fingertips_mean = fingertips.mean(dim=0)  # (3,)

    # 방향 벡터
    hand_vec = fingertips_mean - wrist
    hand_vec = hand_vec / hand_vec.norm()

    # 손가락 평균 방향 벡터 시각화
    rr.log("hand/mean_finger_direction", rr.Arrows3D(
        origins=wrist[None].detach().cpu().numpy(),
        vectors=hand_vec[None].detach().cpu().numpy(),
        colors=[[0, 255, 255]],  # 하늘색
        radii=0.003,
        labels=["Hand Direction"]
    ))

    g_origin = torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0))
    g_vec = torch.mean(post_obj_pc[i, :30][:, gaze_map[i][-1].to(torch.bool)], dim = (0, 1)) -  torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0))

    rr.log(f"gaze", rr.Arrows3D(origins=[g_origin], vectors=[g_vec], colors= [[255, 255, 0]], labels=["gaze_vector"]))
    
    return None