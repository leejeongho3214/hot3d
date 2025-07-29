
import json
import rerun as rr
import numpy as np

import torch

import os

from utils import *
from data_loaders.mano_layer import MANOHandModel

from mano import build_mano_aa
import trimesh

home = os.path.expanduser("~")

with open(os.path.join(home, "Desktop/instance.json"), "r") as f:
    instance_ = json.load(f)

with np.load(home + "/Desktop/mug_white.npz", allow_pickle=True) as data:
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
    
    
def main():
    # rr.init("aa", spawn=True)

    # for instance_id in instance_.keys():
    #     if instance_[instance_id]['instance_name'] not in object_model.object_name or instance_[instance_id]['instance_name'] in ["cellphone", "potato_masher"]:
    #         continue
        
    #     _, obj_pc, _, _ = object_model(instance_[instance_id]['instance_name'])
        
    #     rr.log(
    #             f"world/objects/{instance_[instance_id]['instance_name']}",
    #             rr.Asset3D(
    #                 path=os.path.join(home, f"Desktop/assets/{instance_id}.glb"),
    #             ),
    #         )
        
        # rr.log(
        #     f"world/{instance_[instance_id]['instance_name']}",
        #     rr.Points3D(
        #         positions=obj_pc ,
        #         radii=0.005,
        #     ))
        
    object_model = ObjectModel(os.path.join(home, "Desktop/obj.pkl"))
    _, obj_pc, _, _ = object_model(instance_[str(int(object_idx[0]))]['instance_name'])
    cov = []
    for idx in range(len(rcf_idx)):
        lcov_map = get_contact_map(lcov_idx[idx], 1024, is_lhand[idx])
        rcov_map = get_contact_map(rcov_idx[idx], 1024, is_rhand[idx])
        cov_map = (lcov_map+rcov_map)>0
        cov_map = cov_map.astype(np.float32)
        cov.append(torch.tensor(cov_map))

    vertz = []
    [vertz.append(process_obj_result(torch.tensor(obj_pc), torch.tensor(x_obj[i]))) for i in range(len(x_obj))]
    post_obj_pc = torch.stack(vertz)

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
            
    gaz = torch.zeros([len(gaze_map), 1024])
    for i, idx in enumerate(gaze_map):
        gaz[i][idx] = 1 

    l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
    r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

    joints = []
    wanted_grip = [idx for idx, text in enumerate(action_name) if ("Hook" in text) and ("right" in text)]
    angle = []
    for idx, i in enumerate(wanted_grip):
        r_hand_vertices, r_hand_faces, r_hand_joints = process_hand_result(r_hand_layer, torch.tensor(x_rhand[i]))
        l_hand_vertices, l_hand_faces, l_hand_joints = process_hand_result(l_hand_layer, torch.tensor(x_lhand[i]))
        lmesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)
        rmesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
        
        if "right" in action_name[i]:
            joints.append(r_hand_joints)
        else:
            joints.append(l_hand_joints)
        
        obj_rotmat = rot6d_to_rotmat(torch.tensor(x_obj[i][:, 3:])).reshape(-1, 3, 3)
        
        if len(gaze_map[i]) == 0: continue
        
        obj_rotmat = torch.mean(obj_rotmat[:30], dim = 0)
        
        gaze_vec = torch.mean(post_obj_pc[i, :30][:, gaze_map[i]], dim = (0, 1)) -  torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0, 1))
        obj_vec  = obj_rotmat[:, 0]

        # 2. 정규화
        gaze_vec = F.normalize(gaze_vec, dim=0).to(torch.float64)
        obj_vec = F.normalize(obj_vec, dim=0).to(torch.float64)

        # 3. 코사인 유사도
        cos_sim = torch.dot(gaze_vec, obj_vec).clamp(-1.0, 1.0)

        # 4. 각도 계산 (라디안 → 도)
        angle_rad = torch.acos(cos_sim)
        angle_deg = torch.rad2deg(angle_rad)

        angle.append([idx, angle_deg.item()])
        
    angle = sorted(angle, key = lambda x: x[1])
    for z, (idx, an) in enumerate(angle):
        i = wanted_grip[idx]
        r_hand_vertices, r_hand_faces, r_hand_joints = process_hand_result(r_hand_layer, torch.tensor(x_rhand[i]))
        l_hand_vertices, l_hand_faces, l_hand_joints = process_hand_result(l_hand_layer, torch.tensor(x_lhand[i]))
        lmesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)
        rmesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
        
        if "right" in action_name[i]:
            joints.append(r_hand_joints)
        else:
            joints.append(l_hand_joints)
        
        obj_rotmat = rot6d_to_rotmat(torch.tensor(x_obj[i][:, 3:])).reshape(-1, 3, 3)
        obj_quat = matrix_to_quaternion(obj_rotmat)
        
        colors = np.zeros_like(obj_pc, dtype=np.uint8) # (N, 3)
        colors[cov[i] == 1] = [255, 0, 0]
        colors[cov[i] == 0] = [0, 0, 255]

        rr.log(
            f"world/{z}/cov",
            rr.Points3D(
                positions=obj_pc,
                radii=0.005,
                colors=colors,
            ))
        
        colors = np.zeros_like(obj_pc, dtype=np.uint8) # (N, 3)
        colors[gaz[i] == 1] = [255, 255, 0]
        colors[gaz[i] == 0] = [0, 0, 255]

        rr.log(
            f"world/{z}/gaze_map",
            rr.Points3D(
                positions=obj_pc + [0.2, 0, 0],
                radii=0.005,
                colors=colors,
            ))

        for f in range(len(gaze[0])):
            rr.set_time_sequence("frame", f)

            g_origin = torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0))
            g_vec = torch.mean(post_obj_pc[i, :30][:, gaze_map[i]], dim = (0, 1)) -  torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0))
            
            rr.log(f"world/{z}/gaze", rr.Arrows3D(origins=[g_origin], vectors=[g_vec], colors= [[255, 255, 0]]))
            # rr.log(f"world/{z}/ori_gaze", rr.Arrows3D(origins=[gaze_origin_cam[i][f][0][:3]], vectors=[gaze_dir_cam[i][f][0][:3]], colors= [[255, 0, 255]]))
            
            rr.log(
                f"world/{z}/gaze_point",
                rr.Points3D(
                    positions=[gaze_origin_cam[i][f][0][:3]],
                    colors=[[255, 255, 0]],
                    labels=[an]
                )
            )
            
            rr.log(f"world/{z}/obj_vec",
                rr.Arrows3D(
                    origins=[x_obj[i][f][:3]],
                    vectors=[F.normalize(obj_vec, dim=0)],
                    colors= [[0, 255, 255]]
        ))
            
            rr.log(
            f"world/{z}/obj",
            rr.Points3D(
                positions=post_obj_pc[i][f],
                radii=0.005,
                colors= [0, 0, 255]
            ))
            
            rr.log(
            f"world/{z}/rhand",
                rr.Mesh3D(
                    vertex_positions=r_hand_vertices[f],
                    triangle_indices=r_hand_faces,
                    vertex_normals=rmesh.vertex_normals
                ),)

    rr.notebook_show()

if __name__ == main():
    main()