
import rerun as rr
import torch

import os

from utils import *

from mano import build_mano_aa
import trimesh

from function import object_vis, data_load, contact_gaze, angle_calcu, hand_gaze_calcu

    
def main():
    home = os.path.expanduser("~")
    instance_, object_idx, x_lhand, x_rhand, x_obj, lcov_idx, rcf_idx, rcov_idx, is_lhand, is_rhand, action_name, gaze_map, gaze, cam_pose = data_load()

    object_model = ObjectModel(os.path.join(home, "Desktop/hot3d_vis/obj.pkl"))
    _, obj_pc, _, _ = object_model(instance_[str(int(object_idx[0]))]['instance_name'])

    vertz = []
    for i in range(len(x_obj)):
        vertz.append(process_obj_result(torch.tensor(obj_pc), torch.tensor(x_obj[i])))
    post_obj_pc = torch.stack(vertz)

    contact_map, gaze_map, gaze_origin_cam, gaze_dir_cam = contact_gaze(rcf_idx, rcov_idx, lcov_idx, is_lhand, is_rhand, gaze_map, gaze, cam_pose)

    l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
    r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)

    wanted_grip = [idx for idx, text in enumerate(action_name) if ("handle" in text.lower()) and ("right" in text.lower())]
    
    angle = angle_calcu(wanted_grip, x_obj, gaze_map, gaze_origin_cam, post_obj_pc)
    sorted_angle = sorted(angle, key = lambda x: x[-1])
    
    for idx, (ang_idx, obj_gaze_angle, origin, vector, gaze_angle) in enumerate(sorted_angle):
        rr.set_time_sequence("frame", 0)
        i = wanted_grip[ang_idx]
        
        r_hand_vertices, r_hand_faces, r_hand_joints, r_hand_param = process_hand_result(r_hand_layer, torch.tensor(x_rhand[i]))
        l_hand_vertices, l_hand_faces, l_hand_joints, l_hand_param = process_hand_result(l_hand_layer, torch.tensor(x_lhand[i]))
        lmesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)
        rmesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
        
        obj_rotmat = rot6d_to_rotmat(torch.tensor(x_obj[i][:, 3:])).reshape(-1, 3, 3)
        world_obj_rotmat = torch.einsum('fij,fpj->fpi', torch.tensor(cam_pose[i][:, :3, :3]).to(torch.float), obj_rotmat[:, :3, :3].to(torch.float))  # (B, F, P, 4)
        
        # rr.log(
        #         f"world/{idx}/gaze_lists",
        #         rr.Arrows3D(
        #             origins=[origin.tolist()],      
        #             vectors=[vector.tolist()],
        #             radii=0.002,
        #             colors=[[255, 0, 0]],
        #             labels=["Projected Gaze (XZ-plane)"]
        #         )
        #     )
        
        # hand_gaze_calcu(rr, r_hand_vertices, r_hand_faces, rmesh, r_hand_joints, gaze_origin_cam, post_obj_pc, gaze_map, i)

        for f in range(60):
            rr.set_time_sequence("frame", f)

            g_origin = torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0))
            g_vec = torch.mean(post_obj_pc[i, :30][:, gaze_map[i][-1].to(torch.bool)], dim = (0, 1)) -  torch.mean(gaze_origin_cam[i, :30].squeeze(), dim = (0))
            
            rr.log(f"world/{idx}/gaze", rr.Arrows3D(origins=[g_origin], vectors=[g_vec], colors= [[255, 255, 0]], labels=["gaze_vector"]))
            rr.log(f"world/{idx}/ori_gaze", rr.Arrows3D(origins=[gaze_origin_cam[i][f][0][:3]], vectors=[gaze_dir_cam[i][f][0][:3]], colors= [[255, 0, 255]], labels=["gaze_map"]))
            
            rr.log(
                f"world/{idx}/gaze_point",
                rr.Points3D(
                    positions=[gaze_origin_cam[i][f][0][:3]],
                    colors=[[255, 255, 0]],
                    labels=[gaze_angle]
                )
            )
            
            # rr.log(f"world/{idx}/obj_vec",
            #     rr.Arrows3D(
            #         origins=torch.mean(post_obj_pc[i][f], dim = 0),
            #         vectors=obj_rotmat[f, 0],
            #         colors= [[0, 255, 0], [0, 0, 255]],
            #         labels=["obj_rot_x"]
            # ))
                        
            # rr.log(f"worldy/{idx}",
            #     rr.Arrows3D(
            #         # origins=post_obj_pc[i][f],
            #         origins=[[0, 0, 0]],
            #         vectors=obj_rotmat[f, 1],
            #         colors= [[0, 255, 0]],
            #         labels=[ "y_vec"]
            # ))
            
            # rr.log(f"worldz/{idx}",
            #     rr.Arrows3D(
            #         # origins=post_obj_pc[i][f],
            #         origins=[[0, 0, 0]],
            #         vectors=obj_rotmat[f, 2],
            #         colors= [[0, 0, 255]],
            #         labels=["z_vec"]
            # ))
            
            rr.log(
                f"world/{idx}/obj",
                rr.Points3D(
                    positions=post_obj_pc[i][f],
                    radii=0.005,
                    colors= [0, 0, 255]
            ))
            

            # rr.log(f"world/{idx}/rotation", rr.Arrows3D(
            #     origins=torch.mean(post_obj_pc[i][f], dim = 0).unsqueeze(0).repeat(3, 1),
            #     vectors=(obj_rotmat[f] @ torch.eye(3)).T.tolist(), 
            #     colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            #     labels=["x'", "y'", "z'"]
            # ))
                        
            rr.log(
            f"world/{idx}/rhand",
                rr.Mesh3D(
                    vertex_positions=r_hand_vertices[f],
                    triangle_indices=r_hand_faces,
                    vertex_normals=rmesh.vertex_normals
                ),)
            
            # if f < 30:
            #     colors = np.zeros_like(obj_pc, dtype=np.uint8)
            #     colors[gaze_map[i][f] == 1] = [255, 255, 0]
            #     colors[gaze_map[i][f] == 0] = [0, 0, 255]

            #     rr.log(
            #         f"world/{idx}/gaze_map",
            #         rr.Points3D(
            #             positions=post_obj_pc[i][f],
            #             radii=0.005,
            #             colors=colors,
            #         ))
                
            colors = np.zeros_like(obj_pc, dtype=np.uint8)
            colors[contact_map[i] == 1] = [255, 0, 0]
            colors[contact_map[i] == 0] = [0, 0, 255]
                    
            rr.log(
                f"world/{idx}/contact_map",
                rr.Points3D(
                    positions=post_obj_pc[i][f] + torch.tensor([0.4, 0, 0]),
                    radii=0.005,
                    colors=colors,
                ))

    rr.notebook_show()

if __name__ == "__main__":
    main()