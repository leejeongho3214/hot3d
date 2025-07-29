import json
import rerun as rr
import numpy as np

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
import trimesh


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

with open(os.path.join(home, "Desktop/instance.json"), "r") as f:
    instance_ = json.load(f)
    
object_model = ObjectModel(os.path.join(home, "Desktop/obj.pkl"))
_, obj_pc, _, _ = object_model("mug_patterned")
obj_pc = torch.tensor(obj_pc)


def main():
    with open(f"{home}/Desktop/q.pkl", "rb") as f:
        item = pickle.load(f)
        l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
        r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)
        order = 0
        for x_lhand, x_rhand, x_obj, text in item:
            for batch_idx in range(len(x_lhand)):
                # if ("right" in text[batch_idx] or "Right" in text[batch_idx]):
                #     hand_vertices, hand_faces = process_hand_result(r_hand_layer, x_rhand[batch_idx])
                # else:
                #     hand_vertices, hand_faces = process_hand_result(l_hand_layer, x_lhand[batch_idx])
                    
                r_hand_vertices, r_hand_faces = process_hand_result(r_hand_layer, x_rhand[batch_idx])
                l_hand_vertices, l_hand_faces = process_hand_result(l_hand_layer, x_lhand[batch_idx])
                
                # mesh = trimesh.Trimesh(vertices=hand_vertices[0], faces=hand_faces, process=False)
                r_mesh = trimesh.Trimesh(vertices=r_hand_vertices[0], faces=r_hand_faces, process=False)
                l_mesh = trimesh.Trimesh(vertices=l_hand_vertices[0], faces=l_hand_faces, process=False)
                    
                obj_vertices = process_obj_result(obj_pc, x_obj[batch_idx])
                
                for frame_idx in range(obj_vertices.shape[0]):
                    rr.set_time_sequence("frame", frame_idx)
                    rr.log(
                        f"world/r_hand/{order}",
                        rr.Mesh3D(
                            vertex_positions=r_hand_vertices[frame_idx],
                            triangle_indices=r_hand_faces,
                            vertex_normals=r_mesh.vertex_normals
                        ),
                    )
                    
                    rr.log(
                        f"world/l_hand/{order}",
                        rr.Mesh3D(
                            vertex_positions=l_hand_vertices[frame_idx],
                            triangle_indices=l_hand_faces,
                            vertex_normals=l_mesh.vertex_normals
                        ),
                    )
                                    
                    rr.log(
                        f"world/object/{order}",
                        rr.Points3D(
                            positions=obj_vertices[frame_idx],
                            radii=0.005,
                            colors=[0, 255, 0],
                            labels=[text[batch_idx]]
                        )
                    )
                order += 1
                
                # if order == 10:
                #     return 


if __name__ == main():
    main()
    rr.notebook_show()