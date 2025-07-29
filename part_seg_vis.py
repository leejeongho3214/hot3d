import glob
import json
import pickle
import torch
import tqdm
import trimesh
import numpy as np
import pandas as pd
import rerun as rr

def farthest_point_sample(xyz, npoint, random=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if random:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class ObjectModel:
    def __init__(self, pkl_file):
        pass
    
    def __call__(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_name[object_name]
        point_set = self.point_sets[object_name].copy()
        obj_pc = self.obj_pcs[object_name].copy()
        obj_pc_normal = self.obj_pc_normals[object_name].copy()
        obj_path = self.obj_path[object_name]
        return point_set, obj_pc, obj_pc_normal, obj_path
        
    def sampling(self, object_path):
        self.obj_pcs = {}
        self.obj_pc_normals = {}
        self.point_sets = {}
        self.obj_path = {}
        
        mesh = trimesh.load(object_path, process=False)
        verts = torch.FloatTensor(mesh.vertices).unsqueeze(0)
        normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0)
        normal = normal / torch.norm(normal, dim=2, keepdim=True)
        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].numpy()
        sampled_normal = normal[0, point_set[0]].numpy()
        with open("/Users/ijeongho/Desktop/instance.json", "r") as f:
            instance = json.load(f)
        # object_name = instance[str(object_path.split("/")[-1].split(".")[0])]["instance_name"]
        object_name = "mug_white"
        
        key = f"{object_name}"
        self.obj_pcs[key] = sampled_pc
        self.obj_pc_normals[key] = sampled_normal
        self.point_sets[key] = point_set[0].numpy()
        self.obj_path[key] = "/".join(object_path.split("/")[-2:])

mesh = trimesh.load('/Users/ijeongho/Desktop/part_seg/mug_white/face_labeled_mesh.ply')

object_model = ObjectModel('/Users/ijeongho/Desktop/obj.pkl')
object_model.sampling("/Users/ijeongho/Desktop/part_seg/mug_white/face_labeled_mesh.ply")
point_set, obj_pc, obj_pc_normal, obj_path = object_model("mug_white")

df = pd.read_csv('/Users/ijeongho/Desktop/part_seg/mug_white/face_labeled_rgb_mapping.csv')  # 방금 보여준 형태의 csv

num_vertices = len(mesh.vertices)
vertex_labels = np.full(num_vertices, -1)

for _, row in df.iterrows():
    v1, v2, v3, label = int(row['v1']), int(row['v2']), int(row['v3']), int(row['label'])
    for v in [v1, v2, v3]:
        vertex_labels[v] = label 
vertex_labels[vertex_labels == -1] = 0
                        
rr.init("Input Data", spawn= True)  

fixed_colors = {
    0: [255, 0, 0],    # 빨강
    1: [0, 255, 0],    # 초록
    2: [0, 0, 255],    # 파랑
    3: [255, 255, 0],  # 노랑
    4: [255, 0, 255],  # 마젠타
    5: [0, 255, 255],  # 시안
    6: [128, 0, 0],    # 진한 빨강
    7: [0, 128, 0],    # 진한 초록
    8: [0, 0, 128],    # 진한 파랑
    9: [128, 128, 128] # 회색
}

colors = np.array([
    fixed_colors.get(vertex_labels[o], [255, 255, 255]) for o in point_set
], dtype=np.uint8)

# rerun 시각화
rr.init("Input Data", spawn=True)

rr.log(
    f"world/object",
    rr.Points3D(
        positions=obj_pc,
        radii=0.005,
        colors=colors,
    )
)