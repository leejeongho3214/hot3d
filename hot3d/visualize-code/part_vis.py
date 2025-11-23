import glob
import json
import os
import pickle
import torch
import tqdm
import trimesh
import numpy as np
import pandas as pd
import rerun as rr

from utils import *


rr.init("Input Data", spawn=True)

object_model = ObjectModel('/Users/ijeongho/Desktop/hot3d_vis/obj.pkl')
object_name_list = [
    name for name in os.listdir("/Users/ijeongho/Desktop/hot3d_vis/part")
    if not name.startswith(".")
]

for object_name in object_name_list:
    # if object_model(object_name) is not None:
    point_set, obj_pc, obj_pc_normal, obj_path = object_model(object_name)
    # else: 
    #     continue
    

    df = pd.read_csv(f'/Users/ijeongho/Desktop/hot3d_vis/part/{object_name}/face_labeled_rgb_mapping.csv')  # 방금 보여준 형태의 csv
    mesh = trimesh.load(f'/Users/ijeongho/Desktop/hot3d_vis/part/{object_name}/{object_name}.ply')

    num_vertices = len(mesh.vertices)
    vertex_labels = np.full(num_vertices, -1)

    for _, row in df.iterrows():
        v1, v2, v3, label = int(row['v1']), int(row['v2']), int(row['v3']), int(row['label'])
        for v in [v1, v2, v3]:
            vertex_labels[v] = label 
    vertex_labels[vertex_labels == -1] = 0
                            
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


    rr.log(
        f"world/{object_name}",
        rr.Points3D(
            positions=obj_pc,
            radii=0.005,
            colors=colors,
        )
    )
    
    rr.log(f"world/axis", rr.Arrows3D(
            origins=[torch.mean(torch.tensor(obj_pc), dim = 0)],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"]
        ))
    
    break