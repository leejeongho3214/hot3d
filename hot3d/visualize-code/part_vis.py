import os
import torch
import trimesh
import numpy as np
import pandas as pd
import rerun as rr
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import *


rr.init("Input Data", spawn=True)

base_path = os.path.expanduser("~")
object_model = ObjectModel(os.path.join(base_path, "Desktop/hot3d_vis/obj.pkl"))
object_name_list = [
    name for name in os.listdir(os.path.join(base_path, 'Desktop/hot3d_vis/part'))
    if (not name.startswith(".")) if (name != "cellphone")
]

for object_name in object_name_list:
    # if object_model(object_name) is not None:
    try:
        point_set, obj_pc, obj_pc_normal, obj_path = object_model(object_name)
        # else: 
        #     continue
        
        df = pd.read_csv(os.path.join(base_path, f'Desktop/hot3d_vis/part/{object_name}/face_labeled_rgb_mapping.csv'))  # 방금 보여준 형태의 csv
        mesh = trimesh.load(os.path.join(base_path, f'Desktop/hot3d_vis/part/{object_name}/{object_name}.ply'))
    except Exception as e:
        print(f"Error loading data for {object_name}: {e}")
        continue

    num_vertices = len(mesh.vertices)
    vertex_labels = np.full(num_vertices, -1)

    try:
        for _, row in df.iterrows():
            v1, v2, v3, label = int(row['v1']), int(row['v2']), int(row['v3']), int(row['label'])
            for v in [v1, v2, v3]:
                vertex_labels[v] = label 
        vertex_labels[vertex_labels == -1] = 0

    except Exception as e:
        print(f"Error processing labels for {object_name}: {e}")
        continue
                            
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
    