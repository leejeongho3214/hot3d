import json
import numpy as np
import rerun as rr

# 불러오기
with open('output.json', 'r') as f:
    data = json.load(f)

rr.init("ReplayVisualization", spawn=True)

for object_name, object_data in data.items():
    for a_id, content in object_data.items():
        base_label = f"/{object_name}/{a_id}"

        # Mesh 복원
        vertices, faces, vertex_colors = content["obj_mesh"]
        vertices = np.array(vertices)
        faces = np.array(faces)
        vertex_colors = np.array(vertex_colors)

        rr.log(
            f"{base_label}/heatmap",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=vertex_colors
            )
        )

        # Gaze 복원
        for idx, gaze in enumerate(content["gaze"]):
            rr.set_time_sequence("timestamp", idx)
            rr.log(f"{base_label}/Gaze", rr.Points2D(np.array(gaze), radii=20))

        # 이미지 복원
        for idx, img in enumerate(content["img"]):
            rr.set_time_sequence("timestamp", idx)
            rr.log(f"{base_label}/img", rr.Image(np.array(img)))