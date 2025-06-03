import json
import os
import numpy as np
import rerun as rr
import torch
import scipy.spatial.transform
from data_loaders.loader_object_library import ObjectLibrary
from data_loaders.loader_object_library import load_object_library

# 사용자 홈 디렉토리 경로
home = os.path.expanduser("~")

# 데이터 로딩
with open(home + "/Desktop/analysis/vase.json", "r") as f:
    serializable_list = json.load(f)

# 오브젝트 라이브러리 로딩
object_library_path = home + "/Desktop/assets"
object_library = load_object_library(object_library_folderpath=object_library_path)
object_uid = object_library.object_name_to_id_dict["vase"]

# rerun 초기화
rr.init("HOI-Contact", spawn=True)
object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
    object_library_folderpath=object_library.asset_folder_name,
    object_id=object_uid,
)

# 기준 위치 (모든 오브젝트를 이곳으로 정렬)
target_pos = np.array([0.0, 0.0, 0.0])
target_rot = np.array([0.0, 0.0, 0.0, 1.0])  # 단위 quaternion (w=1)

# 변환 함수 정의
def transform_points(points, translation, quaternion):
    rot = scipy.spatial.transform.Rotation.from_quat(quaternion)
    return rot.apply(points) + translation

def inverse_transform(points, translation, quaternion):
    rot = scipy.spatial.transform.Rotation.from_quat(quaternion)
    return rot.inv().apply(points - translation)

# 시각화 루프
for idx, (hand_, wrist_, obj_, side_) in enumerate(serializable_list):
    rr.set_time_seconds("timestamp", idx)

    # 원래 object pose
    obj_pos = np.array(obj_["translation"])
    obj_quat = np.roll(obj_["quaternion"], -1)  # wxyz → xyzw

    # hand mesh 원본
    verts = np.array(hand_[0], dtype=np.float32)
    tris = np.array(hand_[1], dtype=np.int32)
    norms = np.array(hand_[2], dtype=np.float32)

    # hand 위치를 object 기준으로 역변환 후 → target 위치로 정렬
    local_verts = inverse_transform(verts, obj_pos, obj_quat)
    final_verts = transform_points(local_verts, target_pos, target_rot)

    local_normals = inverse_transform(norms, obj_pos, obj_quat)
    final_normals = transform_points(local_normals, target_pos, target_rot)

    # 손 메시 로깅
    rr.log(
        f"world/hand",
        rr.Mesh3D(
            vertex_positions=final_verts.astype(np.float32),
            triangle_indices=tris.astype(np.int32),
            vertex_normals=final_normals.astype(np.float32),
        ),
    )

    # 오브젝트 위치를 동일 target 위치로 고정
    rr.log(
        f"world/objects/vase/{idx}",
        rr.Transform3D(
            translation=target_pos.tolist(),
            rotation=rr.Quaternion(xyzw=target_rot),
            from_parent=False,
        ),
        static=False
    )

    rr.log(
        f"world/objects/vase/{idx}",
        rr.Asset3D(
            path=object_cad_asset_filepath,
        ),
    )
    
    wrist_pos = np.array(wrist_["translation"])
    wrist_quat = np.roll(wrist_["quaternion"], -1)  # wxyz → xyzw

    # 손 메시
    verts = np.array(hand_[0], dtype=np.float32)
    tris = np.array(hand_[1], dtype=np.int32)
    norms = np.array(hand_[2], dtype=np.float32)

    # 1. 손목 기준으로 정렬 (translation + rotation 제거)
    canonical_verts = inverse_transform(verts, wrist_pos, wrist_quat)
    canonical_normals = inverse_transform(norms, wrist_pos, wrist_quat)

    # ✨ 좌우 뒤집기: 왼손일 경우 X축 뒤집기
    if side_[-1] == "left":
        canonical_verts[:, 0] *= -1
        canonical_normals[:, 0] *= -1

    # ✨ 1.5. mesh 중심 맞추기 → 완전히 겹치게 하기
    mesh_center = np.mean(canonical_verts, axis=0)
    canonical_verts -= mesh_center
    canonical_normals -= mesh_center

    # 2. 기준 위치에 위치시킴 (0,0,0 기준)
    aligned_verts = canonical_verts + target_pos
    aligned_normals = canonical_normals + target_pos

    # 3. 로그
    rr.log(
        f"world/hand_sort",
        rr.Mesh3D(
            vertex_positions=aligned_verts.astype(np.float32),
            triangle_indices=tris.astype(np.int32),
            vertex_normals=aligned_normals.astype(np.float32),
        ),
    )