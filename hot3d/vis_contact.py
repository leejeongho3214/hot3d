import rerun as rr
import numpy as np

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

    # 2. 기준 위치에 위치시킴 (PA-MPJPE처럼 위치만 통일)
    aligned_verts = canonical_verts + target_pos
    aligned_normals = canonical_normals + target_pos

    # 3. 로그
    rr.log(
        f"world/hand_sort/{idx}",
        rr.Mesh3D(
            vertex_positions=aligned_verts.astype(np.float32),
            triangle_indices=tris.astype(np.int32),
            vertex_normals=aligned_normals.astype(np.float32),
        ),
    )
