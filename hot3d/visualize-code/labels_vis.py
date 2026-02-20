import argparse
import json
import os
import pickle
from typing import Dict, List

import numpy as np
import rerun as rr


def _load_obj_points(pkl_path: str, object_key: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    obj_pcs = data.get("obj_pcs", {})
    if object_key not in obj_pcs:
        raise KeyError(f"object_key not found: {object_key}")
    return np.asarray(obj_pcs[object_key], dtype=np.float32)


def _color_for_label(label) -> np.ndarray:
    fixed = {
        0: [255, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 255, 0],
        4: [255, 0, 255],
        5: [0, 255, 255],
        6: [128, 0, 0],
        7: [0, 128, 0],
        8: [0, 0, 128],
        9: [128, 128, 128],
    }
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        label_int = None
    if label_int is not None and label_int in fixed:
        return np.array(fixed[label], dtype=np.uint8)
    seed = abs(hash(str(label))) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=3, dtype=np.uint8)


def _labels_to_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for lab in np.unique(labels):
        colors[labels == lab] = _color_for_label(int(lab))
    return colors


def _load_label_merged(path: str) -> Dict[str, Dict[str, List[int]]]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid label file: {path}")
    return payload


def _map_to_labels(label_map: Dict[str, List[int]], count: int) -> np.ndarray:
    labels = np.zeros(count, dtype=np.int32)
    for i, (k, indices) in enumerate(label_map.items()):
        lab = i + 1
        for idx in indices:
            if 0 <= idx < count:
                labels[idx] = lab
    return labels


def _compute_centroids(points: np.ndarray, label_map: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    for k, indices in label_map.items():
        if not indices:
            continue
        idx = np.asarray(indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < points.shape[0])]
        if idx.size == 0:
            continue
        centroids[str(k)] = points[idx].mean(axis=0)
    return centroids


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize label_merged.json with Rerun.")
    parser.add_argument(
        "--label-merged",
        default=os.path.join(os.path.expanduser("~"), "labels_merged.json"),
        help="path to label_merged.json",
    )
    parser.add_argument(
        "--obj-pkl",
        default=os.path.join(os.path.expanduser("~"), "Desktop/hot3d_vis/obj.pkl"),
        help="path to obj.pkl",
    )
    parser.add_argument(
        "--offset-step",
        type=float,
        default=0.15,
        help="x offset between objects",
    )
    args = parser.parse_args()

    rr.init("labels_vis", spawn=True)
    label_map_all = _load_label_merged(args.label_merged)
    for idx, (object_key, label_map) in enumerate(sorted(label_map_all.items())):
        points = _load_obj_points(args.obj_pkl, object_key)
        labels = _map_to_labels(label_map, points.shape[0])
        colors = _labels_to_colors(labels)

        offset = np.array([args.offset_step * idx, 0.0, 0.0], dtype=np.float32)
        pts = points + offset

        base_path = f"{object_key}"
        rr.log(
            f"{base_path}/points",
            rr.Points3D(
                positions=pts,
                colors=colors,
                radii=0.005,
            ),
            static=True,
        )

        centroids = _compute_centroids(pts, label_map)
        label_positions = []
        label_texts = []
        label_colors = []
        all_labels = list(label_map.keys())
        obj_center = pts.mean(axis=0)
        for i, lab in enumerate(all_labels):
            center = centroids.get(str(lab))
            if center is None:
                center = obj_center + np.array([0.0, 0.0, 0.02 * (i + 1)], dtype=np.float32)
                suffix = " (empty)"
            else:
                suffix = ""
            text = f"{lab}{suffix}"
            label_positions.append(center)
            label_texts.append(text)
            label_colors.append(_color_for_label(lab).tolist())
        if label_positions:
            rr.log(
                f"{base_path}/label_text",
                rr.Points3D(
                    positions=label_positions,
                    colors=label_colors,
                    labels=label_texts,
                    radii=0.01,
                ),
                static=True,
            )


if __name__ == "__main__":
    main()
