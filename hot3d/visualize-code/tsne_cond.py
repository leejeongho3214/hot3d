import argparse
import hashlib
import os
import pickle
from typing import Iterable
from collections import Counter

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from sklearn.cluster import KMeans


def _color_for_text(text: str) -> list[int]:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return [int(digest[0]), int(digest[1]), int(digest[2])]


_PART_KEYWORDS = [  
    "body",
    "bottom",
    "bridge",
    "cap",
    "center",
    "edge",
    "frame",
    "handle",
    "lid",
    "long_edge",
    "rim",
    "roof",
    "short_edge",
    "tail",
    "top"
]
def _extract_part(text: str) -> str:
    lowered = text.lower()
    for keyword in _PART_KEYWORDS:
        if keyword in lowered:
            return keyword
    return "unknown"


def _extract_object(text: str) -> str:
    lowered = text.lower()
    if " of " in lowered:
        after_of = lowered.split(" of ", 1)[1]
        if " grabbed " in after_of:
            return after_of.split(" grabbed ", 1)[0].strip().split(" of a", 1)[1].strip()
        return after_of.split(" with ", 1)[0].strip()
    return "unknown"


def _extract_hand_side(text: str) -> str:
    lowered = text.lower()
    if "both" in lowered:
        return "both"
    if "right" in lowered:
        return "right"
    if "left" in lowered:
        return "left"
    return "unknown"

def _to_numpy_vector(data) -> np.ndarray:
    if torch.is_tensor(data):
        vec = data.detach().cpu().numpy()
    else:
        vec = np.asarray(data)
    return vec.reshape(-1).astype(np.float32)

def _cluster_labels(
    cluster_by: str,
    parts: list[str],
    objects: list[str],
    emb: np.ndarray,
    kmeans_k: int,
    seed: int,
) -> list[str] | None:
    if cluster_by == "part":
        return parts
    if cluster_by == "object":
        return objects
    if cluster_by == "kmeans":
        k = max(2, min(int(kmeans_k), emb.shape[0]))
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(emb)
        return [f"k{lab}" for lab in labels]
    return None

def _draw_cluster_circles(
    ax: plt.Axes,
    emb: np.ndarray,
    labels: list[str],
    alpha: float,
    linestyle: str = "--",
    linewidth: float = 1.4,
    color_for_label=None,
    min_points: int = 2,
    skip_labels: set[str] | None = None,
) -> None:
    if skip_labels is None:
        skip_labels = set()
    for label in sorted(set(labels)):
        if label in skip_labels:
            continue
        idx = [i for i, v in enumerate(labels) if v == label]
        if len(idx) < min_points:
            continue
        pts = emb[idx]
        center = pts.mean(axis=0)
        radius = np.max(np.linalg.norm(pts - center, axis=1))
        if color_for_label is None:
            color = np.array(_color_for_text(label), dtype=np.float32) / 255.0
        else:
            color = color_for_label(label)
        ax.add_patch(
            Circle(
                center,
                radius * 1.05,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
            )
        )

def _draw_nested_object_circles(
    ax: plt.Axes,
    emb: np.ndarray,
    parts: list[str],
    objects: list[str],
    alpha: float,
) -> None:
    part_to_indices: dict[str, list[int]] = {}
    for i, part in enumerate(parts):
        part_to_indices.setdefault(part, []).append(i)
    for indices in part_to_indices.values():
        obj_to_indices: dict[str, list[int]] = {}
        for i in indices:
            obj_to_indices.setdefault(objects[i], []).append(i)
        for obj, obj_indices in obj_to_indices.items():
            if len(obj_indices) < 2:
                continue
            pts = emb[obj_indices]
            center = pts.mean(axis=0)
            radius = np.max(np.linalg.norm(pts - center, axis=1))
            color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ax.add_patch(
                Circle(
                    center,
                    radius * 1.02,
                    fill=False,
                    edgecolor="white",
                    linewidth=8,
                    alpha=alpha,
                    linestyle="-",
                    zorder=4,
                )
            )
            ax.add_patch(
                Circle(
                    center,
                    radius * 1.02,
                    fill=False,
                    edgecolor=color,
                    linewidth=3,
                    alpha=alpha,
                    linestyle="-",
                    zorder=3,
                )
            )


def _load_pairs(obj) -> list[tuple[str, np.ndarray]]:
    if isinstance(obj, dict) and "save_list" in obj:
        obj = obj["save_list"]
    if not isinstance(obj, Iterable):
        raise ValueError("Input does not contain an iterable of [text, enc] pairs.")
    pairs = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text, enc = item[0], item[1]
        pairs.append((str(text), _to_numpy_vector(enc)))
    if not pairs:
        raise ValueError("No valid [text, enc] pairs found in input.")
    return pairs

home = os.path.expanduser("~")
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE visualization for cond_enc vectors.")
    parser.add_argument("--input", default=f"{home}/Desktop/hot3d_vis/clip_sty.pkl",  help="Path to pickle containing save_list.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-S3e for matplotlib.")
    parser.add_argument(
        "--color_by",
        choices=["object", "part"],
        default="object",
        help="Color grouping for points.",
    )
    parser.add_argument(
        "--cluster_by",
        choices=["none", "part", "object", "kmeans"],
        default="none",
        help="Cluster overlay grouping.",
    )
    parser.add_argument(
        "--cluster_shapes",
        nargs="+",
        choices=["circle"],
        default=["circle"],
        help="Cluster shapes to draw when cluster_by is enabled.",
    )
    parser.add_argument(
        "--kmeans_k",
        type=int,
        default=8,
        help="Number of clusters when cluster_by=kmeans.",
    )
    parser.add_argument(
        "--cluster_alpha",
        type=float,
        default=0.35,
        help="Alpha for cluster overlays.",
    )
    parser.add_argument(
        "--part_color_mode",
        choices=["limited", "all"],
        default="limited",
        help="Part coloring: limited uses body/handle/rim + other, all uses unique colors per part.",
    )
    parser.add_argument(
        "--part_stats",
        action="store_true",
        help="Print part distribution percentages and exit.",
    )
    
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE.")
    return parser.parse_args()


def main(args) -> None:

    args.output = os.path.splitext(args.input)[0].split('/')[-1] + f"_t-sne_{args.color_by}.png"
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    pairs = _load_pairs(data)
    texts = [t for t, _ in pairs]
    vectors = np.stack([v for _, v in pairs], axis=0)
    num_samples = vectors.shape[0]

    if num_samples < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")

    perplexity = min(args.perplexity, max(1.0, num_samples - 1))
    if num_samples <= 3:
        perplexity = 1.0

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=args.seed,
    )
    emb = tsne.fit_transform(vectors)
    parts = [_extract_part(t) for t in texts]
    objects = [_extract_object(t) for t in texts]
    hands = [_extract_hand_side(t) for t in texts]
    if args.part_stats:
        counts = Counter(parts)
        total = len(parts)
        for part, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            pct = (cnt / total) * 100.0
            print(f"{part}: {cnt}/{total} ({pct:.2f}%)")
        return
    part_color_map = {
        "body": np.array([31, 119, 180], dtype=np.float32) / 255.0,
        "handle": np.array([255, 127, 14], dtype=np.float32) / 255.0,
        "rim": np.array([44, 160, 44], dtype=np.float32) / 255.0,
        "top": np.array([148, 103, 189], dtype=np.float32) / 255.0,
        "frame": np.array([140, 86, 75], dtype=np.float32) / 255.0,
        "cap": np.array([227, 119, 194], dtype=np.float32) / 255.0,
        "roof": np.array([127, 127, 127], dtype=np.float32) / 255.0,
    }
    dashed_parts = set(part_color_map.keys())
    default_part_color = np.array([180, 180, 180], dtype=np.float32) / 255.0
    if args.part_color_mode == "all":
        part_color_for_label = (
            lambda label: np.array(_color_for_text(label), dtype=np.float32) / 255.0
        )
    else:
        part_color_for_label = lambda label: part_color_map.get(label, default_part_color)
    if args.color_by == "object":
        colors = np.array([_color_for_text(obj) for obj in objects], dtype=np.float32) / 255.0
    else:
        if args.part_color_mode == "all":
            colors = np.array([_color_for_text(part) for part in parts], dtype=np.float32) / 255.0
        else:
            colors = np.array([part_color_map.get(part, default_part_color) for part in parts])

    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)

    part_groups: dict[str, list[int]] = {}
    for i, part in enumerate(parts):
        part_groups.setdefault(part, []).append(i)

    def _part_color(label: str) -> np.ndarray:
        if args.part_color_mode == "all":
            return np.array(_color_for_text(label), dtype=np.float32) / 255.0
        return part_color_map.get(label, default_part_color)

    def _object_color(label: str) -> np.ndarray:
        return np.array(_color_for_text(label), dtype=np.float32) / 255.0

    part_to_objects: dict[str, list[str]] = {}
    for part, obj in zip(parts, objects):
        part_to_objects.setdefault(part, []).append(obj)
    part_to_object = {
        part: Counter(objs).most_common(1)[0][0] for part, objs in part_to_objects.items()
    }

    def _part_circle_color(label: str) -> np.ndarray:
        return _object_color(part_to_object.get(label, "unknown"))

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        s=30,
        c=colors,
        alpha=0.85,
        linewidths=0.2,
        edgecolors="black",
        zorder=1,
    )
    # for x, y, part, hand in zip(emb[:, 0], emb[:, 1], parts, hands):
        # ax.text(x, y, f"{part} / {hand}", fontsize=6, alpha=0.9)
    ax.set_title("t-SNE: cond_enc", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    if args.part_color_mode == "all":
        unique_parts = sorted(set(parts))
        part_handles = [
            Circle(
                (0, 0),
                radius=0.4,
                fill=False,
                edgecolor=_part_circle_color(part),
                linestyle="--",
                linewidth=1.2,
                label=part,
            )
            for part in unique_parts
        ]
    else:
        legend_parts = ["body", "handle", "rim", "top", "frame", "cap", "roof"]
        part_handles = [
            Circle(
                (0, 0),
                radius=0.4,
                fill=False,
                edgecolor=_part_circle_color(part),
                linestyle="--",
                linewidth=1.2,
                label=part,
            )
            for part in legend_parts
        ]
    if part_handles:
        ax.legend(
            handles=part_handles,
            title="Parts (dashed)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.12),
            frameon=True,
            fontsize=10,
            title_fontsize=10,
        )

    cluster_labels = _cluster_labels(
        args.cluster_by, parts, objects, emb, args.kmeans_k, args.seed
    )
    if cluster_labels:
        if "circle" in args.cluster_shapes:
            if args.cluster_by == "part":
                _draw_cluster_circles(
                    ax,
                    emb,
                    cluster_labels,
                    args.cluster_alpha,
                    color_for_label=_part_circle_color,
                    skip_labels=set(cluster_labels) - dashed_parts,
                )
                _draw_nested_object_circles(
                    ax, emb, parts, objects, min(args.cluster_alpha + 0.1, 0.6)
                )
            else:
                _draw_cluster_circles(ax, emb, cluster_labels, args.cluster_alpha)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(args.output, bbox_inches="tight")
    if args.show:
        plt.show()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
    
    args.input = f"{home}/Desktop/hot3d_vis/clip_ori.pkl"
    main(args)
    
    
    
