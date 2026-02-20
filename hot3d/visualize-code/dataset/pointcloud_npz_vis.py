import argparse
import os
import site
import math
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_VIS_DIR = Path.home() / "Desktop" / "hot3d_vis"
DEFAULT_NPZ_PATH = DEFAULT_VIS_DIR / "pointcloud_partseg_labeled_rerun.npz"
DEFAULT_GT_NPZ_PATH = DEFAULT_VIS_DIR / "gt_rerun.npz"
DEFAULT_PRED_NPZ_PATH = DEFAULT_VIS_DIR / "pred_rerun.npz"


def _load_npz(npz_path: Path):
    data = np.load(str(npz_path), allow_pickle=True)

    # Single cloud format.
    if "points_xyz" in data.files and "labels" in data.files:
        points = np.asarray(data["points_xyz"], dtype=np.float32)
        labels = np.asarray(data["labels"]).reshape(-1)
        colors_rgb = (
            np.asarray(data["colors_rgb"], dtype=np.uint8)
            if "colors_rgb" in data.files
            else None
        )
        point_text_labels = (
            np.asarray(data["point_text_labels"]).astype(str)
            if "point_text_labels" in data.files
            else None
        )
        label_text = (
            np.asarray(data["label_text"]).astype(str)
            if "label_text" in data.files
            else None
        )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_xyz must be (N,3), got {points.shape}")
        if labels.shape[0] != points.shape[0]:
            raise ValueError("labels length must match points_xyz length")
        if colors_rgb is not None:
            if colors_rgb.shape != (points.shape[0], 3):
                raise ValueError(
                    f"colors_rgb must be (N,3), got {colors_rgb.shape} (N={points.shape[0]})"
                )
        if point_text_labels is not None:
            if point_text_labels.shape[0] != points.shape[0]:
                raise ValueError(
                    f"point_text_labels length must match points_xyz length (N={points.shape[0]})"
                )
        return {
            "mode": "single",
            "points": points,
            "labels": labels,
            "colors_rgb": colors_rgb,
            "point_text_labels": point_text_labels,
            "label_text": label_text,
        }

    # Partseg batched format.
    required = ["points_xyz_list", "labels_list"]
    if all(k in data.files for k in required):
        object_keys = (
            np.asarray(data["object_keys"]).astype(str)
            if "object_keys" in data.files
            else None
        )
        points_list = data["points_xyz_list"]
        labels_list = data["labels_list"]
        labels_local_list = data["labels_local_list"] if "labels_local_list" in data.files else None
        return {
            "mode": "partseg",
            "object_keys": object_keys,
            "points_list": points_list,
            "labels_list": labels_list,
            "labels_local_list": labels_local_list,
        }

    raise KeyError("Unsupported npz format: expected single or partseg keys.")


def _require_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Rerun import failed. Install/activate environment with rerun-sdk."
        ) from exc

    if not all(hasattr(rr, name) for name in ("init", "log", "Points3D")):
        raise RuntimeError(
            "Loaded `rerun` module does not expose SDK APIs (init/log/Points3D). "
            "Install the official rerun-sdk package in this environment."
        )
    # Ensure bundled viewer executable is discoverable for rr.init(..., spawn=True).
    rerun_cli_dir = os.path.join(
        site.getusersitepackages(), "rerun_sdk", "rerun_cli"
    )
    rerun_cli_exe = os.path.join(rerun_cli_dir, "rerun")
    if os.path.isfile(rerun_cli_exe):
        cur_path = os.environ.get("PATH", "")
        parts = cur_path.split(":") if cur_path else []
        if rerun_cli_dir not in parts:
            os.environ["PATH"] = f"{rerun_cli_dir}:{cur_path}" if cur_path else rerun_cli_dir
    return rr


def _colors_for_labels(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    colors[labels == 1] = np.array([255, 0, 0], dtype=np.uint8)    # red
    colors[labels == 12] = np.array([0, 120, 255], dtype=np.uint8)  # blue
    return colors


def _colors_for_any_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.reshape(-1)
    uniq = np.unique(labels)
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for i, lab in enumerate(uniq.tolist()):
        # Deterministic pseudo-color by class id.
        base = int(lab) * 2654435761 & 0xFFFFFFFF
        r = (base >> 16) & 255
        g = (base >> 8) & 255
        b = base & 255
        # Avoid too-dark colors.
        col = np.array([max(40, r), max(40, g), max(40, b)], dtype=np.uint8)
        colors[labels == lab] = col
    return colors


def _pseudo_two_parts(points: np.ndarray) -> np.ndarray:
    """Fallback visualization-only split when labels are collapsed to one class."""
    center = points.mean(axis=0, keepdims=True)
    x = points - center
    cov = x.T @ x
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    proj = x @ axis
    thr = np.median(proj)
    out = np.zeros(points.shape[0], dtype=np.int32)
    out[proj >= thr] = 1
    return out


def _group_object_indices(point_text_labels: Optional[np.ndarray], n_points: int) -> dict[str, np.ndarray]:
    if point_text_labels is None:
        return {"all": np.arange(n_points, dtype=np.int64)}
    labels = np.asarray(point_text_labels).reshape(-1).astype(str)
    if labels.shape[0] != n_points:
        return {"all": np.arange(n_points, dtype=np.int64)}

    # Expected pattern in this repo: "gt:<object_name>" or "pred:<object_name>".
    obj = np.array(
        [s.split(":", 1)[1] if ":" in s else s for s in labels.tolist()],
        dtype=object,
    )
    uniq = np.unique(obj)
    # Avoid exploding the Rerun tree if labels are overly granular.
    if uniq.size == 0 or uniq.size > 200:
        return {"all": np.arange(n_points, dtype=np.int64)}

    out: dict[str, np.ndarray] = {}
    for name in uniq.tolist():
        idx = np.flatnonzero(obj == name).astype(np.int64)
        if idx.size:
            out[str(name)] = idx
    return out or {"all": np.arange(n_points, dtype=np.int64)}


def _partseg_items(pack, item_index: int, use_local_labels: bool):
    count = len(pack["points_list"])
    if item_index >= count:
        raise IndexError(f"item-index out of range: {item_index} (count={count})")
    indices = list(range(count)) if item_index < 0 else [item_index]
    for idx in indices:
        points_raw = np.asarray(pack["points_list"][idx], dtype=np.float32)
        labels_global = np.asarray(pack["labels_list"][idx]).reshape(-1)
        labels_local = None
        if pack["labels_local_list"] is not None:
            labels_local = np.asarray(pack["labels_local_list"][idx]).reshape(-1)
        if labels_global.shape[0] != points_raw.shape[0]:
            raise ValueError(
                f"labels_list[{idx}] length must match points_xyz_list[{idx}]"
            )
        labels = labels_local if (use_local_labels and labels_local is not None) else labels_global
        if labels.shape[0] != points_raw.shape[0]:
            raise ValueError(f"labels length mismatch at item {idx}")
        name = str(idx)
        if pack["object_keys"] is not None and idx < len(pack["object_keys"]):
            name = str(pack["object_keys"][idx])
        yield idx, name, points_raw, labels


def export_ply(points: np.ndarray, colors: Optional[np.ndarray], out_path: Path) -> None:
    if colors is None:
        colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (points.shape[0], 1))
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def visualize_npz(
    npz_path: Path,
    spawn: bool = True,
    item_index: int = -1,
    use_local_labels: bool = True,
    fallback_split: bool = True,
    radii: float = 0.02,
    show_point_text_labels: bool = False,
) -> None:
    rr = _require_rerun()
    pack = _load_npz(npz_path)

    if pack["mode"] == "single":
        points = pack["points"]
        labels = pack["labels"]
        colors_rgb = pack.get("colors_rgb")
        if colors_rgb is not None:
            colors = colors_rgb
            points_show = points
            text_labels = pack.get("point_text_labels") if show_point_text_labels else None
        else:
            target_mask = np.isin(labels, np.array([1, 12], dtype=labels.dtype))
            if np.any(target_mask):
                points_show = points[target_mask]
                labels = labels[target_mask]
                colors = _colors_for_labels(labels)
                text_labels = None
            else:
                points_show = points
                colors = _colors_for_any_labels(labels)
                text_labels = pack.get("point_text_labels") if show_point_text_labels else None
        rr.init("Pointcloud Viewer", spawn=spawn)
        rr.log(
            "cloud/main",
            rr.Points3D(
                positions=points_show,
                colors=colors,
                radii=radii,
                labels=text_labels,
            ),
        )
        return

    rr.init("Pointcloud PartSeg Viewer", spawn=spawn)
    items = list(_partseg_items(pack, item_index=item_index, use_local_labels=use_local_labels))
    if len(items) == 0:
        return
    max_extent = 0.0
    for _, _, pts, _ in items:
        extent = np.ptp(pts, axis=0).max()
        if float(extent) > max_extent:
            max_extent = float(extent)
    spacing = max(0.2, max_extent * 2.5)
    cols = int(math.ceil(math.sqrt(len(items))))

    for i, (_, name, points_raw, labels) in enumerate(items):
        labels_show = labels
        if fallback_split and np.unique(labels_show).size <= 1:
            labels_show = _pseudo_two_parts(points_raw)
        colors = _colors_for_any_labels(labels_show)
        if len(items) > 1:
            row = i // cols
            col = i % cols
            offset = np.array([col * spacing, -row * spacing, 0.0], dtype=np.float32)
            points_show = points_raw + offset
        else:
            points_show = points_raw
        rr.log(
            f"cloud/{name}",
            rr.Points3D(
                positions=points_show,
                colors=colors,
                radii=radii,
            ),
        )


def visualize_npz_compare(
    gt_npz_path: Path,
    pred_npz_path: Path,
    spawn: bool = True,
    offset_x: float = 0.5,
    radii: float = 0.02,
    show_point_text_labels: bool = False,
) -> None:
    rr = _require_rerun()
    gt_pack = _load_npz(gt_npz_path)
    pred_pack = _load_npz(pred_npz_path)

    if gt_pack["mode"] != "single" or pred_pack["mode"] != "single":
        raise ValueError("compare mode currently supports only single-cloud npz (points_xyz + labels).")

    gt_points = np.asarray(gt_pack["points"], dtype=np.float32)
    pred_points = np.asarray(pred_pack["points"], dtype=np.float32)
    gt_colors = gt_pack.get("colors_rgb")
    pred_colors = pred_pack.get("colors_rgb")

    # Default to label-driven colors if colors_rgb isn't provided.
    if gt_colors is None:
        gt_colors = _colors_for_any_labels(np.asarray(gt_pack["labels"]).reshape(-1))
    if pred_colors is None:
        pred_colors = _colors_for_any_labels(np.asarray(pred_pack["labels"]).reshape(-1))

    # Auto spacing if offset is not provided or too small.
    gt_extent = float(np.ptp(gt_points, axis=0).max()) if gt_points.size else 0.0
    pred_extent = float(np.ptp(pred_points, axis=0).max()) if pred_points.size else 0.0
    extent = max(gt_extent, pred_extent, 1e-6)
    spacing = max(float(offset_x), extent * 2.5)

    gt_offset = np.array([spacing * 0.5, 0.0, 0.0], dtype=np.float32)
    pred_offset = np.array([spacing * 0.5, 0.0, 0.0], dtype=np.float32)

    rr.init("Pointcloud Compare Viewer", spawn=spawn)

    gt_groups = _group_object_indices(gt_pack.get("point_text_labels"), gt_points.shape[0])
    pred_groups = _group_object_indices(pred_pack.get("point_text_labels"), pred_points.shape[0])
    # Prefer logging per-object if possible. If grouping falls back to "all", we still log that.
    for obj_name, idx in gt_groups.items():
        rr.log(
            f"cloud/gt/{obj_name}",
            rr.Points3D(
                positions=gt_points[idx] - gt_offset,
                colors=gt_colors[idx],
                radii=radii,
                labels=(gt_pack.get("point_text_labels")[idx] if show_point_text_labels else None),
            ),
        )
    for obj_name, idx in pred_groups.items():
        rr.log(
            f"cloud/pred/{obj_name}",
            rr.Points3D(
                positions=pred_points[idx] + pred_offset,
                colors=pred_colors[idx],
                radii=radii,
                labels=(pred_pack.get("point_text_labels")[idx] if show_point_text_labels else None),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize pointcloud npz with Rerun")
    parser.add_argument(
        "npz_path",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "path to a single npz file. If omitted, defaults to compare mode "
            f"using {DEFAULT_GT_NPZ_PATH} and {DEFAULT_PRED_NPZ_PATH} (if they exist)."
        ),
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        type=Path,
        default=None,
        metavar=("GT_NPZ", "PRED_NPZ"),
        help=(
            "compare two single-cloud npz files side-by-side in one viewer. "
            "If provided with no args, uses default gt/pred paths under ~/Desktop/hot3d_vis."
        ),
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="do not auto-open rerun viewer window",
    )
    parser.add_argument(
        "--export-ply",
        type=Path,
        default=None,
        help="export colored pointcloud to ascii .ply (for Meshlab/CloudCompare)",
    )
    parser.add_argument(
        "--skip-rerun",
        action="store_true",
        help="only export/analyze without opening rerun",
    )
    parser.add_argument(
        "--item-index",
        type=int,
        default=-1,
        help="for partseg npz: item index to visualize, -1 means all objects",
    )
    parser.add_argument(
        "--global-labels",
        action="store_true",
        help="for partseg npz: use labels_list instead of labels_local_list",
    )
    parser.add_argument(
        "--no-fallback-split",
        action="store_true",
        help="disable pseudo 2-part split when labels collapse to a single class",
    )
    parser.add_argument(
        "--radii",
        type=float,
        default=0.002,
        help="point radius in Rerun (default: 0.02)",
    )
    parser.add_argument(
        "--show-point-text-labels",
        action="store_true",
        help="if npz contains point_text_labels, pass them to rr.Points3D(labels=...)",
    )
    parser.add_argument(
        "--offset-x",
        type=float,
        default=0.5,
        help="for --compare: horizontal offset (auto-adjusted if too small; default: 0.5)",
    )
    args = parser.parse_args()

    if args.compare is not None:
        if len(args.compare) == 0:
            gt_path, pred_path = DEFAULT_GT_NPZ_PATH, DEFAULT_PRED_NPZ_PATH
        elif len(args.compare) == 2:
            gt_path, pred_path = args.compare
        else:
            raise ValueError("--compare expects either 0 args (use defaults) or 2 args (GT_NPZ PRED_NPZ).")
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"PRED file not found: {pred_path}")
        if args.export_ply is not None:
            raise ValueError("--export-ply is not supported with --compare (export from single mode instead).")
        if not args.skip_rerun:
            visualize_npz_compare(
                gt_path,
                pred_path,
                spawn=not args.no_spawn,
                offset_x=args.offset_x,
                radii=args.radii,
                show_point_text_labels=args.show_point_text_labels,
            )
        return

    if args.npz_path is None:
        # Default behavior: if no positional is provided, compare gt vs pred.
        gt_path, pred_path = DEFAULT_GT_NPZ_PATH, DEFAULT_PRED_NPZ_PATH
        if gt_path.exists() and pred_path.exists():
            if not args.skip_rerun:
                visualize_npz_compare(
                    gt_path,
                    pred_path,
                    spawn=not args.no_spawn,
                    offset_x=args.offset_x,
                    radii=args.radii,
                    show_point_text_labels=args.show_point_text_labels,
                )
            return

        # Fallback: if compare inputs don't exist, try the single-file default.
        if DEFAULT_NPZ_PATH.exists():
            args.npz_path = DEFAULT_NPZ_PATH
        else:
            raise FileNotFoundError(
                "No input provided and default files were not found.\n"
                f"- compare defaults: {gt_path} , {pred_path}\n"
                f"- single default: {DEFAULT_NPZ_PATH}\n"
                "Pass a file explicitly, e.g.:\n"
                "python pointcloud_npz_vis.py /path/to/file.npz\n"
                "or compare explicitly, e.g.:\n"
                "python pointcloud_npz_vis.py --compare /path/to/gt.npz /path/to/pred.npz"
            )

    if not args.npz_path.exists():
        raise FileNotFoundError(
            f"File not found: {args.npz_path}\n"
            f"Pass path explicitly, e.g.:\n"
            f"python pointcloud_npz_vis.py /path/to/file.npz"
        )

    pack = _load_npz(args.npz_path)
    if pack["mode"] == "single":
        points = pack["points"]
        labels = pack["labels"]
        colors = pack.get("colors_rgb")
        if colors is None:
            mask = np.isin(labels, np.array([1, 12], dtype=labels.dtype))
            if np.any(mask):
                points = points[mask]
                colors = _colors_for_labels(labels[mask])
            else:
                colors = _colors_for_any_labels(labels)
    else:
        items = list(
            _partseg_items(
                pack,
                item_index=args.item_index,
                use_local_labels=not args.global_labels,
            )
        )
        if len(items) == 0:
            raise ValueError("No partseg items found")
        if len(items) == 1:
            _, _, points, labels = items[0]
            colors = _colors_for_any_labels(labels)
        else:
            pts_list = []
            col_list = []
            for _, _, points_i, labels_i in items:
                pts_list.append(points_i)
                col_list.append(_colors_for_any_labels(labels_i))
            points = np.concatenate(pts_list, axis=0)
            colors = np.concatenate(col_list, axis=0)
    if args.export_ply is not None:
        export_ply(points, colors, args.export_ply)
        print(f"Saved: {args.export_ply}")

    if not args.skip_rerun:
        visualize_npz(
            args.npz_path,
            spawn=not args.no_spawn,
            item_index=args.item_index,
            use_local_labels=not args.global_labels,
            fallback_split=not args.no_fallback_split,
            radii=args.radii,
            show_point_text_labels=args.show_point_text_labels,
        )


if __name__ == "__main__":
    main()
