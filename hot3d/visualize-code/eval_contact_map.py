import argparse
import ast
import csv
import json
import os
import re
from typing import Optional, Tuple

import numpy as np
import torch
import pickle


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_token(token: str, keep_underscores: bool = True) -> str:
    token = token.lower().strip()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]", "", token)
    if not keep_underscores:
        token = token.replace("_", "")
    return token


def _match_key(candidate: str, key_set: set[str]) -> Optional[str]:
    if not candidate:
        return None
    norm_cand = _normalize_token(candidate)
    norm_cand_flat = _normalize_token(candidate, keep_underscores=False)
    for key in key_set:
        norm_key = _normalize_token(key)
        norm_key_flat = _normalize_token(key, keep_underscores=False)
        if norm_cand == norm_key or norm_cand_flat == norm_key_flat:
            return key
        if norm_key and (norm_key in norm_cand or norm_cand in norm_key):
            return key
        if norm_key_flat and (norm_key_flat in norm_cand_flat or norm_cand_flat in norm_key_flat):
            return key
    return None


def _parse_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str):
        text = str(text)
    match = re.search(r"grab\s+(?P<part>.+?)\s+of\s+(?P<object>.+)$", text, re.IGNORECASE)
    if match:
        return match.group("object").strip(), match.group("part").strip()
    # fallback: try "of {object}"
    match = re.search(r"of\s+(?P<object>.+)$", text, re.IGNORECASE)
    if match:
        return match.group("object").strip(), None
    return None, None


def _maybe_parse_list_string(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return value


def _flatten_texts(value):
    value = _maybe_parse_list_string(value)
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten_texts(v))
        return out
    return [value]


def _reduce_contact_map_to_mask(contact_map, num_points: int) -> Optional[np.ndarray]:
    if contact_map is None:
        return None
    arr = _as_numpy(contact_map)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] != num_points:
            return None
        return arr > 0
    if arr.ndim >= 2:
        if arr.shape[-1] == num_points:
            axes = tuple(range(arr.ndim - 1))
            counts = arr.sum(axis=axes)
            return counts > 0
        if arr.shape[0] == num_points:
            axes = tuple(range(1, arr.ndim))
            counts = arr.sum(axis=axes)
            return counts > 0
    return None


def _precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float, float]:
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def _load_part_map(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main(file_name: str):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help="path to label_merged.json (default: ~/label_merged.json or ./label_merged.json)",
    )
    args = parser.parse_args()

    args.file_name = file_name
    home = os.path.expanduser("~")
    file_path = args.file_name
    if not os.path.isabs(file_path):
        file_path = os.path.join(home, file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"pkl not found: {file_path}")

    label_path = args.label_map
    if label_path is None:
        cand1 = os.path.join(home, "label_merged.json")
        cand2 = os.path.join(os.getcwd(), "label_merged.json")
        label_path = cand1 if os.path.exists(cand1) else cand2
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"label_merged not found: {label_path}")

    part_map = _load_part_map(label_path)
    object_keys = set(part_map.keys())

    with open(file_path, "rb") as f:
        item_list = pickle.load(f)

    per_text_stats = {}
    total_prec = []
    total_rec = []
    total_f1 = []

    for idx, (rotated_pc, contact_map, text) in enumerate(item_list):
        raw_texts = _flatten_texts(text)
        text_keys = [str(t) for t in raw_texts]

        # choose per-text contact_map if provided
        contact_arr = None if contact_map is None else _as_numpy(contact_map)
        per_text_contact = (
            contact_arr
            if contact_arr is not None and contact_arr.ndim >= 2 and contact_arr.shape[0] == len(text_keys)
            else None
        )

        # derive per-text point cloud shape for num_points
        obj_points_source = rotated_pc
        if isinstance(obj_points_source, torch.Tensor):
            obj_points_source = obj_points_source.detach().cpu().numpy()
        else:
            obj_points_source = np.asarray(obj_points_source)

        for text_idx, text_key in enumerate(text_keys):
            per_text_points = obj_points_source
            if per_text_points.ndim >= 3 and per_text_points.shape[0] == len(text_keys):
                per_text_points = per_text_points[text_idx]
            if per_text_points.ndim == 3:
                num_points = per_text_points.shape[1]
            else:
                num_points = per_text_points.shape[0]

            obj_str, part_str = _parse_text(text_key)
            obj_key = _match_key(obj_str or "", object_keys)
            if obj_key is None:
                continue

            part_keys = set(part_map.get(obj_key, {}).keys())
            part_key = _match_key(part_str or "", part_keys)
            if part_key is None:
                continue

            gt_mask = np.zeros(num_points, dtype=bool)
            indices = np.asarray(part_map[obj_key][part_key], dtype=np.int64)
            valid = (indices >= 0) & (indices < num_points)
            gt_mask[indices[valid]] = True

            cm = None
            if per_text_contact is not None:
                cm = per_text_contact[text_idx]
            else:
                cm = contact_map

            pred_mask = _reduce_contact_map_to_mask(cm, num_points)
            if pred_mask is None:
                continue

            prec, rec, f1 = _precision_recall(pred_mask, gt_mask)
            per_text_stats.setdefault(text_key, []).append((prec, rec, f1))
            total_prec.append(prec)
            total_rec.append(rec)
            total_f1.append(f1)

    print("[Eval] contact_map vs label_merged")
    for text_key, stats in per_text_stats.items():
        precs = [s[0] for s in stats]
        recs = [s[1] for s in stats]
        f1s = [s[2] for s in stats]
        print(
            f"- {text_key}: "
            f"precision={np.mean(precs):.3f} recall={np.mean(recs):.3f} f1={np.mean(f1s):.3f}"
        )

    if total_prec:
        print(
            f"[Overall] precision={np.mean(total_prec):.3f} "
            f"recall={np.mean(total_rec):.3f} f1={np.mean(total_f1):.3f}"
        )

    # Save per-text metrics to CSV (Excel-friendly)
    out_csv = os.path.join(os.path.expanduser("~"), "contact_map_metrics.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["file_name", "text", "precision", "recall", "f1"])
        for text_key, stats in per_text_stats.items():
            precs = [s[0] for s in stats]
            recs = [s[1] for s in stats]
            f1s = [s[2] for s in stats]
            writer.writerow(
                [
                    os.path.basename(file_path),
                    text_key,
                    f"{np.mean(precs):.6f}",
                    f"{np.mean(recs):.6f}",
                    f"{np.mean(f1s):.6f}",
                ]
            )
    print(f"[Eval] Saved per-text metrics to {out_csv}")


if __name__ == "__main__":
    main("Desktop/hot3d_vis/contact_grab_exc_rot_aug.pkl")
    main("Desktop/hot3d_vis/contact_grab_exc_two_obj.pkl")
    main("Desktop/hot3d_vis/contact_grab_exc_two_obj_rot.pkl")
