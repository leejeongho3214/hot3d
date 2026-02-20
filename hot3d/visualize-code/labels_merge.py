import argparse
import glob
import json
import os
from typing import Dict, List


def _load_label_file(path: str):
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None, None, None
    label_names = payload.get("_label_names", {}) if isinstance(payload.get("_label_names", {}), dict) else {}
    object_keys = [k for k in payload.keys() if k != "_label_names"]
    if not object_keys:
        return None, None, None
    object_key = object_keys[0]
    label_map = payload.get(object_key, {})
    if not isinstance(label_map, dict):
        return None, None, None
    return object_key, label_map, label_names


def _label_to_part(label_id: str, label_names: Dict[str, str]) -> str:
    name = label_names.get(label_id, "").strip()
    return name if name else label_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge label json files into object->part->indices.")
    parser.add_argument(
        "--labels-dir",
        default=os.path.expanduser("~"),
        help="directory with *_labels.json files",
    )
    parser.add_argument(
        "--pattern",
        default="*_labels.json",
        help="glob pattern for label files",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.expanduser("~"), "labels_merged.json"),
        help="output json path",
    )
    args = parser.parse_args()

    label_paths = sorted(glob.glob(os.path.join(args.labels_dir, args.pattern)))
    if not label_paths:
        raise SystemExit(f"No label files found in {args.labels_dir} with {args.pattern}")

    merged: Dict[str, Dict[str, List[int]]] = {}
    for path in label_paths:
        object_key, label_map, label_names = _load_label_file(path)
        if object_key is None:
            continue
        if object_key not in merged:
            merged[object_key] = {}
        for label_id, indices in label_map.items():
            if not isinstance(indices, list):
                continue
            part_key = _label_to_part(str(label_id), label_names)
            merged[object_key].setdefault(part_key, [])
            merged[object_key][part_key].extend(indices)

    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged labels to {args.out}")


if __name__ == "__main__":
    main()
