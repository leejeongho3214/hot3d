import argparse
import glob
import json
import os
from typing import Dict, List


def _sum_indices(label_map: Dict[str, List[int]]) -> int:
    total = 0
    for indices in label_map.values():
        if isinstance(indices, list):
            total += len(indices)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Check label json files for total counts.")
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
        "--expected",
        type=int,
        default=1024,
        help="expected total count",
    )
    args = parser.parse_args()

    label_paths = sorted(glob.glob(os.path.join(args.labels_dir, args.pattern)))
    if not label_paths:
        raise SystemExit(f"No label files found in {args.labels_dir} with {args.pattern}")

    mismatches = []
    for path in label_paths:
        with open(path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        object_keys = [k for k in payload.keys() if k != "_label_names"]
        if not object_keys:
            continue
        object_key = object_keys[0]
        label_map = payload.get(object_key, {})
        total = _sum_indices(label_map)
        if total != args.expected:
            mismatches.append((os.path.basename(path), object_key, total))

    if not mismatches:
        print("All files match expected total.")
        return
    print("Mismatched totals:")
    for fname, obj, total in mismatches:
        print(f"- {fname} ({obj}): {total}")


if __name__ == "__main__":
    main()
