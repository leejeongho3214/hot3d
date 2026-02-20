import json

path = "/Users/jeongho/labels_merged.json"
with open(path, "r") as f:
    merged = json.load(f)

for obj, parts in merged.items():
    print(f"\n[{obj}]")
    total = 0
    for part, indices in parts.items():
        count = len(indices)
        total += count
        print(f"  {part}: {count}")
    print(f"  TOTAL: {total}")

print(len(merged))