import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


BACKGROUND_RAW_IDS = set(range(0, 11))


def auto_build_label_map(raw_ids):
    raw_ids = sorted(set(int(x) for x in raw_ids))
    positive_ids = [x for x in raw_ids if x > 0 and x not in BACKGROUND_RAW_IDS]
    mapping = {0: 0}
    for raw_id in sorted(BACKGROUND_RAW_IDS - {0}):
        mapping[int(raw_id)] = 0
    for idx, raw_id in enumerate(positive_ids, start=1):
        mapping[int(raw_id)] = int(idx)
    return mapping


def collect_files(root_dir, suffix):
    root = Path(root_dir)
    paths = sorted(root.glob(f"*{suffix}"))
    if not paths:
        raise ValueError(f"No files found in {root} with suffix {suffix}")
    return paths


def analyze_labels(labels_dir, label_suffix):
    paths = collect_files(labels_dir, label_suffix)

    voxel_counter = Counter()
    case_counter = Counter()

    for p in paths:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        unique_ids, counts = np.unique(arr, return_counts=True)

        for raw_id, count in zip(unique_ids.tolist(), counts.tolist()):
            raw_id = int(raw_id)
            voxel_counter[raw_id] += int(count)
            case_counter[raw_id] += 1

    raw_ids = sorted(voxel_counter.keys())
    mapping = auto_build_label_map(raw_ids)

    print("Found label IDs:", raw_ids)
    print("Suggested classes:", len(mapping))

    report = {
        "num_cases": len(paths),
        "raw_ids": raw_ids,
        "voxel_count": {int(k): int(v) for k, v in sorted(voxel_counter.items(), key=lambda x: x[0])},
        "case_count": {int(k): int(v) for k, v in sorted(case_counter.items(), key=lambda x: x[0])},
        "suggested_raw_to_train": mapping,
    }
    return report, mapping


def main():
    parser = argparse.ArgumentParser(description="Analyze tooth labels and suggest class mapping")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing label files")
    parser.add_argument("--label_suffix", type=str, default=".mha", help="Label file suffix")
    parser.add_argument(
        "--save_report",
        type=str,
        default=None,
        help="Optional YAML output path for label report",
    )
    parser.add_argument(
        "--save_map",
        type=str,
        default=None,
        help="Optional YAML output for suggested raw_to_train map",
    )
    args = parser.parse_args()

    report, mapping = analyze_labels(args.labels_dir, args.label_suffix)

    if args.save_report is not None:
        with open(args.save_report, "w", encoding="utf-8") as f:
            yaml.dump(report, f, sort_keys=True)
        print(f"Saved report to {args.save_report}")

    if args.save_map is not None and mapping is not None:
        with open(args.save_map, "w", encoding="utf-8") as f:
            yaml.dump(mapping, f, sort_keys=True)
        print(f"Saved suggested map to {args.save_map}")


if __name__ == "__main__":
    main()

