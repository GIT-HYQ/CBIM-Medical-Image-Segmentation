import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


def auto_build_label_map(raw_ids):
    raw_ids = sorted(set(int(x) for x in raw_ids))
    positive_ids = [x for x in raw_ids if x > 0]
    mapping = {0: 0}
    for idx, raw_id in enumerate(positive_ids, start=1):
        mapping[int(raw_id)] = int(idx)
    return mapping


def parse_percentiles(text):
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) == 0:
        raise ValueError("--spacing_percentiles cannot be empty")
    for v in values:
        if v < 0 or v > 100:
            raise ValueError(f"Invalid percentile: {v}")
    return sorted(set(values))


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


def summarize_axis(values, percentiles):
    out = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(values, p))
    return out


def analyze_spacing(images_dir, image_suffix, percentiles):
    paths = collect_files(images_dir, image_suffix)

    spacings = []
    failed = []
    for p in paths:
        try:
            spacing = sitk.ReadImage(str(p)).GetSpacing()  # x, y, z
            spacings.append(spacing)
        except Exception:
            failed.append(str(p))

    if len(spacings) == 0:
        raise ValueError("No readable image files for spacing analysis")

    arr = np.array(spacings, dtype=np.float64)
    ratio = arr.max(axis=1) / np.clip(arr.min(axis=1), 1e-8, None)

    stats = {
        "num_files": len(paths),
        "num_readable": int(arr.shape[0]),
        "num_failed": len(failed),
        "axis": {
            "x": summarize_axis(arr[:, 0], percentiles),
            "y": summarize_axis(arr[:, 1], percentiles),
            "z": summarize_axis(arr[:, 2], percentiles),
        },
        "anisotropy_ratio": summarize_axis(ratio, percentiles),
        "recommended_target_spacing": [
            float(np.median(arr[:, 0])),
            float(np.median(arr[:, 1])),
            float(np.median(arr[:, 2])),
        ],
    }

    print("Spacing summary (x, y, z):")
    print(
        "  median: [{:.4f}, {:.4f}, {:.4f}]".format(
            stats["axis"]["x"]["median"], stats["axis"]["y"]["median"], stats["axis"]["z"]["median"]
        )
    )
    print(
        "  p10:    [{:.4f}, {:.4f}, {:.4f}]".format(
            stats["axis"]["x"].get("p10", stats["axis"]["x"]["median"]),
            stats["axis"]["y"].get("p10", stats["axis"]["y"]["median"]),
            stats["axis"]["z"].get("p10", stats["axis"]["z"]["median"]),
        )
    )
    print(
        "  p90:    [{:.4f}, {:.4f}, {:.4f}]".format(
            stats["axis"]["x"].get("p90", stats["axis"]["x"]["median"]),
            stats["axis"]["y"].get("p90", stats["axis"]["y"]["median"]),
            stats["axis"]["z"].get("p90", stats["axis"]["z"]["median"]),
        )
    )
    print(
        "  anisotropy ratio median/max: {:.3f}/{:.3f}".format(
            stats["anisotropy_ratio"]["median"], stats["anisotropy_ratio"]["max"]
        )
    )

    if len(failed) > 0:
        stats["failed_files"] = failed[:20]
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze tooth labels and/or image spacing")
    parser.add_argument("--report_mode", choices=["labels", "spacing", "both"], default="labels")
    parser.add_argument("--labels_dir", type=str, default=None, help="Directory containing label files")
    parser.add_argument("--label_suffix", type=str, default=".mha", help="Label file suffix")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory containing image files")
    parser.add_argument("--image_suffix", type=str, default=".mha", help="Image file suffix")
    parser.add_argument(
        "--spacing_percentiles",
        type=parse_percentiles,
        default=[10, 50, 90],
        help="Comma-separated percentiles for spacing stats, e.g. 10,50,90",
    )
    parser.add_argument(
        "--save_report",
        type=str,
        default=None,
        help="Optional YAML output path for combined report",
    )
    parser.add_argument(
        "--save_map",
        type=str,
        default=None,
        help="Optional YAML output for suggested raw_to_train map",
    )
    parser.add_argument(
        "--save_spacing_report",
        type=str,
        default=None,
        help="Optional YAML output for spacing report only",
    )

    args = parser.parse_args()

    report = {}
    mapping = None

    if args.report_mode in ["labels", "both"]:
        if args.labels_dir is None:
            raise ValueError("--labels_dir is required when report_mode includes labels")
        label_report, mapping = analyze_labels(args.labels_dir, args.label_suffix)
        report.update(label_report)

    if args.report_mode in ["spacing", "both"]:
        if args.images_dir is None:
            raise ValueError("--images_dir is required when report_mode includes spacing")
        spacing_report = analyze_spacing(args.images_dir, args.image_suffix, args.spacing_percentiles)
        if args.report_mode == "spacing":
            report = spacing_report
        else:
            report["spacing_stats"] = spacing_report

        if args.save_spacing_report is not None:
            with open(args.save_spacing_report, "w", encoding="utf-8") as f:
                yaml.dump(spacing_report, f, sort_keys=True)
            print(f"Saved spacing report to {args.save_spacing_report}")

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

