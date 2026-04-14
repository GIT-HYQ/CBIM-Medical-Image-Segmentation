import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


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
    parser = argparse.ArgumentParser(description="Analyze spacing distribution to choose target_spacing")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--image_suffix", type=str, default=".mha", help="Image file suffix")
    parser.add_argument(
        "--spacing_percentiles",
        type=parse_percentiles,
        default=[10, 50, 90],
        help="Comma-separated percentiles, e.g. 10,50,90",
    )
    parser.add_argument("--save_report", type=str, default=None, help="Optional YAML output path")

    args = parser.parse_args()

    report = analyze_spacing(args.images_dir, args.image_suffix, args.spacing_percentiles)

    if args.save_report is not None:
        with open(args.save_report, "w", encoding="utf-8") as f:
            yaml.dump(report, f, sort_keys=True)
        print(f"Saved spacing report to {args.save_report}")


if __name__ == "__main__":
    main()

