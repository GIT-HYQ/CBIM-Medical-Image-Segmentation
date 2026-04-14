import argparse

import yaml

from analyze_tooth_labels import analyze_spacing, parse_percentiles


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

