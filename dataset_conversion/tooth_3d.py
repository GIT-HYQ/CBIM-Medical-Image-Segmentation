import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from utils import ResampleLabelToRef, ResampleXYZAxis


def parse_spacing(value):
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--target_spacing must be x,y,z")
    return tuple(parts)


def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Allow both flat map {11: 1, ...} and wrapped map {raw_to_train: {...}}
    raw_to_train = data.get("raw_to_train") if isinstance(data, dict) and "raw_to_train" in data else data
    if not isinstance(raw_to_train, dict):
        raise ValueError("Label map must be a dict or contain raw_to_train dict")

    normalized = {}
    for k, v in raw_to_train.items():
        normalized[int(k)] = int(v)

    if 0 not in normalized:
        normalized[0] = 0
    return normalized


def save_label_map(path, raw_to_train):
    train_to_raw = {int(v): int(k) for k, v in raw_to_train.items()}
    payload = {
        "raw_to_train": {int(k): int(v) for k, v in sorted(raw_to_train.items(), key=lambda x: x[0])},
        "train_to_raw": {int(k): int(v) for k, v in sorted(train_to_raw.items(), key=lambda x: x[0])},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=True)


def scan_label_ids(pairs):
    unique_ids = set()
    for _, _, label_path in pairs:
        itk_lab = sitk.ReadImage(str(label_path))
        unique_ids.update(int(x) for x in np.unique(sitk.GetArrayFromImage(itk_lab)).tolist())
    return sorted(unique_ids)


def auto_build_label_map(unique_ids):
    positive_ids = [x for x in unique_ids if x > 0]
    raw_to_train = {0: 0}
    for idx, raw_id in enumerate(sorted(positive_ids), start=1):
        raw_to_train[int(raw_id)] = int(idx)
    return raw_to_train


def remap_label_array(raw_lab, raw_to_train, strict_unmapped=True):
    raw_lab = raw_lab.astype(np.int32)
    max_raw = int(raw_lab.max()) if raw_lab.size > 0 else 0
    lut = np.full(max_raw + 1, -1, dtype=np.int32)

    for raw_id, train_id in raw_to_train.items():
        if raw_id < 0:
            continue
        if raw_id > max_raw:
            continue
        lut[int(raw_id)] = int(train_id)

    mapped = lut[raw_lab]
    unknown_mask = mapped < 0
    if np.any(unknown_mask):
        unknown_ids = sorted(np.unique(raw_lab[unknown_mask]).tolist())
        if strict_unmapped:
            raise ValueError(f"Unmapped label ids found: {unknown_ids}")
        mapped[unknown_mask] = 0

    return mapped.astype(np.uint8)


def normalize_direction(itk_image):
    itk_image = sitk.Image(itk_image)
    itk_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return itk_image


def resample_image_and_label(itk_img, itk_lab, target_spacing):
    if itk_img.GetSize() != itk_lab.GetSize():
        raise ValueError("Image and label size mismatch")

    itk_img = normalize_direction(itk_img)
    itk_lab = normalize_direction(itk_lab)

    if tuple(np.round(itk_img.GetSpacing(), 6)) == tuple(np.round(target_spacing, 6)):
        return itk_img, itk_lab

    rs_img = ResampleXYZAxis(itk_img, space=target_spacing, interp=sitk.sitkBSpline)
    rs_lab = ResampleLabelToRef(itk_lab, rs_img, interp=sitk.sitkNearestNeighbor)
    return rs_img, rs_lab


def validate_label(itk_lab):
    arr = sitk.GetArrayFromImage(itk_lab)
    unique = np.unique(arr)
    if np.any(unique < 0):
        raise ValueError("Label contains negative values")


def convert_case(image_path, label_path, dst_root, case_name, target_spacing, raw_to_train, strict_unmapped):
    itk_img = sitk.ReadImage(str(image_path))
    itk_lab = sitk.ReadImage(str(label_path))

    if itk_img.GetSize() != itk_lab.GetSize():
        raise ValueError(f"{case_name}: image/label size mismatch")

    rs_img, rs_lab = resample_image_and_label(itk_img, itk_lab, target_spacing)
    validate_label(rs_lab)

    raw_lab = sitk.GetArrayFromImage(rs_lab)
    mapped_lab = remap_label_array(raw_lab, raw_to_train, strict_unmapped=strict_unmapped)
    mapped_itk_lab = sitk.GetImageFromArray(mapped_lab)
    mapped_itk_lab.CopyInformation(rs_lab)

    sitk.WriteImage(rs_img, str(dst_root / f"{case_name}.nii.gz"))
    sitk.WriteImage(mapped_itk_lab, str(dst_root / f"{case_name}_gt.nii.gz"))


def collect_cases(images_dir, labels_dir, image_suffix, label_suffix):
    image_paths = sorted(images_dir.glob(f"*{image_suffix}"))
    if not image_paths:
        raise ValueError(f"No image files found in {images_dir} with suffix {image_suffix}")

    pairs = []
    for img_path in image_paths:
        case_name = img_path.name[: -len(image_suffix)]
        lab_path = labels_dir / f"{case_name}{label_suffix}"
        if not lab_path.exists():
            raise FileNotFoundError(f"Label not found for case {case_name}: {lab_path}")
        pairs.append((case_name, img_path, lab_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert tooth mha dataset to CBIM 3D format")
    parser.add_argument("--src_images", type=str, required=True, help="Directory containing image .mha files")
    parser.add_argument("--src_labels", type=str, required=True, help="Directory containing label .mha files")
    parser.add_argument("--dst_root", type=str, required=True, help="Output dataset root")
    parser.add_argument("--image_suffix", type=str, default=".mha", help="Image file suffix")
    parser.add_argument("--label_suffix", type=str, default=".mha", help="Label file suffix")
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help="Path to YAML label map. Supports either {raw:train} or {raw_to_train:{...}}",
    )
    parser.add_argument(
        "--strict_unmapped",
        action="store_true",
        help="Fail conversion if label IDs not covered by label_map",
    )
    parser.add_argument(
        "--save_label_map",
        type=str,
        default=None,
        help="Save used label map YAML path. Default: <dst_root>/list/label_map.yaml",
    )
    parser.add_argument(
        "--target_spacing",
        type=parse_spacing,
        default=(0.4, 0.4, 0.4),
        help="Target spacing in x,y,z order, e.g. 0.4,0.4,0.4",
    )

    args = parser.parse_args()

    images_dir = Path(args.src_images)
    labels_dir = Path(args.src_labels)
    dst_root = Path(args.dst_root)
    list_dir = dst_root / "list"

    dst_root.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_cases(images_dir, labels_dir, args.image_suffix, args.label_suffix)

    if args.label_map is not None:
        raw_to_train = load_label_map(args.label_map)
    else:
        unique_ids = scan_label_ids(pairs)
        raw_to_train = auto_build_label_map(unique_ids)
        print("No --label_map provided. Auto-built map from dataset label IDs.")

    save_map_path = Path(args.save_label_map) if args.save_label_map is not None else (list_dir / "label_map.yaml")
    save_label_map(save_map_path, raw_to_train)
    print(f"Label map saved to {save_map_path}")

    case_names = []
    for case_name, image_path, label_path in pairs:
        convert_case(
            image_path,
            label_path,
            dst_root,
            case_name,
            args.target_spacing,
            raw_to_train=raw_to_train,
            strict_unmapped=args.strict_unmapped,
        )
        case_names.append(case_name)
        print(case_name, "done")

    with open(list_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(case_names, f)

    print(f"Converted {len(case_names)} cases to {dst_root}")


if __name__ == "__main__":
    main()
