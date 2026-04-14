import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from utils import ResampleLabelToRef, ResampleXYZAxis


BACKGROUND_RAW_IDS = set(range(0, 11))


def parse_spacing(value):
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--target_spacing must be x,y,z")
    return tuple(parts)


def parse_context(value):
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--crop_context must be z,y,x")
    if any(v < 0 for v in parts):
        raise ValueError("--crop_context values must be non-negative")
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

    return apply_background_mapping_policy(normalized)


def apply_background_mapping_policy(raw_to_train):
    normalized = {int(k): int(v) for k, v in raw_to_train.items()}
    normalized[0] = 0
    for raw_id in sorted(BACKGROUND_RAW_IDS - {0}):
        normalized[int(raw_id)] = 0
    return normalized


def save_label_map(path, raw_to_train):
    raw_to_train = apply_background_mapping_policy(raw_to_train)
    train_to_raw = {}
    for raw_id, train_id in raw_to_train.items():
        train_to_raw.setdefault(int(train_id), []).append(int(raw_id))
    train_to_raw = {k: sorted(v) for k, v in train_to_raw.items()}
    payload = {
        "raw_to_train": {int(k): int(v) for k, v in sorted(raw_to_train.items(), key=lambda x: x[0])},
        "train_to_raw": {int(k): v for k, v in sorted(train_to_raw.items(), key=lambda x: x[0])},
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
    positive_ids = [x for x in unique_ids if x > 0 and x not in BACKGROUND_RAW_IDS]
    raw_to_train = {0: 0}
    for raw_id in sorted(BACKGROUND_RAW_IDS - {0}):
        raw_to_train[int(raw_id)] = 0
    for idx, raw_id in enumerate(sorted(positive_ids), start=1):
        raw_to_train[int(raw_id)] = int(idx)
    return apply_background_mapping_policy(raw_to_train)


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


def crop_foreground_by_mapped_label(itk_img, itk_lab, crop_context):
    lab = sitk.GetArrayFromImage(itk_lab)
    fg = np.argwhere(lab > 0)
    if fg.size == 0:
        return itk_img, itk_lab

    z0, y0, x0 = fg.min(axis=0)
    z1, y1, x1 = fg.max(axis=0) + 1

    pad_z, pad_y, pad_x = crop_context
    zz, yy, xx = lab.shape

    z0 = max(0, int(z0) - pad_z)
    y0 = max(0, int(y0) - pad_y)
    x0 = max(0, int(x0) - pad_x)
    z1 = min(zz, int(z1) + pad_z)
    y1 = min(yy, int(y1) + pad_y)
    x1 = min(xx, int(x1) + pad_x)

    # RegionOfInterest expects x,y,z index order.
    index = [x0, y0, z0]
    size = [x1 - x0, y1 - y0, z1 - z0]

    cropped_img = sitk.RegionOfInterest(itk_img, size=size, index=index)
    cropped_lab = sitk.RegionOfInterest(itk_lab, size=size, index=index)
    return cropped_img, cropped_lab


def convert_case(
    image_path,
    label_path,
    dst_root,
    case_name,
    target_spacing,
    raw_to_train,
    strict_unmapped,
    crop_foreground=False,
    crop_context=(10, 30, 30),
):
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

    out_img, out_lab = rs_img, mapped_itk_lab
    if crop_foreground:
        out_img, out_lab = crop_foreground_by_mapped_label(out_img, out_lab, crop_context)

    sitk.WriteImage(out_img, str(dst_root / f"{case_name}.nii.gz"))
    sitk.WriteImage(out_lab, str(dst_root / f"{case_name}_gt.nii.gz"))


def collect_cases(images_dir, labels_dir, image_suffix, label_suffix, image_stem_suffix=""):
    image_paths = sorted(images_dir.glob(f"*{image_suffix}"))
    if not image_paths:
        raise ValueError(f"No image files found in {images_dir} with suffix {image_suffix}")

    pairs = []
    for img_path in image_paths:
        image_stem = img_path.name[: -len(image_suffix)]

        case_name = image_stem
        if image_stem_suffix and image_stem.endswith(image_stem_suffix):
            case_name = image_stem[: -len(image_stem_suffix)]
            if case_name == "":
                raise ValueError(f"Invalid image name after trimming suffix: {img_path.name}")

        lab_path = labels_dir / f"{case_name}{label_suffix}"
        # Fallback to unchanged stem to keep backward compatibility.
        if not lab_path.exists() and case_name != image_stem:
            lab_path = labels_dir / f"{image_stem}{label_suffix}"
            case_name = image_stem

        if not lab_path.exists():
            raise FileNotFoundError(
                f"Label not found for image {img_path.name}. Tried: "
                f"{labels_dir / (case_name + label_suffix)} and {labels_dir / (image_stem + label_suffix)}"
            )
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
        "--image_stem_suffix",
        type=str,
        default="_0000",
        help="Suffix in image stem to trim when matching labels, e.g. ToothFairy2F_001_0000 -> ToothFairy2F_001",
    )
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
    parser.add_argument(
        "--crop_foreground",
        action="store_true",
        help="Crop image/label to mapped foreground (>0) bbox after remapping",
    )
    parser.add_argument(
        "--crop_context",
        type=parse_context,
        default=(10, 30, 30),
        help="Foreground crop context in z,y,x order, e.g. 10,30,30",
    )

    args = parser.parse_args()

    images_dir = Path(args.src_images)
    labels_dir = Path(args.src_labels)
    dst_root = Path(args.dst_root)
    list_dir = dst_root / "list"

    dst_root.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_cases(
        images_dir,
        labels_dir,
        args.image_suffix,
        args.label_suffix,
        image_stem_suffix=args.image_stem_suffix,
    )

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
            crop_foreground=args.crop_foreground,
            crop_context=args.crop_context,
        )
        case_names.append(case_name)
        print(case_name, "done")

    with open(list_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(case_names, f)

    print(f"Converted {len(case_names)} cases to {dst_root}")


if __name__ == "__main__":
    main()
