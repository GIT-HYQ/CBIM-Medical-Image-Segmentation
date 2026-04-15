import argparse
from pathlib import Path
import random

import numpy as np
import SimpleITK as sitk
import yaml


def parse_patch_size(value):
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--patch_size must be z,y,x")
    if any(v <= 0 for v in parts):
        raise ValueError("--patch_size values must be positive")
    return tuple(parts)


def clamp_fraction(value, name):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def load_case_names(data_root):
    list_file = Path(data_root) / "list" / "dataset.yaml"
    if not list_file.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {list_file}")
    with open(list_file, "r", encoding="utf-8") as f:
        case_names = yaml.load(f, Loader=yaml.SafeLoader)
    if not isinstance(case_names, list) or not case_names:
        raise ValueError("dataset.yaml must contain a non-empty case-name list")
    return [str(name) for name in case_names]


def resolve_split_file_path(data_root, split_file, k_fold, split_seed):
    if split_file is not None and str(split_file).strip() != "":
        return Path(split_file)
    return Path(data_root) / "list" / f"tooth_split_k{k_fold}_seed{split_seed}.yaml"


def load_train_cases_from_split(split_file_path, fold, k_fold, split_seed, all_case_names):
    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    with open(split_file_path, "r", encoding="utf-8") as f:
        payload = yaml.load(f, Loader=yaml.SafeLoader)

    if not isinstance(payload, dict) or "folds" not in payload:
        raise ValueError(f"Invalid split file format: {split_file_path}")

    if int(payload.get("k_fold", -1)) != int(k_fold):
        raise ValueError(
            f"Split file k_fold={payload.get('k_fold')} does not match --k_fold={k_fold}: {split_file_path}"
        )
    if int(payload.get("seed", -1)) != int(split_seed):
        raise ValueError(
            f"Split file seed={payload.get('seed')} does not match --split_seed={split_seed}: {split_file_path}"
        )

    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != int(k_fold):
        raise ValueError(f"Split file must contain exactly {k_fold} folds: {split_file_path}")
    if int(fold) < 0 or int(fold) >= len(folds):
        raise ValueError(f"Invalid --fold={fold} for k_fold={k_fold}")

    train_cases = folds[int(fold)].get("train", [])
    if not isinstance(train_cases, list) or not train_cases:
        raise ValueError(f"Split file fold={fold} has empty/invalid train list: {split_file_path}")

    case_set = set(all_case_names)
    unknown = [name for name in train_cases if str(name) not in case_set]
    if unknown:
        raise ValueError(f"Split file contains train cases missing from dataset.yaml: {unknown[:5]}")

    return [str(name) for name in train_cases]


def compute_max_start(shape_zyx, patch_size_zyx):
    padded_shape = [max(int(shape_zyx[i]), int(patch_size_zyx[i])) for i in range(3)]
    return [max(padded_shape[i] - int(patch_size_zyx[i]), 0) for i in range(3)]


def sample_random_start(max_start_zyx, rng):
    return tuple(rng.randint(0, max_start_zyx[i]) if max_start_zyx[i] > 0 else 0 for i in range(3))


def compute_patch_fg_stats(label, start_zyx, patch_size_zyx):
    z, y, x = [int(v) for v in start_zyx]
    d, h, w = [int(v) for v in patch_size_zyx]
    z1, y1, x1 = min(z + d, label.shape[0]), min(y + h, label.shape[1]), min(x + w, label.shape[2])

    cropped = label[z:z1, y:y1, x:x1]
    fg_voxels = int(np.count_nonzero(cropped > 0))
    fg_ratio = float(fg_voxels) / float(d * h * w)
    return fg_voxels, fg_ratio


def select_patch(label, patch_size_zyx, want_foreground, fg_threshold, bg_max_fg_ratio, max_attempts, rng):
    max_start_zyx = compute_max_start(label.shape, patch_size_zyx)
    fallback = None
    fallback_distance = None

    for _ in range(max_attempts):
        start_zyx = sample_random_start(max_start_zyx, rng)
        fg_voxels, fg_ratio = compute_patch_fg_stats(label, start_zyx, patch_size_zyx)

        if want_foreground:
            if fg_ratio >= fg_threshold:
                return start_zyx, fg_voxels, fg_ratio, True
            distance = fg_threshold - fg_ratio
        else:
            if fg_ratio <= bg_max_fg_ratio:
                return start_zyx, fg_voxels, fg_ratio, False
            distance = fg_ratio - bg_max_fg_ratio

        if fallback is None or distance < fallback_distance:
            fallback = (start_zyx, fg_voxels, fg_ratio, fg_ratio >= fg_threshold)
            fallback_distance = distance

    if fallback is not None:
        return fallback

    start_zyx = (0, 0, 0)
    fg_voxels, fg_ratio = compute_patch_fg_stats(label, start_zyx, patch_size_zyx)
    return start_zyx, fg_voxels, fg_ratio, fg_ratio >= fg_threshold


def build_case_patch_records(case_name, label, patch_size_zyx, patches_per_case, foreground_fraction, fg_threshold, bg_max_fg_ratio, max_attempts, rng):
    num_foreground = int(round(patches_per_case * foreground_fraction))
    num_foreground = min(max(num_foreground, 0), patches_per_case)
    num_background = patches_per_case - num_foreground

    records = []
    for _ in range(num_foreground):
        start_zyx, fg_voxels, fg_ratio, accepted_as_foreground = select_patch(
            label, patch_size_zyx, True, fg_threshold, bg_max_fg_ratio, max_attempts, rng
        )
        records.append((case_name, start_zyx, fg_voxels, fg_ratio, int(accepted_as_foreground)))

    for _ in range(num_background):
        start_zyx, fg_voxels, fg_ratio, accepted_as_foreground = select_patch(
            label, patch_size_zyx, False, fg_threshold, bg_max_fg_ratio, max_attempts, rng
        )
        records.append((case_name, start_zyx, fg_voxels, fg_ratio, int(accepted_as_foreground)))

    return records


def save_patch_index(index_path, meta_path, patch_size_zyx, records, args):
    case_name = np.asarray([record[0] for record in records])
    start_zyx = np.asarray([record[1] for record in records], dtype=np.int32)
    size_zyx = np.repeat(np.asarray([patch_size_zyx], dtype=np.int32), len(records), axis=0)
    fg_voxels = np.asarray([record[2] for record in records], dtype=np.int32)
    fg_ratio = np.asarray([record[3] for record in records], dtype=np.float32)
    is_foreground = np.asarray([record[4] for record in records], dtype=np.uint8)

    np.savez_compressed(
        index_path,
        case_name=case_name,
        start_zyx=start_zyx,
        size_zyx=size_zyx,
        fg_voxels=fg_voxels,
        fg_ratio=fg_ratio,
        is_foreground=is_foreground,
    )

    meta = {
        "version": 1,
        "index_file": str(index_path.name),
        "patch_size_zyx": [int(v) for v in patch_size_zyx],
        "patches_per_case": int(args.patches_per_case),
        "foreground_fraction": float(args.foreground_fraction),
        "foreground_threshold": float(args.foreground_threshold),
        "background_max_fg_ratio": float(args.background_max_fg_ratio),
        "max_attempts": int(args.max_attempts),
        "seed": int(args.seed),
        "num_cases": int(len(set(case_name.tolist()))),
        "num_patches": int(len(records)),
        "split_file": str(args.split_file) if args.split_file is not None else None,
        "fold": int(args.fold) if args.fold is not None else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Pre-generate tooth 3D patch indices from converted labels")
    parser.add_argument("--data_root", type=str, required=True, help="Converted CBIM tooth dataset root")
    parser.add_argument("--patch_size", type=parse_patch_size, required=True, help="Patch size in z,y,x order")
    parser.add_argument("--patches_per_case", type=int, default=8, help="Number of indexed patches per case")
    parser.add_argument(
        "--foreground_fraction",
        type=lambda v: clamp_fraction(v, "--foreground_fraction"),
        default=0.875,
        help="Fraction of patches per case that should prefer foreground",
    )
    parser.add_argument(
        "--foreground_threshold",
        type=lambda v: clamp_fraction(v, "--foreground_threshold"),
        default=0.01,
        help="Minimum foreground ratio for a foreground patch",
    )
    parser.add_argument(
        "--background_max_fg_ratio",
        type=lambda v: clamp_fraction(v, "--background_max_fg_ratio"),
        default=0.0,
        help="Maximum foreground ratio for a background patch",
    )
    parser.add_argument("--max_attempts", type=int, default=64, help="Maximum sampling attempts per indexed patch")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic index generation")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional .npz output path. Default: <data_root>/list/tooth_patch_index.npz",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Optional fixed split file. Used with --fold to generate indices only for that fold's train cases.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for split-aware generation (generate only train cases in this fold).",
    )
    parser.add_argument("--k_fold", type=int, default=5, help="Expected k_fold in split file when --fold is used")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Expected split seed in split file when --fold is used",
    )
    args = parser.parse_args()

    if args.patches_per_case <= 0:
        raise ValueError("--patches_per_case must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max_attempts must be positive")
    if args.fold is None and args.split_file is not None:
        raise ValueError("--split_file requires --fold")
    if args.fold is not None and args.k_fold <= 1:
        raise ValueError("--k_fold must be greater than 1 when --fold is used")

    data_root = Path(args.data_root)
    list_dir = data_root / "list"
    index_path = Path(args.save_path) if args.save_path is not None else (list_dir / "tooth_patch_index.npz")
    meta_path = index_path.with_suffix("")
    meta_path = meta_path.parent / f"{meta_path.name}_meta.yaml"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    all_case_names = load_case_names(data_root)
    if args.fold is not None:
        split_path = resolve_split_file_path(data_root, args.split_file, args.k_fold, args.split_seed)
        args.split_file = str(split_path)
        case_names = load_train_cases_from_split(
            split_path,
            fold=args.fold,
            k_fold=args.k_fold,
            split_seed=args.split_seed,
            all_case_names=all_case_names,
        )
        print(f"Using split file {split_path}, fold={args.fold}, train cases={len(case_names)}")
    else:
        case_names = all_case_names

    rng = random.Random(args.seed)
    records = []

    for case_name in case_names:
        label_path = data_root / f"{case_name}_gt.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        itk_lab = sitk.ReadImage(str(label_path))
        label = sitk.GetArrayFromImage(itk_lab).astype(np.uint8)

        case_records = build_case_patch_records(
            case_name,
            label,
            args.patch_size,
            args.patches_per_case,
            args.foreground_fraction,
            args.foreground_threshold,
            args.background_max_fg_ratio,
            args.max_attempts,
            rng,
        )
        records.extend(case_records)
        print(f"{case_name}: generated {len(case_records)} patch indices")

    if not records:
        raise ValueError("No patch indices were generated")

    save_patch_index(index_path, meta_path, args.patch_size, records, args)
    print(f"Saved patch index to {index_path}")
    print(f"Saved patch-index metadata to {meta_path}")


if __name__ == "__main__":
    main()

