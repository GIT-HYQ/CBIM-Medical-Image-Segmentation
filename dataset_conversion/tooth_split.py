import argparse
from pathlib import Path
import random

import yaml


def build_split_payload(case_names, k_fold, seed, holdout_ratio=0.15):
    shuffled = list(case_names)
    random.Random(seed).shuffle(shuffled)
    length = len(shuffled)

    holdout_num = int(round(length * float(holdout_ratio)))
    holdout_num = max(1, holdout_num)
    holdout_num = min(length - 1, holdout_num) if length > 1 else length

    holdout_test = shuffled[:holdout_num]
    trainval = shuffled[holdout_num:]

    fold_len = max(1, len(trainval) // k_fold)

    folds = []
    for fold_idx in range(k_fold):
        val_start = fold_idx * fold_len
        val_end = (fold_idx + 1) * fold_len if fold_idx < k_fold - 1 else len(trainval)

        val_names = trainval[val_start:val_end]
        val_set = set(val_names)
        train_names = [name for name in trainval if name not in val_set]

        folds.append(
            {
                "fold": int(fold_idx),
                "train": list(train_names),
                "val": list(val_names),
            }
        )

    return {
        "split_version": 2,
        "k_fold": int(k_fold),
        "seed": int(seed),
        "holdout_ratio": float(holdout_ratio),
        "num_cases": int(length),
        "num_holdout_test": int(len(holdout_test)),
        "num_trainval": int(len(trainval)),
        "holdout_test": list(holdout_test),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fixed fold split file for tooth dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Converted CBIM tooth dataset root")
    parser.add_argument("--k_fold", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument("--holdout_ratio", type=float, default=0.15, help="Global holdout test ratio, e.g. 0.15")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional output YAML path. Default: <data_root>/list/tooth_split_k{k_fold}_seed{seed}.yaml",
    )
    args = parser.parse_args()

    if args.k_fold <= 1:
        raise ValueError("--k_fold must be greater than 1")
    if args.holdout_ratio <= 0 or args.holdout_ratio >= 1:
        raise ValueError("--holdout_ratio must be in (0, 1)")

    data_root = Path(args.data_root)
    list_dir = data_root / "list"
    dataset_file = list_dir / "dataset.yaml"
    if not dataset_file.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {dataset_file}")

    with open(dataset_file, "r", encoding="utf-8") as f:
        case_names = yaml.load(f, Loader=yaml.SafeLoader)

    if not isinstance(case_names, list) or not case_names:
        raise ValueError("dataset.yaml must contain a non-empty case-name list")

    payload = build_split_payload([str(name) for name in case_names], args.k_fold, args.seed, holdout_ratio=args.holdout_ratio)

    save_path = (
        Path(args.save_path)
        if args.save_path is not None
        else (list_dir / f"tooth_split_k{args.k_fold}_seed{args.seed}.yaml")
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False)

    print(f"Saved fixed split file: {save_path}")


if __name__ == "__main__":
    main()

