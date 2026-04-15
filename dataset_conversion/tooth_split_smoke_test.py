import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "dataset_conversion" / "tooth_split.py"

    with tempfile.TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "tooth_3d"
        list_dir = data_root / "list"
        list_dir.mkdir(parents=True, exist_ok=True)

        case_names = [f"case_{i:03d}" for i in range(20)]
        with open(list_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.dump(case_names, f)

        cmd = [
            sys.executable,
            str(script),
            "--data_root",
            str(data_root),
            "--k_fold",
            "5",
            "--seed",
            "0",
            "--holdout_ratio",
            "0.15",
        ]
        subprocess.check_call(cmd, cwd=str(repo_root))

        split_file = list_dir / "tooth_split_k5_seed0.yaml"
        assert split_file.exists(), "split file was not created"

        with open(split_file, "r", encoding="utf-8") as f:
            payload = yaml.load(f, Loader=yaml.SafeLoader)

        assert int(payload["k_fold"]) == 5
        assert int(payload["seed"]) == 0
        assert int(payload["num_cases"]) == len(case_names)
        assert int(payload["num_holdout_test"]) == 3
        assert int(payload["num_trainval"]) == 17
        assert len(payload["folds"]) == 5
        assert len(payload["holdout_test"]) == 3

        all_case_set = set(case_names)
        holdout_set = set(payload["holdout_test"])
        for fold in payload["folds"]:
            train_set = set(fold["train"])
            val_set = set(fold["val"])
            assert "test" not in fold
            assert train_set.isdisjoint(val_set)
            assert len(val_set) > 0
            assert holdout_set.isdisjoint(train_set)
            assert holdout_set.isdisjoint(val_set)
            assert train_set.union(val_set).union(holdout_set) == all_case_set

    print("tooth_split smoke test passed")


if __name__ == "__main__":
    main()

