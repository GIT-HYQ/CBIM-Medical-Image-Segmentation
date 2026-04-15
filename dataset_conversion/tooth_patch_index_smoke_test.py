import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
import yaml


def write_nifti(path, arr, spacing=(0.4, 0.4, 0.4)):
    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing(spacing)
    sitk.WriteImage(itk, str(path))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "dataset_conversion" / "tooth_patch_index.py"

    with tempfile.TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "tooth_3d"
        list_dir = data_root / "list"
        data_root.mkdir(parents=True, exist_ok=True)
        list_dir.mkdir(parents=True, exist_ok=True)

        case_names = []
        for i in range(4):
            case_name = f"case_{i:03d}"
            case_names.append(case_name)

            image = np.random.normal(300, 50, size=(16, 24, 24)).astype(np.float32)
            label = np.zeros((16, 24, 24), dtype=np.uint8)
            label[4:12, 6:18, 6:18] = 1 if i % 2 == 0 else 2

            write_nifti(data_root / f"{case_name}.nii.gz", image)
            write_nifti(data_root / f"{case_name}_gt.nii.gz", label)

        with open(list_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.dump(case_names, f)

        split_file = list_dir / "tooth_split_k2_seed0.yaml"
        split_payload = {
            "k_fold": 2,
            "seed": 0,
            "num_cases": 4,
            "folds": [
                {"fold": 0, "train": ["case_000", "case_001"], "val": ["case_002"], "test": ["case_003"]},
                {"fold": 1, "train": ["case_002", "case_003"], "val": ["case_000"], "test": ["case_001"]},
            ],
        }
        with open(split_file, "w", encoding="utf-8") as f:
            yaml.dump(split_payload, f, sort_keys=False)

        cmd = [
            sys.executable,
            str(script),
            "--data_root",
            str(data_root),
            "--patch_size",
            "8,16,16",
            "--patches_per_case",
            "4",
            "--foreground_fraction",
            "0.75",
            "--foreground_threshold",
            "0.01",
            "--background_max_fg_ratio",
            "0.0",
            "--seed",
            "0",
            "--split_file",
            str(split_file),
            "--fold",
            "0",
            "--k_fold",
            "2",
            "--split_seed",
            "0",
        ]
        subprocess.check_call(cmd, cwd=str(repo_root))

        patch_index = data_root / "list" / "tooth_patch_index.npz"
        patch_meta = data_root / "list" / "tooth_patch_index_meta.yaml"
        assert patch_index.exists(), "patch index was not created"
        assert patch_meta.exists(), "patch index metadata was not created"

        with np.load(patch_index, allow_pickle=False) as data:
            assert set(data.files) >= {"case_name", "start_zyx", "size_zyx", "fg_voxels", "fg_ratio", "is_foreground"}
            assert len(data["case_name"]) == 8, "unexpected number of patch records"
            assert data["start_zyx"].shape == (8, 3)
            assert np.all(data["size_zyx"] == np.array([8, 16, 16], dtype=np.int32))
            assert set(data["case_name"].tolist()) == {"case_000", "case_001"}

        sys.path.insert(0, str(repo_root))
        from training.dataset.dim3.dataset_tooth import ToothDataset

        args = SimpleNamespace(
            data_root=str(data_root),
            training_size=[8, 16, 16],
            patch_index_enabled=True,
            patch_index_file=None,
            patch_index_cache_cases=1,
            classes=3,
            aug_device="cpu",
            proc_idx=0,
            scale=[0.0, 0.0, 0.0],
            rotate=[0, 0, 0],
            translate=[0.0, 0.0, 0.0],
        )

        trainset = ToothDataset(args, mode="train", k_fold=2, k=0, seed=0)
        train_img, train_lab = trainset[0]
        assert tuple(train_img.shape) == (1, 8, 16, 16)
        assert tuple(train_lab.shape) == (1, 8, 16, 16)
        assert len(trainset) == 4, "train fold should expose indexed patches from one selected case"

        valset = ToothDataset(args, mode="val", k_fold=2, k=0, seed=0)
        val_img, val_lab, spacing, name = valset[0]
        assert val_img.shape[0] == 1 and val_lab.shape[0] == 1
        assert spacing.shape == (3,)
        assert name.endswith(".nii.gz")

    print("tooth_patch_index smoke test passed")


if __name__ == "__main__":
    main()

