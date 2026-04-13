import subprocess
import sys
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk
import yaml


def write_mha(path, arr, spacing=(0.5, 0.5, 0.6)):
    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing(spacing)
    sitk.WriteImage(itk, str(path))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "dataset_conversion" / "tooth_3d.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src_images = tmp_dir / "images"
        src_labels = tmp_dir / "labels"
        dst_root = tmp_dir / "tooth_3d"
        label_map_file = tmp_dir / "label_map.yaml"

        src_images.mkdir(parents=True, exist_ok=True)
        src_labels.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            case = f"case_{i:03d}"
            image = np.random.normal(300, 100, size=(20, 32, 32)).astype(np.float32)
            label = np.zeros((20, 32, 32), dtype=np.uint8)
            label[5:10, 8:20, 8:20] = 11 if i == 0 else 48

            write_mha(src_images / f"{case}.mha", image)
            write_mha(src_labels / f"{case}.mha", label)

        with open(label_map_file, "w", encoding="utf-8") as f:
            yaml.dump({0: 0, 11: 1, 48: 2}, f)

        cmd = [
            sys.executable,
            str(script),
            "--src_images",
            str(src_images),
            "--src_labels",
            str(src_labels),
            "--dst_root",
            str(dst_root),
            "--label_map",
            str(label_map_file),
            "--strict_unmapped",
            "--target_spacing",
            "0.5,0.5,0.6",
        ]

        subprocess.check_call(cmd, cwd=str(repo_root / "dataset_conversion"))

        list_file = dst_root / "list" / "dataset.yaml"
        assert list_file.exists(), "dataset.yaml not created"

        with open(list_file, "r", encoding="utf-8") as f:
            names = yaml.load(f, Loader=yaml.SafeLoader)

        assert len(names) == 2, "unexpected number of converted cases"

        for case in names:
            assert (dst_root / f"{case}.nii.gz").exists()
            assert (dst_root / f"{case}_gt.nii.gz").exists()

        mapped_0 = sitk.GetArrayFromImage(sitk.ReadImage(str(dst_root / "case_000_gt.nii.gz")))
        mapped_1 = sitk.GetArrayFromImage(sitk.ReadImage(str(dst_root / "case_001_gt.nii.gz")))
        assert set(np.unique(mapped_0).tolist()) <= {0, 1}
        assert set(np.unique(mapped_1).tolist()) <= {0, 2}

    print("tooth_3d smoke test passed")


if __name__ == "__main__":
    main()

