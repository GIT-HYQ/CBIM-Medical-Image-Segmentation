import math
import os
import random
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk
import torch
import yaml
from torch.utils.data import Dataset

from training import augmentation


class ToothDataset(Dataset):
    def __init__(self, args, mode="train", k_fold=5, k=0, seed=0):
        self.mode = mode
        self.args = args
        self.patch_size = tuple(int(v) for v in self.args.training_size)
        self.use_patch_index = mode == "train" and bool(getattr(args, "patch_index_enabled", False))
        self.patch_index_file = None
        self.patch_records = []
        self.case_cache = OrderedDict()
        self.case_cache_size = max(0, int(getattr(args, "patch_index_cache_cases", 2)))

        assert mode in ["train", "val", "test"]

        with open(os.path.join(args.data_root, "list", "dataset.yaml"), "r") as f:
            img_name_list = yaml.load(f, Loader=yaml.SafeLoader)

        split_file_path = self.resolve_split_file_path(k_fold, seed)
        split_payload = None
        if os.path.exists(split_file_path):
            split_payload = self.load_split_file(split_file_path, img_name_list, k_fold, seed)
        else:
            split_payload = self.build_split_payload(img_name_list, k_fold, seed)
            if bool(getattr(self.args, "save_split_file", True)):
                self.save_split_file(split_file_path, split_payload)

        if int(k) < 0 or int(k) >= len(split_payload["folds"]):
            raise ValueError(f"Invalid fold index k={k} for k_fold={k_fold}")

        if mode == "test":
            # New split schema stores independent holdout once for all folds.
            if "holdout_test" in split_payload:
                selected_names = split_payload.get("holdout_test", [])
            else:
                # Backward compatibility with legacy folds[].test schema.
                selected_names = split_payload["folds"][int(k)].get("test", [])
        else:
            fold_split = split_payload["folds"][int(k)]
            if mode == "train":
                selected_names = fold_split["train"]
            else:
                selected_names = fold_split["val"] if len(fold_split["val"]) > 0 else fold_split["train"]

        if len(selected_names) == 0:
            raise ValueError(f"Selected split is empty for mode={mode}, fold={k}")

        print("Start loading %s data" % self.mode)
        print(selected_names)

        self.selected_names = list(selected_names)
        self.case_files = {
            name: (
                os.path.join(args.data_root, f"{name}.nii.gz"),
                os.path.join(args.data_root, f"{name}_gt.nii.gz"),
            )
            for name in self.selected_names
        }

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []
        self.name_list = []

        if self.use_patch_index:
            self.patch_index_file = self.resolve_patch_index_path()
            self.patch_records = self.load_patch_index(self.patch_index_file, self.selected_names)
            self.name_list = [f"{name}.nii.gz" for name in self.selected_names]
            print("Load done, length of indexed patches:", len(self.patch_records))
            return

        if self.mode == "train":
            for name in self.selected_names:
                itk_img, itk_lab = self.read_case(name)
                img, lab = self.preprocess(itk_img, itk_lab)

                self.img_list.append(img)
                self.lab_list.append(lab)
                self.name_list.append(f"{name}.nii.gz")
        else:
            self.name_list = [f"{name}.nii.gz" for name in self.selected_names]

        loaded_length = len(self.img_list) if self.mode == "train" else len(self.name_list)
        print("Load done, length of dataset:", loaded_length)

    def __len__(self):
        if self.mode == "train":
            if self.use_patch_index:
                return len(self.patch_records)
            return len(self.img_list) * 100000
        return len(self.name_list)

    def resolve_split_file_path(self, k_fold, seed):
        split_file = getattr(self.args, "split_file", None)
        if split_file is not None and str(split_file).strip() != "":
            return str(split_file)
        return os.path.join(self.args.data_root, "list", f"tooth_split_k{k_fold}_seed{seed}.yaml")

    def build_split_payload(self, img_name_list, k_fold, seed):
        shuffled = list(img_name_list)
        random.Random(seed).shuffle(shuffled)
        length = len(shuffled)

        holdout_ratio = float(getattr(self.args, "split_holdout_ratio", 0.15))
        holdout_num = int(round(length * holdout_ratio))
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

    def load_split_file(self, split_file_path, img_name_list, k_fold, seed):
        with open(split_file_path, "r", encoding="utf-8") as f:
            payload = yaml.load(f, Loader=yaml.SafeLoader)

        if not isinstance(payload, dict) or "folds" not in payload:
            raise ValueError(f"Invalid split file format: {split_file_path}")

        if int(payload.get("k_fold", -1)) != int(k_fold):
            raise ValueError(
                f"split file k_fold={payload.get('k_fold')} does not match current k_fold={k_fold}: {split_file_path}"
            )
        if int(payload.get("seed", -1)) != int(seed):
            raise ValueError(
                f"split file seed={payload.get('seed')} does not match current split seed={seed}: {split_file_path}"
            )

        folds = payload.get("folds")
        if not isinstance(folds, list) or len(folds) != int(k_fold):
            raise ValueError(f"split file must contain exactly {k_fold} folds: {split_file_path}")

        all_cases = set(str(name) for name in img_name_list)
        holdout_cases = payload.get("holdout_test", None)
        if holdout_cases is not None:
            if not isinstance(holdout_cases, list):
                raise ValueError(f"split file holdout_test must be list: {split_file_path}")
            unknown = [name for name in holdout_cases if str(name) not in all_cases]
            if unknown:
                raise ValueError(f"split file holdout_test contains unknown cases: {unknown[:5]}")

        for fold in folds:
            # For backward compatibility, allow optional legacy test field.
            split_keys = ["train", "val"] + (["test"] if "test" in fold else [])
            for split_key in split_keys:
                names = fold.get(split_key, [])
                if not isinstance(names, list):
                    raise ValueError(f"split file field folds[].{split_key} must be list: {split_file_path}")
                unknown = [name for name in names if str(name) not in all_cases]
                if unknown:
                    raise ValueError(
                        f"split file contains cases not found in dataset.yaml for folds[].{split_key}: {unknown[:5]}"
                    )

        return payload

    def save_split_file(self, split_file_path, split_payload):
        os.makedirs(os.path.dirname(split_file_path), exist_ok=True)
        with open(split_file_path, "w", encoding="utf-8") as f:
            yaml.dump(split_payload, f, sort_keys=False)
        print(f"Saved fixed split file: {split_file_path}")

    def resolve_patch_index_path(self):
        patch_index_file = getattr(self.args, "patch_index_file", None)
        if patch_index_file is None or str(patch_index_file).strip() == "":
            patch_index_file = os.path.join(self.args.data_root, "list", "tooth_patch_index.npz")
        return patch_index_file

    def load_patch_index(self, patch_index_file, selected_names):
        if not os.path.exists(patch_index_file):
            raise FileNotFoundError(
                f"Patch index file not found: {patch_index_file}. "
                "Please generate it first or disable patch_index_enabled."
            )

        selected_name_set = set(selected_names)
        with np.load(patch_index_file, allow_pickle=False) as patch_index:
            if "case_name" not in patch_index or "start_zyx" not in patch_index:
                raise ValueError("Patch index must contain case_name and start_zyx arrays")

            case_names = patch_index["case_name"]
            starts = patch_index["start_zyx"]
            sizes = patch_index["size_zyx"] if "size_zyx" in patch_index else None

            patch_records = []
            for idx, case_name in enumerate(case_names.tolist()):
                case_name = str(case_name)
                if case_name not in selected_name_set:
                    continue

                crop_start = tuple(int(v) for v in starts[idx].tolist())
                crop_size = self.patch_size if sizes is None else tuple(int(v) for v in sizes[idx].tolist())
                if tuple(crop_size) != tuple(self.patch_size):
                    raise ValueError(
                        f"Patch index crop size {crop_size} does not match training_size {self.patch_size} "
                        f"for case {case_name}."
                    )
                patch_records.append((case_name, crop_start, crop_size))

        if not patch_records:
            raise ValueError(
                f"No indexed patches found for mode={self.mode}. "
                f"Selected cases: {sorted(selected_name_set)}"
            )

        return patch_records

    def read_case(self, name):
        img_path, lab_path = self.case_files[name]
        itk_img = sitk.ReadImage(img_path)
        itk_lab = sitk.ReadImage(lab_path)

        assert itk_img.GetSize() == itk_lab.GetSize()
        return itk_img, itk_lab

    def load_case_tensors(self, name):
        itk_img, itk_lab = self.read_case(name)
        spacing = np.array(itk_lab.GetSpacing()).tolist()[::-1]
        img, lab = self.preprocess(itk_img, itk_lab)
        return img, lab, np.array(spacing), f"{name}.nii.gz"

    def get_case_tensors_from_cache(self, name):
        if name in self.case_cache:
            img, lab = self.case_cache.pop(name)
            self.case_cache[name] = (img, lab)
            return img, lab

        img, lab, _, _ = self.load_case_tensors(name)
        if self.case_cache_size > 0:
            self.case_cache[name] = (img, lab)
            while len(self.case_cache) > self.case_cache_size:
                self.case_cache.popitem(last=False)
        return img, lab

    def preprocess(self, itk_img, itk_lab):
        img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        lab = sitk.GetArrayFromImage(itk_lab).astype(np.uint8)

        # Robust normalization for CBCT-like intensity range.
        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        img = (img - p1) / max(1e-8, (p99 - p1))

        if hasattr(self.args, "classes"):
            max_allowed = int(self.args.classes) - 1
            if int(lab.max()) > max_allowed:
                raise ValueError(
                    f"Label value {int(lab.max())} exceeds classes-1 ({max_allowed}). "
                    "Please check class mapping in conversion."
                )

        z, y, x = img.shape

        if z < self.args.training_size[0]:
            diff = int(math.ceil((self.args.training_size[0] - z) / 2))
            img = np.pad(img, ((diff, diff), (0, 0), (0, 0)))
            lab = np.pad(lab, ((diff, diff), (0, 0), (0, 0)))
        if y < self.args.training_size[1]:
            diff = int(math.ceil((self.args.training_size[1] + 2 - y) / 2))
            img = np.pad(img, ((0, 0), (diff, diff), (0, 0)))
            lab = np.pad(lab, ((0, 0), (diff, diff), (0, 0)))
        if x < self.args.training_size[2]:
            diff = int(math.ceil((self.args.training_size[2] + 2 - x) / 2))
            img = np.pad(img, ((0, 0), (0, 0), (diff, diff)))
            lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff)))

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        assert tensor_img.shape == tensor_lab.shape
        return tensor_img, tensor_lab

    def crop_tensor_by_start(self, tensor_img, tensor_lab, crop_start, crop_size):
        z, y, x = [int(v) for v in crop_start]
        d, h, w = [int(v) for v in crop_size]

        _, _, depth, height, width = tensor_img.shape
        z = max(0, min(z, max(depth - d, 0)))
        y = max(0, min(y, max(height - h, 0)))
        x = max(0, min(x, max(width - w, 0)))

        cropped_img = tensor_img[:, :, z:z + d, y:y + h, x:x + w]
        cropped_lab = tensor_lab[:, :, z:z + d, y:y + h, x:x + w]
        return cropped_img.contiguous(), cropped_lab.contiguous()

    def apply_train_augmentations(self, tensor_img, tensor_lab, crop_start=None, crop_size=None):
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
        tensor_lab = tensor_lab.unsqueeze(0).unsqueeze(0)

        if self.args.aug_device == "gpu":
            tensor_img = tensor_img.cuda(self.args.proc_idx)
            tensor_lab = tensor_lab.cuda(self.args.proc_idx)

        d, h, w = self.args.training_size

        if crop_start is None:
            if np.random.random() < 0.2:
                tensor_img, tensor_lab = augmentation.crop_3d(
                    tensor_img, tensor_lab, [d + 32, h + 32, w + 32], mode="random"
                )
                tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(
                    tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate
                )
                tensor_img, tensor_lab = augmentation.crop_3d(
                    tensor_img, tensor_lab, self.args.training_size, mode="center"
                )
            else:
                tensor_img, tensor_lab = augmentation.crop_3d(
                    tensor_img, tensor_lab, self.args.training_size, mode="random"
                )
        else:
            crop_size = self.patch_size if crop_size is None else crop_size
            tensor_img, tensor_lab = self.crop_tensor_by_start(tensor_img, tensor_lab, crop_start, crop_size)
            if np.random.random() < 0.2:
                tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(
                    tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate
                )

        if np.random.random() < 0.3:
            tensor_img = augmentation.mirror(tensor_img, axis=2)
            tensor_lab = augmentation.mirror(tensor_lab, axis=2)
        if np.random.random() < 0.3:
            tensor_img = augmentation.mirror(tensor_img, axis=1)
            tensor_lab = augmentation.mirror(tensor_lab, axis=1)

        if np.random.random() < 0.2:
            tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
        if np.random.random() < 0.2:
            tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
        if np.random.random() < 0.2:
            tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.0])
        if np.random.random() < 0.2:
            tensor_img = augmentation.gaussian_noise(tensor_img, std=float(np.random.random() * 0.05))

        tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()
        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape == tensor_lab.shape
        return tensor_img, tensor_lab

    def __getitem__(self, idx):
        if self.mode == "train":
            if self.use_patch_index:
                idx = idx % len(self.patch_records)
                case_name, crop_start, crop_size = self.patch_records[idx]
                tensor_img, tensor_lab = self.get_case_tensors_from_cache(case_name)
                tensor_img, tensor_lab = self.apply_train_augmentations(
                    tensor_img, tensor_lab, crop_start=crop_start, crop_size=crop_size
                )
            else:
                idx = idx % len(self.img_list)
                tensor_img = self.img_list[idx]
                tensor_lab = self.lab_list[idx]
                tensor_img, tensor_lab = self.apply_train_augmentations(tensor_img, tensor_lab)
            return tensor_img, tensor_lab

        name = self.selected_names[idx]
        tensor_img, tensor_lab, spacing, img_name = self.load_case_tensors(name)
        tensor_img = tensor_img.unsqueeze(0)
        tensor_lab = tensor_lab.unsqueeze(0)

        assert tensor_img.shape == tensor_lab.shape
        return tensor_img, tensor_lab, spacing, img_name

