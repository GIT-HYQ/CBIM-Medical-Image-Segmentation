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

        random.Random(seed).shuffle(img_name_list)

        length = len(img_name_list)
        fold_len = max(1, length // k_fold)
        test_start = k * fold_len
        test_end = (k + 1) * fold_len if k < k_fold - 1 else length

        test_name_list = img_name_list[test_start:test_end]
        test_name_set = set(test_name_list)
        train_name_list = [name for name in img_name_list if name not in test_name_set]

        val_len = max(1, int(len(train_name_list) * 0.2)) if len(train_name_list) > 1 else 0
        val_name_list = train_name_list[:val_len]
        train_name_list = train_name_list[val_len:] if val_len > 0 else train_name_list

        if mode == "train":
            selected_names = train_name_list
        elif mode == "val":
            selected_names = val_name_list if len(val_name_list) > 0 else test_name_list
        else:
            selected_names = test_name_list

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

