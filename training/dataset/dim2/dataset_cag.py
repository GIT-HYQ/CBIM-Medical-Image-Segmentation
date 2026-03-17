import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
import logging
import copy
import datetime
import pandas as pd
import cv2
class CAGDataset2(Dataset):
    def __init__(self, args, mode = 'train', k_fold=5, k=0, seed=0):
        data_path = args.data_root
        if mode=="train":
            self.name_list = sorted(os.listdir(data_path + '/images/training/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/training/'))
            self.data = []
            for i in range(len(self.name_list)):
                img_path = data_path + '/images/training/' + self.name_list[i]
                mask_path = data_path + '/annotations/training/' + self.label_list[i]
                self.data.append([img_path, mask_path])
        elif mode=="val":
            self.name_list = sorted(os.listdir(data_path + '/images/validation/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/validation/'))
            self.data = []
            for i in range(len(self.name_list)):
                img_path = data_path + '/images/validation/' + self.name_list[i]
                mask_path = data_path + '/annotations/validation/' + self.label_list[i]
                self.data.append([img_path, mask_path])
        elif mode=="test":
            self.name_list = sorted(os.listdir(data_path + '/images/test/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/test/'))
            self.data = []
            for i in range(len(self.name_list)):
                img_path = data_path + '/images/test/' + self.name_list[i]
                mask_path = data_path + '/annotations/test/' + self.label_list[i]
                self.data.append([img_path, mask_path])
        else:
            print("Error, invalid split type")

        # mode_map = {
        #     "train": "Training",
        #     'val': 'Val',
        #     'test': "Test"
        # }
        # in_mode = mode_map.get(mode)
        # df = pd.read_csv(os.path.join(args.data_root, 'CAG_231215_' + in_mode + '_GroundTruth.csv'), encoding='gbk')
        # self.name_list = df.iloc[:,1].tolist()
        # self.label_list = df.iloc[:,2].tolist()
        
        self.mode = mode
        self.args = args
        
        logging.info(f"Start loading {self.mode} data")

    def __len__(self):
        return len(self.name_list)

    def preprocess(self, img, lab):      
        img = img / 255

        img = img.astype(np.float32)
        lab = lab.astype(np.uint8)

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        return tensor_img, tensor_lab


    def __getitem__(self, index):

        index = index % len(self)
        name = self.name_list[index]
        img_path, msk_path = self.data[index]
        
        # mask_name = self.label_list[index]
        # msk_path = os.path.join(self.args.data_root, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype("float32")
        label = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE).astype("float32")
        image_size = (self.args.training_size[0], self.args.training_size[1])
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, image_size, interpolation=cv2.INTER_NEAREST)
        image = image.reshape((1, image.shape[0], image.shape[1]))
        label = label.reshape((1, label.shape[0], label.shape[1]))

        tensor_img, tensor_lab = self.preprocess(image, label)

        if self.mode == 'train':
            # print(tensor_img.shape, tensor_lab.shape)
            tensor_img = tensor_img.unsqueeze(0)
            tensor_lab = tensor_lab.unsqueeze(0)
          
            # # Gaussian Noise
            # tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)
            # # Additive brightness
            # tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)
            # # gamma
            # tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)

            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_2d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_2d(tensor_img, tensor_lab, self.args.training_size, mode='random')

            tensor_img, tensor_lab = tensor_img.squeeze(0), tensor_lab.squeeze(0)
            # print(tensor_img.shape, tensor_lab.shape)
        # else:
        #     tensor_img, tensor_lab = self.center_crop(tensor_img, tensor_lab)
        
        assert tensor_img.shape == tensor_lab.shape
        
        if self.mode == 'train':
            return tensor_img, tensor_lab
        else:
            return tensor_img, tensor_lab, np.array((1.0, 1.0, 1.0)), name.split('/')[-1]


    def center_crop(self, img, label):
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)
        D, H, W = img.shape

        diff_H = H - self.args.training_size[0]
        diff_W = W - self.args.training_size[1]

        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[:, rand_x:rand_x+self.args.training_size[0], rand_y:rand_y+self.args.training_size[0]]
        croped_lab = label[:, rand_x:rand_x+self.args.training_size[1], rand_y:rand_y+self.args.training_size[1]]

        return croped_img, croped_lab

class CAGDataset(Dataset):
    def __init__(self, args, mode = 'train', k_fold=5, k=0, seed=0):

        mode_map = {
            "train": "Training",
            'val': 'Val',
            'test': "Test"
        }
        in_mode = mode_map.get(mode)
        df = pd.read_csv(os.path.join(args.data_root, 'CAG_231215_' + in_mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        
        self.mode = mode
        self.args = args
        
        # random.Random(seed).shuffle(img_name_list)

        # length = len(img_name_list)
        # test_name_list = img_name_list[k*(length//k_fold):(k+1)*(length//k_fold)]
        # train_name_list = img_name_list
        # train_name_list = list(set(img_name_list) - set(test_name_list))

        # if mode == 'train':
        #     img_name_list = train_name_list
        # else:
        #     img_name_list = test_name_list
        
        logging.info(f"Start loading {self.mode} data")
        
        # path = args.data_root

        # img_list = []
        # lab_list = []
        # spacing_list = []
        
        # for name in img_name_list:
        #     for idx in [0, 1]:
                
        #         img_name = name + '_%d.nii.gz'%idx
        #         lab_name = name + '_%d_gt.nii.gz'%idx
                
        #         itk_img = sitk.ReadImage(os.path.join(path, img_name))
        #         itk_lab = sitk.ReadImage(os.path.join(path, lab_name))

        #         spacing = np.array(itk_lab.GetSpacing()).tolist()
        #         spacing_list.append(spacing[::-1])

        #         assert itk_img.GetSize() == itk_lab.GetSize()

        #         img, lab = self.preprocess(itk_img, itk_lab)

        #         img_list.append(img)
        #         lab_list.append(lab)
      
        # self.img_slice_list = []
        # self.lab_slice_list = []
        # if self.mode == 'train':
        #     for i in range(len(img_list)):

        #         z, x, y = img_list[i].shape

        #         for j in range(z):
        #             self.img_slice_list.append(copy.deepcopy(img_list[i][j]))
        #             self.lab_slice_list.append(copy.deepcopy(lab_list[i][j]))
        #     del img_list
        #     del lab_list
        # else:
        #     self.img_slice_list = img_list
        #     self.lab_slice_list = lab_list
        #     self.spacing_list = spacing_list
        
        
        # logging.info(f"Load done, length of dataset: {len(self.img_slice_list)}")



    def __len__(self):
        return len(self.name_list)

    def preprocess(self, img, lab):
        
        # img = sitk.GetArrayFromImage(itk_img)
        # lab = sitk.GetArrayFromImage(itk_lab)

        # max98 = np.percentile(img, 98)
        # img = np.clip(img, 0, max98)
            
        # z, y, x = img.shape
        # if x < self.args.training_size[0]:
        #     diff = (self.args.training_size[0] + 10 - x) // 2
        #     img = np.pad(img, ((0,0), (0,0), (diff, diff)))
        #     lab = np.pad(lab, ((0,0), (0,0), (diff,diff)))
        # if y < self.args.training_size[1]:
        #     diff = (self.args.training_size[1] + 10 -y) // 2
        #     img = np.pad(img, ((0,0), (diff, diff), (0,0)))
        #     lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))

        # img = img / max98
        img = img / 255

        img = img.astype(np.float32)
        lab = lab.astype(np.uint8)

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        return tensor_img, tensor_lab


    def __getitem__(self, index):

        index = index % len(self)
        name = self.name_list[index]
        img_path = os.path.join(self.args.data_root, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.args.data_root, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype("float32")
        label = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE).astype("float32")
        image_size = (self.args.training_size[0], self.args.training_size[1])
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, image_size, interpolation=cv2.INTER_NEAREST)
        image = image.reshape((1, image.shape[0], image.shape[1]))
        label = label.reshape((1, label.shape[0], label.shape[1]))

        tensor_img, tensor_lab = self.preprocess(image, label)

        # tensor_img = self.img_slice_list[idx]
        # tensor_lab = self.lab_slice_list[idx]

        if self.mode == 'train':
            # print(tensor_img.shape, tensor_lab.shape)
            tensor_img = tensor_img.unsqueeze(0)
            tensor_lab = tensor_lab.unsqueeze(0)
          
            # # Gaussian Noise
            # tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)
            # # Additive brightness
            # tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)
            # # gamma
            # tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)

            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_2d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_2d(tensor_img, tensor_lab, self.args.training_size, mode='random')

            tensor_img, tensor_lab = tensor_img.squeeze(0), tensor_lab.squeeze(0)
            # print(tensor_img.shape, tensor_lab.shape)
        # else:
        #     tensor_img, tensor_lab = self.center_crop(tensor_img, tensor_lab)
        
        assert tensor_img.shape == tensor_lab.shape
        
        if self.mode == 'train':
            return tensor_img, tensor_lab
        else:
            return tensor_img, tensor_lab, np.array((1.0, 1.0, 1.0)), name.split('/')[-1]


    def center_crop(self, img, label):
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)
        D, H, W = img.shape

        diff_H = H - self.args.training_size[0]
        diff_W = W - self.args.training_size[1]

        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[:, rand_x:rand_x+self.args.training_size[0], rand_y:rand_y+self.args.training_size[0]]
        croped_lab = label[:, rand_x:rand_x+self.args.training_size[1], rand_y:rand_y+self.args.training_size[1]]

        return croped_img, croped_lab

class CAGDataset3(Dataset):
    """
    基于 CAGDataset2:
    - 图像输入支持 2 通道: [gray, prior_2c]
    - prior 路径规则:
      data_root/annotations_2c/{split}/{image_name}  (同名同后缀)
    """
    def __init__(self, args, mode='train', k_fold=5, k=0, seed=0):
        data_path = args.data_root
        if mode == "train":
            self.name_list = sorted(os.listdir(data_path + '/images/training/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/training/'))
            self.data = []
            for i in range(len(self.name_list)):
                img_path = data_path + '/images/training/' + self.name_list[i]
                mask_path = data_path + '/annotations/training/' + self.label_list[i]
                self.data.append([img_path, mask_path, "training"])
        elif mode == "val":
            self.name_list = sorted(os.listdir(data_path + '/images/validation/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/validation/'))
            self.data = []
            for i in range(len(self.name_list)):
                img_path = data_path + '/images/validation/' + self.name_list[i]
                mask_path = data_path + '/annotations/validation/' + self.label_list[i]
                self.data.append([img_path, mask_path, "validation"])
        elif mode == "test":
            self.name_list = sorted(os.listdir(data_path + '/images/test/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/test/'))
            self.data = []
            for i in range(len(self.name_list)):
                img_path = data_path + '/images/test/' + self.name_list[i]
                mask_path = data_path + '/annotations/test/' + self.label_list[i]
                self.data.append([img_path, mask_path, "test"])
        else:
            raise ValueError("Error, invalid split type")

        self.mode = mode
        self.args = args
        self.use_prior_input = getattr(args, "use_prior_input", True)  # 默认开启
        self.prior_dir_name = getattr(args, "prior_dir_name", "annotations_2c")  # 新增
        self._prior_warned = False
        logging.info(f"Start loading {self.mode} data")
        logging.info(f"[CAGDataset3] prior_dir_name={self.prior_dir_name}")

    def __len__(self):
        return len(self.name_list)

    def preprocess(self, img, lab):
        # img: [C,H,W], lab: [1,H,W]
        # 通道0灰度图按255归一化；通道1 prior做自适应归一化
        img = img.astype(np.float32)
        if img.shape[0] >= 1:
            img[0] = img[0] / 255.0
        if img.shape[0] >= 2:
            # 若prior最大值>1，按255缩放；否则视为已是概率图[0,1]
            if img[1].max() > 1.0:
                img[1] = img[1] / 255.0
            img[1] = np.clip(img[1], 0.0, 1.0)

        lab = lab.astype(np.uint8)

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()
        return tensor_img, tensor_lab

    def _load_prior(self, split_name, file_name, out_h, out_w):
        # 固定规则：data_root/annotations_2c/{split}/{stem}_manual1{ext}
        stem, ext = os.path.splitext(file_name)
        prior_path = os.path.join(
            self.args.data_root,
            self.prior_dir_name,
            split_name,
            f"{stem}_manual1{ext}"
        )

        prior = cv2.imread(prior_path, cv2.IMREAD_GRAYSCALE)
        if prior is None:
            raise FileNotFoundError(f"[CAGDataset3] prior not found/readable: {prior_path}")

        prior = prior.astype(np.float32)
        if prior.shape != (out_h, out_w):
            prior = cv2.resize(prior, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return prior

    def __getitem__(self, index):
        index = index % len(self)
        name = self.name_list[index]
        img_path, msk_path, split_name = self.data[index]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype("float32")
        label = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE).astype("float32")

        image_size = (self.args.training_size[0], self.args.training_size[1])
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, image_size, interpolation=cv2.INTER_NEAREST)

        if self.use_prior_input:
            prior = self._load_prior(split_name, os.path.basename(img_path), image.shape[0], image.shape[1])
            image = np.stack([image, prior], axis=0)      # [2,H,W]
        else:
            image = image.reshape((1, image.shape[0], image.shape[1]))  # [1,H,W]

        label = label.reshape((1, label.shape[0], label.shape[1]))       # [1,H,W]

        # 可选：调试阶段检查标签值
        # u = np.unique(label)
        # assert set(u.tolist()).issubset({0, 1, 2}), f"Unexpected label values: {u}"

        tensor_img, tensor_lab = self.preprocess(image, label)

        if self.mode == 'train':
            tensor_img = tensor_img.unsqueeze(0)  # [1,C,H,W]
            tensor_lab = tensor_lab.unsqueeze(0)  # [1,1,H,W]

            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_2d(
                tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate
            )
            tensor_img, tensor_lab = augmentation.crop_2d(
                tensor_img, tensor_lab, self.args.training_size, mode='random'
            )

            tensor_img, tensor_lab = tensor_img.squeeze(0), tensor_lab.squeeze(0)

        # 仅检查空间尺寸一致（2通道输入时不能再比较整体shape）
        assert tensor_img.shape[-2:] == tensor_lab.shape[-2:]

        if self.mode == 'train':
            return tensor_img, tensor_lab
        else:
            return tensor_img, tensor_lab, np.array((1.0, 1.0, 1.0)), name.split('/')[-1]
