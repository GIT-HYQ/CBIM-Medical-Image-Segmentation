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
          
            # Gaussian Noise
            tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)
            # Additive brightness
            tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)
            # gamma
            tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)

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
