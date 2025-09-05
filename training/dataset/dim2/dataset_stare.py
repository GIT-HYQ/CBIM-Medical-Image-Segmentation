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


class StareDataset(Dataset):
    def __init__(self, args, mode = 'train', k_fold=5, k=0, seed=0):
        data_path = args.data_root
        if mode=="train":
            self.name_list = sorted(os.listdir(data_path + '/images/training/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/training/'))
            self.data = []
            for i, image_name in enumerate(self.name_list):
                img_path = data_path + '/images/training/' + self.name_list[i]
                mask_name = image_name.replace('.png', '.ah.png')
                mask_path = data_path + '/annotations/training/' + mask_name
                self.data.append([img_path, mask_path])
        elif mode=="val":
            self.name_list = sorted(os.listdir(data_path + '/images/validation/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/validation/'))
            self.data = []
            for i, image_name in enumerate(self.name_list):
                img_path = data_path + '/images/validation/' + self.name_list[i]
                mask_name = image_name.replace('.png', '.ah.png')
                mask_path = data_path + '/annotations/validation/' + mask_name
                self.data.append([img_path, mask_path])
        elif mode=="test":
            self.name_list = sorted(os.listdir(data_path + '/images/test/'))
            self.label_list = sorted(os.listdir(data_path + '/annotations/test/'))
            self.data = []
            for i, image_name in enumerate(self.name_list):
                img_path = data_path + '/images/test/' + self.name_list[i]
                mask_name = image_name.replace('.png', '.ah.png')
                mask_path = data_path + '/annotations/test/' + mask_name
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
        image, label = self.center_crop_2d(image, label)
        # x, y = image.shape[1], image.shape[0]
        # if x < self.args.training_size[0]:
        #     diff = (self.args.training_size[0] - x)
        #     image = np.pad(image, ((0,0), (diff//2, diff-diff//2)))
        #     label = np.pad(label, ((0,0), (diff//2, diff-diff//2)))
        # if y < self.args.training_size[1]:
        #     diff = (self.args.training_size[1] + 10 - y)
        #     image = np.pad(image, ((diff//2, diff-diff//2), (0,0)))
        #     label = np.pad(label, ((diff//2, diff-diff//2), (0,0)))
        # # image_size = (self.args.training_size[0], self.args.training_size[1])
        # # image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
        # # label = cv2.resize(label, image_size, interpolation=cv2.INTER_NEAREST)
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

    def center_crop_2d(self, img, label):
        H, W = img.shape

        diff_H = H - self.args.training_size[0]
        diff_W = W - self.args.training_size[1]

        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[rand_x:rand_x+self.args.training_size[0], rand_y:rand_y+self.args.training_size[1]]
        croped_lab = label[rand_x:rand_x+self.args.training_size[0], rand_y:rand_y+self.args.training_size[1]]

        return croped_img, croped_lab


