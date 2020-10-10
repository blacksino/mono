# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
import PIL
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from path import Path
import cv2
import time

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 root_path,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 with_pose ,
                 is_train=False):
        super(MonoDataset, self).__init__()

        self.root_path = Path(root_path)
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.is_train = is_train
        self.with_pose = with_pose
        self.scene_list_path = self.root_path / 'train.txt' if is_train else self.root_path/ 'val.txt'
        self.to_tensor = transforms.ToTensor()
        self.scenes = [Path(folder[:-1]) for folder in open(self.scene_list_path)]
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.samples = self.crawl_folders(3)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = False

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    inputs[(n,im,i)].save()

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        sample = self.samples[index].copy()
        do_color_aug = self.is_train and random.random()>0.5
        do_flip = self.is_train and random.random() > 0.5

        for i in self.frame_idxs:
            if i == 0:
                inputs[("color", i, -1)] = self.get_color(sample['tgt_left'],do_flip)
            if i == "s":
                inputs[("color", i, -1)] = self.get_color(sample['tgt_right'],do_flip)
            if i == -1:
                inputs[("color", i, -1)] = self.get_color(sample['ref_left'][0],do_flip)
            if i == 1:
                inputs[("color", i, -1)] = self.get_color(sample['ref_left'][1],do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):

            K = sample['intrinsics']
            zeros = np.zeros((3,1))
            K = np.c_[K,zeros]
            zeros = np.array([0,0,0,1]).reshape(1,4)
            K = np.r_[K,zeros].astype(np.float32)
            K[0, :] /= 1280
            K[1, :] /= 1024
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)


        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(sample['left_depth'], do_flip)
            depth_gt = torch.from_numpy(depth_gt).float()
            inputs["depth_gt"] = depth_gt

        if "s" in self.frame_idxs:
            factor = 0.1
            if sample['LR_Distance'] > 4.36:
                factor = 0.105
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1
            # if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * factor

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
        return inputs

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2  # demi_length = 1
        shifts = list(range(-demi_length, demi_length + 1))  # (-1,1)
        shifts.pop(demi_length)  # 除去0，前后帧的索引
        for scene in self.scenes:
            intrinsics = np.load(scene / 'rec_intrinsic.npy').astype(np.float32).reshape((3, 3))
            imgs_left = sorted(Path(scene / 'data/rec_rgb').glob('Left*'))
            imgs_right = sorted(Path(scene / 'data/rec_rgb').glob('Right*'))
            current_depth = []
            for img in imgs_left:
                d = scene / 'data' / 'rec_gt_with_mask' / (
                        img.name[0:5] + 'gt' + img.name[10:16] + '.tiff')
                assert (d.isfile()), "depth file {} not found".format(str(d))
                current_depth.append(d)
            LR_Matrix_path = scene / 'endoscope_calibration.yaml'
            LR_Matrix = cv2.FileStorage(LR_Matrix_path, cv2.FileStorage_READ)
            T = np.array(LR_Matrix.getNode('T').mat()).astype(float)
            T = np.linalg.norm(T)
            if len(imgs_left) < sequence_length:
                continue
            for i in range(demi_length, len(imgs_left) - demi_length):  # 确保待处理的图片，每一帧都有前后帧
                sample = {'intrinsics': intrinsics, 'tgt_left': imgs_left[i], 'tgt_right': imgs_right[i],
                          'ref_left': [], 'ref_right': [],'left_depth': current_depth[i],'LR_Distance': T}
                if self.with_pose:
                    sample['tgt_json'] = []
                    sample['refs_jsons'] = []
                    name = imgs_left[i].name
                    number = Path(self.getting_number_from_path(name))
                    json_path = scene / 'data/frame_data/frame_data' + number + '.json'
                    sample['tgt_json'].append(json_path)
                for j in shifts:
                    sample['ref_left'].append(imgs_left[i + j])
                    sample['ref_right'].append(imgs_right[i + j])
                    if self.with_pose:
                        name = imgs_left[i + j].name
                        number = Path(self.getting_number_from_path(name))
                        json_path = scene / 'data/frame_data/frame_data' + number + '.json'
                        sample['refs_jsons'].append(json_path)
                sequence_set.append(sample)  # 第i帧的DICT集合
        random.shuffle(sequence_set)
        return  sequence_set

    def load_as_float(self,path):
        return cv2.imread(path).astype(np.float32)

    def get_color(self, folder, do_flip):
        raise NotImplementedError

    def get_depth(self, folder,do_flip):
        raise NotImplementedError

    def getting_number_from_path(path):
        name = path.name
        number = filter(Path.isdigit, name)
        number_str = ''.join(list(number))
        return number_str



