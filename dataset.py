#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:53:00 2020

@author: melike
"""
import numpy as np
import os
import os.path as osp
import torch
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt
from collections import Counter
import constants as C

"""
Both train and test sets should train.png and test.png
"""

class RSDataset(Dataset):
    r""" Pavia Dataset """
    
    def __init__(self, name, mode, split, patch_size=9, transform=None):
        """
        Args:
            name (String): Loads Reykjavik or Pavia dataset. Can be 'reykjavik' or 'pavia'. 
            mode (String): Loads train or test set. Can be 'train' or 'test'. 
            split (Integer): Split type. Can 'original', 'horizontal' or 'vertical'.
            patch_size (Integer, optional): Patch size, must be odd number. 
            transform (Callable, optional): Optional transform to be applied on a sample. 
        """
        self.name = name.lower()
        self.mode = mode.lower()
        self.split = split.lower()
        self.patch_size = patch_size
        self.pad = int(self.patch_size / 2)
        self.ps = []
        self.labels = []
        self.transform = transform
        self._set_paths()
        self._load_points()
        self.load_imgs()
        self._check_items()
    
    def __len__(self):
        return len(self.ps)
    
    def __getitem__(self, index):
        p = self.ps[index]
        patch = self.load_patch(p)
        if self.transform:
            patch = self.transform(patch)
            
        return torch.from_numpy(patch).float(), torch.tensor(self.labels[index]), p

    """
    Checks and removes samples of a class in case of they are in insufficient number. 
    """
    def _check_items(self):
        if self.split != 'original':
            mask = [True] * len(self.labels)
            inds = []
            if self.name == 'reykjavik':
                if self.split == 'vertical':
                    inds = [i for i, label in enumerate(self.labels) if label in C.REY_VER_REMOVE_LABELS]
            elif self.name == 'pavia':
                if self.split == 'horizontal':
                    inds = [i for i, label in enumerate(self.labels) if label in C.PAV_HOR_REMOVE_LABELS]
                elif self.split == 'vertical':
                    inds = [i for i, label in enumerate(self.labels) if label in C.PAV_VER_REMOVE_LABELS]
            for i in inds:
                mask[i] = False
            
            self.labels = np.array(self.labels)[mask]                           # Remove eliminated class samples.
            self.ps = np.array(self.ps)[mask]
    
    def _set_paths(self):
        if self.name == 'reykjavik':
            self.c = 1
            self.ts = C.REY_TS
            self.img_dir = C.REYKJAVIK_DIR_PATH
            if self.mode == 'train':
                self.ann_path = osp.join(self.img_dir, 'train.png')
            elif self.mode == 'test':
                self.ann_path = osp.join(self.img_dir, 'GT.png')
            
        elif self.name == 'pavia':
            self.c = 4
            self.ts = C.PAV_TS
            self.img_dir = C.PAVIA_DIR_PATH
            if self.mode == 'train':
                self.ann_path = osp.join(self.img_dir, 'Train_University.bmp')
            elif self.mode == 'test':
                self.ann_path = osp.join(self.img_dir, 'Test_University.bmp')
        
        if self.split == 'original':
            self.ap_dir = osp.join(self.img_dir, 'APs')
        else: 
            self.ap_dir = osp.join(self.img_dir, 'split_APs')
            self.ann_path = osp.join(self.img_dir, self.mode + "_" + self.split + ".png" )
            
    """
    Loads labeled points from given set for given split. 
    """
    def _load_points(self):
        ann_img = io.imread(self.ann_path)
        self.h, self.w = ann_img.shape
        for i in range(self.h):
            for j in range(self.w):
                if ann_img[i, j] != 0:
                    self.ps.append([i, j])
                    self.labels.append(ann_img[i, j])
    
    """
    Returns AP images of the dataset. 
    """
    def _get_ap_path(self, pc, t):
        path = osp.join(self.ap_dir, 'a_area=' + str(t) + '_pca=' + str(pc))
        if self.split != 'original':
            path += '_' + self.split + '_' + self.mode
        return path + '.png'
    
    """
    Returns PC images of the dataset. 
    """
    def _get_pc_path(self, pc):
        if self.name == 'pavia':
            path = osp.join(self.img_dir, 'paviaPCA' + str(pc))
        elif self.name == 'reykjavik':
            path = osp.join(self.img_dir, 'pan')
        if self.split != 'original':
            path += '_' + self.split + '_' + self.mode
        return path + '.png'
    
    """
    Loads AP and PC images. 
    """
    def load_imgs(self):
        self.pcs = []
        self.aps = {}
        for i in range(1, self.c + 1):
            pc = io.imread(self._get_pc_path(i))
            self.pcs.append(np.pad(pc, (self.pad, self.pad), 'reflect'))        # Beware the padded images...
            _imgs = []
            for t in self.ts:
                ap = io.imread(self._get_ap_path(i, t))
                _imgs.append(np.pad(ap, (self.pad, self.pad), 'reflect'))
            self.aps[i-1] = _imgs
        return self.pcs, self.aps
    
    """
    Takes a point and returns its patch.
    """
    def load_patch(self, p):
        tl_x = p[0]                                                             # it's p[0] - pad, but due to padding it becomes p[0].
        tl_y = p[1]                                                             # same as above. 
        patch = np.zeros((self.c, len(self.ts) + 1, self.patch_size, self.patch_size))
        
        for i in range(self.c):
            for j, ap in enumerate(self.aps[i]):
                patch[i][j] = ap[tl_x : tl_x + self.patch_size, tl_y : tl_y + self.patch_size]
                patch[i][j] = patch[i][j] - patch[i][j].mean()
            patch[i][len(self.ts)] = self.pcs[i][tl_x : tl_x + self.patch_size, tl_y : tl_y + self.patch_size]
            patch[i][len(self.ts)] = patch[i][len(self.ts)] - patch[i][len(self.ts)].mean()
        return patch

# reykTr = RSDataset(name='reykjavik', mode='train', split='original')
# pavHorTest = RSDataset(name='pavia', mode='test', split='horizontal')
# sorted(Counter(pavHorTest.labels).items())