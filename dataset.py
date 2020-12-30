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
from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt
from collections import Counter
import constants as C

class RSDataset(Dataset):
    r""" Pavia Dataset """
    
    def __init__(self, name, mode, split, patch_size=9, transform=None, tree=None):
        """
        Args:
            name (String): Loads Reykjavik or Pavia dataset. Can be 'reykjavik', 'pavia' or 'pavia_full'. 
            mode (String): Loads train or test set. Can be 'train' or 'test'. 
            split (Integer): Split type. Can be 'original', 'horizontal' or 'vertical'.
            patch_size (Integer, optional): Patch size, must be odd number. 
            transform (Callable, optional): Optional transform to be applied on a sample. 
        """
        self._param_check(name, tree)
        self.name = name.lower()
        self.mode = mode.lower()
        self.split = split.lower()
        self.tree = tree
        if self.tree is not None:
            self.tree = self.tree.lower()
        self.patch_size = patch_size
        self.pad = int(self.patch_size / 2)
        self.ps = []
        self.labels = []
        self.transform = transform
        self._set_paths()
        self._load_points()                                                     # Loads annotated points of given set. 
        self.load_imgs()                                                        # Loads AP and PC images (if possible). 
        self._remove_underrep_class_samples()                                   # In case of split images, removes samples of classes with insufficient number. 
        self._update_labels()                                                   # Makes labels continuos in [0, num_class] range and makes them start from 0.
    
    def __len__(self):
        return len(self.ps)
    
    def __getitem__(self, index):
        p = self.ps[index]
        patch = self.load_patch(p)
        patch = torch.from_numpy(patch).float()                                 # Convert to PyTorch tensor.
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, torch.tensor(self.labels[index]).type(torch.LongTensor)
    
    def print(self):
        print('name: {}, mode: {}, split: {}, tree: {}, len: {}'.format(self.name, self.mode, self.split, self.tree, len(self)))
    
    """
    Checks params.
    """
    def _param_check(self, name, tree):
        if name == 'pavia_full' and tree:
            raise Exception('pavia_full dataset does not work with tree param.')

    """
    Removes samples of a class in case of they are in insufficient number. 
    """
    def _remove_underrep_class_samples(self):
        if self.split != 'original':
            mask = [True] * len(self.labels)
            inds = []
            if self.name == 'reykjavik':
                if self.split == 'vertical':
                    inds = [i for i, label in enumerate(self.labels) if label in C.REY_VER_REMOVE_LABELS]
            elif self.name == 'pavia' or self.name == 'pavia_full':
                if self.split == 'horizontal':
                    inds = [i for i, label in enumerate(self.labels) if label in C.PAV_HOR_REMOVE_LABELS]
                elif self.split == 'vertical':
                    inds = [i for i, label in enumerate(self.labels) if label in C.PAV_VER_REMOVE_LABELS]
            for i in inds:
                mask[i] = False
            
            self.labels = np.array(self.labels)[mask]                           # Remove eliminated class samples.
            self.ps = np.array(self.ps)[mask]
            
    """
    Updates labels to make them continous in range in case of removal of 
    insufficient classes' samples and start labels from 0.
    """
    def _update_labels(self):
        if self.split != 'original':
            if self.name == 'reykjavik':
                if self.split == 'vertical':                                    # 6 -> 5
                    for i, label in enumerate(self.labels):
                        if label == 6:                      self.labels[i] = 5
            elif self.name == 'pavia' or self.name == 'pavia_full':
                if self.split == 'horizontal':                                  # 4 -> 3, 6 -> 4, 8 -> 5, 9 -> 6
                    for i, label in enumerate(self.labels):
                        if label == 4:                      self.labels[i] = 3
                        elif label == 6:                    self.labels[i] = 4
                        elif label == 8:                    self.labels[i] = 5
                        elif label == 9:                    self.labels[i] = 6
                        
                elif self.split == 'vertical':                                  # 4 -> 3, 5 -> 4, 6 -> 5, 9 -> 6 
                    for i, label in enumerate(self.labels):
                        if label == 4:                      self.labels[i] = 3
                        elif label == 5:                    self.labels[i] = 4
                        elif label == 6:                    self.labels[i] = 5
                        elif label == 9:                    self.labels[i] = 6
        
        # Finally, start labels from 0. 
        if isinstance(self.labels, list):                                       # Labels are already converted to nd.array in case of split.
            self.labels = np.array(self.labels)
        self.labels = self.labels - 1
        
    
    def _set_paths(self):
        if self.name == 'reykjavik':
            self.c = 1
            self.ts = C.REY_TS
            self.img_dir = C.REYKJAVIK_DIR_PATH
            if self.mode == 'train':
                self.ann_path = osp.join(self.img_dir, 'train.png')
            elif self.mode == 'test':
                self.ann_path = osp.join(self.img_dir, 'GT.png')
            if self.split == 'vertical':
                self.num_classes = 5
            else:
                self.num_classes = 6
            
        else:
            if self.name == 'pavia':
                self.c = 4
                self.ts = C.PAV_TS
                self.img_dir = C.PAVIA_DIR_PATH
                
            elif self.name == 'pavia_full':
                self.c = 103
                self.ts = None                                                  # No thresholds since AP is not applied. 
                self.img_dir = C.PAVIA_FULL_DIR_PATH
            
            if self.mode == 'train':
                self.ann_path = osp.join(self.img_dir, 'Train_University.bmp')
            elif self.mode == 'test':
                self.ann_path = osp.join(self.img_dir, 'Test_University.bmp')
            if self.split == 'horizontal' or self.split == 'vertical':
                self.num_classes = 6
            else:
                self.num_classes = 9
        
            
        if self.name == 'pavia_full':                                           # Pavia with spectral signature
            self.L = 1
        else:                                                                   # Pavia and Reykjavik
            if self.tree is None:
                self.L = len(self.ts) + 1                                       # Thresholds + original PC image. 
            else:
                self.L = 2 * len(self.ts) + 1                                   # Minmax has 2 * ts + 1
        
            if self.split == 'original':
                if self.tree is None:
                    self.ap_dir = osp.join(self.img_dir, 'APs')
                else:
                    self.ap_dir = osp.join(self.img_dir, 'APs_minmax')
            else: 
                if self.tree is None:
                    self.ap_dir = osp.join(self.img_dir, 'split_APs')
                else:
                    self.ap_dir = osp.join(self.img_dir, 'split_APs_minmax')
        if self.split != 'original':
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
    def _get_ap_path(self, pc, t, j=None):
        if self.tree is None:
            path = osp.join(self.ap_dir, 'a_area=' + str(t) + '_pca=' + str(pc))
        else:
            j_str = 'o' if j == 1 else 'c'
            path = osp.join(self.ap_dir, 'm_' + j_str + '_' + 'area=' + str(t) + '_pca=' + str(pc))
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
        elif self.name == 'pavia_full':
            path = osp.join(self.img_dir, 'pavia' + str(pc))
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
            
            if self.name != 'pavia_full':
                _imgs = []
                if self.tree is None:
                    for t in self.ts:
                        ap = io.imread(self._get_ap_path(i, t))
                        _imgs.append(np.pad(ap, (self.pad, self.pad), 'reflect'))
                else:
                    for t in self.ts:
                        for j in range(2):
                            ap = io.imread(self._get_ap_path(i, t, j))
                            _imgs.append(np.pad(ap, (self.pad, self.pad), 'reflect'))
                self.aps[i-1] = _imgs
        return self.pcs, self.aps
    
    """
    Crops point p's patch from the 2d image.
    """
    def _crop_2d_patch(self, p, img):
        tl_x, tl_y = p                                                          # it's p - pad, but due to padding it becomes p.
        patch = img[tl_x : tl_x + self.patch_size, tl_y : tl_y + self.patch_size]
        return patch - patch.mean()
    
    """
    Takes a point and returns its multi-dim patch.
    """
    def load_patch(self, p):
        patch = np.zeros((self.c, self.L, self.patch_size, self.patch_size))
        if self.name == 'pavia_full':                                           # Only has spectral images, no APs. 
            for i, pc in enumerate(self.pcs):
                patch[i][0] = self._crop_2d_patch(p, pc)                        # 2nd dim is always 0 since it's AP dimension and 'full pavia' has no APs.
        else:
            for i in range(self.c):
                for j, ap in enumerate(self.aps[i]):
                    patch[i][j] = self._crop_2d_patch(p, ap)
                patch[i][len(self.ts)] = self._crop_2d_patch(p, self.pcs[i])    # PC image is after the AP images. 
        return patch
    
    """
    Returns model name for the dataset.
    """
    def get_model_name(self):
        model_name = self.name + '_' + self.split + '_'
        if self.tree is not None:
            model_name += self.tree + '_'
        return model_name

# reykTr = RSDataset(name='reykjavik', mode='train', split='original')
# pavHorTest = RSDataset(name='pavia', mode='test', split='horizontal')
# sorted(Counter(pavHorTest.labels).items())