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
import constants as C

"""
Both train and test sets should train.png and test.png
"""

class RSDataset(Dataset):
    r""" Pavia Dataset """
    
    def __init__(self, name, mode, split, transform=None):
        """
        Args:
            name (String): Loads Reykjavik or Pavia dataset. Can be 'reykjavik' or 'pavia'. 
            mode (String): Loads train or test set. Can be 'train' or 'test'. 
            split (Integer): Split type. Can 'original', 'horizontal' or 'vertical'.
            transform (Callable, optional): Optional transform to be applied on a sample. 
        """
        self.name = name.lower()
        self.mode = mode.lower()
        self.split = split.lower()
        self.transform = transform
        self._set_paths()
        self.points = self._load_points()
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        # transform'u unutma
        pass
    
    """
    Sets paths for given dataset name. 
    """
    def _set_paths(self):
        if self.name == 'reykjavik':
            self.img_dir = C.REYKJAVIK_DIR_PATH
            self.pc_paths = [osp.join(self.img_dir, 'pan')]
            if self.split == 'original':
                self.ap_paths = [osp.join(self.img_dir, 'APs', 'a_area=' + str(t) + '_pca=1.png') for t in C.REY_TS]
                if self.mode == 'train':
                    self.ann_path = osp.join(self.img_dir, 'train.png')
                elif self.mode == 'test':
                    self.ann_path = osp.join(self.img_dir, 'GT.png')
            else:
                self.ap_paths = [osp.join(self.img_dir, 'split_APs', 'a_area=' + str(i) + '_pca=1_' + \
                                          self.split + '_' + self.mode + '.png') for i in C.REY_TS]

        elif self.name == 'pavia':
            self.img_dir = C.PAVIA_DIR_PATH
            self.pc_paths = [osp.join(self.img_dir, 'paviaPCA' + str(i)) for i in range(1, 5)]
            if self.split == 'original':
                self.ap_paths = [osp.join(self.img_dir, 'APs', 'a_area=' + str(t) + '_pca=' + str(i) + \
                                          '.png') for t in C.PAV_TS for i in range(1, 5)]
                if self.mode == 'train':
                    self.ann_path = osp.join(self.img_dir, 'Train_University.bmp')
                elif self.mode == 'test':
                    self.ann_path = osp.join(self.img_dir, 'Test_University.bmp')
            else:
                self.ap_paths = [osp.join(self.img_dir, 'split_APs', 'a_area=' + str(t) + '_pca=' + str(i) + '_' \
                                          + self.split + '_' + self.mode + '.png') for t in C.PAV_TS for i in range(1, 5)]
        
        if self.split != 'original':
            self.ann_path = osp.join(self.img_dir, self.mode + "_" + self.split + ".png" )
            self.pc_paths = [p + '_' + self.split + '_' + self.mode for p in self.pc_paths]
            
        self.pc_paths = [p + '.png' for p in self.pc_paths]
    
    """
    Loads labeled points from given set for given split. 
    """
    def _load_points(self):
        pass
