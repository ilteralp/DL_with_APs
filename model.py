#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:25:20 2020

@author: melike

Resources:
    1. https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
"""

import torch
import torch.nn as nn

K1, K2, K3 = 5, 3, 3                                                            # Kernel sizes of corresponding conv layers.


class APNet(nn.Module):
    r""" 
    Neural network architecture of Deep Learning With Attribute Profiles for 
    Hyperspectral Image Classification paper. 
    """
    
    def __init__(self, in_channels, num_classes, L):
        """
        Args:
            in_channels (int): Number of channels of a sample patch. 
            num_classes (int): Number of classes within the dataset. 
            L (int): Number of AP images (length of AP thresholds + PC image of that channel).
        """
        super(APNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.convd_depth = L - (K1 - 1 + K2 - 1 + K3 - 1)                       # L (number of thresholds of AP) varies, so calculate convd depth.
        self.features = nn.Sequential(
            nn.Conv3d(self.in_channels, 48, kernel_size=K1),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 96, kernel_size=K2),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 96, kernel_size=K3),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(96 * self.convd_depth, 1024),                             # convd_depth varies but convd_h and convd_w are constant, being 9-(5-1+3-1+3-1)=1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)                                                 # Flatten starts from 1st dim, leave batch_size dim.
        x = self.classifier(x)                                                  # No softmax due to using cross entropy loss.
        return x