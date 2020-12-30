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

conv1 = {'ks': 5, 'pad': 2, 'st': 1}                                            # Kernel sizes are from the article. Padding and stride are set in order to make output shape same as input shape. 
conv2 = {'ks': 3, 'pad': 1, 'st': 1}
conv3 = {'ks': 3, 'pad': 1, 'st': 1}

class APNet(nn.Module):
    r""" 
    Neural network architecture of Deep Learning With Attribute Profiles for 
    Hyperspectral Image Classification paper. 
    """
    
    def __init__(self, in_channels, L, H, W, num_classes):
        """
        Args:
            in_channels (int): Number of channels of a sample patch. 
            L (int): Number of AP images (length of AP thresholds + PC image of that channel) or 1 for spectral signature for a sample. 
            H (int): Height of a sample patch. 
            W (int): Width of a sample patch. 
            num_classes (int): Number of classes within the dataset. 
        """
        super(APNet, self).__init__()
        self.in_channels = in_channels
        self.last_conv_d = self.calc_last_conv(L)                               # L (number of thresholds of AP) varies, so calculate the last conv layer's depth.
        self.last_conv_h = self.calc_last_conv(H)
        self.last_conv_w = self.calc_last_conv(W)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv3d(self.in_channels, 48, kernel_size=conv1['ks'], padding=conv1['pad'], stride=conv1['st']),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 96, kernel_size=conv2['ks'], padding=conv2['pad'], stride=conv2['st']),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 96, kernel_size=conv3['ks'], padding=conv3['pad'], stride=conv3['st']),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(96 * self.last_conv_d * self.last_conv_h * self.last_conv_w, 1024),    # input to the 1st linear should be num_filters * depth * h * w.
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
    
    """
    Takes an input shape, returns output shape of it after convolutions. See 
    https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    """
    def calc_last_conv(self, shape):
        dil  = 1                                                                # Default value of dilation in PyTorch.
        for layer in [conv1, conv2, conv3]:
            shape = int((shape + 2 * layer['pad'] - dil * (layer['ks'] - 1) - 1) / layer['st']) + 1
        return shape 