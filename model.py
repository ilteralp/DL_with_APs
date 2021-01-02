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

# Patch_size is 9x9
conv1_ps9 = {'kernel_size': (5, 5, 5), 'padding': (2, 2, 2), 'stride': (1, 1, 1)} # (D, H, W)
conv2_ps9 = {'kernel_size': (3, 3, 3), 'padding': (1, 1, 1), 'stride': (1, 1, 1)} # Kernel sizes are from the article. Padding and stride are set in order to make output shape same as input shape. 
conv3_ps9 = {'kernel_size': (3, 3, 3), 'padding': (1, 1, 1), 'stride': (1, 1, 1)}

# Patch_size is 3x3
conv1_ps3 = {'kernel_size': (5, 3, 3), 'padding': (2, 1, 1), 'stride': (1, 1, 1)}
conv2_ps3 = {'kernel_size': (3, 3, 3), 'padding': (1, 1, 1), 'stride': (1, 1, 1)}
conv3_ps3 = {'kernel_size': (3, 3, 3), 'padding': (1, 1, 1), 'stride': (1, 1, 1)}

# Patch_size is 1x1
conv1_ps1 = {'kernel_size': (5, 1, 1), 'padding': (2, 0, 0), 'stride': (1, 1, 1)}
conv2_ps1 = {'kernel_size': (3, 1, 1), 'padding': (1, 0, 0), 'stride': (1, 1, 1)}
conv3_ps1 = {'kernel_size': (3, 1, 1), 'padding': (1, 0, 0), 'stride': (1, 1, 1)}

ps_layers = {'9': [conv1_ps9, conv2_ps9, conv3_ps9],
             '7': [conv1_ps9, conv2_ps9, conv3_ps9], # Same as patch_size=9
             '5': [conv1_ps9, conv2_ps9, conv3_ps9], # Same as patch_size=9
             '3': [conv1_ps3, conv2_ps3, conv3_ps3],
             '1': [conv1_ps1, conv2_ps1, conv3_ps1]}

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
        self._check_params(H, W)
        self.in_channels = in_channels
        self.params = ps_layers[str(H)]
        self.last_conv_d, self.last_conv_h, self.last_conv_w = self.calc_last_conv(L, H, W)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv3d(self.in_channels, 48, **self.params[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 96, **self.params[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 96, **self.params[2]),
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
    
    def _check_params(self, H, W):
        if H != W:
            raise Exception('Height and weight of the patch should be equal.')
        if H != 1 and H != 9:
            raise Exception('Convolution params are set for patch_size={1, 9}.')
    
    """
    Takes an input shape, returns output shape of it after convolutions. See 
    https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    """
    def calc_last_conv(self, *args):
        dil = (1, 1, 1)                                                         # Default value of dilation in PyTorch.
        args = list(args)
        for layer in self.params:
            for i, shape in enumerate(args):
                args[i] = int((shape + 2 * layer['padding'][i] - dil[i] *\
                              (layer['kernel_size'][i] - 1) - 1) / layer['stride'][i]) + 1     
        return args