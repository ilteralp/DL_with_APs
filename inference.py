#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 04:35:15 2020

@author: melike
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os.path as osp
import time
import argparse
from model import APNet
from dataset import RSDataset
import constants as C

"""
Usage:
python inference.py --name reykjavik \
                    --split horizontal \
                    --model_name best
"""

# parser = argparse.ArgumentParser(description='Train a APNet model')            # Parse command-line arguments
# parser.add_argument('--name', required=True)
# parser.add_argument('--split', required=True)
# parser.add_argument('--model')
# args = parser.parse_args()

def test():
    for batch_samples, batch_labels in test_loader:
        batch_samples, batch_labels = batch_samples.to(device), batch_labels.to(device)
        output = model(batch_samples)
        print('output[0]:', output[0], 'batch_labels[0]:', batch_labels)
        

if __name__ == "__main__":
    model_name = 'best'
    use_cuda = torch.cuda.is_available()                                        # Use GPU if available
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    params = {'num_workers': 4,
              'batch_size': 50,
              'shuffle': False}
    
    test_set = RSDataset(name='reykjavik', mode='test', split='original')
    test_loader = DataLoader(test_set, **params)
    
    model = APNet(in_channels=test_set.c, num_classes=test_set.num_classes, L=test_set.L).to(device)
    model.load_state_dict(torch.load(osp.join(C.MODEL_DIR, test_set.get_model_name() + model_name + '.pth')))
    test()