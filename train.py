#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:51:56 2020

@author: melike
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import argparse
from model import APNet
from dataset import RSDataset

"""
Usage:
python train.py --name reykjavik \
                --mode train \
                --split horizontal
"""

# parser = argparse.ArgumentParser(description='Train a SegNet model')            # Parse command-line arguments
# parser.add_argument('--name', required=True)
# parser.add_argument('--mode', required=True)
# parser.add_argument('--split', required=True)
# parser.add_argument('--patch_size', type=int)
# parser.add_argument('--checkpoint')
# parser.add_argument('--gpu', type=int)
# args = parser.parse_args()

train_transforms = {                                                            # Transformations to be applied on the train set to augment data. 
    'original': None,
    'hor': transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
    'ver': transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
    'mirror': transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                  transforms.RandomVerticalFlip(p=1)]),
    'rot45': transforms.Compose([transforms.RandomRotation(degrees=[45, 45])]),
    'rot135': transforms.Compose([transforms.RandomRotation(degrees=[135, 135])])
}

def train(model, criterion, optimizer):
    is_better = True
    best_loss = float('inf')
    
    for epoch in range(max_epochs):
        batch_loss = 0
        t_start = time.time()
        
        for batch_samples, batch_labels in train_dataloader:
            batch_samples, batch_labels = batch_samples.to(device), batch_labels.to(device)
            optimizer.zero_grad()                                               # Set grads to zero
            output = model(batch_samples)                                       # Feed input to model
            loss = criterion(output, batch_labels)                              # Calculate loss
            loss.backward()                                                     # Calculate grads via backprop
            optimizer.step()                                                    # Update weights
            batch_loss += loss
            
        delta = time.time() - t_start
        is_better = batch_loss < best_loss
        if is_better:
            best_loss = batch_loss
                
        print("Epoch #{}\tLoss: {:.4f}\t Time: {:2f} seconds".format(epoch, batch_loss, delta))
        

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()                                            # Use GPU if available
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    params = {'batch_size': 50,                                                     # Training parameters
              'shuffle': True, 
              'num_workers': 4}
    max_epochs = 2
    
    train_set = RSDataset(name='reykjavik', mode='train', split='original')
    train_dataloader = DataLoader(train_set, **params)
    
    model = APNet(in_channels=train_set.c, num_classes=train_set.num_classes,L=train_set.L).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(model, criterion, optimizer)
    
# ConcatDataset ile train'de augmentation yap. 