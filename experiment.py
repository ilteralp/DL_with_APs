#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:39:11 2020

@author: melike
"""

import torch
from torch.utils.data import DataLoader
import os.path as osp
from skimage import io
import matplotlib.pyplot as plt
from model import APNet
from dataset import RSDataset
from train import train
from inference import test
from report import Report
import constants as C


"""
Takes an input image folder, base image name, output folder and channels. 
Creates vertical and horizontal image splits. Saves them to output folder. 
"""
def create_splits(in_dir, out_dir, base_img_name, c):
    for i in range(1, c+1):
        img_name = base_img_name + str(i)
        img = io.imread(osp.join(in_dir, img_name) + ".png")
        h, w = img.shape
        mid_h, mid_w = int(h/2), int(w/2)
        hor_tr = img[0:mid_h, 0:w]
        ver_tr = img[0:h, 0:mid_w]
        hor_test = img[mid_h:h, 0:w]
        ver_test = img[0:h, mid_w:w]
        base_out_path = osp.join(out_dir, img_name)
        h_tr_path = base_out_path + '_horizontal_train.png'
        h_test_path = base_out_path + '_horizontal_test.png'
        v_tr_path = base_out_path + '_vertical_train.png'
        v_test_path = base_out_path + '_vertical_test.png'
        io.imsave(h_tr_path, hor_tr)
        io.imsave(h_test_path, hor_test)
        io.imsave(v_tr_path, ver_tr)
        io.imsave(v_test_path, ver_test)
        

if __name__ == "__main__":
    
    TRAIN = True
    datasets = ['pavia', 'reykjavik', 'pavia_full']
    splits = ['original', 'vertical', 'horizontal']
    trees = {'pavia': [None, 'minmax'], 'reykjavik': [None, 'minmax'], 'pavia_full': [None]}
    model_names = ['best', 'last_epoch']
    patch_sizes = [1, 3, 5, 7, 9]
    
    use_cuda = torch.cuda.is_available()                                            # Use GPU if available
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    train_params = {'batch_size': 50,                                               # Training parameters
              'shuffle': True, 
              'num_workers': 4}
    max_epochs = 100
    
    test_params = {'num_workers': 4,                                                # Test parameters    
                  'batch_size': 10,
                  'shuffle': False}
    report = Report()
    """ ============================ Train ============================ """
    if TRAIN:
        for name in datasets:
            for split in splits:
                for tree in trees[name]:
                    for patch_size in patch_sizes:
                        train_set = RSDataset(name=name, mode='train', split=split, tree=tree, patch_size=patch_size)
                        train_set.print()
                        train_loader = DataLoader(train_set, **train_params)
                        model = APNet(*(train_set[0][0].shape), num_classes=train_set.num_classes).to(device)
                        criterion = torch.nn.CrossEntropyLoss().to(device)
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                        loss_file = open(osp.join(C.MODEL_DIR, train_set.get_model_name() + 'lost.txt'), 'w')
                        train(model=model, criterion=criterion, optimizer=optimizer, model_name=train_set.get_model_name(),\
                              train_loader=train_loader, max_epochs=max_epochs, device=device, loss_file=loss_file)
                        loss_file.close()
                    
    """ ============================ Test ============================= """
    for name in datasets:
        for split in splits:
            for tree in trees[name]:
                for model_name in model_names:
                    for patch_size in patch_sizes:
                        test_set = RSDataset(name=name, mode='test', split=split, tree=tree, patch_size=patch_size)
                        test_set.print()
                        test_loader = DataLoader(test_set, **test_params)
                        model = APNet(*(test_set[0][0].shape), num_classes=test_set.num_classes).to(device)
                        model_path = osp.join(C.MODEL_DIR, test_set.get_model_name() + model_name + '.pth')
                        model.load_state_dict(torch.load(model_path))
                        scores = test(model=model, num_classes=test_set.num_classes, model_path=model_path,\
                                      test_loader=test_loader, device=device)
                        report.add(test_set, scores, model_name)
    report.save()

