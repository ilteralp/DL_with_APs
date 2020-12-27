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
from scikit.metrics import cohen_kappa_score
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

"""
Calculates and updates confusion matrix for each batch. 
"""
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

"""
Prints and returns confusion metrics
"""
def get_confusion_matrix(conf_matrix):
    TP = conf_matrix.diag()
    num_classes = len(conf_matrix)
    s_TP, s_TN, s_FP, s_FN = 0, 0, 0, 0
    for c in range(num_classes):
        idx = torch.ones(num_classes).bool()                                    # Converts to bool for accessing indices.
        idx[c] = 0
        TN = conf_matrix[torch.nonzero(idx)[:, None], torch.nonzero(idx)].sum()
        FP = conf_matrix[idx, c].sum()
        FN = conf_matrix[c, idx].sum()
        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))
        s_TP += TP[c]
        s_TN += TN
        s_FP += FP
        s_FN += FN
    return s_TP, s_TN, s_FP, s_FN

def test():
    conf_matrix = torch.zeros(test_set.num_classes, test_set.num_classes, dtype=torch.long)
    all_preds = torch.tensor([], dtype=torch.long)
    all_labels = torch.tensor([], dtype=torch.long)
    for batch_samples, batch_labels in test_loader:
        batch_samples, batch_labels = batch_samples.to(device), batch_labels.to(device)
        output = model(batch_samples)
        conf_matrix = confusion_matrix(output, batch_labels, conf_matrix)
        preds = torch.argmax(output, 1)                                         # Convert to (num_samples, num_classes) -> (num_samples)
        all_preds = torch.cat((all_preds, preds), dim=0)                        # Keep all preds to calculate kappa. 
        all_labels = torch.cat((all_labels, batch_labels), dim=0)
    TP, TN, FP, FN = get_confusion_matrix(conf_matrix)
    kappa = cohen_kappa_score(all_preds, all_labels)
    print('shapes:', all_preds.shape, all_labels.shape)
    print('Final, TP {}, TN {}, FP {}, FN {}'.format(TP, TN, FP, FN))
    print('kappa:', kappa)

if __name__ == "__main__":
    model_name = 'best'
    use_cuda = torch.cuda.is_available()                                        # Use GPU if available
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    params = {'num_workers': 4,
              'batch_size': 10,
              'shuffle': False}
    
    test_set = RSDataset(name='reykjavik', mode='test', split='original')
    test_loader = DataLoader(test_set, **params)
    
    model = APNet(in_channels=test_set.c, num_classes=test_set.num_classes, L=test_set.L).to(device)
    model.load_state_dict(torch.load(osp.join(C.MODEL_DIR, test_set.get_model_name() + model_name + '.pth')))
    test()