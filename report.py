#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:25:42 2020

@author: melike
"""

import os
import os.path as osp
from xlwt import Workbook
import xlwt
import constants as C

class Report:
    def __init__(self):
        self.wb = Workbook()
        self.sheet = self.wb.add_sheet('Scores')
        self.header = xlwt.easyxf('font: bold 1', 'align: vert centre, horiz centre', 'alignment: wrap True')
        self._init_report()
        
    """
    Initializes report fields. 
    """
    def _init_report(self):
        self.sheet.write(0, 0, 'Dataset', self.header)
        self.sheet.write(0, 1, 'Tree', self.header)
        self.sheet.write(0, 2, 'Split', self.header)
        self.sheet.write(0, 3, 'Number of Classes', self.header)
        self.sheet.write(0, 4, 'Input Shape', self.header)
        self.sheet.write(0, 5, 'Model Name', self.header)
        self.sheet.write(0, 6, 'Kappa', self.header)
        self.sheet.write(0, 7, 'F1', self.header)
        self.sheet.write(0, 8, 'Recall', self.header)
        self.sheet.write(0, 9, 'Precision', self.header)
        
    """
    Adds given dataset, its score and model name to the report. 
    """
    def add(self, dataset, scores, model_name):
        rid = len(self.sheet._Worksheet__rows)
        self.sheet.write(rid, 0, dataset.name)
        self.sheet.write(rid, 1, 'alpha' if dataset.tree == None and dataset.name != 'pavia_full' else dataset.tree)
        self.sheet.write(rid, 2, dataset.split)
        self.sheet.write(rid, 3, dataset.num_classes)
        self.sheet.write(rid, 4, str(dataset[0][0].shape))
        self.sheet.write(rid, 5, model_name)
        self.sheet.write(rid, 6, scores['kappa'])
        self.sheet.write(rid, 7, scores['f1'])
        self.sheet.write(rid, 8, scores['recall'])
        self.sheet.write(rid, 9, scores['precision'])
        
    """
    Returns current report's id. 
    """
    def _get_report_id(self):
        return len([name for name in os.listdir(C.REPORT_PATH) if osp.isfile(os.path.join(C.REPORT_PATH, name))])
    
    """
    Saves the report. 
    """
    def save(self):
        path = osp.join(C.REPORT_PATH, str(self._get_report_id()) + '.xls')
        self.wb.save(path)
        print('Report saved to', path)
        
        
    