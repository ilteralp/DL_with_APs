#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 22:28:17 2020

@author: melike
"""

LOCAL_ENV = False
ROG_ENV = not LOCAL_ENV

""" Melike local """
if LOCAL_ENV:
    REYKJAVIK_DIR_PATH = r'/home/melike/rs/reykjavik'
    PAVIA_DIR_PATH = r'/home/melike/rs/pavia'
    PAVIA_FULL_DIR_PATH = r'/home/melike/rs/pavia uni'
    MODEL_DIR = r'/home/melike/repos/DL_with_APs/models'

""" ROG """
if ROG_ENV:
    REYKJAVIK_DIR_PATH = r'/home/rog/rs/reykjavik'
    PAVIA_DIR_PATH = r'/home/rog/rs/pavia'
    PAVIA_FULL_DIR_PATH = r'/home/rog/rs/pavia uni'
    MODEL_DIR = r'/home/rog/repos/DL_with_APs/models'

REY_TS = [25, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 150000]
PAV_TS = [770, 1538, 2307, 3076, 3846, 4615, 5384, 6153, 6923, 7692, 8461, 9230, 10000, 10769]

REY_VER_REMOVE_LABELS = [5]
PAV_VER_REMOVE_LABELS = [3, 7, 8]
PAV_HOR_REMOVE_LABELS = [3, 5, 7]