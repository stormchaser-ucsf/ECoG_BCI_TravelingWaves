#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 22:53:37 2026

@author: user
"""


import os

if os.name=='nt':
    os.chdir('C:/Users/nikic/Documents/GitHub/ECoG_BCI_TravelingWaves/CNN3D')
else:       
    #os.chdir('/home/reza/Repositories/ECoG_BCI_TravelingWaves/CNN3D')
    #os.chdir('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/CNN3D/')
    os.chdir('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/CNN3D/')

    



from iAE_utils_models import *
import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import mat73
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import h5py
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
from sklearn.metrics import balanced_accuracy_score as balan_acc
from sklearn.preprocessing import MinMaxScaler


#%% LOADING MATLAB FILES FOR ECHO SYNCH

filepath = '/media/user/Data/ecog_data/ECoG BCI/Testing BlackRock/B1/SynchData/20210623/'
filename ='Raw_BlackRockData_Block1.mat'
#filename = 'Raw_BlackRockData_Block1_EventMarkers.mat'
filename ='StreamedData_Block1.mat'

filename = filepath + filename

data_dict = mat73.loadmat(filename)

data = data_dict.get('StreamedData_Block1')

C= data

idx=[]
for i in range(len(C)):
    if C[i] is not None and C[i][0] is not None:
        idx.append(i)
        print(i, C[i][0])
        
        
        
        