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
filename0 ='Raw_BlackRockData_Block4.mat'
filename = 'Raw_BlackRockData_Block4_EventMarkers.mat'
filename2 ='StreamedData_Block4.mat'

filename0 = filepath + filename0
filename = filepath + filename
filename2 = filepath + filename2

data_dict = mat73.loadmat(filename)
data_markers = data_dict.get('Raw_BlackRockData_Block1_EventMarkers')
idx = [i for i, x in enumerate(data_markers) if x[0] not in [None, '']]

data_dict = mat73.loadmat(filename0)
data = data_dict.get('Raw_BlackRockData_Block4')

data_dict = mat73.loadmat(filename2)
data_streamed = data_dict.get('StreamedData_Block4')


#data = data_dict.get('StreamedData_Block1')
#data = data_dict.get('Raw_BlackRockData_Block1')


#%% FOOOF stuff

import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import mat73
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fooof import FOOOF

from fooof.utils.download import load_fooof_data

# Download example data files needed for this example
freqs = load_fooof_data('freqs.npy', folder='data')
spectrum = load_fooof_data('spectrum.npy', folder='data')


#
plt.figure()
plt.plot(freqs,np.log(spectrum))

# Initialize a FOOOF object
fm = FOOOF()

# Set the frequency range to fit the model
freq_range = [2, 40]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
fm.report(freqs, spectrum, freq_range)


spectrum=[0.012626249644428,
0.006021088406225,
0.001509473720073,
0.008401041898429,
0.022016214313101,
0.010018892931448,
0.007961131145752,
0.022151974653030,
0.021472622740165,
0.011105977012877,
0.038580801770172,
0.013529180475394,
0.012412925543013,
0.006662298221361,
0.008280735415854,
0.011700274855954,
0.004298883214370,
0.003396480591897,
0.005657454515378,
0.003517161092838,
0.002777846807726,
0.004988539986736,
0.003312615902613,
0.004029902260475,
0.006967248236186,
0.001342972423428,
0.001559125082361,
0.000663224505375,
0.004609897495534,
0.004411620066941,
0.002271891040513,
0.002423767381149,
0.001369058452299,
0.000925132347623,
0.001601969600951,
0.001211855565707,
0.002244694675862,
0.002046508822038,
0.000845990578372,
0.000707804269734]

spectrum=np.array(spectrum)

freqs=[0.993410742187500,
1.986821484375000,
2.980232226562500,
3.973642968750000,
4.967053710937500,
5.960464453125000,
6.953875195312500,
7.947285937499999,
8.940696679687500,
9.934107421875000,
10.927518164062500,
11.920928906249999,
12.914339648437499,
13.907750390624999,
14.901161132812499,
15.894571874999999,
16.887982617187500,
17.881393359375000,
18.874804101562500,
19.868214843750000,
20.861625585937499,
21.855036328124999,
22.848447070312499,
23.841857812499999,
24.835268554687499,
25.828679296874999,
26.822090039062498,
27.815500781249998,
28.808911523437498,
29.802322265624998,
30.795733007812498,
31.789143749999997,
32.782554492187501,
33.775965234375001,
34.769375976562500,
35.762786718750000,
36.756197460937500,
37.749608203125000,
38.743018945312500,
39.736429687499999]









        