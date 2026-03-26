#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:38:12 2026

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
from scipy.linalg import eigh

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import h5py
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
from sklearn.metrics import balanced_accuracy_score as balan_acc
from sklearn.preprocessing import MinMaxScaler

import matplotlib.cm as cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from fooof import FOOOF
from scipy.signal import welch

import warnings
warnings.filterwarnings("ignore", module="fooof")

#%% LOAD DATA

filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210/'
#filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN/'
#filepath='/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189/'
filename ='lfp_epochs_holdState.mat'
#filename='lfp_epochs_moveState.mat'
filename = filepath + filename

data_dict = mat73.loadmat(filename)
lfp = data_dict.get('lfp_epochs')
Fs = data_dict.get('Fs')
bad_chI  = data_dict.get('bad_chI')

#%% RUN FOOOF

freq_range = [2, 40]
osc_clus=[]
for i in np.arange(len(lfp)):
    data = lfp[i]
    
    spectral_peaks={}
    
    for ch in np.arange(data.shape[1]):
        x = data[:,ch]
        F, Pxx = welch(
                x,
                fs=Fs,
                window='hamming',
                nperseg=512,
                noverlap=256,
                nfft=512,                
                return_onesided=True,
                detrend=False
            )
        
        # Initialize a FOOOF object
        fm = FOOOF()
        # run FOOF
        #fm.report(F,Pxx,freq_range)
        fm.fit(F, Pxx, freq_range)
        
        if fm.peak_params_.size > 0:
            peaks = fm.peak_params_
            peaks =  peaks[:,0]
            spectral_peaks[ch] = peaks
            
        else:
            spectral_peaks[ch] = np.array([0])
                
            
    osc_clus_tmp = []        
    for f in np.arange(2,41):
        lo,hi = f-1,f+1
        count = 0
        
        for j in np.arange(len(spectral_peaks)):
            if bad_chI[j]:                    
                freqs = spectral_peaks[j]
                for k in np.arange(len(freqs)):
                    if lo <= freqs[k] <= hi:
                        count+=1
                        
        osc_clus_tmp.append(count)            
            
        
    osc_clus.append(osc_clus_tmp)        
            

# plotting
f=np.arange(2,41)
plt.figure();
plt.plot(f,np.median(osc_clus,axis=0)/np.sum(bad_chI))
   
        


#%% PARALLEL ANALYSES

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from fooof import FOOOF
from joblib import Parallel, delayed


freq_range = [2, 40]


def process_channel(x, Fs, freq_range):
    """Compute PSD + FOOOF peaks for one channel."""
    F, Pxx = welch(
        x,
        fs=Fs,
        window='hamming',
        nperseg=512,
        noverlap=256,
        nfft=512,
        return_onesided=True,
        detrend=False
    )

    fm = FOOOF(verbose=False)
    fm.fit(F, Pxx, freq_range)

    if fm.peak_params_.size > 0:
        return fm.peak_params_[:, 0]   # center freqs
    else:
        return np.array([0.0])


def process_trial(data, Fs, freq_range, bad_chI):
    """Process all channels in one trial and return histogram counts."""
    n_ch = data.shape[1]

    # Parallel over channels
    peaks_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_channel)(data[:, ch], Fs, freq_range)
        for ch in range(n_ch)
    )

    spectral_peaks = {ch: peaks_list[ch] for ch in range(n_ch)}

    osc_clus_tmp = []
    for f in range(2, 41):
        lo, hi = f - 1, f + 1
        count = 0

        for j in range(n_ch):
            if bad_chI[j]:
                freqs = spectral_peaks[j]
                count += np.sum((freqs >= lo) & (freqs <= hi))

        osc_clus_tmp.append(count)

    return osc_clus_tmp


osc_clus = []
for i in range(len(lfp)):
    data = lfp[i]
    osc_clus_tmp = process_trial(data, Fs, freq_range, bad_chI)
    osc_clus.append(osc_clus_tmp)

osc_clus = np.array(osc_clus)

# plotting
f = np.arange(2, 41)
x = np.median(osc_clus, axis=0) / np.sum(bad_chI)
plt.figure()
plt.plot(f, x)
plt.show()

