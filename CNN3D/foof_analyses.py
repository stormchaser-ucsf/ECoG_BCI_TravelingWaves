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

#filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210/'
filepath = '/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN/'
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
        
        # plot over same figure
        plt.clf()
        fm.plot(plt_log=False)
        plt.draw()
        plt.waitforbuttonpress()
        
        
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
plt.plot(f,osc_clus.T/ np.sum(bad_chI),color=(0.75,0.75,0.75,0.5))
plt.plot(f, x,color=(0,0,1))
plt.show()
plt.xlim((2,20))
plt.xticks(np.arange(0,41,2))
plt.yticks(np.arange(0,1,0.1))
plt.ylim((-0.02,.9))
#plt.gca().tick_params(axis='y', labelleft=False)
#plt.gca().tick_params(axis='x', labelbottom=False)

idx=np.argmax(x)
peak_freq = f[idx]

print(f"Peak Freq. is {peak_freq}Hz")
#plt.savefig("EC189_Foof_AllCh.svg", format="svg", dpi=300, bbox_inches="tight", pad_inches=0)

"""
The peak frequencies were 7Hz in EC189 and EC210 and 17Hz in EC176. 
"""


#%% SEEING SINGLE TRIAL STUFF

import matplotlib.pyplot as plt
from fooof import FOOOF
from scipy.signal import welch
import numpy as np

freq_range = [2, 40]
osc_clus = []

plt.ion()

fig, ax = plt.subplots(num=1)
plt.show(block=False)

for i in np.arange(len(lfp)):
    data = lfp[i]
    spectral_peaks = {}

    for ch in np.arange(data.shape[1]):

        x = data[:, ch]

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

        fm = FOOOF()
        fm.fit(F, Pxx, freq_range)

        # clear same axes
        ax.clear()

        # force FOOOF to plot into same axes
        fm.plot(ax=ax, plt_log=False)

        ax.set_title(f"LFP {i}, channel {ch}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

        print("Press key/mouse in figure to continue...")
        plt.waitforbuttonpress()

#%% DOMINANT PEAKS ONLY WITHIN 7 AND 11 HZ

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from fooof import FOOOF

freq_range = [2, 40]
alpha_range = [7, 11]

examples = []

# FOOOF settings: adjust if needed
fooof_settings = dict(
    peak_width_limits=[1, 8],
    max_n_peaks=6,
    min_peak_height=0.1,
    peak_threshold=2.0,
    verbose=False
)

for trial_idx in np.arange(len(lfp)):

    data = lfp[trial_idx]   # time x channels

    for ch in np.arange(data.shape[1]):

        x = data[:, ch]

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

        fm = FOOOF(**fooof_settings)
        fm.fit(F, Pxx, freq_range)

        peaks = fm.peak_params_

        if peaks.shape[0] == 0:
            continue

        cf = peaks[:, 0]   # center frequency
        pw = peaks[:, 1]   # peak power
        bw = peaks[:, 2]   # bandwidth

        in_alpha = (cf >= alpha_range[0]) & (cf <= alpha_range[1])
        outside_alpha = ~in_alpha

        # condition 1: at least one 7-11 Hz peak
        has_alpha_peak = np.any(in_alpha)

        # condition 2: strongest peak is in 7-11 Hz
        strongest_peak_idx = np.argmax(pw)
        strongest_peak_in_alpha = in_alpha[strongest_peak_idx]

        # condition 3: no other fitted peaks outside 7-11 Hz
        #no_other_peaks = not np.any(outside_alpha)
        
        # condition 3: allow only small outside peaks
        outside_power_thresh = 0.1   # adjust this        
        no_meaningful_other_peaks = not np.any(pw[outside_alpha] > outside_power_thresh)

        #if has_alpha_peak and strongest_peak_in_alpha and no_other_peaks:
        if has_alpha_peak and strongest_peak_in_alpha and no_meaningful_other_peaks:

            examples.append({
                "trial": trial_idx,
                "channel": ch,
                "alpha_cf": cf[strongest_peak_idx],
                "alpha_pw": pw[strongest_peak_idx],
                "alpha_bw": bw[strongest_peak_idx],
                "n_peaks": peaks.shape[0],
                "fooof_model": fm
            })

print(f"Found {len(examples)} clean alpha-only examples.")


for ex in examples:
    print(
        f"Trial {ex['trial']}, Ch {ex['channel']}: "
        f"CF={ex['alpha_cf']:.2f} Hz, "
        f"PW={ex['alpha_pw']:.3f}, "
        f"BW={ex['alpha_bw']:.2f} Hz"
    )



# plot them

plt.ion()
fig, ax = plt.subplots(num=1)
plt.show(block=False)

for ex in examples:

    trial_idx = ex["trial"]
    ch = ex["channel"]

    data = lfp[trial_idx]
    x = data[:, ch]

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

    fm = FOOOF(**fooof_settings)
    fm.fit(F, Pxx, freq_range)

    ax.clear()
    fm.plot(ax=ax, plt_log=False)
    ax.set_title(
        f"Trial {trial_idx}, Ch {ch} | "
        f"Peak {ex['alpha_cf']:.2f} Hz"
    )

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

    print("Press key/mouse in figure to continue...")
    plt.waitforbuttonpress()
    
#plot specific ones
ex=[2,12,17]

for i in np.arange(len(ex)):
    
    plt.figure()   
    fig, ax = plt.subplots(num=1)
    plt.show(block=False)

    idx = ex[i]
    tmp = examples[idx]
    
    trial_idx = tmp["trial"]
    ch = tmp["channel"]

    data = lfp[trial_idx]
    x = data[:, ch]

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

    fm = FOOOF(**fooof_settings)
    fm.fit(F, Pxx, freq_range)
   
    fm.plot(ax=ax, plt_log=False)
    plt.xticks(np.arange(0,41,6))
    plt.xlim((2,40))
  
    fig.canvas.draw()
    fig.canvas.flush_events()
    


image_format = 'svg' 
image_name = 'foof3.svg'
plt.savefig(image_name, format=image_format, dpi=300,bbox_inches='tight', pad_inches=0)



