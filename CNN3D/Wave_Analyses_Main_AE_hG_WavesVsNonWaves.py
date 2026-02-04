# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 11:53:15 2025

@author: nikic
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:13:24 2025

@author: nikic
"""

#%% PRELIMS

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


#%% LOAD THE DATA 

# load the data 
if os.name=='nt':
    #filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'    
    filename='alpha_dynamics_200Hz_AllDays_M1_Complex_ArtifactCorr_SinglePrec.mat'
    filepath = 'F:\\DATA\\ecog data\\ECoG BCI\\GangulyServer\\Multistate B3\\'
    filename = filepath + filename
else:
    
    #filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/'
    #filename = 'B3_Wave_NonWave_hG_For_AE.mat'
    
    # filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/'
    # filename = 'B1_Wave_NonWave_hG_For_AE.mat'
    
    filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/'
    filename = 'B6_Wave_NonWave_hG_For_AE.mat'
    
    
    filename = filepath + filename
    
        



#filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_AllDays_DaysLabeled.mat'
#filename = '/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/alpha_dynamics_200Hz_AllDays_DaysLabeled'



data_dict = mat73.loadmat(filename)

# from scipy.io import loadmat
# data_dict = loadmat(filename)

condn_data = data_dict.get('condn_data')

iterations = 5

decoding_accuracy=[]
balanced_decoding_accuracy=[]





#%% TRAIN MODEL

waves_var=[]
nonwaves_var=[]

for iterr in np.arange(iterations):    
    
    print(f"Iteration number: {iterr+1} of {iterations}")
    
    
   
    # parse into training, validation and testing datasets 70% training, 15% val and testing   
    Xtrain,Xval,Xtest = training_split_wavesAE(condn_data,0.7)
    
    # model parameters for the MLP based autoencoder
    input_size=253
    hidden_size=64 #96 originally 
    latent_dims=2 #3 originally 
    num_classes = 7
    
    from iAE_utils_models import *
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() 

    
    if 'model' in locals():
        del model    
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        

    # training params 
    num_epochs=150    
    learning_rate = 1e-3
    batch_val=512
    patience=6
    gradient_clipping=10    
    nn_filename = 'iAE_B6_Waves.pth' 

    #get number of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    # train the model using monte carlo sampling for balanced minibatch for a set number of 
    # epochs    
    
    model,acc = training_loop_iAE_waves(model,num_epochs,learning_rate,batch_val,
                          patience,gradient_clipping,nn_filename,
                          Xtrain,Xval,
                          input_size,hidden_size,latent_dims,num_classes)
    
    # test the model on held out data, trial level 
    wave_nonwave_dict = {
    "targetID": [],
    "latent_wave": [],
    "latent_nonwave": []
        }
    model.eval()    
    labels = np.array(Xtest["targetID"])
    for i in np.arange(1,8):
        idx = np.where(labels==i)[0]
        target_data =[]
        target_data_nw=[]
        for j in np.arange(len(idx)):
            trial_idx = idx[j]
            tmp = Xtest["wave_neural"][trial_idx]
            tmp = tmp.T
            tmp = torch.from_numpy(tmp).to(device).float()
            with torch.no_grad():
                latent = model.encoder(tmp)
            latent = latent.detach().cpu().numpy()
            latent = np.mean(latent,axis=0)[:,None]            
            target_data.append(latent)
            
            tmp = Xtest["nonwave_neural"][trial_idx]
            tmp = tmp.T
            tmp = torch.from_numpy(tmp).to(device).float()
            with torch.no_grad():
                latent = model.encoder(tmp)
            latent = latent.detach().cpu().numpy()
            latent = np.mean(latent,axis=0)[:,None]            
            target_data_nw.append(latent)
        
        target_data = np.concatenate(target_data,axis=1)
        target_data_nw = np.concatenate(target_data_nw,axis=1)
        wave_nonwave_dict["targetID"].append(i)
        wave_nonwave_dict["latent_wave"].append(target_data)
        wave_nonwave_dict["latent_nonwave"].append(target_data_nw)
        
    
    # get latent variances and store results
    wav_var=[]
    nonwave_var=[]
    for i in np.arange(len(wave_nonwave_dict["targetID"])):
        tmp = wave_nonwave_dict["latent_wave"][i]
        C = np.cov(tmp)
        eigvals,eigvec = eigh(C)
        wav_var.append(np.prod(eigvals))
        
        tmp = wave_nonwave_dict["latent_nonwave"][i]
        C = np.cov(tmp)
        eigvals,eigvec = eigh(C)
        nonwave_var.append(np.prod(eigvals))
    
    waves_var.append(wav_var)
    nonwaves_var.append(nonwave_var)

waves_var = np.vstack(waves_var)
nonwaves_var = np.vstack(nonwaves_var)

plt.figure()
#plt.boxplot((waves_var.flatten(),nonwaves_var.flatten()))
plt.boxplot((np.mean(np.log(waves_var),axis=1),np.mean(np.log(nonwaves_var),axis=1)))
#plt.ylim(-0.01,0.05)
#plt.ylim(-5,-3.5)

stat,p = stats.wilcoxon(np.mean(np.log(waves_var),axis=1),
                        np.mean(np.log(nonwaves_var),axis=1))
print(p)

# perhaps just plot a few movements in 2d or 3d, showcasing gaussian ellipses
# difference in trial to trial variance b/w movements wave and non wave 

#%% PLOTTING
    
# do PCA and plot top 3 dimensions
Z_wave_all = []
labels_wave = []

for k, m in enumerate(wave_nonwave_dict["targetID"]):
    Z = wave_nonwave_dict["latent_wave"][k]    # [latent_dim, n_trials]
    Z = Z.T                                    # [n_trials, latent_dim]

    Z_wave_all.append(Z)
    labels_wave.extend([m] * Z.shape[0])

Z_wave_all = np.vstack(Z_wave_all)              # [N_wave_trials, latent_dim]
labels_wave = np.array(labels_wave)

Z_nonwave_all = []
labels_nonwave = []

for k, m in enumerate(wave_nonwave_dict["targetID"]):
    Z = wave_nonwave_dict["latent_nonwave"][k]
    Z = Z.T

    Z_nonwave_all.append(Z)
    labels_nonwave.extend([m] * Z.shape[0])

Z_nonwave_all = np.vstack(Z_nonwave_all)
labels_nonwave = np.array(labels_nonwave)



#pca = PCA(n_components=5)
#pca.fit(np.vstack([Z_wave_all, Z_nonwave_all]))

#Z_wave_pca = pca.transform(Z_wave_all)
#Z_nonwave_pca = pca.transform(Z_nonwave_all)
Z_wave_pca = Z_wave_all
Z_nonwave_pca = Z_nonwave_all


### 2D version
colors = cm.tab10(np.linspace(0, 1, 7))

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# ---- Wave ----
for i, m in enumerate(range(1, 8)):
    idx = labels_wave == m
    axs[0].scatter(
        Z_wave_pca[idx, 0],
        Z_wave_pca[idx, 1],
        s=25,
        alpha=0.7,
        color=colors[i],
        label=f"Move {m}"
    )

axs[0].set_title("Wave epochs")
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")
axs[0].legend()

# ---- Non-wave ----
for i, m in enumerate(range(1, 8)):
    idx = labels_nonwave == m
    axs[1].scatter(
        Z_nonwave_pca[idx, 0],
        Z_nonwave_pca[idx, 1],
        s=25,
        alpha=0.7,
        color=colors[i]
    )

axs[1].set_title("Non-wave epochs")
axs[1].set_xlabel("PC1")
axs[1].set_ylabel("PC2")

plt.tight_layout()
plt.show()



### 3D version
import matplotlib.pyplot as plt

import matplotlib.cm as cm    
movements_to_plot = [1,2 ,3,4,5,6, 7]
colors = cm.tab10(np.linspace(0, 1, 7))

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# ---- Wave ----
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for m in movements_to_plot:
    color_idx = m - 1  # assuming movements are 1–7

    idx = labels_wave == m
    ax1.scatter(
        Z_wave_pca[idx, 0],
        Z_wave_pca[idx, 1],
        Z_wave_pca[idx, 2],
        s=30,
        alpha=0.7,
        color=colors[color_idx],
        label=f"Move {m}"
    )

ax1.set_title("Wave epochs")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")
ax1.legend()    
ax1.set_xlim(-6,8)
ax1.set_ylim(-8,6)
ax1.set_zlim(-6,8)
ax1.view_init(elev=30, azim=135)


# ---- Non-wave ----
for m in movements_to_plot:
    color_idx = m - 1

    idx = labels_nonwave == m
    ax2.scatter(
        Z_nonwave_pca[idx, 0],
        Z_nonwave_pca[idx, 1],
        Z_nonwave_pca[idx, 2],
        s=30,
        alpha=0.7,
        color=colors[color_idx]
    )

ax2.set_title("Non-wave epochs")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")    
ax2.set_xlim(-6,8)
ax2.set_ylim(-8,6)
ax2.set_zlim(-6,8)
ax2.view_init(elev=30, azim=135)


for m in movements_to_plot:
    mu = Z_wave_pca[labels_wave == m].mean(axis=0)
    ax1.scatter(mu[0], mu[1], mu[2],
                color=colors[m-1], s=150, marker='X')

    mu = Z_nonwave_pca[labels_nonwave == m].mean(axis=0)
    ax2.scatter(mu[0], mu[1], mu[2],
                color=colors[m-1], s=150, marker='X')


#set_equal_3d_axes(ax1, Z_wave_pca[ np.isin(labels_wave, movements_to_plot) ])
#set_equal_3d_axes(ax2, Z_nonwave_pca[ np.isin(labels_nonwave, movements_to_plot) ])    

plt.tight_layout()
plt.show()


########### movement comparison, wave and non wave
# plot_movement_wave_vs_nonwave_3d(wave_nonwave_dict, movement_id=1)

plot_movement_wave_vs_nonwave_3d_with_ellipses(
wave_nonwave_dict, movement_id=1)       

plot_movement_wave_vs_nonwave_2d_with_ellipses(
wave_nonwave_dict,
movement_id=2)





          
#%% SAVING 

np.savez('Alpha_200Hz_AllDays_B3_Complex_DataAug_1Iter_New', 
          ce_loss = ce_loss,
          balanced_acc_days = balanced_acc_days,
          ol_mse_days = ol_mse_days,
          cl_mse_days=cl_mse_days,
          balanced_decoding_accuracy=balanced_decoding_accuracy,
          decoding_accuracy=decoding_accuracy
          )

#%% saving variables to reload and do analyses

os.chdir('/media/user/Data/ECoG_BCI_TravelingWave_Data/')
#os.chdir('/mnt/DataDrive/ECoG_TravelingWaveProject_Nik/Analyses_Data/')
np.savez('WaveAnalyses_Nov15_2025_B3Complex_New',
         Xval=Xval,
         Yval=Yval,
         labels_val=labels_val,
         Xtest=Xtest,
         Ytest=Ytest,
         labels_test=labels_test,
         labels_test_days=labels_test_days,
         model=model,
         nn_filename=nn_filename)

#%% LOADING DATA BACK FROM ABOVE

data = np.load('/media/user/Data/ecog_data/ECoG BCI/Spyder_Data/WaveAnalyses_Nov15_2025_B3Complex_New.npz',
               allow_pickle=True)

Xval = data.get('Xval')
Yval = data.get('Yval')
Xtest = data.get('Xtest')
Ytest = data.get('Ytest')
labels_test_days = data.get('labels_test_days')
model = data.get('model')
labels_test = data.get('labels_test')
labels_val = data.get('labels_val')
nn_filename = 'i3DAE_B3_Complex_New_V2.pth' 


# get the CNN architecture model
num_classes=1    
input_size=384*2
lstm_size=32
ksize=2;
model_class = Autoencoder3D_Complex_deep

#asas
# data = {
#     'Xtest': Xtest,
#     'labels_test': labels_test,
#     'labels_test_days':labels_test_days
#     }
# scipy.io.savemat('xtest_data.mat',data)

#%% LOADING DATA BACK AND PLOTTING

data=np.load('Alpha_200Hz_AllDays_B3_New_L2Norm_AE_Model_ArtCorrData_Complex_v2.npz')
print(data.files)
ol_mse_days = data.get('ol_mse_days')
cl_mse_days = data.get('cl_mse_days')
balanced_acc_days = data.get('balanced_acc_days')

tmp = np.median(ol_mse_days,axis=0)
tmp1 = np.median(cl_mse_days,axis=0)
plt.figure();
plt.plot(tmp)    
plt.plot(tmp1)
plt.show()
tmp = tmp/(11*23*40)
tmp1 = tmp1/(11*23*40)

# now same but with regression line
from sklearn.linear_model import LinearRegression
days = np.arange(10)+1
x = days
x = x.reshape(-1,1)
y = tmp
mdl = LinearRegression()
mdl.fit(x,y)
plt.figure();
plt.scatter(x,y,color='blue')
#x = np.concatenate((np.ones((10,1)),x),axis=1)
yhat = mdl.predict(x)
plt.plot(x,yhat,color='blue')
y = tmp1
mdl = LinearRegression()
mdl.fit(x,y)
plt.scatter(x,y,color='red')
#x = np.concatenate((np.ones((10,1)),x),axis=1)
yhat = mdl.predict(x)
plt.plot(x,yhat,color='red')
plt.show()


ol_mse_days = ol_mse_days/(11*23*40)
cl_mse_days = cl_mse_days/(11*23*40)
plt.figure();
plt.boxplot([(ol_mse_days.flatten()),(cl_mse_days.flatten())])

res = stats.ttest_rel(tmp, tmp1)
print(res)
res = stats.wilcoxon(tmp, tmp1)
print(res)

res = stats.wilcoxon(ol_mse_days.flatten(), cl_mse_days.flatten())
print(res)


plt.figure();
plt.boxplot([(tmp),(tmp1)])


#%% (MAIN MAIN) PLOTTING LAYER ACTIVATIONS AS MOVIE AND PHASORS


for h in hook_handles:
    h.remove()


torch.cuda.empty_cache()
torch.cuda.ipc_collect() 


# get the CNN architecture model
num_classes=1    
input_size=384*2
lstm_size=32
ksize=2;


from iAE_utils_models import *

if 'model' in locals():
    del model 

#model = Autoencoder3D_Complex_ROI(num_classes,input_size,lstm_size).to(device)
model = Autoencoder3D_Complex_deep(ksize,num_classes,input_size,lstm_size).to(device)


# nn_filename = 'i3DAE_B3_Complex_New.pth' 
model.load_state_dict(torch.load(nn_filename))




#model=model_bkup
#model=model.eval()
# print(model.encoder.conv1._forward_hooks)


# Container to hold the activation
activation = {}
# ---- Hook function ----
def hook_fn(module, input, output):
    out_r, out_i = output
    activation["real"] = out_r
    activation["imag"] = out_i

# Register hook to conv4 layer
hook_handle = model.encoder.conv6.register_forward_hook(hook_fn) # change to different conv layers


# get the data
days1 = np.where(labels_test_days==1)[0]
X_day = Xtest[days1,:]
Y_day = Ytest[days1,:]
labels_day = labels_test[days1]

ol_idx = np.where(labels_day == 0)[0]
cl_idx = np.where(labels_day == 1)[0]

ol_xtest = X_day[ol_idx,:]
ol_ytest = Y_day[ol_idx,:]
ol_xtest_real = np.real(ol_xtest)
ol_xtest_imag = np.imag(ol_xtest)
ol_ytest_real = np.real(ol_ytest)
ol_ytest_imag = np.imag(ol_ytest)

cl_xtest = X_day[cl_idx,:]
cl_ytest = Y_day[cl_idx,:]
cl_xtest_real = np.real(cl_xtest)
cl_xtest_imag = np.imag(cl_xtest)
cl_ytest_real = np.real(cl_ytest)
cl_ytest_imag = np.imag(cl_ytest)

l=np.arange(0,128)
tmpx_r = torch.from_numpy(ol_xtest_real[l,:]).to(device).float()
tmpx_i = torch.from_numpy(ol_xtest_imag[l,:]).to(device).float()
tmpy_r = torch.from_numpy(ol_ytest_real[l,:]).to(device).float()
tmpy_i = torch.from_numpy(ol_ytest_imag[l,:]).to(device).float()

# tmpx_r = torch.from_numpy(cl_xtest_real[l,:]).to(device).float()
# tmpx_i = torch.from_numpy(cl_xtest_imag[l,:]).to(device).float()
# tmpy_r = torch.from_numpy(cl_ytest_real[l,:]).to(device).float()
# tmpy_i = torch.from_numpy(cl_ytest_imag[l,:]).to(device).float()

model.eval()
with torch.no_grad():
    out_real,out_imag,logits = model(tmpx_r, tmpx_i)
    


# # tmp plotting
# tmpx = torch.squeeze(out_real[10,:]).to('cpu').detach().numpy()
# tmpy = torch.squeeze(tmpy_r[10,:]).to('cpu').detach().numpy()
# fig,ax = plt.subplots(1,2)
# ax[0].imshow(tmpx[:,:,0])
# ax[1].imshow(tmpy[:,:,0])
# plt.tight_layout()
# plt.show()

# Access activation for the target filter
target_filter=0;
out_r = activation["real"]      # shape: [N, C, D, H, W]
out_i = activation["imag"]
magnitude = torch.sqrt(out_r**2 + out_i**2)       # shape: [N, C, D, H, W]
act_strength = magnitude[:,target_filter,:].mean()
print(act_strength)
hook_handle.remove()

del tmpx_r ,tmpx_i ,tmpy_r ,tmpy_i
torch.cuda.empty_cache()
torch.cuda.ipc_collect()  # helps reduce fragmentation 

#x=torch.flatten(magnitude,1,4)
#x=torch.mean(x,axis=1)

# plot a movie of the activation 
target_ch=target_filter
trial=61
x = out_r.to('cpu').detach().numpy()
y = out_i.to('cpu').detach().numpy()
x = (x[trial,target_ch,:])
y = (y[trial,target_ch,:])

x1 = np.moveaxis(x, -1, 0)  # Shape: (40, 11, 23)
#x1 = y # if already in time shape first

# Normalize for visualization
#x1 = (x1 - x1.min()) / (x1.max() - x1.min())

# Plotting
fig, ax = plt.subplots()
im = ax.imshow(x1[0], cmap='magma', animated=True)
title = ax.set_title("Time: 0", fontsize=12)
#ax.set_title("Optimized Input Over Time")
ax.axis('off')

def update(frame):
    im.set_array(x1[frame])
    #im.set_clim(vmin=0.09, vmax=0.15)
    title.set_text(f"Time: {frame}/{x1.shape[0]}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=x1.shape[0], interval=100, blit=False)

# Show the animation
plt.show()
# save the animation
ani.save("RealPart_Layer6_ch0_CL.gif", writer="pillow", fps=6)

# phasor animation
xreal = x;
ximag = y;
#xreal = 2 * ((xreal - xreal.min()) / (xreal.max() - xreal.min())) - 1
#ximag = 2 * ((ximag - ximag.min()) / (ximag.max() - ximag.min())) - 1

fig, ax = plt.subplots(figsize=(6, 6))

def update(t):
    #plot_phasor_frame_time(xreal, ximag, t, ax)
    plot_phasor_frame(xreal, ximag, t, ax)
    return []

#ani = animation.FuncAnimation(fig, update, frames=xreal.shape[0], blit=False)
ani = animation.FuncAnimation(fig, update, frames=xreal.shape[2], blit=False)

plt.show()

# save the animation
ani.save("Layer6_Ch0_Phasor_CL.gif", writer="pillow", fps=4)


# 
x1=xreal[4,4,:]
y1=ximag[4,4,:]
plt.plot(x1)
plt.plot(y1)
plt.show();

plt.figure();
plt.plot(x1,y1);
plt.show();



#%% (MAIn MAIN) OPTIMIZING INPUT TO MAXIMALL ACTIVATE A CERTAIN LAYER CHANNEL



# get the CNN architecture model    
num_classes=1    
input_size=384*2
lstm_size=32
ksize=2;

from iAE_utils_models import *
if 'model' in locals():
    del model    
model = Autoencoder3D_Complex_deep(ksize,num_classes,input_size,lstm_size).to(device)

nn_filename = 'i3DAE_B3_Complex_New.pth' 
model.load_state_dict(torch.load(nn_filename))

model=model.eval()
# print(model.encoder.conv1._forward_hooks)

# Choose the target filter index
target_filter = 0

# Container to hold the activation
activation = {}
# ---- Hook function ----
def hook_fn(module, input, output):
    out_r, out_i = output
    activation["real"] = out_r
    activation["imag"] = out_i

# Register hook to conv4 layer
hook_handle = model.encoder.conv5.register_forward_hook(hook_fn)

# Initialize complex input with gradients
input_shape = (1, 1, 11, 23, 40)  # (batch, channels, H, W, T)
x_real = torch.randn(input_shape, requires_grad=True, device=device)
x_imag = torch.randn(input_shape, requires_grad=True,device=device)

optimizer = optim.AdamW([x_real, x_imag], lr=1e-3)

# Optimization loop
losses=[];
for step in range(3000):
    
    optimizer.zero_grad()
    #out_r, out_i = conv(x_real, x_imag)
    
    _ = model(x_real, x_imag)

    # Access activation for the target filter
    out_r = activation["real"]      # shape: [1, C, D, H, W]
    out_i = activation["imag"]
    
    magnitude = torch.sqrt(out_r**2 + out_i**2)    
    # Maximize average activation of target filter
    reg = 1e-4 * (x_real**2 + x_imag**2).mean()
    loss = -magnitude[0, target_filter].mean() + reg
    loss.backward()
    optimizer.step()
    #print(x_real.abs().mean().item(), x_imag.abs().mean().item())
    losses.append(loss.item())
    
    if step % 50 == 0:
        print(f"Step {step}, Loss: {-loss.item():.4f}")
        
    with torch.no_grad():
        norm = torch.sqrt(x_real**2 + x_imag**2).mean()
        x_real.div_(norm + 1e-8)
        x_imag.div_(norm + 1e-8)
        
plt.plot(losses)
hook_handle.remove()

# Detach optimized input
opt_real = x_real.to('cpu').detach().squeeze().numpy()
opt_imag = x_imag.to('cpu').detach().squeeze().numpy()
opt_complex = opt_real + 1j * opt_imag

# Compute magnitude and phase
magnitude = np.abs(opt_complex)
phase = np.angle(opt_complex)

# plot movie
x = np.real(opt_complex)
x = np.moveaxis(x, -1, 0)  # Shape: (40, 11, 23)

# Normalize for visualization
x = (x - x.min()) / (x.max() - x.min())

# Plotting
fig, ax = plt.subplots()
im = ax.imshow(x[0], cmap='viridis', animated=True)
title = ax.set_title("Time: 0", fontsize=12)
#ax.set_title("Optimized Input Over Time")
ax.axis('off')

def update(frame):
    im.set_array(x[frame])
    title.set_text(f"Time: {frame}/40")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=x.shape[0], interval=100, blit=True)

# Show the animation
plt.show()
# save the animation
ani.save("optimized_input_ch0_Layer5_hook.gif", writer="pillow", fps=6)



# phasor animation
xreal = opt_real;
ximag = opt_imag;
fig, ax = plt.subplots(figsize=(6, 6))

def update(t):
    #plot_phasor_frame_time(xreal, ximag, t, ax)
    plot_phasor_frame(xreal, ximag, t, ax)
    return []

#ani = animation.FuncAnimation(fig, update, frames=xreal.shape[0], blit=False)
ani = animation.FuncAnimation(fig, update, frames=xreal.shape[2], blit=False)

plt.show()

# save the animation
ani.save("optimized_input_ch0__Layer5_hook_phasor.gif", writer="pillow", fps=4)



#%% OLD VERSION OF ABOVE
model=model.to(device)
model=model.eval()
conv = model.encoder.conv1.to(device)
conv=conv.eval()
# Choose the target filter index
target_filter = 0

# Initialize complex input with gradients
input_shape = (1, 1, 11, 23, 40)  # (batch, channels, H, W, T)
x_real = torch.randn(input_shape, requires_grad=True, device=device)
x_imag = torch.randn(input_shape, requires_grad=True,device=device)

optimizer = optim.AdamW([x_real, x_imag], lr=1e-3,weight_decay=1e-4)

# Optimization loop
losses=[];
for step in range(20000):
    
    optimizer.zero_grad()
    out_r, out_i = conv(x_real, x_imag)
    
    magnitude = torch.sqrt(out_r**2 + out_i**2)    
    # Maximize average activation of target filter
    reg = 1e-2 * (x_real**2 + x_imag**2).mean()
    loss = -magnitude[0, target_filter].mean() + reg
    loss.backward()
    optimizer.step()
    #print(x_real.abs().mean().item(), x_imag.abs().mean().item())
    losses.append(loss.item())
    
    if step % 50 == 0:
        print(f"Step {step}, Loss: {-loss.item():.4f}")
        
plt.plot(losses)

# Detach optimized input
opt_real = x_real.to('cpu').detach().squeeze().numpy()
opt_imag = x_imag.to('cpu').detach().squeeze().numpy()
opt_complex = opt_real + 1j * opt_imag

# Compute magnitude and phase
magnitude = np.abs(opt_complex)
phase = np.angle(opt_complex)

# plot movie
x = np.real(opt_complex)
x = np.moveaxis(x, -1, 0)  # Shape: (40, 11, 23)

# Normalize for visualization
x = (x - x.min()) / (x.max() - x.min())

# Plotting
fig, ax = plt.subplots()
im = ax.imshow(x[0], cmap='viridis', animated=True)
ax.set_title("Optimized Input Over Time")
ax.axis('off')

def update(frame):
    im.set_array(x[frame])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=x.shape[0], interval=100, blit=True)

# Show the animation
plt.show()
# save the animation
ani.save("optimized_input7.gif", writer="pillow", fps=10)




#%% plotting data back for comparison sake

data=np.load('Alpha_STG_ROI_200Hz_AllDays_B3_New_L2Norm_AE_Model_ArtCorrData.npz')
data1=np.load('Alpha_M1_ROI_200Hz_AllDays_B3_New_L2Norm_AE_Model_ArtCorrData.npz')

stg = data.get('ol_mse_days')
m1 = data1.get('ol_mse_days')


plt.figure();
plt.boxplot((stg.flatten(),m1.flatten()))


stg =np.mean(stg,axis=0)
m1 =np.mean(m1,axis=0)


plt.figure()
plt.plot(stg)    
plt.plot(m1)
plt.legend(('stg','m1'))
plt.show()

stg = data.get('balanced_acc_days')
m1 = data1.get('balanced_acc_days')

plt.figure();
plt.boxplot((stg.flatten(),m1.flatten()))

#%% plotting data back


data=np.load('Alpha_200Hz_2nd_5days.npz')
data1=np.load('Alpha_200Hz_2nd_5days_nullModel.npz')
ol_mse = data.get('ol_mse')
cl_mse = data.get('cl_mse')
null_mse = data1.get('ol_mse_null')
null_mse2 = data1.get('cl_mse_null')
null_mse = np.concatenate((null_mse, null_mse2))

decoding_acc = data.get('decoding_acc')

plt.figure();
plt.boxplot([np.log(ol_mse),np.log(cl_mse),np.log(null_mse)])
hfont = {'fontname':'Arial'}
plt.xticks(ticks=[1,2,3],labels=('OL','CL','null'),**hfont)
plt.ylabel('Mean Sq. prediction error (log)')



#%% ABLATION EXPERIMENTS 


#%% LOOKING AT ACTIVATION THROUGH HIDDEN LAYERS OF THE N/W

#%% EXAMINING THE MODEL WEIGHTS PHASE AND MAGNITUDE PLOTS

# xx=model.encoder.conv1
# xr = xx.real_conv.weight
# xi = xx.imag_conv.weight

# # plot as quiver plots
# for i in np.arange(xi.shape[0]):
#     rw = torch.squeeze(xr[i]).to('cpu').detach().numpy()
#     iw = torch.squeeze(xi[i]).to('cpu').detach().numpy()
#     rw = rw[0]
#     iw = iw[0]
#     w = rw + 1j*iw
#     mag=np.abs(w)
#     phs = np.angle(w)    
#     H, W, T = mag.shape
#     for t_slice in np.arange(T):                
#         U = mag[:, :, t_slice] * np.cos(phs[:, :, t_slice])  # x-component
#         V = mag[:, :, t_slice] * np.sin(phs[:, :, t_slice])  # y-component
#         X, Y = np.meshgrid(np.arange(W), np.arange(H))
#         plt.figure(figsize=(3, 3))
#         plt.quiver(X, Y, U, V, angles='xy')
#         plt.title("Phase and Magnitude of Kernel at t=0")
#         #plt.gca().invert_yaxis()
#         plt.axis("equal")
#         plt.show()


# plotting with time panels rolled out
xx=model.encoder.conv4 # out channel, in channel
xr = xx.real_conv.weight
xi = xx.imag_conv.weight

w = xr+1j*xi
print(w.shape)

#plot_phasor_kernels_side_by_side_v2(xr, xi,1,0)

# perform PCA on all the kernels for an individual layer
w_ch = w[0,:].to('cpu').detach().numpy()
orig_shape=w_ch.shape
w_ch = w_ch.reshape(w_ch.shape[0],-1)
#w_ch_r = w_ch1.reshape(orig_shape)

x = np.transpose(w_ch)
# mean center
x = x-np.mean(x,axis=0)

# complex covariance
C = (x.conj().T @ x) / x.shape[0]

# eigendecomposition
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# project onto first PC
Z = x @ eigvecs[:, 0]

# reshape it
Z = Z.reshape(orig_shape[1:])

# plot it
plot_phasor_kernels_side_by_side_PCA(Z)





        

    

