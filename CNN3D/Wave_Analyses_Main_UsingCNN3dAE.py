# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:13:24 2025

@author: nikic
"""

#%% PRELIMS
import os

os.chdir('C:/Users/nikic/Documents/GitHub/ECoG_BCI_TravelingWaves/CNN3D')


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




#%% LOAD THE DATA



# load the data 
filename = 'F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_2nd_5Days_Rawzscore.mat'
#filename = 'F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_2nd_5Days.mat'
#filename = 'F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_1st5Days.mat'
data_dict = mat73.loadmat(filename)

xdata = data_dict.get('xdata')
ydata = data_dict.get('ydata')
labels = data_dict.get('labels')

xdata = np.concatenate(xdata)
ydata = np.concatenate(ydata)


decoding_acc=[]
cl_mse=[]
ol_mse=[]
for iter in np.arange(6):    
    
    
   
    # parse into training, validation and testing datasets
    Xtrain,Xtest,Xval,Ytrain,Ytest,Yval,labels_train,labels_test,labels_val = training_test_val_split_CNN3DAE(xdata,ydata,labels,0.8)                        
    #del xdata, ydata
    
    # # circular shifting the data for null stats
    # random_shifts = np.random.randint(0,Xtrain.shape[-1],size=Xtrain.shape[0])
    # for i in np.arange(len(random_shifts)):
    #     Xtrain[i,:] = np.roll(Xtrain[i,:],shift=random_shifts[i],axis=-1) 
    

    
    # expand dimensions for cnn 
    Xtrain= np.expand_dims(Xtrain,axis=1)
    Xtest= np.expand_dims(Xtest,axis=1)
    Xval= np.expand_dims(Xval,axis=1)
    Ytrain= np.expand_dims(Ytrain,axis=1)
    Ytest= np.expand_dims(Ytest,axis=1)
    Yval= np.expand_dims(Yval,axis=1)
    Xtrain = np.transpose(Xtrain,(0,1,3,4,2)) 
    Xtest = np.transpose(Xtest,(0,1,3,4,2)) 
    Xval = np.transpose(Xval,(0,1,3,4,2)) 
    Ytrain = np.transpose(Ytrain,(0,1,3,4,2)) 
    Ytest = np.transpose(Ytest,(0,1,3,4,2)) 
    Yval = np.transpose(Yval,(0,1,3,4,2)) 
    
    
    # convert labels to indicators
    labels_train = one_hot_convert(labels_train)
    labels_test = one_hot_convert(labels_test)
    labels_val = one_hot_convert(labels_val)
    
    
    
    # get the CNN architecture model
    num_classes=2
    input_size=378
    lstm_size=64
    ksize=2;
    if 'model' in locals():
        del model 
   
    model = Autoencoder3D(ksize,num_classes,input_size,lstm_size).to(device)
    
    # getparams and train the model 
    num_epochs=150
    batch_size=128
    learning_rate=2e-3
    batch_val=512
    patience=5
    gradient_clipping=10
    nn_filename = 'i3DAE.pth' 
    
    model,acc = training_loop_iAE3D(model,num_epochs,batch_size,learning_rate,batch_val,
                        patience,gradient_clipping,nn_filename,
                        Xtrain,Ytrain,labels_train,Xval,Yval,labels_val,
                        input_size,num_classes,ksize,lstm_size)
    
    # test the model on held out data 
    # recon acc
    Xtest1 = torch.from_numpy(Xtest).to(device).float()
    model.eval()
    with torch.no_grad():
        recon, decodes = model(Xtest1)
    
    idx = convert_to_ClassNumbers(torch.from_numpy(labels_test)).detach().numpy()
    idx_cl = np.where(idx==1)[0]
    idx_ol = np.where(idx==0)[0]
    
    recon_cl = recon[idx_cl,:].cpu().detach().numpy()
    Ytest_cl = Ytest[idx_cl,:]
    cl_error = (np.sum((recon_cl - Ytest_cl)**2)) / Ytest_cl.shape[0]
    print(cl_error)
    cl_mse.append(cl_error)
    
    recon_ol = recon[idx_ol,:].cpu().detach().numpy()
    Ytest_ol = Ytest[idx_ol,:]
    ol_error = (np.sum((recon_ol - Ytest_ol)**2)) / Ytest_ol.shape[0]
    print(ol_error)
    ol_mse.append(ol_error)
    
    # decoding accuracy
    decodes1 = convert_to_ClassNumbers(decodes).cpu().detach().numpy()           
    accuracy = np.sum(idx == decodes1) / len(decodes1)
    print(accuracy*100)
    decoding_acc.append(accuracy*100)

plt.figure();
plt.boxplot([ol_mse,cl_mse])

ol_mse_null=ol_mse
cl_mse_null = cl_mse
decoding_acc_null = decoding_acc

np.savez('Alpha_200Hz_2nd_5days_RawZscore', 
          ol_mse = ol_mse,
          cl_mse = cl_mse,
          decoding_acc = decoding_acc)


#%% plotting amplitude differences
from scipy.stats import gaussian_kde

ol_idx = np.where(labels==0)[0]
cl_idx = np.where(labels==1)[0]

ol = xdata[ol_idx,:].flatten()
cl = xdata[cl_idx,:].flatten()
plt.figure();
#plt.boxplot([ol,cl]);
kde = gaussian_kde(ol)
x_range = np.linspace(min(ol), max(ol), 100)
density_values = kde(x_range)

# Plot KDE
plt.plot(x_range, density_values, label="Density Estimate", color='red')
plt.fill_between(x_range, density_values, alpha=0.3, color='red')  # Shaded area
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Scipy Gaussian KDE")
plt.legend()
plt.show()

#%% plotting data back


data=np.load('Alpha_200Hz_2nd_5days.npz')
ol_mse = data.get('ol_mse')
cl_mse = data.get('cl_mse')
decoding_acc = data.get('decoding_acc')
plt.figure();
plt.boxplot([ol_mse,cl_mse]);


#%% simple methods to get activations at hidden layers 

activations = {}  # Dictionary to store activations

def hook_fn(name):
    """Hook function to store activations."""
    def hook(module, input, output):
        activations[name] = output.detach().clone()  # Store activation
    return hook

# Attach hooks to all Conv3d layers
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv3d):  # Only hook Conv3d layers
        layer.register_forward_hook(hook_fn(name))

Xtest1 = torch.from_numpy(Xtest).to(device).float()
model.eval()
with torch.no_grad():
    recon, decodes = model(Xtest1)
    
    
print("Stored activations:", activations.keys())
for layer_name, activation in activations.items():
    print(f"Layer: {layer_name}, Activation Shape: {activation.shape}")


keys_list = list(activations.keys())  # Convert keys to a list
key = keys_list[0] 

layer_activation = activations[key]  # Retrieve activation

print(f"last Activation - Layer: {key}, Shape: {layer_activation.shape}")


# access images of this last layer activations
tmp = layer_activation[120,2,:]

# plot the activation over time
num_time_steps = tmp.shape[-1]

# Create subplots
#fig, axes = plt.subplots(2, round(num_time_steps/2), figsize=(num_time_steps * 2, 3))
# Arrange subplots into 2 rows
num_cols = math.ceil(num_time_steps / 2)  # Compute number of columns needed
fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 2, 6))  # 2-row layout

# Flatten axes array in case num_time_steps is odd
axes = axes.flatten()

for t in range(num_time_steps):
    ax = axes[t]
    ax.imshow(tmp[:,:,t].cpu().numpy(), cmap="viridis")  # Convert to NumPy
    #ax.set_title(f"Time {t}")
    ax.axis("off")  # Hide axis


# Hide any unused subplot spaces (if odd number of time steps)
for t in range(num_time_steps, len(axes)):
    axes[t].axis("off")
    
plt.tight_layout()
plt.show()


#%% movie



keys_list = list(activations.keys())  # Convert keys to a list
key = keys_list[3] 

layer_activation = activations[key]  # Retrieve activation
print(f"last Activation - Layer: {key}, Shape: {layer_activation.shape}")


a=0
    
tmp = layer_activation[120,a,:]
num_time_steps = tmp.shape[-1]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(tmp[:,:,0].cpu().numpy(), cmap="viridis")  # Initial frame
ax.set_title(f"Time 0")
ax.axis("off")

# Update function for animation
def update(frame):
    im.set_array(tmp[:,:,frame].cpu().numpy())  # Update image
    ax.set_title(f"Time {frame}")

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=500)  # 500ms per frame

# Show animation
plt.show()


#%% do it in a loop to get a general distribution of recon errors and decoding accuracies

