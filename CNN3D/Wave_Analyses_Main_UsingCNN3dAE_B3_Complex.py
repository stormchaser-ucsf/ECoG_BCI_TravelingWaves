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






#%% LOAD THE DATA 

# load the data 
if os.name=='nt':
    #filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'    
    filename='alpha_dynamics_200Hz_AllDays_M1_Complex_ArtifactCorr_SinglePrec.mat'
    filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3/'
    filename = filepath + filename
else:
    filepath ='/mnt/DataDrive/ECoG_TravelingWaveProject_Nik/'
    filename = 'alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'
    #filename = filepath + filename
    #filename = 'alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'
    #filepath = '/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/'
    filename = filepath + filename
    
        



#filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_AllDays_DaysLabeled.mat'
#filename = '/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/alpha_dynamics_200Hz_AllDays_DaysLabeled'



data_dict = mat73.loadmat(filename)

xdata = data_dict.get('xdata')
ydata = data_dict.get('ydata')
labels = data_dict.get('labels')
labels_days = data_dict.get('days')
labels_batch = data_dict.get('labels_batch')

xdata = np.concatenate(xdata)
ydata = np.concatenate(ydata)

iterations = 15
days = np.unique(labels_days)

decoding_accuracy=[]
balanced_decoding_accuracy=[]

decoding_acc=[]
balanced_decoding_acc=[];
cl_mse=[]
ol_mse=[]
cl_mse_days=np.zeros((iterations,len(days)))
ol_mse_days=np.zeros((iterations,len(days)))
balanced_acc_days=np.zeros((iterations,len(days)))
ce_loss = np.zeros((iterations,len(days)))

print(xdata.shape)

del data_dict



#%% TRAIN MODEL

for iterr in np.arange(iterations):    
    
    
   
    # parse into training, validation and testing datasets
    
    Xtrain,Xtest,Xval,Ytrain,Ytest,Yval,labels_train,labels_test,labels_val,labels_test_days=training_test_val_split_CNN3DAE_equal(xdata,ydata,labels,0.7,labels_days)                        
    #del xdata, ydata
    
    # # circular shifting the data for null stats
    # random_shifts = np.random.randint(0,Xtrain.shape[-1],size=Xtrain.shape[0])
    # for i in np.arange(len(random_shifts)):
    #     Xtrain[i,:] = np.roll(Xtrain[i,:],shift=random_shifts[i],axis=-1) 
    
    # data augmentation of the training dataset
    
    

    
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
    
    # data augmentation
    # augmentation_factor=2
    # noise_var=0.025
    # Xtrain,Ytrain,labels_train = complex_data_augmentation(Xtrain,Ytrain,labels_train,noise_var,augmentation_factor)
    # 
    
    # # convert labels to indicators
    # labels_train = one_hot_convert(labels_train)
    # labels_test = one_hot_convert(labels_test)
    # labels_val = one_hot_convert(labels_val)
    
    
    
    # get the CNN architecture model
    num_classes=1    
    input_size=384*2
    lstm_size=32
    ksize=2;
    
    from iAE_utils_models import *
    
    if 'model' in locals():
        del model 
    
        
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() 
   
    model = Autoencoder3D_Complex_deep(ksize,num_classes,input_size,lstm_size).to(device)
    model_class = Autoencoder3D_Complex_deep
    
    #get number of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    # getparams and train the model 
    num_epochs=100
    batch_size=128
    learning_rate=2e-3
    batch_val=2048
    patience=6
    gradient_clipping=10
    nn_filename = 'i3DAE_B3_Complex_New.pth' 
    alp_factor=25
    aug_flag=True
    if aug_flag==True:
        batch_size=64
        sigma=0.025
        aug_factor=4
    else:
        batch_size=128
        sigma=0
        aug_factor=0
            
    
    # model_goat = Autoencoder3D_Complex(ksize,num_classes,input_size,lstm_size)
    # #model_goat = Autoencoder3D_B1(ksize,num_classes,input_size,lstm_size)    
    # model.load_state_dict(torch.load(nn_filename))
    # model_goat=model_goat.to(device)
    # model_goat.eval()
    
    
    
    model,acc,recon_loss_epochs,classif_loss_epochs,total_loss_epochs = training_loop_iAE3D_Complex(model,num_epochs,batch_size,
                            learning_rate,batch_val,patience,gradient_clipping,nn_filename,
                            Xtrain,Ytrain,labels_train,Xval,Yval,labels_val,
                            input_size,num_classes,ksize,lstm_size,alp_factor,aug_flag,
                            sigma,aug_factor,model_class)
    
    # test the model on held out data 
    # recon acc
    recon_r,recon_i,decodes = test_model_complex(model,Xtest)
    dec_output = np.array(convert_to_ClassNumbers_sigmoid_list(decodes))
    x=np.array(labels_test)[:,None]
    decoding_accuracy.append( (np.sum(dec_output==x)/dec_output.shape[0]*100).tolist())
    balanced_decoding_accuracy.append(balan_acc(dec_output,x)*100)
    # get the accuracy across days 
        
    for i in np.arange(len(days)): # loop over the 10 days 
        idx_days = np.where(labels_test_days == days[i])[0]
        tmp_labels = labels_test[idx_days]
        tmp_ydata_r,tmp_ydata_i = Ytest[idx_days,:].real, Ytest[idx_days,:].imag
        tmp_recon_r = recon_r[idx_days,:]
        tmp_recon_i = recon_i[idx_days,:]
        tmp_decodes = decodes[idx_days,:]
        #decodes1 = convert_to_ClassNumbers(tmp_decodes).cpu().detach().numpy()           
        decodes1 = convert_to_ClassNumbers_sigmoid_list(tmp_decodes)
                
        idx = (tmp_labels)
        idx_cl = np.where(idx==1)[0]
        idx_ol = np.where(idx==0)[0]
    
        recon_r_cl = tmp_recon_r[idx_cl,:]
        recon_i_cl = tmp_recon_i[idx_cl,:]        
        Ytest_r_cl = tmp_ydata_r[idx_cl,:]
        Ytest_i_cl = tmp_ydata_i[idx_cl,:]        
        r_cl_error = (np.sum((recon_r_cl - Ytest_r_cl)**2)) / Ytest_r_cl.shape[0]
        i_cl_error = (np.sum((recon_i_cl - Ytest_i_cl)**2)) / Ytest_i_cl.shape[0]
        cl_mse_days[iterr,i] = r_cl_error + i_cl_error
        #print(cl_error)
        #cl_mse.append(cl_error)
        
            
        recon_r_ol = tmp_recon_r[idx_ol,:]
        recon_i_ol = tmp_recon_i[idx_ol,:]        
        Ytest_r_ol = tmp_ydata_r[idx_ol,:]
        Ytest_i_ol = tmp_ydata_i[idx_ol,:]        
        r_ol_error = (np.sum((recon_r_ol - Ytest_r_ol)**2)) / Ytest_r_ol.shape[0]
        i_ol_error = (np.sum((recon_i_ol - Ytest_i_ol)**2)) / Ytest_i_ol.shape[0]
        ol_mse_days[iterr,i] = r_ol_error + i_ol_error
        
                
        # balanced accuracy
        balanced_acc = balan_acc(idx,decodes1)
        balanced_acc_days[iterr,i]=balanced_acc
        #print(balanced_acc*100)
        
        #balanced_decoding_acc.append(balanced_acc*100)
        
        # cross entropy loss
        #classif_criterion = nn.CrossEntropyLoss(reduction='mean')    
        classif_criterion = nn.BCEWithLogitsLoss(reduction='mean')# input. target
        # classif_loss = (classif_criterion(tmp_decodes,
        #                                   torch.from_numpy(tmp_labels).to(device))).item()
        classif_loss = (classif_criterion(torch.from_numpy(tmp_decodes.squeeze()).float(),
                                          torch.from_numpy(tmp_labels).float())).item()
       
        
        ce_loss[iterr,i]= classif_loss
        
        del tmp_decodes, tmp_labels,classif_loss
    
    del Xtrain,Xtest,Xval,Ytrain,Ytest,Yval,labels_train,labels_test,labels_val,labels_test_days

# classif_loss = (classif_criterion(torch.from_numpy(tmp_labels[:1,:]).to(device),
#                                    tmp_decodes[:1,:])).item()
# print(classif_loss)


# tmp = torch.from_numpy(tmp_labels)
# tmp1 = tmp_decodes.cpu()

# classif_loss = classif_criterion(tmp1,tmp).item()
# print(classif_loss)

# val=[];
# for i in np.arange(tmp.shape[0]):
#     val.append(classif_criterion(tmp1[i,:],tmp[i,:]).item())

# val=np.array(val)
# print(np.mean(val))

# plt.figure();
# plt.stem(val)

torch.cuda.empty_cache()
torch.cuda.ipc_collect() 

tmp = np.mean(ol_mse_days,axis=0)
tmp1 = np.mean(cl_mse_days,axis=0)
plt.figure();
plt.plot(tmp)    
plt.plot(tmp1)
plt.show()

# now same but with regression line
from sklearn.linear_model import LinearRegression
x = days
x = x.reshape(-1,1)
y = tmp1
mdl = LinearRegression()
mdl.fit(x,y)
plt.figure();
plt.scatter(x,y)
#x = np.concatenate((np.ones((10,1)),x),axis=1)
yhat = mdl.predict(x)
plt.plot(x,yhat,color='red')
plt.show()


tmp = np.mean(balanced_acc_days,axis=0)
plt.figure();
plt.plot(tmp)

plt.figure();
plt.boxplot(tmp)

plt.figure();
plt.boxplot([(ol_mse_days.flatten()),(cl_mse_days.flatten())])


plt.figure();
plt.boxplot([(ol_mse_days[0,:].flatten()),(cl_mse_days[0,:].flatten())])

# ol_mse_days_null=ol_mse_days
# cl_mse_days_null = cl_mse_days
# balanced_acc_days_null = balanced_acc_days
# cd_loss_null = ce_loss


#%% SAVING 

np.savez('Alpha_200Hz_AllDays_B3_Complex_DataAug_15Iter', 
          ce_loss = ce_loss,
          balanced_acc_days = balanced_acc_days,
          ol_mse_days = ol_mse_days,
          cl_mse_days=cl_mse_days,
          balanced_decoding_accuracy=balanced_decoding_accuracy,
          decoding_accuracy=decoding_accuracy
          )



#%% LOADING DATA BACK

data=np.load('Alpha_200Hz_AllDays_B3_New_L2Norm_AE_Model_ArtCorrData_Complex_v2.npz')






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







