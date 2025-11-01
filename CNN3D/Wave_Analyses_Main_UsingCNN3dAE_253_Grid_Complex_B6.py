#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 09:53:30 2025

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 10:15:12 2025

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
#alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled_ArtifactCorr
#alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled
#filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate clicker/alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled_ArtifactCorr.mat'

#filename = '/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled_ArtifactCorr.mat'

# filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B6/'
# filename='alpha_dynamics_B6_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex.mat'

filepath = '/mnt/DataDrive/ECoG_TravelingWaveProject_Nik/'
#filename = 'alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'
filename = 'alpha_dynamics_B1_253_Arrow_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex.mat'

filename = filepath + filename

data_dict = mat73.loadmat(filename)

xdata = data_dict.get('xdata')
ydata = data_dict.get('ydata')
labels = data_dict.get('labels')
labels_days = data_dict.get('days')
labels_batch = data_dict.get('labels_batch')

xdata = np.concatenate(xdata)
ydata = np.concatenate(ydata)

iterations = 1
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


#del data_dict

# testing some stuff out
# idx = np.where(labels_days==1)
# labels_days[idx]=10
# idx = np.where(labels_days==4)
# labels_days[idx]=1
# idx = np.where(labels_days==10)
# labels_days[idx]=4

#%% TRAIN MODEL

for iterr in np.arange(iterations):    
    
    print(iterr)
    
    
   
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
    
    # transfer learning
    b3Trf_filename = 'i3DAE_B3_Complex_New.pth'  
    nn_filename = 'i3DAE_B6_Complex_New_tmp.pth' 
    
    model = Autoencoder3D_Complex_deep(ksize,num_classes,input_size,lstm_size)
    #model.load_state_dict(torch.load(b3Trf_filename))
    model=model.to(device)
    model.train()
    model_class = Autoencoder3D_Complex_deep
    
    #get number of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    # getparams and train the model 
    num_epochs=100
    batch_size=128
    learning_rate=1e-3
    batch_val=2048
    patience=6
    gradient_clipping=10    
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
    
    
    model,acc,recon_loss_epochs,classif_loss_epochs,total_loss_epochs = training_loop_iAE3D_Complex(model,num_epochs,batch_size,
                            learning_rate,batch_val,patience,gradient_clipping,nn_filename,
                            Xtrain,Ytrain,labels_train,Xval,Yval,labels_val,
                            input_size,num_classes,ksize,lstm_size,alp_factor,aug_flag,
                            sigma,aug_factor,model_class)
    
    
    
    
    
    # test the model on held out data 
    # recon acc
    recon_r,recon_i,decodes = test_model_complex(model,Xtest)
    recon = recon_r + 1j*recon_i
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
        tmp_ydata = Ytest[idx_days,:]
        tmp_recon = recon[idx_days,:]
                
        idx = (tmp_labels)
        idx_cl = np.where(idx==1)[0]
        idx_ol = np.where(idx==0)[0]
    
        recon_r_cl = tmp_recon_r[idx_cl,:]
        recon_i_cl = tmp_recon_i[idx_cl,:]        
        Ytest_r_cl = tmp_ydata_r[idx_cl,:]
        Ytest_i_cl = tmp_ydata_i[idx_cl,:]        
        # r_cl_error = ((recon_r_cl - Ytest_r_cl)**2).sum()/Ytest_r_cl.shape[0]
        # i_cl_error = ((recon_i_cl - Ytest_i_cl)**2).sum()/Ytest_i_cl.shape[0]
        r_cl_error = (np.sum((recon_r_cl - Ytest_r_cl)**2)) / Ytest_r_cl.shape[0]
        i_cl_error = (np.sum((recon_i_cl - Ytest_i_cl)**2)) / Ytest_i_cl.shape[0]
        # r_cl_error = (np.sum((recon_r_cl - Ytest_r_cl)**2)) / np.sum((Ytest_r_cl**2))
        # i_cl_error = (np.sum((recon_i_cl - Ytest_i_cl)**2)) / np.sum((Ytest_i_cl**2))
        
        cl_mse_days[iterr,i] = r_cl_error + i_cl_error
        
        #cl_error = lin.norm((tmp_recon[idx_cl,:] - tmp_ydata[idx_cl,:]).ravel())/lin.norm(tmp_ydata[idx_cl,:].ravel())        
        #cl_mse_days[iterr,i]  = cl_error
        #print(cl_error)
        #cl_mse.append(cl_error)
        
            
        recon_r_ol = tmp_recon_r[idx_ol,:]
        recon_i_ol = tmp_recon_i[idx_ol,:]        
        Ytest_r_ol = tmp_ydata_r[idx_ol,:]
        Ytest_i_ol = tmp_ydata_i[idx_ol,:]        
        r_ol_error = (np.sum((recon_r_ol - Ytest_r_ol)**2)) / Ytest_r_ol.shape[0]
        i_ol_error = (np.sum((recon_i_ol - Ytest_i_ol)**2)) / Ytest_i_ol.shape[0]
        # r_ol_error = (np.sum((recon_r_ol - Ytest_r_ol)**2)) / np.sum((Ytest_r_ol**2))
        # i_ol_error = (np.sum((recon_i_ol - Ytest_i_ol)**2)) / np.sum((Ytest_i_ol**2)) 
        
        #ol_error = lin.norm((tmp_recon[idx_ol,:] - tmp_ydata[idx_ol,:]).ravel())/lin.norm(tmp_ydata[idx_ol,:].ravel())
        
        ol_mse_days[iterr,i] = r_ol_error + i_ol_error
        #ol_mse_days[iterr,i] = ol_error
        
                
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
    
    if iterations > 1:
        del Xtrain,Xtest,Xval,Ytrain,Ytest,Yval,labels_train,labels_test,labels_val,labels_test_days
        
    

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
res = stats.ttest_rel(cl_mse_days, ol_mse_days,axis=1)
print(res)
res = stats.wilcoxon(cl_mse_days, ol_mse_days,axis=1)
print(res)
print(cl_mse_days.mean() - ol_mse_days.mean())

plt.figure();
plt.boxplot([(ol_mse_days[0,:].flatten()),(cl_mse_days[0,:].flatten())])

#%% SAVING MAIN RESLULTS

np.savez('Alpha_200Hz_AllDays_B6_Hand_1Iteration', 
          ce_loss = ce_loss,
          balanced_acc_days = balanced_acc_days,
          ol_mse_days = ol_mse_days,
          cl_mse_days=cl_mse_days)



#%% saving variables to reload and do analyses

os.chdir('/media/user/Data/ecog_data/ECoG BCI/Spyder_Data/')
np.savez('WaveAnalyses_B6_Oct28_20205',
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

data = np.load('/media/user/Data/ecog_data/ECoG BCI/Spyder_Data/WaveAnalyses_Sept0920205.npz',
               allow_pickle=True)

Xval = data.get('Xval')
Yval = data.get('Yval')
Xtest = data.get('Xtest')
Ytest = data.get('Ytest')
labels_test_days = data.get('labels_test_days')
model = data.get('model')
labels_test = data.get('labels_test')
labels_val = data.get('labels_val')
nn_filename = 'i3DAE_B3_Complex_New.pth' 


# get the CNN architecture model
num_classes=1    
input_size=384*2
lstm_size=32
ksize=2;
model_class = Autoencoder3D_Complex_deep

#%% PLOT BACK RESULTS


data=np.load('Alpha_200Hz_AllDays_B1_253Grid_Arrow_10Iterations.npz')
print(data.files)

data1=np.load('Alpha_200Hz_AllDays_B1_253Grid_Arrow_25Iterations_Null.npz')
print(data1.files)

ol_mse = data.get('ol_mse_days')
cl_mse = data.get('cl_mse_days')
null_mse = data1.get('ol_mse_days_null')
null_mse2 = data1.get('cl_mse_days_null')
null_mse = np.concatenate((null_mse, null_mse2))
cl_mse = np.concatenate((cl_mse, ol_mse))

decoding_acc = data.get('decoding_acc')


plt.figure();
plt.boxplot([np.log(cl_mse.flatten()), np.log(null_mse.flatten())])
hfont = {'fontname':'Arial'}
plt.xticks(ticks=[1,2],labels=('Real','Null'),**hfont)
plt.ylabel('Mean Sq. prediction error (log)')


plt.figure();
plt.boxplot([np.log(ol_mse.flatten()),np.log(cl_mse.flatten()),np.log(null_mse.flatten())])
hfont = {'fontname':'Arial'}
plt.xticks(ticks=[1,2,3],labels=('OL','CL','null'),**hfont)
plt.ylabel('Mean Sq. prediction error (log)')


# plotting differences in decoding accuracy 
decoding_acc = data.get('balanced_acc_days')
decoding_acc_null = data1.get('balanced_acc_days_null')
plt.figure();
plt.boxplot([decoding_acc.flatten(), decoding_acc_null.flatten()])
#plt.axhline(y=0.5, color='k', linestyle=':', linewidth=1)
hfont = {'fontname':'Arial'}
plt.xticks(ticks=[1,2],labels=('Real','Null'),**hfont)
plt.ylabel('Balanced Accuracy')

# plotting MSE error between OL and CL, 1st two days
ol_mse = data.get('ol_mse_days')
cl_mse = data.get('cl_mse_days')
tmp1 = ol_mse.flatten()
tmp2 = cl_mse.flatten()
plt.figure();
plt.boxplot([tmp1,tmp2])
hfont = {'fontname':'Arial'}
plt.xticks(ticks=[1,2],labels=('OL','CL'),**hfont)
plt.ylabel('Mean Sq. prediction error ')



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


# all closed loop activations
tmp = layer_activation[idx_cl,4,:]
tmp = torch.mean(tmp,axis=0)

# access images of this last layer activations
#tmp = layer_activation[100,0,:]



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
key = keys_list[0] 

layer_activation = activations[key]  # Retrieve activation
print(f"last Activation - Layer: {key}, Shape: {layer_activation.shape}")


a=0
    
#tmp = layer_activation[3,a,:]

tmp = layer_activation[idx_cl,1,:]
tmp = torch.mean(tmp,axis=0)



num_time_steps = tmp.shape[-1]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(tmp[:,:,0].cpu().numpy(), cmap="viridis")  # Initial frame
ax.set_title(f"Time 0")
ax.axis("off")

#cbar = fig.colorbar(im, ax=ax)

# Update function for animation
def update(frame):
    im.set_array(tmp[:,:,frame].cpu().numpy())  # Update image
    ax.set_title(f"Time {frame}")

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=500)  # 500ms per frame

# Show animation
plt.show()

# Save as GIF
ani.save("Raw.gif", writer="pillow", fps=10)


#%% TESTING OF CNN ARCHITECTURES



# # testing the various sizes of the convolutional layers 
input = torch.randn(32,1,11,23,40)


# layer 1: 9,21,99 is output
m=nn.Conv3d(1,12,kernel_size=2,stride=(1,1,1))
a = nn.AvgPool3d(kernel_size=2,stride=1)
r = nn.ELU()
out = m(input)
out = r(out)
#out = a(out)

#layer 2
m = nn.Conv3d(12,12,kernel_size=2,stride=(1,1,1))
a = nn.AvgPool3d(kernel_size=2,stride=1)
out = m(out)
out = r(out)
#out = a(out)

# layer 3
m = nn.Conv3d(12,12,kernel_size=2,stride=(1,2,2))
out = m(out)
out = r(out)

# layer 4
m = nn.Conv3d(12,6,kernel_size=2,stride=(1,1,2))
out = m(out)
out = r(out)

# pass to lstm for classification
tmp = torch.flatten(out,start_dim=1,end_dim=3)
x=tmp
x = torch.permute(x,(0,2,1))
rnn1 = nn.LSTM(input_size=378,hidden_size=48,batch_first=True,bidirectional=False)
output,(hn,cn) = rnn1(x)
#output1,(hn1,cn1) = rnn2(output)
hn=torch.squeeze(hn)
linear0 = nn.Linear(48,2)
out=linear0(hn)


# tmp = out.view()

# # bottleneck enc side
# out = out.view(out.size(0), -1) 
# m = nn.Linear(out.shape[1], 128)    
# out = m(out)

# # bottleneck dec side
# m = nn.Linear(128,6*7*19*25)    
# out = m(out)

# layer 4
#out = out.view(out.size(0), 6,7, 19,25)

m = nn.ConvTranspose3d(6,12,kernel_size=2,stride=(1,1,1),output_padding=(0,0,0))
out = m(out)
out = r(out)

# layer 3
m = nn.ConvTranspose3d(12,12,kernel_size=2,stride=(1,1,2),output_padding=(0,0,0))
out = m(out)
out = r(out)

out_tmp=out;

# layer 2 want  10,22,100
out=out_tmp
m = nn.ConvTranspose3d(12,12,kernel_size=2,stride=(1,2,2),
                      padding=(0,0,0), output_padding=(0,0,0))
out = m(out)
out = r(out)
print(out.shape)

out_tmp=out;


# layer 1 #11,23,200
out=out_tmp
m = nn.ConvTranspose3d(12,1,kernel_size=2,stride=(1,1,2),output_padding=(0,0,0))
out = m(out)
out = r(out)
print(out.shape)




# CNN 3D AE encoder
class Encoder3D(nn.Module):
    def __init__(self,ksize):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 12, kernel_size=ksize, stride=(1, 1, 2)) 
        self.conv2 = nn.Conv3d(12, 12, kernel_size=ksize, stride=(1, 1, )) # downsampling h
        self.conv3 = nn.Conv3d(12, 12, kernel_size=ksize, stride=(1, 1, 2))
        self.conv4 = nn.Conv3d(12, 6, kernel_size=ksize, stride=(1, 1, 1))
        #self.fc = nn.Linear(6 * 7 * 9 *25, num_nodes)    # 6 filters, w, h, d    
        self.elu = nn.ELU()
        #self.bn1 = torch.nn.BatchNorm3d(num_features=12)
        #self.bn2 = torch.nn.BatchNorm3d(num_features=6)
        #self.pool = nn.AvgPool3d(kernel_size=ksize,stride=1)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x = self.conv2(x)
        #x = self.bn1(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x=self.conv3(x)
        #x = self.bn1(x)
        x=self.elu(x)
        
        x=self.conv4(x)
        #x = self.bn2(x)
        x=self.elu(x)
        return x

# CNN 3D AE decoder
class Decoder3D(nn.Module):
    def __init__(self,ksize):
        super(Decoder3D, self).__init__()
        #self.fc = nn.Linear(num_nodes, 6 * 7 * 9 *25)
        self.deconv1 = nn.ConvTranspose3d(6, 12, kernel_size=ksize, stride=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 1, 2))        
        self.deconv3 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 2, 2))                                           
        self.deconv4 = nn.ConvTranspose3d(12, 1, kernel_size=ksize, stride=(1, 1, 2))                                           
        self.elu = nn.ELU()        
        #self.bn1 = torch.nn.BatchNorm3d(num_features=12)
        #self.bn2 = torch.nn.BatchNorm3d(num_features=6)

    def forward(self, x):
        #x = self.fc(x)
        #x = self.elu(x)
        #x = x.view(x.size(0), 6, 7, 9, 25)
        x = self.deconv1(x)
        #x = self.bn1(x)        
        x = self.elu(x)
        
        x = self.deconv2(x)
        #x = self.bn1(x)        
        x = self.elu(x)
        
        x = self.deconv3(x)
        #x = self.bn1(x)
        x = self.elu(x)
        
        x = self.deconv4(x)
        #x = torch.tanh(x) # squish between 0 and 1
        return x
    
    
# lstm model working in latent space of CNN3D AE
class rnn_lstm(nn.Module):
    def __init__(self,num_classes,input_size,lstm_size):
        super(rnn_lstm,self).__init__()
        self.num_classes = num_classes        
        self.input_size = round(input_size)
        self.lstm_size = round(lstm_size)
        
        self.rnn1=nn.LSTM(input_size=input_size,hidden_size=self.lstm_size,
                          num_layers=1,batch_first=True,bidirectional=False)        
        # self.rnn2=nn.LSTM(input_size=round(self.lstm_size*2),
        #                   hidden_size=round(self.lstm_size/2),
        #                   num_layers=1,batch_first=True,bidirectional=False)      
        self.linear0 = nn.Linear(round(self.lstm_size),num_classes)
                
    
    def forward(self,x):        
        # convert to batch, seq, feature
        x = torch.flatten(x,start_dim=1,end_dim=3)
        x = torch.permute(x,(0,2,1))
        output1, (hn1,cn1) = self.rnn1(x) 
        #output2, (hn2,cn2) = self.rnn2(output1) 
        hn1 = torch.squeeze(hn1)        
        out = self.linear0(hn1)        
        return out
    
class Autoencoder3D(nn.Module):
    def __init__(self, ksize,num_classes,input_size,lstm_size):
        super(Autoencoder3D, self).__init__()
        self.encoder = Encoder3D(ksize)
        self.decoder = Decoder3D(ksize)
        #self.classifier = nn.Linear(num_nodes, num_classes)  
        self.classifier = rnn_lstm(num_classes,input_size,lstm_size)
        
    def forward(self,x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        logits = self.classifier(latent)
        return recon,logits 

#%%


for xx in  np.arange(10):
    print('hello')
    
    
    

