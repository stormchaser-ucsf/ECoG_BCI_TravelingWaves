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
    os.chdir('/home/reza/Repositories/ECoG_BCI_TravelingWaves/CNN3D')

    



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


#%% testing stuff

tmp=Xtrain[:32,:]
tmp_real,tmp_imag = tmp.real,tmp.imag
tmp_real = torch.from_numpy(tmp_real).float()
tmp_imag = torch.from_numpy(tmp_imag).float()

from iAE_utils_models import *
ksize=(3,5,7)
if 'model' in locals():
      del model 
model = Encoder3D_Complex(ksize)
a,b = model(tmp_real,tmp_imag)


num_classes =1
input_size = 32*2
lstm_size = 16

if 'model' in locals():
      del model 
model = rnn_lstm_complex(num_classes,input_size,lstm_size)
out=model(a,b)

if 'model' in locals():
      del model 
mode11 = Decoder3D_Complex(ksize)
ar,br = mode11(a,b)



if 'model' in locals():
      del model 
model = Autoencoder3D_Complex(ksize,num_classes,input_size,lstm_size)
real_recon,imag_recon,logits = model(tmp_real,tmp_imag)


# more testing stuff 
tmp=Xtrain[:32,:]
tmp_real,tmp_imag = tmp.real,tmp.imag
tmp_real = torch.from_numpy(tmp_real).float()
tmp_imag = torch.from_numpy(tmp_imag).float()

print(tmp_real.shape)

model = nn.Conv3d(1, 8, kernel_size=(3,3,3),dilation=(1,1,2))
out=model(tmp_real)
print(out.shape)

model = nn.Conv3d(8, 8, kernel_size=(3,3,3),dilation=(1,1,2))
out=model(out)
print(out.shape)

model = nn.Conv3d(8, 8, kernel_size=(3,3,3),dilation=(1,1,2))
out=model(out)
print(out.shape)

model = nn.Conv3d(8, 16, kernel_size=(3,5,7),dilation=(1,2,2))
out=model(out)
print(out.shape)

model = nn.Conv3d(16, 32, kernel_size=(3,5,7),dilation=(1,2,2))
out=model(out)
print(out.shape)
out1=out;


# #pass to lstm for classification
# tmp = torch.flatten(out,start_dim=1,end_dim=3)
# x=tmp
# x = torch.permute(x,(0,2,1))
# rnn1 = nn.LSTM(input_size=32,hidden_size=8,batch_first=True,bidirectional=False)
# output,(hn,cn) = rnn1(x)
# #output1,(hn1,cn1) = rnn2(output)
# hn=torch.squeeze(hn)
# linear0 = nn.Linear(8,2)
# out=linear0(hn)


# build the decoder layers to get back the original data 
# bottleneck enc side
# out = out.view(out.size(0), -1) 
# m = nn.Linear(out.shape[1], 128)    
# out = m(out)

# # bottleneck dec side
# m = nn.Linear(128,6*7*19*25)    
# out = m(out)

# layer 5
#out = out.view(out.size(0), 6,7, 19,25)
out=out1;
m = nn.ConvTranspose3d(32,16,kernel_size=(3,5,7),dilation=(1,2,2),output_padding=(0,0,0))
out = m(out)
print(out.shape)

# layer 4
m = nn.ConvTranspose3d(16,16,kernel_size=(3,5,7),dilation=(1,2,2),output_padding=(0,0,0))
out = m(out)
print(out.shape)

# layer 3
m = nn.ConvTranspose3d(16,8,kernel_size=(3,3,3),dilation=(1,1,2),output_padding=(0,0,0))
out = m(out)
print(out.shape)

# layer 2
m = nn.ConvTranspose3d(8,8,kernel_size=(3,3,3),dilation=(1,1,2),output_padding=(0,0,0))
out = m(out)
print(out.shape)

# layer 1
m = nn.ConvTranspose3d(8,1,kernel_size=(3,3,3),dilation=(1,1,2),output_padding=(0,0,0))
out = m(out)
print(out.shape)




#%% LOAD THE DATA 

# load the data 
if os.name=='nt':
    filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_Day1to3.mat'    
else:
    filename = '/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex.mat'
    
        



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

iterations = 5
days = np.unique(labels_days)

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
    
    
    # # convert labels to indicators
    # labels_train = one_hot_convert(labels_train)
    # labels_test = one_hot_convert(labels_test)
    # labels_val = one_hot_convert(labels_val)
    
    
    
    # get the CNN architecture model
    num_classes=1    
    input_size=80*2
    lstm_size=24
    ksize=2;
    
    from iAE_utils_models import *
    
    if 'model' in locals():
        del model 
   
    model = Autoencoder3D_Complex(ksize,num_classes,input_size,lstm_size).to(device)
    
    # getparams and train the model 
    num_epochs=150
    batch_size=128
    learning_rate=1e-3
    batch_val=512
    patience=6
    gradient_clipping=10
    nn_filename = 'i3DAE_B3_Complex.pth' 
    
    # model_goat = Autoencoder3D_Complex(ksize,num_classes,input_size,lstm_size)
    # #model_goat = Autoencoder3D_B1(ksize,num_classes,input_size,lstm_size)    
    # model.load_state_dict(torch.load(nn_filename))
    # model_goat=model_goat.to(device)
    # model_goat.eval()
    
    
    
    model,acc,recon_loss_epochs,classif_loss_epochs,total_loss_epochs = training_loop_iAE3D_Complex(model,num_epochs,batch_size,
                            learning_rate,batch_val,patience,gradient_clipping,nn_filename,
                            Xtrain,Ytrain,labels_train,Xval,Yval,labels_val,
                            input_size,num_classes,ksize,lstm_size)
    
    # test the model on held out data 
    # recon acc
    recon_r,recon_i,decodes = test_model_complex(model,Xtest)
        
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
        cl_mse_days[iter,i] = r_cl_error + i_cl_error
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
plt.boxplot([(ol_mse_days.flatten()),(cl_mse_days.flatten())])


plt.figure();
plt.boxplot([(ol_mse_days[0,:].flatten()),(cl_mse_days[0,:].flatten())])

# ol_mse_days_null=ol_mse_days
# cl_mse_days_null = cl_mse_days
# balanced_acc_days_null = balanced_acc_days
# cd_loss_null = ce_loss


#%% SAVING 

np.savez('Alpha_200Hz_AllDays_B3_New_L2Norm_AE_Model_ArtCorrData_Complex_v2', 
          ce_loss = ce_loss,
          balanced_acc_days = balanced_acc_days,
          ol_mse_days = ol_mse_days,
          cl_mse_days=cl_mse_days)

#%% ABLATION EXPERIMENTS 


#%% LOADING DATA BACK

data=np.load('Alpha_200Hz_AllDays_B3_New_L2Norm_AE_Model_ArtCorrData_Complex_v2.npz')


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

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import deque

# 1. Create 10x10 grid of electrodes
rows, cols = 10, 10
electrodes = [[f"E{r}{c}" for c in range(cols)] for r in range(rows)]

# 2. Assign random power values (structured to show clusters)
power_map = {}
for r in range(rows):
    for c in range(cols):
        # Simulate structured power (stronger in center)
        dist = np.sqrt((r - 5)**2 + (c - 5)**2)
        base_power = np.exp(-dist / 2) * 10  # peak near center
        noise = np.random.normal(0, 0.5)
        power_map[electrodes[r][c]] = base_power + noise

# 3. Apply threshold (e.g., top 10%)
all_powers = np.array(list(power_map.values()))
threshold = np.percentile(all_powers, 90)
active_nodes = {k for k, v in power_map.items() if v >= threshold}

# 4. Build neighbor graph (4-connectivity)
neighbor_graph = {}
for r in range(rows):
    for c in range(cols):
        e = electrodes[r][c]
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                n = electrodes[nr][nc]
                if n in active_nodes:
                    neighbors.append(n)
        if e in active_nodes:
            neighbor_graph[e] = neighbors

# 5. Connected components using DFS
def find_clusters(graph):
    visited = set()
    clusters = []

    def dfs(node, cluster):
        visited.add(node)
        cluster.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, cluster)

    for node in graph:
        if node not in visited:
            cluster = []
            dfs(node, cluster)
            clusters.append(cluster)
    return clusters

clusters = find_clusters(neighbor_graph)

# 6. Plot
plt.figure(figsize=(8, 8))
for r in range(rows):
    for c in range(cols):
        e = electrodes[r][c]
        x, y = c, -r
        power = power_map[e]
        if e in active_nodes:
            # Color by cluster
            for i, cluster in enumerate(clusters):
                if e in cluster:
                    plt.scatter(x, y, c=plt.cm.tab10(i % 10), s=100, label=f"Cluster {i}" if i == 0 else "")
                    break
        else:
            plt.scatter(x, y, color='lightgray', s=30)

plt.title(f"Spatial Clusters of Electrodes (Top 10% Power ≥ {threshold:.2f})")
plt.axis('off')
plt.grid(False)
plt.show()

    

