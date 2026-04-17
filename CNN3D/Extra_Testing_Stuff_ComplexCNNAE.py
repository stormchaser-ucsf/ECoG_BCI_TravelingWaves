# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 09:37:35 2025

@author: nikic
"""




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


#%% testing ARCHITECTURE

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

model = nn.Conv3d(1, 8, kernel_size=(3,3,3),dilation=(2,1,1))
model=model.to(device)
tmp_real=tmp_real.to(device)
out=model(tmp_real)
print(out.shape)

model = nn.Conv3d(8, 8, kernel_size=(3,3,3),dilation=(2,1,1))
out=model(out)
print(out.shape)

model = nn.Conv3d(8, 8, kernel_size=(3,3,3),dilation=(2,1,1))
out=model(out)
print(out.shape)

model = nn.Conv3d(8, 16, kernel_size=(3,5,7),dilation=(2,1,1))
out=model(out)
print(out.shape)

model = nn.Conv3d(16, 32, kernel_size=(3,5,7),dilation=(2,2,2))
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


#%% testing ARCHITECTURE SMALLER NETWORK

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




# computing receptive field sizes

# time
dilation =[2,2,2,1,1,1]
stride=[1,1,1,1,1,1]
kernel_size=[5,5,5,5,5,5]
r=compute_RF(dilation, stride,kernel_size)
print(r)
print(np.cumsum(r))

# spatial
dilation =[1,1,1,1,1,1]
stride=[1,1,1,1,1,1]
kernel_size=[2,2,2,2,2,2]
r=compute_RF(dilation, stride,kernel_size)
print(r)
np.cumsum(r)


# more testing stuff 
tmp=Xtrain[:64,:]
total_params=0
tmp_real,tmp_imag = tmp.real,tmp.imag

#tmp_real = rnd.randn(64,1,40,11,23)
#tmp_real = rnd.randn(64,1,40,11,23)
tmp_real = torch.from_numpy(tmp_real).float()
tmp_imag = torch.from_numpy(tmp_imag).float()


print(tmp_real.shape)

model = nn.Conv3d(1, 8, kernel_size=(3,2,2),dilation=(2,1,1))
out=model(tmp_real)
print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params

model = nn.Conv3d(8, 8, kernel_size=(3,2,2),dilation=(2,1,1))
out=model(out)
print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params

model = nn.Conv3d(8, 12, kernel_size=(3,2,2),dilation=(2,1,1))
out=model(out)
print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params

model = nn.Conv3d(12, 12, kernel_size=(3,2,2),dilation=(2,1,1),stride=(1,1,1 ))
out=model(out)
print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params

model = nn.Conv3d(12, 16, kernel_size=(4,2,2),dilation=(3,1,1),stride=(1,1,1))
out=model(out)
print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params

model = nn.Conv3d(16, 32, kernel_size=(4,2,2),dilation=(3,1,1),stride=(1,1,1))
out=model(out)
print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params

print(total_params)

# model = nn.Conv3d(32, 32, kernel_size=(2,2,3),dilation=(1,1,2),stride=(1,1,2))
# out=model(out)
# print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params



out1=out;


#pass to lstm for classification
tmp = torch.squeeze(out)
#tmp = torch.flatten(out,start_dim=1,end_dim=3)
x=tmp
x = torch.permute(x,(0,2,1))
rnn1 = nn.LSTM(input_size=32,hidden_size=8,batch_first=True,bidirectional=False)
output,(hn,cn) = rnn1(x)
#output1,(hn1,cn1) = rnn2(output)
hn=torch.squeeze(hn)
linear0 = nn.Linear(8,2)
out=linear0(hn)


# build the decoder layers to get back the original data 
# bottleneck enc side
# out = out.view(out.size(0), -1) 
# m = nn.Linear(out.shape[1], 128)    
# out = m(out)

# # bottleneck dec side
# m = nn.Linear(128,6*7*19*25)    
# out = m(out)

# layer 6
#out = out.view(out.size(0), 6,7, 19,25)
out=out1;
m = nn.ConvTranspose3d(32,16,kernel_size=(4,2,2),dilation=(3,1,1),stride=(1,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 5
m = nn.ConvTranspose3d(16,12,kernel_size=(4,2,2),dilation=(3,1,1),stride=(1,1,1),
                       output_padding=(0,0,0))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 4
m = nn.ConvTranspose3d(12,12,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 3
m = nn.ConvTranspose3d(12,8,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 2
m = nn.ConvTranspose3d(8,8,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params

# layer 1
m = nn.ConvTranspose3d(8,1,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


print(total_params)


from iAE_utils_models import *
m1 = Encoder3D_Complex_ROI()
tmp=Xtrain[:32]
tmpr = torch.from_numpy(np.real(tmp)).float()
tmpi = torch.from_numpy(np.imag(tmp)).float()
a,b = m1(tmpr,tmpi)


#num_classes,input_size,lstm_size
num_classes=1
input_size = 32*2
lstm_size=16
model = Autoencoder3D_Complex_ROI(num_classes,input_size,lstm_size)
recon_a,recon_b,logits = model(tmpr,tmpi)

# with time first
from iAE_utils_models import *
tmp=Xtrain[:32]
tmpr = torch.from_numpy(np.real(tmp)).float()
tmpi = torch.from_numpy(np.imag(tmp)).float()
num_classes=1
input_size = 32*2
lstm_size=16
model = Autoencoder3D_Complex_ROI_time(num_classes,input_size,lstm_size)
recon_a,recon_b,logits = model(tmpr,tmpi)

#%% DESIGNING A CNN MODEL WITH HIGHER CONVOLUTIONS LAYERS TO TREAT THE ENTIRE DATA
#temp
# h,w,t
tmp=Xtrain[:128,:] # get this from the original xdata, just the first 128 samples
total_params=0
tmp = tmp[:,:,:,:,:20]
tmp_real,tmp_imag = np.real(tmp),np.imag(tmp)

#tmp_real = rnd.randn(64,1,40,11,23)
#tmp_real = rnd.randn(64,1,40,11,23)

tmp_real = torch.from_numpy(tmp_real).float().to(device)
tmp_imag = torch.from_numpy(tmp_imag).float().to(device)

#r,i = model(tmp_real,tmp_imag)


print(tmp_real.shape)

# self.conv1 = ComplexConv3D(1, 12, (2,2,3), (1, 1, 1),0,(1,1,2)) 
# self.conv2 = ComplexConv3D(12, 12, (2,2,3), (1, 1, 1),0,(1,1,2)) 
# self.conv3 = ComplexConv3D(12, 12, (2,2,3), (1, 1, 1),0,(1,1,2))  
# self.conv4 = ComplexConv3D(12, 16, (2,3,3), (1, 1, 1),0,(1,2,2))  
# self.conv5 = ComplexConv3D(16, 16, (2,3,3), (1, 1, 1),0,(1,2,2))  
# self.conv6 = ComplexConv3D(16, 16, (2,3,3), (1, 1, 1),0,(1,2,2))  
# self.conv7 = ComplexConv3D(16, 24, (2,3,4), (1, 1, 1),0,(1,2,3))  

model = nn.Conv3d(1, 8, kernel_size=(2,2,2),dilation=(1,1,1)).to(device)
out=model(tmp_real)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(8,8, kernel_size=(2,2,2),dilation=(1,1,1)).to(device)
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(8, 12, kernel_size=(2,2,2),dilation=(1,1,1)).to(device)
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(12, 12, kernel_size=(2,3,2),dilation=(1,2,2),stride=(1,1,1 )).to(device)
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(12, 12, kernel_size=(2,3,2),dilation=(1,2,2),stride=(1,1,1)).to(device)
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(12, 16, kernel_size=(2,3,3),dilation=(1,2,2),stride=(1,1,1)).to(device)
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(16, 16, kernel_size=(2,3,3),dilation=(1,2,2),stride=(1,1,1)).to(device)
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

print(total_params*4)

# model = nn.Conv3d(32, 32, kernel_size=(2,2,3),dilation=(1,1,2),stride=(1,1,2))
# out=model(out)
# print(out.shape)
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = total_params+trainable_params



out1=out;


#pass to lstm for classification
rnn1 = nn.LSTM(input_size=384,hidden_size=32,batch_first=True,bidirectional=False).to(device)
tmp = torch.flatten(out,start_dim=1,end_dim=3)
x=tmp
x = torch.permute(x,(0,2,1))
#rnn1 = nn.GRU(input_size=256,hidden_size=32,batch_first=True,bidirectional=False)
output,(hn,cn) = rnn1(x)
#output1,(hn1,cn1) = rnn2(output)
hn=torch.squeeze(hn)
trainable_params = sum(p.numel() for p in rnn1.parameters() if p.requires_grad)
total_params = 4*total_params+trainable_params

linear0 = nn.Linear(32,1)
# out=linear0(hn)
trainable_params = sum(p.numel() for p in linear0.parameters() if p.requires_grad)
total_params = total_params+trainable_params
print(total_params)

# build the decoder layers to get back the original data 
# bottleneck enc side
# out = out.view(out.size(0), -1) 
# m = nn.Linear(out.shape[1], 128)    
# out = m(out)

# # bottleneck dec side
# m = nn.Linear(128,6*7*19*25)    
# out = m(out)

# layer 6
#out = out.view(out.size(0), 6,7, 19,25)
out=out1;
m = nn.ConvTranspose3d(32,16,kernel_size=(4,2,2),dilation=(3,1,1),stride=(1,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 5
m = nn.ConvTranspose3d(16,12,kernel_size=(4,2,2),dilation=(3,1,1),stride=(1,1,1),
                       output_padding=(0,0,0))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 4
m = nn.ConvTranspose3d(12,12,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 3
m = nn.ConvTranspose3d(12,8,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


# layer 2
m = nn.ConvTranspose3d(8,8,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params

# layer 1
m = nn.ConvTranspose3d(8,1,kernel_size=(3,2,2),dilation=(2,1,1))
out = m(out)
print(out.shape)
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = total_params+trainable_params


print(total_params)



model =   Encoder3D_Complex_deep(2)
tmp=Xtrain[:128,:] # get this from the original xdata, just the first 128 samples
tmp_real,tmp_imag = np.real(tmp),np.imag(tmp)
#tmp_real = rnd.randn(64,1,40,11,23)
#tmp_real = rnd.randn(64,1,40,11,23)
tmp_real = torch.from_numpy(tmp_real).float()
tmp_imag = torch.from_numpy(tmp_imag).float()

r,i = model(tmp_real,tmp_imag)


num_classes,input_size,lstm_size=1,384*2,32

classifier = rnn_lstm_complex_deep(num_classes,input_size,lstm_size)
logits = classifier(r,i)


# putting it all together 
from iAE_utils_models import *
ksize=2
model = Autoencoder3D_Complex_deep(ksize,num_classes,input_size,lstm_size).to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable_params)
tmp_real=tmp_real.to(device)
tmp_imag = tmp_imag.to(device)
r,i,logits=model(tmp_real,tmp_imag)

#%% tarjan alogrithm to find clusters

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

#%% COMPLEX BATCH NORM TESTING HERE

tmp = Xtrain[:32,:]
input_real = torch.from_numpy(tmp.real).float().to(device)
input_imag = torch.from_numpy(tmp.imag).float().to(device)
m1 = nn.Conv3d(1, 8, kernel_size=(3,2,2),dilation=(2,1,1)).to(device)
m2 = nn.Conv3d(1, 8, kernel_size=(3,2,2),dilation=(2,1,1)).to(device)

input_real=m1(input_real)
input_imag=m1(input_imag)
B,C,D,H,W = input_real.shape
input = torch.stack([input_real, input_imag], dim=-1)  # (B, C, D, H, W, 2)
inp = input.permute(1, 0, 2, 3, 4, 5)
inp = inp.reshape(C, -1, 2)

mean = torch.mean(inp,dim=1)
centered = inp - mean[:, None, :]
N = centered.shape[1]

# compute covariance per channel 
cov = torch.matmul(torch.transpose(centered,1,2),centered)/(N-1)

# regularize
eye = torch.eye(2,device = cov.device)

# regular covariance, add eps per channel 
eps=1e-9
cov = cov + eps*eye[None,:,:]

# cholesky factor
cov_chol = torch.linalg.cholesky(cov)

# chol inv
cov_chol_inv = torch.linalg.inv(cov_chol)


# multiply centered data with cholesky factor to whiten
whitened = torch.matmul(centered,torch.transpose(cov_chol_inv,2,1))
#check
cov_ckh = torch.matmul(torch.transpose(whitened,1,2),whitened)/(N-1)

# scale and add shift (affine transform)
a_init = 1.0 / math.sqrt(3)
c_init = 1.0 / math.sqrt(2)
b_init = 0.0
num_features = 8
a = nn.Parameter(torch.full((num_features,), a_init))
b = nn.Parameter(torch.full((num_features,), b_init))
c = nn.Parameter(torch.full((num_features,), c_init))

weight = torch.stack([
    torch.stack([a, b], dim=1),
    torch.stack([b, c], dim=1)
], dim=1).to(device)  # (C, 2, 2)

bias = nn.Parameter(torch.rand(num_features, 2)).to(device)

# apply affine transform
out = torch.matmul(whitened,weight) + bias[:,None,:]

# reshape
out = out.reshape(C, B, D, H, W, 2).permute(1, 0, 2, 3, 4, 5)


#%% TESTING CNN CLASSIFIER WITH COMPLEX DATA CNN3D WAVE VS NON WAVES

from iAE_utils_models import *


class ComplexWaveCNN8_Reduced(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        # Input: (B, 1, 8, 11, 23)

        self.conv1 = ComplexConv3D(1,  12, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1))
        self.conv2 = ComplexConv3D(12, 20, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1))
        self.conv3 = ComplexConv3D(20, 28, (3, 3, 3), (1, 2, 2), (1, 1, 1), (1, 1, 1))
        self.conv4 = ComplexConv3D(28, 40, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1))
        self.conv5 = ComplexConv3D(40, 48, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1))

        # Bottleneck to cut parameter count
        self.conv6 = ComplexConv3D(48, 16, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1))

        self.act1 = ModReLU(12)
        self.act2 = ModReLU(20)
        self.act3 = ModReLU(28)
        self.act4 = ModReLU(40)
        self.act5 = ModReLU(48)
        self.act6 = ModReLU(16)

        self.dropout = nn.Dropout(dropout)

        # After conv3 stride:
        # T: 8 -> 8
        # H: 11 -> 6
        # W: 23 -> 12
        feat_dim = 16 * 8 * 6 * 12   # 9216 per stream

        self.fc1 = nn.Linear(2 * feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, real, imag):
        real, imag = self.conv1(real, imag)
        real, imag = self.act1(real, imag)

        real, imag = self.conv2(real, imag)
        real, imag = self.act2(real, imag)

        real, imag = self.conv3(real, imag)
        real, imag = self.act3(real, imag)

        real, imag = self.conv4(real, imag)
        real, imag = self.act4(real, imag)

        real, imag = self.conv5(real, imag)
        real, imag = self.act5(real, imag)

        real, imag = self.conv6(real, imag)
        real, imag = self.act6(real, imag)

        real = torch.flatten(real, start_dim=1)
        imag = torch.flatten(imag, start_dim=1)

        z = torch.cat([real, imag], dim=1)
        z = self.dropout(z)
        z = torch.relu(self.fc1(z))
        z = self.dropout(z)
        z = torch.relu(self.fc2(z))
        logits = self.fc3(z)

        return logits

model = ComplexWaveCNN8_Reduced().to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

Xtrain_batch =Xtrain[:32]

xr = torch.from_numpy(Xtrain_batch.real).to(device).float()   # shape (B,1,5,11,23)
xi = torch.from_numpy(Xtrain_batch.imag).to(device).float()

logits = model(xr, xi)



def train_model(model, Xtrain, labels_train, Xval, labels_val,
                epochs=20, batch_size=128, lr=1e-3):

    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    for epoch in range(epochs):
        # -----------------
        # Train
        # -----------------
        model.train()
        idx = np.random.permutation(N)

        train_loss_sum = 0.0
        train_correct = 0

        for i in range(0, N, batch_size):
            batch_idx = idx[i:i + batch_size]

            x = Xtrain[batch_idx]                 # (B, 1, 5, 11, 23), complex
            y = labels_train[batch_idx]           # (B, 1)

            xr = torch.from_numpy(x.real).to(device).float()
            xi = torch.from_numpy(x.imag).to(device).float()
            y = torch.from_numpy(y).to(device).float()

            optimizer.zero_grad()

            logits = model(xr, xi)                # (B, 1)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()        # (B, 1)
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        # -----------------
        # Validation
        # -----------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                x = Xval[i:i + batch_size]        # (B, 1, 5, 11, 23), complex
                y = labels_val[i:i + batch_size]  # (B, 1)

                xr = torch.from_numpy(x.real).to(device).float()
                xi = torch.from_numpy(x.imag).to(device).float()
                y = torch.from_numpy(y).to(device).float()

                logits = model(xr, xi)            # (B, 1)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()    # (B, 1)
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return model



model = train_model(
    model,
    Xtrain, labels_train,
    Xval, labels_val,
    epochs=15,
    batch_size=64,
    lr=1e-3
)



#%% REAL MODEL ONLY (ABOVE CONTROL)


import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Real-valued 3D CNN baseline
# ----------------------------
class RealWaveCNN3D(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 12, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv2 = nn.Conv3d(12, 20, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv3 = nn.Conv3d(20, 28, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv4 = nn.Conv3d(28, 40, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv5 = nn.Conv3d(40, 48, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv6 = nn.Conv3d(48, 16, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # input: (B,1,8,11,23)
        # after conv3 stride (1,2,2): (B,28,8,6,12)
        # conv4/5/6 keep shape -> (B,16,8,6,12)
        feat_dim = 16 * 8 * 6 * 12

        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ----------------------------
# Training function
# ----------------------------
def train_real_model(model, Xtrain, labels_train, Xval, labels_val,
                     epochs=15, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_acc = -np.inf
    best_state = None

    # use only real part
    Xtrain_real = Xtrain.real.astype(np.float32)
    Xval_real   = Xval.real.astype(np.float32)

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(N)

        train_loss_sum = 0.0
        train_correct = 0

        for i in range(0, N, batch_size):
            batch_idx = idx[i:i+batch_size]

            x = torch.from_numpy(Xtrain_real[batch_idx]).to(device)
            y = torch.from_numpy(labels_train[batch_idx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)              # (B,1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                x = torch.from_numpy(Xval_real[i:i+batch_size]).to(device)
                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


# ----------------------------
# Run
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RealWaveCNN3D(dropout=0.30)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters:", n_params)

model, best_val_acc = train_real_model(
    model,
    Xtrain, labels_train,
    Xval, labels_val,
    epochs=15,
    batch_size=128,
    lr=1e-3,
    device=device
)

print("Best validation accuracy:", best_val_acc)


#%% ABSOLUTE VALUE MAGNITUDE (FROM ABOVE)

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Amplitude-only 3D CNN
# ----------------------------
class AmplitudeWaveCNN3D(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 12, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv2 = nn.Conv3d(12, 20, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv3 = nn.Conv3d(20, 28, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv4 = nn.Conv3d(28, 40, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv5 = nn.Conv3d(40, 48, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv6 = nn.Conv3d(48, 16, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # input: (B,1,8,11,23)
        # after conv3 stride (1,2,2): (B,28,8,6,12)
        # conv4/5/6 keep shape -> (B,16,8,6,12)
        feat_dim = 16 * 8 * 6 * 12

        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ----------------------------
# Training function
# ----------------------------
def train_amplitude_model(model, Xtrain, labels_train, Xval, labels_val,
                          epochs=15, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_acc = -np.inf
    best_state = None

    # amplitude only
    Xtrain_amp = np.abs(Xtrain).astype(np.float32)
    Xval_amp   = np.abs(Xval).astype(np.float32)

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(N)

        train_loss_sum = 0.0
        train_correct = 0

        for i in range(0, N, batch_size):
            batch_idx = idx[i:i+batch_size]

            x = torch.from_numpy(Xtrain_amp[batch_idx]).to(device)
            y = torch.from_numpy(labels_train[batch_idx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)   # (B,1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                x = torch.from_numpy(Xval_amp[i:i+batch_size]).to(device)
                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


# ----------------------------
# Run
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AmplitudeWaveCNN3D(dropout=0.30)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters:", n_params)

model, best_val_acc = train_amplitude_model(
    model,
    Xtrain, labels_train,
    Xval, labels_val,
    epochs=15,
    batch_size=128,
    lr=1e-3,
    device=device
)

print("Best validation accuracy:", best_val_acc)


#%% phase only model (from abvove)
#(MAIN)

Xtrain_phase = Xtrain / (np.abs(Xtrain) + 1e-8)
Xval_phase   = Xval   / (np.abs(Xval)   + 1e-8)

import torch
import torch.nn as nn


class PhaseWaveCNN3D(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(2, 12, (3,3,3), (1,1,1), (1,1,1))
        self.conv2 = nn.Conv3d(12,20, (3,3,3), (1,1,1), (1,1,1))
        self.conv3 = nn.Conv3d(20,28, (3,3,3), (1,2,2), (1,1,1))
        self.conv4 = nn.Conv3d(28,40, (3,3,3), (1,1,1), (1,1,1))
        self.conv5 = nn.Conv3d(40,48, (3,3,3), (1,1,1), (1,1,1))
        self.conv6 = nn.Conv3d(48,16, (1,1,1), (1,1,1), (0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        feat_dim = 16 * 8 * 6 * 12

        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    
def train_phase_model(model, Xtrain, labels_train, Xval, labels_val,
                      epochs=15, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_acc = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(N)

        train_correct = 0
        train_loss_sum = 0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            xr = Xtrain[bidx].real.astype(np.float32)
            xi = Xtrain[bidx].imag.astype(np.float32)

            x = np.concatenate([xr, xi], axis=1)   # (B,2,8,11,23)
            x = torch.from_numpy(x).to(device)

            y = torch.from_numpy(labels_train[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        model.eval()
        val_correct = 0
        val_loss_sum = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                xr = Xval[i:i+batch_size].real.astype(np.float32)
                xi = Xval[i:i+batch_size].imag.astype(np.float32)

                x = np.concatenate([xr, xi], axis=1)
                x = torch.from_numpy(x).to(device)

                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_acc

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PhaseWaveCNN3D()
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

model, best_val_acc = train_phase_model(
    model,
    Xtrain_phase,
    labels_train,
    Xval_phase,
    labels_val,
    device=device
)

print("Best val acc:", best_val_acc)


#%% real only after amplitude normalization 

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Model (same as real baseline)
# ----------------------------
class RealNormWaveCNN3D(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 12, (3,3,3), (1,1,1), (1,1,1))
        self.conv2 = nn.Conv3d(12,20, (3,3,3), (1,1,1), (1,1,1))
        self.conv3 = nn.Conv3d(20,28, (3,3,3), (1,2,2), (1,1,1))
        self.conv4 = nn.Conv3d(28,40, (3,3,3), (1,1,1), (1,1,1))
        self.conv5 = nn.Conv3d(40,48, (3,3,3), (1,1,1), (1,1,1))
        self.conv6 = nn.Conv3d(48,16, (1,1,1), (1,1,1), (0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        feat_dim = 16 * 8 * 6 * 12

        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ----------------------------
# Training function
# ----------------------------
def train_realnorm_model(model, Xtrain, labels_train, Xval, labels_val,
                         epochs=15, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_acc = -1
    best_state = None

    # ---- real normalized input ----
    Xtrain_realnorm = (Xtrain.imag / (np.abs(Xtrain) + 1e-8)).astype(np.float32)
    Xval_realnorm   = (Xval.imag   / (np.abs(Xval)   + 1e-8)).astype(np.float32)

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(N)

        train_loss_sum = 0
        train_correct = 0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            x = torch.from_numpy(Xtrain_realnorm[bidx]).to(device)
            y = torch.from_numpy(labels_train[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        model.eval()
        val_loss_sum = 0
        val_correct = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                x = torch.from_numpy(Xval_realnorm[i:i+batch_size]).to(device)
                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_acc


# ----------------------------
# Run
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RealNormWaveCNN3D(dropout=0.30)
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

model, best_val_acc = train_realnorm_model(
    model,
    Xtrain,
    labels_train,
    Xval,
    labels_val,
    device=device
)

print("Best validation accuracy:", best_val_acc)

#%% CONTROL SHUFFLING OF TIME

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Phase-only model
# ----------------------------
class PhaseWaveCNN3D(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(2, 12, (3,3,3), (1,1,1), (1,1,1))
        self.conv2 = nn.Conv3d(12,20, (3,3,3), (1,1,1), (1,1,1))
        self.conv3 = nn.Conv3d(20,28, (3,3,3), (1,2,2), (1,1,1))
        self.conv4 = nn.Conv3d(28,40, (3,3,3), (1,1,1), (1,1,1))
        self.conv5 = nn.Conv3d(40,48, (3,3,3), (1,1,1), (1,1,1))
        self.conv6 = nn.Conv3d(48,16, (1,1,1), (1,1,1), (0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        feat_dim = 16 * 8 * 6 * 12
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ----------------------------
# Phase-only preprocessing
# ----------------------------
def make_phase_only(X):
    return X / (np.abs(X) + 1e-8)


# ----------------------------
# Temporal shuffle along time axis
# X shape: (N, 1, T, H, W), complex
# same permutation applied to all samples
# ----------------------------
def temporal_shuffle(X):
    Xs = X.copy()
    perm = np.random.permutation(X.shape[2])
    Xs = Xs[:, :, perm, :, :]
    return Xs, perm


# ----------------------------
# Training
# ----------------------------
def train_phase_model(model, Xtrain, labels_train, Xval, labels_val,
                      epochs=15, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_acc = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(N)

        train_correct = 0
        train_loss_sum = 0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            xr = Xtrain[bidx].real.astype(np.float32)
            xi = Xtrain[bidx].imag.astype(np.float32)

            x = np.concatenate([xr, xi], axis=1)   # (B,2,T,H,W)
            x = torch.from_numpy(x).to(device)

            y = torch.from_numpy(labels_train[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        model.eval()
        val_correct = 0
        val_loss_sum = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                xr = Xval[i:i+batch_size].real.astype(np.float32)
                xi = Xval[i:i+batch_size].imag.astype(np.float32)

                x = np.concatenate([xr, xi], axis=1)
                x = torch.from_numpy(x).to(device)

                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_acc


# ----------------------------
# Run temporal-shuffle control
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# phase only
Xtrain_phase = make_phase_only(Xtrain)
Xval_phase   = make_phase_only(Xval)

# temporal shuffle
# Xtrain_phase_shuf, perm_train = temporal_shuffle(Xtrain_phase)
# Xval_phase_shuf, perm_val     = temporal_shuffle(Xval_phase)

perm = np.random.permutation(Xtrain_phase.shape[2])

Xtrain_phase_shuf = Xtrain_phase[:, :, perm, :, :]
Xval_phase_shuf   = Xval_phase[:, :, perm, :, :]

print("Shared time permutation:", perm)

print("Train time permutation:", perm_train)
print("Val time permutation:", perm_val)

model = PhaseWaveCNN3D(dropout=0.30)
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

model, best_val_acc = train_phase_model(
    model,
    Xtrain_phase_shuf,
    labels_train,
    Xval_phase_shuf,
    labels_val,
    epochs=15,
    batch_size=128,
    lr=1e-3,
    device=device
)

print("Best validation accuracy after temporal shuffle:", best_val_acc)

#%% differing size of model


import numpy as np
import torch
import torch.nn as nn


# ============================
# Model (~2.5M params)
# ============================
class PhaseWaveCNN3D_2p5M(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(2,  12, (3,3,3), (1,1,1), (1,1,1))
        self.conv2 = nn.Conv3d(12, 20, (3,3,3), (1,1,1), (1,1,1))
        self.conv3 = nn.Conv3d(20, 28, (3,3,3), (1,2,2), (1,1,1))
        self.conv4 = nn.Conv3d(28, 40, (3,3,3), (1,1,1), (1,1,1))
        self.conv5 = nn.Conv3d(40, 48, (3,3,3), (1,1,1), (1,1,1))
        self.conv6 = nn.Conv3d(48, 16, (1,1,1), (1,1,1), (0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        feat_dim = 16 * 8 * 6 * 12  # 9216

        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ============================
# Utilities
# ============================
def make_phase_only(X):
    return X / (np.abs(X) + 1e-8)

def pack_phase(X):
    xr = X.real.astype(np.float32)
    xi = X.imag.astype(np.float32)
    return np.concatenate([xr, xi], axis=1)


# ============================
# Training
# ============================
def train_model(model, Xtrain, labels_train, Xval, labels_val,
                epochs=20, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_loss = np.inf
    best_state = None

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        idx = np.random.permutation(N)

        train_loss_sum = 0
        train_correct = 0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            x = torch.from_numpy(pack_phase(Xtrain[bidx])).to(device)
            y = torch.from_numpy(labels_train[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        # ===== VALIDATION =====
        model.eval()
        val_loss_sum = 0
        val_correct = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                x = torch.from_numpy(pack_phase(Xval[i:i+batch_size])).to(device)
                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        # ===== SAVE BEST (BY VAL LOSS) =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_loss


# ============================
# Run
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# prepare phase-only data
Xtrain_phase = make_phase_only(Xtrain)
Xval_phase   = make_phase_only(Xval)

# init model
model = PhaseWaveCNN3D_2p5M()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters:", n_params)

# train
model, best_val_loss = train_model(
    model,
    Xtrain_phase,
    labels_train,
    Xval_phase,
    labels_val,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    device=device
)

print("Best validation loss:", best_val_loss)

#%% MODEL WITH MINOR CONTROLS 

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Model
# ----------------------------
class PhaseWaveCNN3D(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(2, 12, (3,3,3), (1,1,1), (1,1,1))
        self.conv2 = nn.Conv3d(12,20, (3,3,3), (1,1,1), (1,1,1))
        self.conv3 = nn.Conv3d(20,28, (3,3,3), (1,2,2), (1,1,1))
        self.conv4 = nn.Conv3d(28,40, (3,3,3), (1,1,1), (1,1,1))
        self.conv5 = nn.Conv3d(40,48, (3,3,3), (1,1,1), (1,1,1))
        self.conv6 = nn.Conv3d(48,16, (1,1,1), (1,1,1), (0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        feat_dim = 16 * 8 * 6 * 12
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ----------------------------
# Utilities
# ----------------------------
def make_phase_only(X):
    return X / (np.abs(X) + 1e-8)


def apply_temporal_permutation(X, perm):
    return X[:, :, perm, :, :]


# ----------------------------
# Training
# ----------------------------
def train_model(model, Xtrain, labels_train, Xval, labels_val,
                epochs=15, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_acc = -1

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(N)

        train_correct = 0
        train_loss_sum = 0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            xr = Xtrain[bidx].real.astype(np.float32)
            xi = Xtrain[bidx].imag.astype(np.float32)

            x = np.concatenate([xr, xi], axis=1)
            x = torch.from_numpy(x).to(device)

            y = torch.from_numpy(labels_train[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_acc = train_correct / N

        model.eval()
        val_correct = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                xr = Xval[i:i+batch_size].real.astype(np.float32)
                xi = Xval[i:i+batch_size].imag.astype(np.float32)

                x = np.concatenate([xr, xi], axis=1)
                x = torch.from_numpy(x).to(device)

                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_acc = val_correct / Nv
        best_val_acc = max(best_val_acc, val_acc)

        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return best_val_acc


# ----------------------------
# Run experiment
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Phase-only data
Xtrain_phase = make_phase_only(Xtrain)
Xval_phase   = make_phase_only(Xval)

print("\n=== BASELINE (PHASE ONLY) ===")
model = PhaseWaveCNN3D()
baseline_acc = train_model(
    model, Xtrain_phase, labels_train, Xval_phase, labels_val, device=device
)

print("\n=== TEMPORAL SHUFFLE ===")
T = Xtrain_phase.shape[2]
perm = np.random.permutation(T)

Xtrain_shuf = apply_temporal_permutation(Xtrain_phase, perm)
Xval_shuf   = apply_temporal_permutation(Xval_phase, perm)

print("Permutation:", perm)

model = PhaseWaveCNN3D()
shuffle_acc = train_model(
    model, Xtrain_shuf, labels_train, Xval_shuf, labels_val, device=device
)

print("\nRESULTS")
print("Baseline accuracy:", baseline_acc)
print("Shuffled accuracy:", shuffle_acc)

#%% MODEL WITH CONTROLS


import numpy as np
import torch
import torch.nn as nn


# ============================
# Model
# ============================
class PhaseWaveCNN3D(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        self.conv1 = nn.Conv3d(2, 12, (3,3,3), (1,1,1), (1,1,1))
        self.conv2 = nn.Conv3d(12,20, (3,3,3), (1,1,1), (1,1,1))
        self.conv3 = nn.Conv3d(20,28, (3,3,3), (1,2,2), (1,1,1))
        self.conv4 = nn.Conv3d(28,40, (3,3,3), (1,1,1), (1,1,1))
        self.conv5 = nn.Conv3d(40,48, (3,3,3), (1,1,1), (1,1,1))
        self.conv6 = nn.Conv3d(48,16, (1,1,1), (1,1,1), (0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        feat_dim = 16 * 8 * 6 * 12
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ============================
# Utilities
# ============================
def make_phase_only(X):
    return X / (np.abs(X) + 1e-8)


def pack_phase_channels(X):
    xr = X.real.astype(np.float32)
    xi = X.imag.astype(np.float32)
    return np.concatenate([xr, xi], axis=1)   # (N,2,T,H,W)


# 1) Destroy cross-time consistency at each spatial location
# same independent voxelwise permutation applied to all samples
def shuffle_time_independently_per_voxel(X):
    Xs = X.copy()
    N, C, T, H, W = X.shape
    perms = np.zeros((H, W, T), dtype=int)

    for h in range(H):
        for w in range(W):
            perm = np.random.permutation(T)
            perms[h, w] = perm
            Xs[:, :, :, h, w] = X[:, :, perm, h, w]

    return Xs, perms


# 2) Single-frame repeated across time
def repeat_single_frame(X, frame_idx=0):
    Xr = X.copy()
    frame = X[:, :, frame_idx:frame_idx+1, :, :]      # (N,C,1,H,W)
    Xr = np.repeat(frame, X.shape[2], axis=2)         # repeat to full T
    return Xr


# ============================
# Training
# ============================
def train_model(model, Xtrain, labels_train, Xval, labels_val,
                epochs=15, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]
    best_val_acc = -1

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(N)

        train_correct = 0
        train_loss_sum = 0.0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            x = torch.from_numpy(pack_phase_channels(Xtrain[bidx])).to(device)
            y = torch.from_numpy(labels_train[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_acc = train_correct / N
        train_loss = train_loss_sum / N

        model.eval()
        val_correct = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                x = torch.from_numpy(pack_phase_channels(Xval[i:i+batch_size])).to(device)
                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_acc = val_correct / Nv
        val_loss = val_loss_sum / Nv
        best_val_acc = max(best_val_acc, val_acc)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return best_val_acc


# ============================
# Run all 3 conditions
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# phase only baseline
Xtrain_phase = make_phase_only(Xtrain)
Xval_phase   = make_phase_only(Xval)

print("\n=== BASELINE PHASE-ONLY ===")
model = PhaseWaveCNN3D()
baseline_acc = train_model(model, Xtrain_phase, labels_train, Xval_phase, labels_val, device=device)

# 1) voxelwise temporal shuffle
print("\n=== VOXELWISE TIME SHUFFLE ===")
Xtrain_voxshuf, perms_train = shuffle_time_independently_per_voxel(Xtrain_phase)
Xval_voxshuf, perms_val     = shuffle_time_independently_per_voxel(Xval_phase)

model = PhaseWaveCNN3D()
voxshuf_acc = train_model(model, Xtrain_voxshuf, labels_train, Xval_voxshuf, labels_val, device=device)

# 2) single-frame repeated
print("\n=== SINGLE FRAME REPEATED ===")
frame_idx = 0   # you can try 0, 3, 7, etc.
Xtrain_repeat = repeat_single_frame(Xtrain_phase, frame_idx=frame_idx)
Xval_repeat   = repeat_single_frame(Xval_phase, frame_idx=frame_idx)

model = PhaseWaveCNN3D()
repeat_acc = train_model(model, Xtrain_repeat, labels_train, Xval_repeat, labels_val, device=device)

print("\nRESULTS")
print("Baseline phase-only accuracy       :", baseline_acc)
print("Voxelwise time-shuffle accuracy    :", voxshuf_acc)
print("Single-frame repeated accuracy     :", repeat_acc)
print("Repeated frame index used          :", frame_idx)


#%%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utilities
# =========================================================
def make_phase_only(X):
    # X: complex array, shape (N, 1, T, H, W)
    return X / (np.abs(X) + 1e-8)


def pack_phase_channels(X):
    # returns real-valued array of shape (N, 2, T, H, W)
    xr = X.real.astype(np.float32)
    xi = X.imag.astype(np.float32)
    return np.concatenate([xr, xi], axis=1)


# =========================================================
# Model: explicit temporal-consistency head
# =========================================================
class PhaseWaveCNN3D_TemporalConsistency(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()

        # Input: (B, 2, 8, 11, 23)

        self.conv1 = nn.Conv3d(2,  12, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv2 = nn.Conv3d(12, 20, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        # spatial downsample only
        self.conv3 = nn.Conv3d(20, 28, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))

        self.conv4 = nn.Conv3d(28, 40, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv5 = nn.Conv3d(40, 48, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        # bottleneck
        self.conv6 = nn.Conv3d(48, 24, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # features:
        # mean_t   -> 24
        # std_t    -> 24
        # diff_t   -> 24
        # sim_mean -> 1
        # sim_std  -> 1
        feat_dim = 24 * 3 + 2

        self.fc1 = nn.Linear(feat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, 2, T, H, W)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        # x shape ~ (B, 24, T, H', W')
        # pool only over space, keep time
        z = x.mean(dim=(3, 4))   # (B, 24, T)

        # temporal summary features
        mean_t = z.mean(dim=2)                  # (B, 24)
        std_t  = z.std(dim=2, unbiased=False)   # (B, 24)

        diff_t = torch.abs(z[:, :, 1:] - z[:, :, :-1]).mean(dim=2)   # (B, 24)

        # cosine similarity between adjacent time embeddings
        z1 = F.normalize(z[:, :, :-1], dim=1)   # (B, 24, T-1)
        z2 = F.normalize(z[:, :, 1:],  dim=1)   # (B, 24, T-1)
        sim = (z1 * z2).sum(dim=1)              # (B, T-1)

        sim_mean = sim.mean(dim=1, keepdim=True)                 # (B, 1)
        sim_std  = sim.std(dim=1, unbiased=False, keepdim=True)  # (B, 1)

        feats = torch.cat([mean_t, std_t, diff_t, sim_mean, sim_std], dim=1)

        feats = self.dropout(feats)
        feats = torch.relu(self.fc1(feats))
        feats = self.dropout(feats)
        feats = torch.relu(self.fc2(feats))
        logits = self.fc3(feats)

        return logits


# =========================================================
# Training loop
# =========================================================
def train_model(model, Xtrain, labels_train, Xval, labels_val,
                epochs=20, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = Xtrain.shape[0]
    Nv = Xval.shape[0]

    best_val_loss = np.inf
    best_val_acc = -np.inf
    best_state = None

    for epoch in range(epochs):
        # -----------------------------
        # Train
        # -----------------------------
        model.train()
        idx = np.random.permutation(N)

        train_loss_sum = 0.0
        train_correct = 0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            x = torch.from_numpy(pack_phase_channels(Xtrain[bidx])).to(device)
            y = torch.from_numpy(labels_train[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.shape[0]
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / N
        train_acc = train_correct / N

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                x = torch.from_numpy(pack_phase_channels(Xval[i:i+batch_size])).to(device)
                y = torch.from_numpy(labels_val[i:i+batch_size].astype(np.float32)).to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.shape[0]
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y).sum().item()

        val_loss = val_loss_sum / Nv
        val_acc = val_correct / Nv

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    return model, best_val_loss, best_val_acc


# =========================================================
# Run
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# phase-only input
Xtrain_phase = make_phase_only(Xtrain)
Xval_phase   = make_phase_only(Xval)

model = PhaseWaveCNN3D_TemporalConsistency(dropout=0.30)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters:", n_params)

model, best_val_loss, best_val_acc = train_model(
    model,
    Xtrain_phase,
    labels_train,
    Xval_phase,
    labels_val,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    device=device
)

print("Best validation loss:", best_val_loss)
print("Best validation accuracy:", best_val_acc)


#%% USING CNN TO GET STABLE PHASE GRADIENT OVER TIME

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utilities
# =========================================================
def pack_real_imag(X):
    # X: complex, shape (N, 1, T, H, W)
    xr = X.real.astype(np.float32)
    xi = X.imag.astype(np.float32)
    return np.concatenate([xr, xi], axis=1)   # (N, 2, T, H, W)


def to_label_vector(y):
    y = np.asarray(y).astype(np.float32)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    return y


# =========================================================
# Model: dense phase-gradient predictor
# output shape = (B, 2, T, H, W)
# =========================================================
class PhaseGradientCNN3D(nn.Module):
    def __init__(self, dropout=0.15):
        super().__init__()

        self.conv1 = nn.Conv3d(2,  16, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 24, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(24, 32, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv5 = nn.Conv3d(32, 24, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(24, 16, kernel_size=(3,3,3), stride=1, padding=1)

        self.head  = nn.Conv3d(16, 2, kernel_size=(1,1,1), stride=1, padding=0)

        self.act = nn.ReLU()
        self.drop = nn.Dropout3d(dropout)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.drop(x)

        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.drop(x)

        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))

        out = self.head(x)   # (B,2,T,H,W)
        return out


# =========================================================
# Loss: smooth L1 + direction cosine loss
# =========================================================
def gradient_loss(pred, target, eps=1e-8, lambda_dir=0.25):
    """
    pred, target: (B, 2, T, H, W)
    """
    loss_mag = F.smooth_l1_loss(pred, target)

    pred_n = pred / (torch.sqrt(torch.sum(pred**2, dim=1, keepdim=True)) + eps)
    targ_n = target / (torch.sqrt(torch.sum(target**2, dim=1, keepdim=True)) + eps)

    cos = torch.sum(pred_n * targ_n, dim=1)   # (B,T,H,W)

    targ_mag = torch.sqrt(torch.sum(target**2, dim=1))   # (B,T,H,W)
    mask = (targ_mag > 1e-6).float()

    loss_dir = ((1.0 - cos) * mask).sum() / (mask.sum() + eps)

    return loss_mag + lambda_dir * loss_dir


# =========================================================
# Stability score from predicted gradients
# =========================================================
def gradient_stability_score(G, eps=1e-8):
    """
    G: numpy array, shape (N, 2, T, H, W)

    Returns:
        score: shape (N,)
        pixel_stability: shape (N, H, W)
    """
    gx = G[:, 0]
    gy = G[:, 1]

    mag = np.sqrt(gx**2 + gy**2 + eps)
    ux = gx / mag
    uy = gy / mag

    mean_ux = np.mean(ux, axis=1)   # average over time
    mean_uy = np.mean(uy, axis=1)

    pixel_stability = np.sqrt(mean_ux**2 + mean_uy**2)   # (N,H,W), in [0,1]
    score = pixel_stability.mean(axis=(1, 2))            # one score per epoch

    return score, pixel_stability


# =========================================================
# Threshold selection
# =========================================================
def best_threshold_from_scores(scores, labels):
    labels = to_label_vector(labels)
    best_thr = 0.5
    best_acc = -1.0

    for thr in np.linspace(scores.min(), scores.max(), 201):
        preds = (scores >= thr).astype(np.float32)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_thr, best_acc


# =========================================================
# Prediction helper
# =========================================================
def predict_gradients(model, X, batch_size=128, device="cuda"):
    model.eval()
    Xp = pack_real_imag(X)
    N = Xp.shape[0]
    outs = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(Xp[i:i+batch_size]).to(device)
            yb = model(xb)
            outs.append(yb.cpu().numpy())

    return np.concatenate(outs, axis=0)   # (N,2,T,H,W)


# =========================================================
# Training loop
# =========================================================
def train_gradient_model(model,
                         Xtrain, Gtrain, labels_train,
                         Xval, Gval, labels_val,
                         epochs=20, batch_size=128, lr=1e-3,
                         device="cuda"):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Xtrain_p = pack_real_imag(Xtrain)
    Xval_p   = pack_real_imag(Xval)

    labels_train_vec = to_label_vector(labels_train)
    labels_val_vec   = to_label_vector(labels_val)

    N = Xtrain_p.shape[0]
    Nv = Xval_p.shape[0]

    best_val_loss = np.inf
    best_state = None
    best_thr = None
    best_val_acc = None

    for epoch in range(epochs):
        # ---------------------
        # Train
        # ---------------------
        model.train()
        idx = np.random.permutation(N)

        train_loss_sum = 0.0

        for i in range(0, N, batch_size):
            bidx = idx[i:i+batch_size]

            xb = torch.from_numpy(Xtrain_p[bidx]).to(device)
            gb = torch.from_numpy(Gtrain[bidx].astype(np.float32)).to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = gradient_loss(pred, gb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.shape[0]

        train_loss = train_loss_sum / N

        # ---------------------
        # Validation loss
        # ---------------------
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for i in range(0, Nv, batch_size):
                xb = torch.from_numpy(Xval_p[i:i+batch_size]).to(device)
                gb = torch.from_numpy(Gval[i:i+batch_size].astype(np.float32)).to(device)

                pred = model(xb)
                loss = gradient_loss(pred, gb)
                val_loss_sum += loss.item() * xb.shape[0]

        val_loss = val_loss_sum / Nv

        # ---------------------
        # Validation wave/non-wave from stability
        # ---------------------
        Gpred_val = predict_gradients(model, Xval, batch_size=batch_size, device=device)
        val_scores, _ = gradient_stability_score(Gpred_val)
        thr, val_acc = best_threshold_from_scores(val_scores, labels_val_vec)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr
            best_val_acc = val_acc

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train GradLoss: {train_loss:.4f} | "
            f"Val GradLoss: {val_loss:.4f} | "
            f"Val WaveAcc: {val_acc:.4f} | "
            f"Thr: {thr:.4f}"
        )

    model.load_state_dict(best_state)
    return model, best_val_loss, best_thr, best_val_acc


# =========================================================
# Final evaluation helper
# =========================================================
def evaluate_wave_detection(model, X, labels, thr, batch_size=128, device="cuda"):
    labels = to_label_vector(labels)

    Gpred = predict_gradients(model, X, batch_size=batch_size, device=device)
    scores, pixel_stability = gradient_stability_score(Gpred)

    preds = (scores >= thr).astype(np.float32)
    acc = np.mean(preds == labels)

    return {
        "accuracy": acc,
        "scores": scores,
        "preds": preds,
        "pixel_stability": pixel_stability,
        "pred_gradients": Gpred,
    }


# =========================================================
# Run
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PhaseGradientCNN3D(dropout=0.15)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters:", n_params)

import numpy as np

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_phase_gradients_complex(X, sigma=1.0):
    """
    Compute spatial phase gradients from complex data and smooth them spatially.

    Parameters
    ----------
    X : np.ndarray
        Complex array of shape (N, 1, T, H, W)

    sigma : float
        Spatial Gaussian smoothing parameter.
        This is an approximation to MATLAB smoothn.

    Returns
    -------
    G : np.ndarray
        Real array of shape (N, 2, T, H, W)
        channel 0 = gx
        channel 1 = gy
    """
    assert X.ndim == 5, f"Expected (N,1,T,H,W), got {X.shape}"
    assert X.shape[1] == 1, f"Expected channel dim = 1, got {X.shape[1]}"

    z = X[:, 0]   # (N,T,H,W)

    # forward phase differences using complex ratio
    gx_fwd = np.angle(z[..., 1:] * np.conj(z[..., :-1]))         # (N,T,H,W-1)
    gy_fwd = np.angle(z[:, :, 1:, :] * np.conj(z[:, :, :-1, :])) # (N,T,H-1,W)

    gx = np.zeros_like(z.real, dtype=np.float32)
    gy = np.zeros_like(z.real, dtype=np.float32)

    # central-style fill
    gx[..., 1:-1] = 0.5 * (gx_fwd[..., :-1] + gx_fwd[..., 1:])
    gx[..., 0]    = gx_fwd[..., 0]
    gx[..., -1]   = gx_fwd[..., -1]

    gy[:, :, 1:-1, :] = 0.5 * (gy_fwd[:, :, :-1, :] + gy_fwd[:, :, 1:, :])
    gy[:, :, 0, :]    = gy_fwd[:, :, 0, :]
    gy[:, :, -1, :]   = gy_fwd[:, :, -1, :]

    # spatial smoothing per time frame
    # sigma only over H,W, not over N or T
    for n in range(gx.shape[0]):
        for t in range(gx.shape[1]):
            gx[n, t] = gaussian_filter(gx[n, t], sigma=sigma, mode="nearest")
            gy[n, t] = gaussian_filter(gy[n, t], sigma=sigma, mode="nearest")

    G = np.stack([gx, gy], axis=1)   # (N,2,T,H,W)
    return G.astype(np.float32)

# ---------------------------------------------------
# Example usage
# ---------------------------------------------------
# Xtrain, Xval are complex arrays of shape (N,1,T,H,W)
Gtrain = compute_phase_gradients_complex(Xtrain, sigma=1.0)
Gval   = compute_phase_gradients_complex(Xval, sigma=1.0)

print(Gtrain.shape)   # (N,2,T,H,W)
print(Gval.shape)

print("Xtrain shape:", Xtrain.shape)
print("Gtrain shape:", Gtrain.shape)
print("Xval shape:", Xval.shape)
print("Gval shape:", Gval.shape)

model, best_val_loss, best_thr, best_val_acc = train_gradient_model(
    model,
    Xtrain, Gtrain, labels_train,
    Xval,   Gval,   labels_val,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    device=device
)

print("Best validation gradient loss:", best_val_loss)
print("Best validation wave accuracy:", best_val_acc)
print("Best stability threshold:", best_thr)

# Optional final eval on validation set using saved best model
val_out = evaluate_wave_detection(
    model, Xval, labels_val, thr=best_thr, batch_size=128, device=device
)
print("Final val wave accuracy:", val_out["accuracy"])
