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
tmp_real,tmp_imag = np.real(tmp),np.imag(tmp)

#tmp_real = rnd.randn(64,1,40,11,23)
#tmp_real = rnd.randn(64,1,40,11,23)
tmp_real = torch.from_numpy(tmp_real).float().to(device)
tmp_imag = torch.from_numpy(tmp_imag).float().to(device)

r,i = model(tmp_real,tmp_imag)


print(tmp_real.shape)

model = nn.Conv3d(1, 8, kernel_size=(2,2,3),dilation=(1,1,2))
out=model(tmp_real)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(8,8, kernel_size=(2,2,3),dilation=(1,1,2))
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(8, 12, kernel_size=(2,2,3),dilation=(1,1,2))
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(12, 12, kernel_size=(2,3,3),dilation=(1,2,2),stride=(1,1,1 ))
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(12, 12, kernel_size=(2,3,3),dilation=(1,2,2),stride=(1,1,1))
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(12, 16, kernel_size=(2,3,3),dilation=(1,2,2),stride=(1,1,1))
out=model(out)
print(out.shape)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = total_params+trainable_params

model = nn.Conv3d(16, 16, kernel_size=(2,3,4),dilation=(1,2,3),stride=(1,1,1))
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