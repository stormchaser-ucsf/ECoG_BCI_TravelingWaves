# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:22:15 2025

@author: nikic
"""

# BUILD A CNN MODEL USING SIMULATED DATA FIRST

#%% PRELIMS
import os

os.chdir('C:/Users/nikic/Documents/GitHub/ECoG_BCI_TravelingWaves/CNN3D')


#from iAE_utils_models import *

import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import mat73
import matplotlib.pyplot as plt


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import h5py
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader

#%%

# get files and labels
file_paths = [
    ('simulate_rotate_CCW.mat', 0),
    ('simulate_rotate_CW.mat', 1),
    ('simulate_sink_wave.mat', 2),
    ('simulate_source_wave.mat', 3),
    ('simulate_spiral_in_CCW.mat', 4),
    ('simulate_spiral_in_CW.mat', 5),
    ('simulate_spiral_out_CCW.mat', 6),
    ('simulate_spiral_out_CW.mat', 7),
    ('simulate_planar_wave.mat', 8)
]

x_data, y_data, labels = [], [], []

for file_path, label in file_paths:
    with h5py.File(file_path, 'r') as mat_file:
        x = np.array(mat_file['x'])  # Load x data
        y = np.array(mat_file['y'])  # Load y data

    # transform shape -- to trial*channel*f1*f2*time
    x = x.transpose(0, 1, 4, 3, 2)
    y = y.transpose(0, 1, 4, 3, 2)
    # x = x.transpose(0, 1, 2, 4, 3)
    # y = y.transpose(0, 1, 2, 4, 3)

    x_data.append(x)
    y_data.append(y)
    labels.extend([label] * x.shape[0]) 

x_data = np.concatenate(x_data, axis=0)
y_data = np.concatenate(y_data, axis=0)
labels = np.array(labels)

x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(x_tensor, y_tensor, labels_tensor)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"x_data: {x_data.shape}")
print(f"Train Data: {len(train_dataset)} samples")
print(f"Validation Data: {len(val_dataset)} samples")



#%% BUILD AND TRAIN THE CNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class Encoder3D(nn.Module):
    def __init__(self,ksize):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 12, kernel_size=ksize, stride=(1, 1, 2), padding=0) 
        self.conv2 = nn.Conv3d(12, 12, kernel_size=ksize, stride=(1, 2, 2), padding=0) # downsampling h
        self.conv3 = nn.Conv3d(12, 12, kernel_size=ksize, stride=(1, 1, 2), padding=0)
        self.conv4 = nn.Conv3d(12, 6, kernel_size=ksize, stride=(1, 1, 2), padding=0)
        #self.fc = nn.Linear(6 * 7 * 9 *25, num_nodes)    # 6 filters, w, h, d    
        self.elu = nn.ELU()
        #self.pool = nn.AvgPool3d(kernel_size=ksize,stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x = self.conv2(x)
        x = self.elu(x)
        #x = self.pool(x)
        
        x=self.conv3(x)
        x=self.elu(x)
        
        x=self.conv4(x)
        x=self.elu(x)
        
        #x = x.view(x.size(0), -1) # can replace this with a LSTM and a classification layer b/w OL and CL
        #x=self.fc(x)
        
        return x

# Define the Decoder network to reconstruct the original shape
class Decoder3D(nn.Module):
    def __init__(self,ksize):
        super(Decoder3D, self).__init__()
        #self.fc = nn.Linear(num_nodes, 6 * 7 * 9 *25)
        self.deconv1 = nn.ConvTranspose3d(6, 12, kernel_size=ksize, stride=(1, 1, 2))
        self.deconv2 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 1, 2))
        # self.deconv3 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 2, 2),
        #                                   dilation = (2,3,3),padding=0,output_padding=(0,1,1))
        self.deconv3 = nn.ConvTranspose3d(12, 12, kernel_size=ksize, stride=(1, 2, 2))
                                           #dilation = (2,3,7),padding=0,output_padding=(0,1,1))
        self.deconv4 = nn.ConvTranspose3d(12, 1, kernel_size=ksize, stride=(1, 1, 2))
                                           #output_padding=(0, 0, 1),dilation=(2,2,2))
        self.elu = nn.ELU()

    def forward(self, x):
        #x = self.fc(x)
        #x = self.elu(x)
        #x = x.view(x.size(0), 6, 7, 9, 25)
        x = self.deconv1(x)
        x = self.elu(x)
        x = self.deconv2(x)
        x = self.elu(x)
        x = self.deconv3(x)
        x = self.elu(x)
        x = self.deconv4(x)
        x = torch.tanh(x)
        return x

# Initialize the models
#num_nodes=256;
ksize=2;
encoder = Encoder3D(ksize).to(device)
decoder = Decoder3D(ksize).to(device)

# testing it out
input = torch.randn(32,1,11,23,40).to(device)
out1 = encoder(input)
output = decoder(out1)


# creating a bidir LSTM -> GRU -> MLP, with GRU having half the number of hidden nodes
# (batch, seq, feature)
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

# latent test
num_classes=2
input_size=378
lstm_size=64
latent_classifier = rnn_lstm(num_classes,input_size,lstm_size).to(device)
latent = latent_classifier(out1)
    
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

num_classes=2
input_size=378
lstm_size=64
ksize=2;
autoencoder = Autoencoder3D(ksize,num_classes,input_size,lstm_size).to(device)

# testing it oot
recon,out = autoencoder(input)

# class Autoencoder3D(nn.Module):
#     def __init__(self, encoder, decoder, num_classes):
#         super(Autoencoder3D, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.classifier = nn.Linear(128, num_classes)  

#     def forward(self, x):
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         logits = self.classifier(latent)
#         return reconstructed, logits
    

#num_classes = 9  # modify based on types of waves
#autoencoder = Autoencoder3D(encoder, decoder, num_classes).to(device)

# 定义损失函数
criterion_reconstruction = nn.MSELoss()
criterion_classification = nn.CrossEntropyLoss()  

optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

# weights
alpha = 1  
beta = 1   

# training
num_epochs = 100
patience = 6
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(num_epochs):
    autoencoder.train()
    train_reconstruction_loss = 0
    train_classification_loss = 0

    for batch_x, batch_y, batch_labels in train_loader:
        batch_x, batch_y, batch_labels = batch_x.to(device), batch_y.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        reconstructed, logits = autoencoder(batch_x)

        reconstruction_loss = criterion_reconstruction(reconstructed, batch_y)
        classification_loss = criterion_classification(logits, batch_labels)
        total_loss = alpha * reconstruction_loss + beta * classification_loss

        total_loss.backward()
        optimizer.step()

        train_reconstruction_loss += reconstruction_loss.item()
        train_classification_loss += classification_loss.item()

    train_reconstruction_loss /= len(train_loader)
    train_classification_loss /= len(train_loader)

    autoencoder.eval()
    val_reconstruction_loss = 0
    val_classification_loss = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_labels in val_loader:
            batch_x, batch_y, batch_labels = batch_x.to(device), batch_y.to(device), batch_labels.to(device)
            reconstructed, logits = autoencoder(batch_x)

            reconstruction_loss = criterion_reconstruction(reconstructed, batch_y)
            classification_loss = criterion_classification(logits, batch_labels)

            val_reconstruction_loss += reconstruction_loss.item()
            val_classification_loss += classification_loss.item()

    val_reconstruction_loss /= len(val_loader)
    val_classification_loss /= len(val_loader)
    val_total_loss = alpha * val_reconstruction_loss + beta * val_classification_loss

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Reconstruction Loss: {train_reconstruction_loss:.4f}")
    print(f"  Train Classification Loss: {train_classification_loss:.4f}")
    print(f"  Val Reconstruction Loss: {val_reconstruction_loss:.4f}")
    print(f"  Val Classification Loss: {val_classification_loss:.4f}")
    print(f"  Val Total Loss: {val_total_loss:.4f}")


    # early stop
    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        epochs_no_improve = 0
        best_model = autoencoder.state_dict()  
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        autoencoder.load_state_dict(best_model)
        break


#%% viusalization

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# File paths and labels
file_paths = [
    ('simulate_rotate_CCW.mat', 0),
    ('simulate_rotate_CW.mat', 1),
    ('simulate_sink_wave.mat', 2),
    ('simulate_source_wave.mat', 3),
    ('simulate_spiral_in_CCW.mat', 4),
    ('simulate_spiral_in_CW.mat', 5),
    ('simulate_spiral_out_CCW.mat', 6),
    ('simulate_spiral_out_CW.mat', 7),
    ('simulate_planar_wave.mat', 8)
]

# Generate label descriptions with all uppercase for 'CW' and 'CCW'
label_descriptions = {label: name.replace("simulate_", "").replace(".mat", "").replace("_", " ").title()
                      for name, label in file_paths}
label_descriptions = {k: v.replace("Ccw", "CCW").replace("Cw", "CW") for k, v in label_descriptions.items()}

# Function to visualize PCA-transformed latent space in 2D or 3D
def visualize_pca(latent_space, labels, n_components=2):
    """
    Visualize PCA-transformed latent space with labels.

    Args:
        latent_space: PCA-transformed latent space.
        labels: Corresponding labels for the data points.
        n_components: Number of dimensions for visualization (2 or 3).
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            idx = labels == label
            plt.scatter(latent_space[idx, 0], latent_space[idx, 1], color=colors[i], 
                        label=label_descriptions[label], alpha=0.7)
        plt.title("PCA Visualization of Latent Space (2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            idx = labels == label
            ax.scatter(latent_space[idx, 0], latent_space[idx, 1], latent_space[idx, 2], 
                       color=colors[i].tolist(), label=label_descriptions[label], alpha=0.7)
        ax.set_title("PCA Visualization of Latent Space (3D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.legend()
        plt.show()
        
def extract_latent_space(encoder, data_loader):
    encoder.eval()  # Set encoder to evaluation mode
    latent_spaces = []
    labels = []
    with torch.no_grad():
        for batch_x, _, batch_labels in data_loader:
            batch_x = batch_x.to(device)
            latent = encoder(batch_x)  # Pass through the encoder
            latent_spaces.append(latent.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return np.concatenate(latent_spaces, axis=0), np.array(labels)

# Extract the latent space and corresponding labels
latent_space, train_labels = extract_latent_space(encoder, train_loader)

# Standardize the latent space
scaler = StandardScaler()
latent_space_scaled = scaler.fit_transform(latent_space)

# Perform PCA and visualize
n_components = 3  # Change to 2 for 2D visualization
pca = PCA(n_components=n_components)
latent_space_pca = pca.fit_transform(latent_space_scaled)

# Print the shape of the transformed latent space
print(latent_space_pca.shape)

# Visualize the PCA-transformed latent space
visualize_pca(latent_space_pca, train_labels, n_components=n_components)


#%% one more viz

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# File paths and labels
file_paths = [
    ('simulate_rotate_CCW.mat', 0),
    ('simulate_rotate_CW.mat', 1),
    ('simulate_sink_wave.mat', 2),
    ('simulate_source_wave.mat', 3),
    ('simulate_spiral_in_CCW.mat', 4),
    ('simulate_spiral_in_CW.mat', 5),
    ('simulate_spiral_out_CCW.mat', 6),
    ('simulate_spiral_out_CW.mat', 7),
    ('simulate_planar_wave.mat', 8)
]

# Generate label descriptions with all uppercase for 'CW' and 'CCW'
label_descriptions = {label: name.replace("simulate_", "").replace(".mat", "").replace("_", " ").title()
                      for name, label in file_paths}
label_descriptions = {k: v.replace("Ccw", "CCW").replace("Cw", "CW") for k, v in label_descriptions.items()}

# Function to visualize PCA-transformed latent space in 2D or 3D
def visualize_pca(latent_space, labels, n_components=2):
    """
    Visualize PCA-transformed latent space with labels.

    Args:
        latent_space: PCA-transformed latent space.
        labels: Corresponding labels for the data points.
        n_components: Number of dimensions for visualization (2 or 3).
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            idx = labels == label
            plt.scatter(latent_space[idx, 0], latent_space[idx, 1], color=colors[i], 
                        label=label_descriptions[label], alpha=0.7)
        plt.title("PCA Visualization of Latent Space (2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            idx = labels == label
            ax.scatter(latent_space[idx, 0], latent_space[idx, 1], latent_space[idx, 2], 
                       color=colors[i].tolist(), label=label_descriptions[label], alpha=0.7)
        ax.set_title("PCA Visualization of Latent Space (3D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.legend()
        plt.show()
        
def extract_latent_space(encoder, data_loader):
    encoder.eval()  # Set encoder to evaluation mode
    latent_spaces = []
    labels = []
    with torch.no_grad():
        for batch_x, _, batch_labels in data_loader:
            batch_x = batch_x.to(device)
            latent = encoder(batch_x)  # Pass through the encoder
            latent_spaces.append(latent.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return np.concatenate(latent_spaces, axis=0), np.array(labels)

# Extract the latent space and corresponding labels
latent_space, val_labels = extract_latent_space(encoder, val_loader)

# Standardize the latent space
scaler = StandardScaler()
latent_space_scaled = scaler.fit_transform(latent_space)

# Perform PCA and visualize
n_components = 3  # Change to 2 for 2D visualization
pca = PCA(n_components=n_components)
latent_space_pca = pca.fit_transform(latent_space_scaled)

# Print the shape of the transformed latent space
print(latent_space_pca.shape)

# Visualize the PCA-transformed latent space
visualize_pca(latent_space_pca, val_labels, n_components=n_components)


#%% real data

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re

# Extract label from file path

def extract_label_from_path(file_path):
    """
    Extract label from file path based on its naming convention.
    Args:
        file_path: File path containing the label.
    Returns:
        A human-readable label string.
    """
    # Adjusted regex to account for any text between 'online' or 'imagine' and 'day'
    match = re.search(r'(imagine|online).*_day(\d+)', file_path, re.IGNORECASE)
    if match:
        phase, day = match.groups()
        return f"{phase.capitalize()} Day {day}"  # Format as "Imagine Day X" or "Online Day X"
    else:
        raise ValueError(f"Invalid file name format: {file_path}")


# Load the imagine/online data
file_path = 'b3_hand_imagine_alpha_day1.mat'  # Example file path
with h5py.File(file_path, 'r') as mat_file:
    x_new = np.array(mat_file['x'])
    y_new = np.array(mat_file['y'])

# Transpose the data to match PyTorch input shape
x_new = x_new.transpose(0, 1, 4, 3, 2)
y_new = y_new.transpose(0, 1, 4, 3, 2)

# Convert the new data to PyTorch tensors
x_new_tensor = torch.tensor(x_new, dtype=torch.float32).to(device)

# Extract latent space for the new data
def extract_new_latent_space(encoder, x_new_tensor):
    encoder.eval()  # Set the encoder to evaluation mode
    latent_spaces = []
    with torch.no_grad():
        for i in range(x_new_tensor.size(0)):  # Iterate through each sample
            sample_x = x_new_tensor[i:i+1]  # Add batch dimension
            latent = encoder(sample_x)
            latent_spaces.append(latent.cpu().numpy())
    return np.concatenate(latent_spaces, axis=0)

# Extract latent space for new data
latent_space_new = extract_new_latent_space(encoder, x_new_tensor)

# Standardize latent spaces
scaler = StandardScaler()
# Perform PCA on combined data
n_components = 3  # Set to 2 for 2D visualization
pca = PCA(n_components=n_components)
latent_space_pca_val = pca.fit_transform(latent_space_scaled)  # Fit PCA on latent_space_scaled

# Project latent_space_new into the PCA space learned from latent_space_scaled
latent_space_pca_new = pca.transform(latent_space_new)  

# Extract label from file path
new_label = extract_label_from_path(file_path)

# Plot combined PCA-transformed data
def visualize_combined_pca(latent_space_val, labels_val, latent_space_new, new_label, n_components=3):
    """
    Visualize combined PCA-transformed latent spaces for val and new data.
    
    Args:
        latent_space_val: PCA-transformed latent space for validation data.
        labels_val: Labels for validation data.
        latent_space_new: PCA-transformed latent space for new data.
        new_label: Label for new data extracted from file path.
        n_components: Number of dimensions for visualization (2 or 3).
    """
    unique_labels = np.unique(labels_val)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            idx = labels_val == label
            plt.scatter(latent_space_val[idx, 0], latent_space_val[idx, 1], 
                        color=colors[i], label=label_descriptions[label], alpha=0.7)
        plt.scatter(latent_space_new[:, 0], latent_space_new[:, 1], 
                    color="black", label=new_label, alpha=0.7, marker='x')
        plt.title("PCA Visualization of Latent Space (2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            idx = labels_val == label
            ax.scatter(latent_space_val[idx, 0], latent_space_val[idx, 1], latent_space_val[idx, 2], 
                       color=colors[i].tolist(), label=label_descriptions[label], alpha=0.7)
        ax.scatter(latent_space_new[:, 0], latent_space_new[:, 1], latent_space_new[:, 2], 
                   color="black", label=new_label, alpha=0.7, marker='x')
        ax.set_title("PCA Visualization of Latent Space (3D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.legend()
        plt.show()

# Visualize the combined latent space
visualize_combined_pca(latent_space_pca_val, val_labels, latent_space_pca_new, new_label, n_components=n_components)


#%% plot activations

sample_x = x_new_tensor[0]
sample_x = sample_x.unsqueeze(0)

activations = {}

# Define a hook function to capture the output of each layer
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()  # Store activation and move to CPU
    return hook

# Register hooks for each convolutional layer in the encoder
encoder.conv1.register_forward_hook(get_activation("conv1"))
encoder.conv2.register_forward_hook(get_activation("conv2"))
encoder.conv3.register_forward_hook(get_activation("conv3"))

# Perform a forward pass to get activations
with torch.no_grad():
    _ = autoencoder.encoder(sample_x)

import matplotlib.pyplot as plt
import math

# Function to visualize activation maps over time in a 2D grid layout
def plot_activations_2d(activations, layer_name, n_cols=25):
    activation = activations[layer_name].squeeze(0)  # Remove batch dimension
    n_time = activation.size(-1)  # Number of time points

    # Calculate the number of rows and columns for the grid
    n_rows = math.ceil(n_time / n_cols)

    # Create a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.suptitle(f"Activation Maps over Time for {layer_name}", fontsize=16)

    for t in range(n_time):
        # Calculate the row and column index in the grid
        row, col = divmod(t, n_cols)
        averaged_activation = activation[:, :, :, t].mean(dim=0)  # Average across channels

        ax = axes[row, col]
        ax.imshow(averaged_activation, cmap="viridis", aspect="auto")  # Plot averaged activation
        ax.set_title(f"Time {t+1}")
        ax.axis("off")

    # Turn off unused subplots
    for i in range(n_time, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()

# Visualize activations for each convolutional layer in a 2D grid
plot_activations_2d(activations, "conv1")
plot_activations_2d(activations, "conv2")
plot_activations_2d(activations, "conv3")

#%% one more


import matplotlib.pyplot as plt
import math

# Function to visualize activation maps for each channel over time
def plot_each_channel(activations, layer_name, n_cols=25):
    activation = activations[layer_name].squeeze(0)  # Remove batch dimension
    n_channels = activation.size(0)  # Number of channels
    n_time = activation.size(-1)  # Number of time points

    # Calculate the number of rows needed (one subplot for each channel)
    n_rows = math.ceil(n_time / n_cols)

    for ch in range(n_channels):    
        # Create a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        fig.suptitle(f"Activation Maps for Each Channel Over Time in {layer_name}", fontsize=16)

        # Plot each time step as a subplot
        for t in range(n_time):
            # Get the activations for the current channel across all time points
            row, col = divmod(t, n_cols)
            channel_activation = activation[ch, :, :, t]

            ax = axes[row, col]
            ax.imshow(channel_activation, cmap="viridis", aspect="auto")  # Plot the activation map
            
            # Add titles and axis details
            ax.set_title(f"Channel {ch+1}, Time {t+1}")
            ax.axis("off")

    # Turn off unused subplots
    for i in range(n_time, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()

# Visualize activations for a specific convolutional layer
plot_each_channel(activations, "conv1", n_cols=25)


#%%

#loading files from matlab

import scipy.signal as signal
filename = 'F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/20230510/HandImagined/114718/Imagined/Data0001.mat'
data_dict = mat73.loadmat(filename)
data_imagined = data_dict.get('TrialData')

tmp = data_imagined.get('BroadbandData')

x=np.concatenate(tmp)

fs=1000

lo=8
hi=10
order=4
b,a = signal.butter(order,[lo/(fs/2), hi/(fs/2)],btype='band')

y = signal.filtfilt(b,a,x,axis=0)



