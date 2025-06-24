
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from iAE_utils_models import *

# ==== USER DEFINED IMPORTS ====
# from your_model import ComplexAutoencoder3D, classifier
# from your_dataset import YourDatasetClass

# ==== DEVICE ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== LOAD MODEL ====
# model = ComplexAutoencoder3D(...)
# model.load_state_dict(torch.load('your_trained_model.pth'))
# model = model.to(device)
# model.eval()

# ==== LOAD DATASET ====
# dataset = YourDatasetClass(...)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# ==== ABLATION HOOK ====
def make_channel_ablation_hook(indices):
    def hook(module, input, output):
        output[:, indices] = 0
        return output
    return hook

# ==== FIND CONV LAYERS ====
def find_conv_layers_encoder_real(model):
    conv_layers_real = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and ('real' in name):
            conv_layers_real[name] = module
    return conv_layers_real

def find_conv_layers_encoder_imag(model):
    conv_layers_imag = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and ( 'imag' in name):
            conv_layers_imag[name] = module
    return conv_layers_imag



def find_conv_layers_decoder_real(model):
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ConvTranspose3d) and ('real' in name ):
            conv_layers[name] = module
    return conv_layers


def find_conv_layers_decoder_imag(model):
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ConvTranspose3d) and ( 'imag' in name):
            conv_layers[name] = module
    return conv_layers


# ==== ABLATION LOOP ====
def ablate_kernels_encoder(model, Xtest,Ytest):
    conv_layers_real = find_conv_layers_encoder_real(model)
    conv_layers_imag = find_conv_layers_encoder_imag(model)
    a=iter(conv_layers_real.items())
    b=iter(conv_layers_imag.items())
    results = {}
    tmp_ydata_r,tmp_ydata_i = Ytest.real, Ytest.imag
    tmp_ydata_r = torch.from_numpy(tmp_ydata_r)
    tmp_ydata_i = torch.from_numpy(tmp_ydata_i)
    
    for i in np.arange(len(conv_layers_real)):
        real_layer_name,real_layer = next(a)
        imag_layer_name,imag_layer = next(b)        
        lay_name = 'conv' + str(i+1)
        results[lay_name] = {}
        num_kernels = imag_layer.out_channels
        
        for k in range(num_kernels):
            print(f"Ablating complex {lay_name}[{k}]...")
            hook_real = real_layer.register_forward_hook(make_channel_ablation_hook([k]))
            hook_imag = imag_layer.register_forward_hook(make_channel_ablation_hook([k]))
            
            recon_r,recon_i,decodes = test_model_complex(model,Xtest)
            
            recon_r = torch.from_numpy(recon_r)            
            recon_i = torch.from_numpy(recon_i)            
            
            loss =  F.mse_loss(recon_r, tmp_ydata_r) + F.mse_loss(recon_i, tmp_ydata_i)
            results[lay_name][k] = loss
            hook_real.remove()           
            hook_imag.remove()   

    
    return results

def ablate_kernels_decoder(model, Xtest,Ytest):
    conv_layers_real = find_conv_layers_decoder_real(model)
    conv_layers_imag = find_conv_layers_decoder_imag(model)
    a=iter(conv_layers_real.items())
    b=iter(conv_layers_imag.items())
    results = {}
    tmp_ydata_r,tmp_ydata_i = Ytest.real, Ytest.imag
    tmp_ydata_r = torch.from_numpy(tmp_ydata_r)
    tmp_ydata_i = torch.from_numpy(tmp_ydata_i)
    
    for i in np.arange(len(conv_layers_real)):
        real_layer_name,real_layer = next(a)
        imag_layer_name,imag_layer = next(b)        
        lay_name = 'deconv' + str(i+1)
        results[lay_name] = {}
        num_kernels = imag_layer.out_channels
        
        for k in range(num_kernels):
            print(f"Ablating complex {lay_name}[{k}]...")
            hook_real = real_layer.register_forward_hook(make_channel_ablation_hook([k]))
            hook_imag = imag_layer.register_forward_hook(make_channel_ablation_hook([k]))
            
            recon_r,recon_i,decodes = test_model_complex(model,Xtest)
            
            recon_r = torch.from_numpy(recon_r)            
            recon_i = torch.from_numpy(recon_i)            
            
            loss =  F.mse_loss(recon_r, tmp_ydata_r) + F.mse_loss(recon_i, tmp_ydata_i)
            results[lay_name][k] = loss
            hook_real.remove()           
            hook_imag.remove()   

    
    return results



#%% plotting kernels as heat maps

#%% TESTING STUFF

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate synthetic alpha traveling wave
# -----------------------------
def generate_traveling_wave(H=11, W=23, T=40, f=9, direction='right', velocity=2):
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    t = np.linspace(0, 0.2, T)  # 200 ms duration

    X, Y, T_mesh = np.meshgrid(x, y, t, indexing='ij')

    if direction == 'right':
        phase = 2 * np.pi * (f * T_mesh - X * velocity)
    elif direction == 'down':
        phase = 2 * np.pi * (f * T_mesh - Y * velocity)
    else:
        phase = 2 * np.pi * f * T_mesh

    wave = np.sin(phase)
    wave = np.transpose(wave, (2, 1, 0))  # (T, H, W)
    return wave

# Create dataset (just 100 synthetic samples)
N = 100
waves = []
for _ in range(N):
    wave = generate_traveling_wave()
    waves.append(wave)

waves = np.stack(waves)  # shape: (N, T, H, W)
waves = waves[:, np.newaxis, :, :, :]  # (N, 1, T, H, W)
waves_tensor = torch.tensor(waves, dtype=torch.float32)

# -----------------------------
# 2. Define Tiny 3D CNN Autoencoder
# -----------------------------
class Tiny3DAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=2, stride=1),  # (N, 4, 39, 10, 22)
            nn.ReLU(),
            nn.Conv3d(4, 8, kernel_size=2, stride=1),  # (N, 8, 38, 9, 21)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(8, 4, kernel_size=2, stride=1),  # (N, 4, 39, 10, 22)
            nn.ReLU(),
            nn.ConvTranspose3d(4, 1, kernel_size=2, stride=1),  # (N, 1, 40, 11, 23)
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out

# -----------------------------
# 3. Training loop
# -----------------------------
model = Tiny3DAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training
epochs = 30
batch_size = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for i in range(0, N, batch_size):
        inputs = waves_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# -----------------------------
# 4. Visualize one reconstruction
# -----------------------------
model.eval()
with torch.no_grad():
    x = waves_tensor[0:1]
    recon = model(x)

x_np = x[0, 0].numpy()        # (T, H, W)
recon_np = recon[0, 0].numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(x_np[20], cmap='viridis')
plt.title('Original (t=20)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(recon_np[20], cmap='viridis')
plt.title('Reconstruction (t=20)')
plt.colorbar()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
conv1_weights = model.encoder[0].weight.data.cpu().numpy()

for i in range(conv1_weights.shape[0]):  # loop over filters
    kernel = conv1_weights[i, 0]  # shape: (2, 2, 2)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(kernel[0], cmap='bwr', vmin=-1, vmax=1)
    axs[0].set_title(f'Filter {i} - Depth 0')
    axs[1].imshow(kernel[1], cmap='bwr', vmin=-1, vmax=1)
    axs[1].set_title(f'Filter {i} - Depth 1')
    plt.tight_layout()
    plt.show()


# ==== RUN ABLATION ====
# results = ablate_kernels(model, dataloader)

# ==== SAVE RESULTS ====
# import json
# with open('ablation_results.json', 'w') as f:
#     json.dump(results, f, indent=2)

# ==== OPTIONAL: PLOT ====
# import matplotlib.pyplot as plt
# for layer, losses in results.items():
#     plt.figure()
#     plt.bar(losses.keys(), losses.values())
#     plt.title(f"Ablation Impact - {layer}")
#     plt.xlabel('Kernel Index')
#     plt.ylabel('Reconstruction Loss')
#     plt.show()
