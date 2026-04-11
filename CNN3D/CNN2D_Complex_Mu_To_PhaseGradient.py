#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:43:12 2026

@author: user
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

subj='B1'

# load the data 
if os.name=='nt':
    #filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'    
    filename='alpha_dynamics_200Hz_AllDays_M1_Complex_ArtifactCorr_SinglePrec.mat'
    filepath = 'F:\\DATA\\ecog data\\ECoG BCI\\GangulyServer\\Multistate B3\\'
    filename = filepath + filename
else:
    #filepath ='/mnt/DataDrive/ECoG_TravelingWaveProject_Nik/'
    #filepath = '/media/user/Data/ECoG_BCI_TravelingWave_Data/'
    #filename = 'alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'
    #filename = filepath + filename
    #filename = 'alpha_dynamics_200Hz_AllDays_DaysLabeled_ArtifactCorr_Complex_SinglePrec.mat'
    #filepath = '/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/'
    
    if subj=='B3':
            
        filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/'
        #filename ='all_data.mat'
        #filename ='all_data_B3_arrow.mat'
        filename ='all_data_B3_arrow_ol_cl.mat'
        filename = filepath + filename
        filename1 = 'all_data_B3_Hand_ol_cl.mat'
        filename1 = filepath + filename1 
        
    elif subj=='B1':
        filepath = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate clicker/'
        #filename ='all_data.mat'
        #filename ='all_data_B3_arrow.mat'
        filename ='Mu_Phase_Grad_Data_CNN.mat'
        filename = filepath + filename
    
        
        



#filename='F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate B3/alpha_dynamics_200Hz_AllDays_DaysLabeled.mat'
#filename = '/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/alpha_dynamics_200Hz_AllDays_DaysLabeled'



data_dict = mat73.loadmat(filename)
X = data_dict.get('Xtrain')
Y = data_dict.get('Ytrain')
labels = data_dict.get('labels')


if subj=='B3':        
    data_dict1 = mat73.loadmat(filename1)
    data1 =  data_dict1.get('all_data_hand')


# xdata = data_dict.get('xdata')
# ydata = data_dict.get('ydata')
# labels = data_dict.get('labels')
# labels_days = data_dict.get('days')
# labels_batch = data_dict.get('labels_batch')

# xdata = np.concatenate(xdata)
# ydata = np.concatenate(ydata)

iterations = 1
decoding_accuracy=[]
balanced_decoding_accuracy=[]


if subj=='B3':
    del data_dict,data_dict1
else:
    del data_dict

#%% GET DATA

Xtrain,Xval,Xtest,
Ytrain,Yval,Ytest,
idx_train,idx_val,idx_test =  training_test_split_Mu_PhaseGradient_equal(X,Y,0.7)
   
Xtrain = np.concatenate((Xtrain,Xtest))       
Ytrain = np.concatenate((Ytrain,Ytest))       
    
# expand dimensions for cnn 
Xtrain= np.expand_dims(Xtrain,axis=1)
Xval= np.expand_dims(Xval,axis=1)
Ytrain= np.expand_dims(Ytrain,axis=1)
Yval= np.expand_dims(Yval,axis=1)


#%% build and train model

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Dataset
# ============================================================

class ComplexGradientDataset(Dataset):
    """
    X: complex input, shape (N, 1, H, W)
    Y: complex target, shape (N, 1, H, W)

    Target interpretation:
        Y.real = gx
        Y.imag = gy
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        super().__init__()

        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(f"Expected X and Y to be 4D, got X={X.shape}, Y={Y.shape}")

        if X.shape != Y.shape:
            raise ValueError(f"X and Y must have same shape, got X={X.shape}, Y={Y.shape}")

        if X.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got X.shape[1]={X.shape[1]}")

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        if not torch.is_complex(self.X):
            raise ValueError("X must be complex-valued")
        if not torch.is_complex(self.Y):
            raise ValueError("Y must be complex-valued")

        if self.X.dtype == torch.complex128:
            self.X = self.X.to(torch.complex64)
        if self.Y.dtype == torch.complex128:
            self.Y = self.Y.to(torch.complex64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class RealGradientDataset(Dataset):
    """
    Converts complex input/output into 2 real channels.

    Input:
        X complex, shape (N, 1, H, W)
        Y complex, shape (N, 1, H, W)

    Returned:
        X_real: (N, 2, H, W) = [real(mu), imag(mu)]
        Y_real: (N, 2, H, W) = [gx, gy]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        super().__init__()

        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(f"Expected X and Y to be 4D, got X={X.shape}, Y={Y.shape}")

        if X.shape != Y.shape:
            raise ValueError(f"X and Y must have same shape, got X={X.shape}, Y={Y.shape}")

        if X.shape[1] != 1:
            raise ValueError(f"Expected 1 complex channel, got X.shape[1]={X.shape[1]}")

        X2 = np.concatenate([X.real, X.imag], axis=1)  # (N,2,H,W)
        Y2 = np.concatenate([Y.real, Y.imag], axis=1)  # (N,2,H,W)

        self.X = torch.from_numpy(X2).float()
        self.Y = torch.from_numpy(Y2).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]

# ============================================================
# 2. Complex layers
# ============================================================

class ComplexConv2d(nn.Module):
    """
    Complex convolution:
        (Wr + iWi) * (xr + ixi)
        = (Wr*xr - Wi*xi) + i(Wr*xi + Wi*xr)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        bias: bool = True
    ):
        super().__init__()

        self.real_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.imag_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x.real
        xi = x.imag

        yr = self.real_conv(xr) - self.imag_conv(xi)
        yi = self.real_conv(xi) + self.imag_conv(xr)

        return torch.complex(yr, yi)


class ModReLU(nn.Module):
    """
    2D modReLU
    """
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        mag = torch.sqrt(a ** 2 + b ** 2 + self.eps)
        scale = torch.relu(mag + self.bias) / (mag + self.eps)
        return a * scale, b * scale


class ComplexModReLU(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.modrelu = ModReLU(num_features=num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.modrelu(x.real, x.imag)
        return torch.complex(a, b)


# ============================================================
# 3. Model
# ============================================================

class ComplexPhaseGradientCNN(nn.Module):
    """
    Input:
        complex tensor, shape (B, 1, 11, 23)

    Output:
        complex tensor, shape (B, 1, 11, 23)

    Interpretation:
        output.real = gx
        output.imag = gy
    """
    def __init__(self, hidden_channels: int = 24):
        super().__init__()

        # 7-layer CNN
        self.conv1 = ComplexConv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.act1 = ComplexModReLU(hidden_channels)

        self.conv2 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act2 = ComplexModReLU(hidden_channels)

        self.conv3 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act3 = ComplexModReLU(hidden_channels)

        self.conv4 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act4 = ComplexModReLU(hidden_channels)

        self.conv5 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act5 = ComplexModReLU(hidden_channels)

        self.conv6 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act6 = ComplexModReLU(hidden_channels)

        self.conv7 = ComplexConv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.conv7(x)
        return x

class RealPhaseGradientCNN(nn.Module):
    """
    Input:
        real tensor, shape (B, 2, 11, 23)
        channels = [real(mu), imag(mu)]

    Output:
        real tensor, shape (B, 2, 11, 23)
        channels = [gx, gy]
    """
    def __init__(self, hidden_channels: int = 24):
        super().__init__()

        self.conv1 = nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act6 = nn.ReLU()

        self.conv7 = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.conv7(x)
        return x

# ============================================================
# 4. Loss
# ============================================================

class ComplexMSELoss(nn.Module):
    """
    MSE on real and imaginary parts separately.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_r = self.mse(pred.real, target.real)
        loss_i = self.mse(pred.imag, target.imag)
        return loss_r + loss_i


class RealMSELoss(nn.Module):
    """
    MSE on gx and gy channels.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_x = self.mse(pred[:, 0:1], target[:, 0:1])
        loss_y = self.mse(pred[:, 1:2], target[:, 1:2])
        return loss_x + loss_y


def gradient_smoothness_loss_real(pred):
    gx = pred[:, 0:1]
    gy = pred[:, 1:2]

    dx_gx = gx[:, :, :, 1:] - gx[:, :, :, :-1]
    dy_gx = gx[:, :, 1:, :] - gx[:, :, :-1, :]

    dx_gy = gy[:, :, :, 1:] - gy[:, :, :, :-1]
    dy_gy = gy[:, :, 1:, :] - gy[:, :, :-1, :]

    return (
        dx_gx.pow(2).mean() +
        dy_gx.pow(2).mean() +
        dx_gy.pow(2).mean() +
        dy_gy.pow(2).mean()
    )

# ============================================================
# 5. Early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None
        self.stop = False

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# 6. Train / eval
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        
        pred = model(xb)
        #loss = loss_fn(pred, yb)

        #mse_loss = loss_fn(pred, yb)
        
        # --- weighted MSE ---
        weights = torch.sqrt(yb.real**2 + yb.imag**2 + 1e-6)
        
        vec_error = ((pred.real - yb.real)**2 +
                     (pred.imag - yb.imag)**2)
        
        mse_loss = (vec_error * weights).mean()
        
        
        
        smooth_loss = gradient_smoothness_loss(pred)        
        loss = mse_loss + 0.1 * smooth_loss
        
        
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = xb.shape[0]
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        bs = xb.shape[0]
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    patience: int = 6,
    grad_clip: Optional[float] = 1.0,
    save_path: str = "best_complex_phase_gradient_cnn.pt",
) -> Dict[str, list]:
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = ComplexMSELoss()
    early_stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        early_stopper.step(val_loss, model)

        if early_stopper.best_state is not None:
            torch.save(early_stopper.best_state, save_path)

        if early_stopper.stop:
            print(f"Early stopping at epoch {epoch}: no val improvement for {patience} consecutive epochs.")
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    return history


# ============================================================
# 7. Prediction helper
# ============================================================

@torch.no_grad()
def predict_complex(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()

    X_tensor = torch.from_numpy(X)
    if X_tensor.dtype == torch.complex128:
        X_tensor = X_tensor.to(torch.complex64)

    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)

    preds = []
    for xb in loader:
        xb = xb.to(device)
        yhat = model(xb)
        preds.append(yhat.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    return preds


# ============================================================
# 8. Plotting: overlay quiver plots
# ============================================================

def plot_quiver_overlay(
    y_true_complex: np.ndarray,
    y_pred_complex: np.ndarray,
    sample_idx: int = 0,
    stride: int = 1,
    scale: Optional[float] = None,
    title_prefix: str = "Validation Sample",
):
    """
    y_true_complex, y_pred_complex: shape (M, 1, H, W), complex

    real = gx
    imag = gy
    """
    gt = y_true_complex[sample_idx, 0]
    pr = y_pred_complex[sample_idx, 0]

    gx_true = np.real(gt)
    gy_true = np.imag(gt)

    gx_pred = np.real(pr)
    gy_pred = np.imag(pr)

    H, W = gx_true.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    yy_s = yy[::stride, ::stride]
    xx_s = xx[::stride, ::stride]

    gx_true_s = gx_true[::stride, ::stride]
    gy_true_s = gy_true[::stride, ::stride]

    gx_pred_s = gx_pred[::stride, ::stride]
    gy_pred_s = gy_pred[::stride, ::stride]

    plt.figure(figsize=(10, 5))

    # Ground truth
    plt.quiver(
        xx_s, yy_s,
        gx_true_s, gy_true_s,
        angles='xy',
        scale_units='xy',
        scale=scale,
        color='blue',
        alpha=0.8,
        label='Actual'
    )

    # Prediction
    plt.quiver(
        xx_s, yy_s,
        gx_pred_s, gy_pred_s,
        angles='xy',
        scale_units='xy',
        scale=scale,
        color='red',
        alpha=0.6,
        label='Estimated'
    )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.xlim(-0.5, W - 0.5)
    plt.ylim(H - 0.5, -0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{title_prefix} {sample_idx}: Actual vs Estimated Phase Gradient")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def gradient_smoothness_loss(pred):
    gx = pred.real
    gy = pred.imag

    dx_gx = gx[:, :, :, 1:] - gx[:, :, :, :-1]
    dy_gx = gx[:, :, 1:, :] - gx[:, :, :-1, :]

    dx_gy = gy[:, :, :, 1:] - gy[:, :, :, :-1]
    dy_gy = gy[:, :, 1:, :] - gy[:, :, :-1, :]

    return (
        dx_gx.pow(2).mean() +
        dy_gx.pow(2).mean() +
        dx_gy.pow(2).mean() +
        dy_gy.pow(2).mean()
    )


def plot_multiple_validation_overlays(
    y_true_complex: np.ndarray,
    y_pred_complex: np.ndarray,
    sample_indices=(0, 1, 2, 3),
    stride: int = 1,
    scale: Optional[float] = None,
):
    n = len(sample_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        gt = y_true_complex[idx, 0]
        pr = y_pred_complex[idx, 0]

        gx_true = np.real(gt)
        gy_true = np.imag(gt)

        gx_pred = np.real(pr)
        gy_pred = np.imag(pr)

        H, W = gx_true.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        yy_s = yy[::stride, ::stride]
        xx_s = xx[::stride, ::stride]

        ax.quiver(
            xx_s, yy_s,
            gx_true[::stride, ::stride],
            gy_true[::stride, ::stride],
            angles='xy',
            scale_units='xy',
            scale=scale,
            color='blue',
            alpha=0.8,
            label='Actual'
        )

        ax.quiver(
            xx_s, yy_s,
            gx_pred[::stride, ::stride],
            gy_pred[::stride, ::stride],
            angles='xy',
            scale_units='xy',
            scale=scale,
            color='red',
            alpha=0.6,
            label='Estimated'
        )

        ax.invert_yaxis()
        ax.axis("equal")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_title(f"Val sample {idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. Main usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # ASSUMPTION:
    # Xtrain, Ytrain, Xval, Yval already exist as numpy arrays
    # with shapes:
    #   Xtrain: (N, 1, 11, 23), complex
    #   Ytrain: (N, 1, 11, 23), complex
    #   Xval  : (M, 1, 11, 23), complex
    #   Yval  : (M, 1, 11, 23), complex
    # --------------------------------------------------------

    # Example sanity checks:
    print("Xtrain shape:", Xtrain.shape, Xtrain.dtype)
    print("Ytrain shape:", Ytrain.shape, Ytrain.dtype)
    print("Xval shape  :", Xval.shape, Xval.dtype)
    print("Yval shape  :", Yval.shape, Yval.dtype)

    train_ds = ComplexGradientDataset(Xtrain, Ytrain)
    val_ds = ComplexGradientDataset(Xval, Yval)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ComplexPhaseGradientCNN(hidden_channels=24).to(device)
    
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=5e-4,
        weight_decay=1e-4,
        max_epochs=20,
        patience=6,
        grad_clip=1.0,
        save_path="best_complex_phase_gradient_cnn.pt",
    )

    # Predict on validation set
    Yval_pred = predict_complex(
        model=model,
        X=Xval,
        device=device,
        batch_size=128,
    )

    print("Validation prediction shape:", Yval_pred.shape)

    # Plot one overlay
    plot_quiver_overlay(
        y_true_complex=Yval,
        y_pred_complex=Yval_pred,
        sample_idx=100,
        stride=1,
        scale=None,
        title_prefix="Validation"
    )

    # Plot several overlays
    plot_multiple_validation_overlays(
        y_true_complex=Yval,
        y_pred_complex=Yval_pred,
        sample_indices=(100, 101, 102, 223),
        stride=1,
        scale=None,
    )

#%% REAL / COMPLEX MODELS

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. CONFIG
# ============================================================

MODEL_TYPE = "complex"   # "complex" or "real"
HIDDEN_CHANNELS = 32


# ============================================================
# 1. Dataset
# ============================================================

class ComplexGradientDataset(Dataset):
    """
    X: complex input, shape (N, 1, H, W)
    Y: complex target, shape (N, 1, H, W)

    Target interpretation:
        Y.real = gx
        Y.imag = gy
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        super().__init__()

        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(f"Expected X and Y to be 4D, got X={X.shape}, Y={Y.shape}")

        if X.shape != Y.shape:
            raise ValueError(f"X and Y must have same shape, got X={X.shape}, Y={Y.shape}")

        if X.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got X.shape[1]={X.shape[1]}")

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        if not torch.is_complex(self.X):
            raise ValueError("X must be complex-valued")
        if not torch.is_complex(self.Y):
            raise ValueError("Y must be complex-valued")

        if self.X.dtype == torch.complex128:
            self.X = self.X.to(torch.complex64)
        if self.Y.dtype == torch.complex128:
            self.Y = self.Y.to(torch.complex64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class RealGradientDataset(Dataset):
    """
    Converts complex X,Y into 2-channel real tensors.

    Input:
        X complex, shape (N, 1, H, W)
        Y complex, shape (N, 1, H, W)

    Output:
        Xr shape (N, 2, H, W): [real(mu), imag(mu)]
        Yr shape (N, 2, H, W): [gx, gy]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        super().__init__()

        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(f"Expected X and Y to be 4D, got X={X.shape}, Y={Y.shape}")

        if X.shape != Y.shape:
            raise ValueError(f"X and Y must have same shape, got X={X.shape}, Y={Y.shape}")

        if X.shape[1] != 1:
            raise ValueError(f"Expected 1 complex channel, got X.shape[1]={X.shape[1]}")

        X2 = np.concatenate([X.real, X.imag], axis=1)  # (N,2,H,W)
        Y2 = np.concatenate([Y.real, Y.imag], axis=1)  # (N,2,H,W)

        self.X = torch.from_numpy(X2).float()
        self.Y = torch.from_numpy(Y2).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ============================================================
# 2. Complex layers
# ============================================================

class ComplexConv2d(nn.Module):
    """
    Complex convolution:
        (Wr + iWi) * (xr + ixi)
        = (Wr*xr - Wi*xi) + i(Wr*xi + Wi*xr)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        bias: bool = True
    ):
        super().__init__()

        self.real_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.imag_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x.real
        xi = x.imag

        yr = self.real_conv(xr) - self.imag_conv(xi)
        yi = self.real_conv(xi) + self.imag_conv(xr)

        return torch.complex(yr, yi)


class ModReLU(nn.Module):
    """
    2D modReLU
    """
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        mag = torch.sqrt(a ** 2 + b ** 2 + self.eps)
        scale = torch.relu(mag + self.bias) / (mag + self.eps)
        return a * scale, b * scale


class ComplexModReLU(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.modrelu = ModReLU(num_features=num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.modrelu(x.real, x.imag)
        return torch.complex(a, b)


# ============================================================
# 3. Models
# ============================================================

class ComplexPhaseGradientCNN(nn.Module):
    """
    Input:
        complex tensor, shape (B, 1, 11, 23)

    Output:
        complex tensor, shape (B, 1, 11, 23)

    Interpretation:
        output.real = gx
        output.imag = gy
    """
    def __init__(self, hidden_channels: int = 24):
        super().__init__()

        self.conv1 = ComplexConv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.act1 = ComplexModReLU(hidden_channels)

        self.conv2 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act2 = ComplexModReLU(hidden_channels)

        self.conv3 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act3 = ComplexModReLU(hidden_channels)

        self.conv4 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act4 = ComplexModReLU(hidden_channels)

        self.conv5 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act5 = ComplexModReLU(hidden_channels)

        self.conv6 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act6 = ComplexModReLU(hidden_channels)

        self.conv7 = ComplexConv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.conv7(x)
        return x


class RealPhaseGradientCNN(nn.Module):
    """
    Input:
        real tensor, shape (B, 2, 11, 23)
        channels = [real(mu), imag(mu)]

    Output:
        real tensor, shape (B, 2, 11, 23)
        channels = [gx, gy]
    """
    def __init__(self, hidden_channels: int = 24):
        super().__init__()

        self.conv1 = nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.act6 = nn.ReLU()

        self.conv7 = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.conv7(x)
        return x


# ============================================================
# 4. Loss
# ============================================================

class ComplexMSELoss(nn.Module):
    """
    MSE on real and imaginary parts separately.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_r = self.mse(pred.real, target.real)
        loss_i = self.mse(pred.imag, target.imag)
        return loss_r + loss_i


class RealMSELoss(nn.Module):
    """
    MSE on gx and gy channels separately.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_x = self.mse(pred[:, 0:1], target[:, 0:1])
        loss_y = self.mse(pred[:, 1:2], target[:, 1:2])
        return loss_x + loss_y


# ============================================================
# 5. Early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None
        self.stop = False

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# 6. Smoothness
# ============================================================

def gradient_smoothness_loss(pred):
    gx = pred.real
    gy = pred.imag

    dx_gx = gx[:, :, :, 1:] - gx[:, :, :, :-1]
    dy_gx = gx[:, :, 1:, :] - gx[:, :, :-1, :]

    dx_gy = gy[:, :, :, 1:] - gy[:, :, :, :-1]
    dy_gy = gy[:, :, 1:, :] - gy[:, :, :-1, :]

    return (
        dx_gx.pow(2).mean() +
        dy_gx.pow(2).mean() +
        dx_gy.pow(2).mean() +
        dy_gy.pow(2).mean()
    )


def gradient_smoothness_loss_real(pred):
    gx = pred[:, 0:1]
    gy = pred[:, 1:2]

    dx_gx = gx[:, :, :, 1:] - gx[:, :, :, :-1]
    dy_gx = gx[:, :, 1:, :] - gx[:, :, :-1, :]

    dx_gy = gy[:, :, :, 1:] - gy[:, :, :, :-1]
    dy_gy = gy[:, :, 1:, :] - gy[:, :, :-1, :]

    return (
        dx_gx.pow(2).mean() +
        dy_gx.pow(2).mean() +
        dx_gy.pow(2).mean() +
        dy_gy.pow(2).mean()
    )


# ============================================================
# 7. Train / eval
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)

        if torch.is_complex(yb):
            # --- weighted MSE ---
            weights = torch.sqrt(yb.real**2 + yb.imag**2 + 1e-6)

            vec_error = ((pred.real - yb.real)**2 +
                         (pred.imag - yb.imag)**2)

            mse_loss = (vec_error * weights).mean()

            smooth_loss = gradient_smoothness_loss(pred)

        else:
            # real model / real tensors with 2 channels
            weights = torch.sqrt(yb[:, 0:1]**2 + yb[:, 1:2]**2 + 1e-6)

            vec_error = ((pred[:, 0:1] - yb[:, 0:1])**2 +
                         (pred[:, 1:2] - yb[:, 1:2])**2)

            mse_loss = (vec_error * weights).mean()

            smooth_loss = gradient_smoothness_loss_real(pred)

        loss = mse_loss + 0.05 * smooth_loss

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = xb.shape[0]
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        bs = xb.shape[0]
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    patience: int = 6,
    grad_clip: Optional[float] = 1.0,
    save_path: str = "best_phase_gradient_cnn.pt",
) -> Dict[str, list]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sample_x, sample_y = train_loader.dataset[0]
    if torch.is_complex(sample_y):
        loss_fn = ComplexMSELoss()
    else:
        loss_fn = RealMSELoss()

    early_stopper = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        early_stopper.step(val_loss, model)

        if early_stopper.best_state is not None:
            torch.save(early_stopper.best_state, save_path)

        if early_stopper.stop:
            print(f"Early stopping at epoch {epoch}: no val improvement for {patience} consecutive epochs.")
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    return history


# ============================================================
# 8. Prediction helper
# ============================================================

@torch.no_grad()
def predict_complex(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()

    if MODEL_TYPE == "complex":
        X_tensor = torch.from_numpy(X)
        if X_tensor.dtype == torch.complex128:
            X_tensor = X_tensor.to(torch.complex64)

    elif MODEL_TYPE == "real":
        X2 = np.concatenate([X.real, X.imag], axis=1)
        X_tensor = torch.from_numpy(X2).float()

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)

    preds = []
    for xb in loader:
        xb = xb.to(device)
        yhat = model(xb)
        preds.append(yhat.cpu())

    preds = torch.cat(preds, dim=0)

    if MODEL_TYPE == "complex":
        return preds.numpy()

    elif MODEL_TYPE == "real":
        preds_np = preds.numpy()  # (N,2,H,W)
        return preds_np[:, 0:1] + 1j * preds_np[:, 1:2]


# ============================================================
# 9. Plotting: overlay quiver plots
# ============================================================

def plot_quiver_overlay(
    y_true_complex: np.ndarray,
    y_pred_complex: np.ndarray,
    sample_idx: int = 0,
    stride: int = 1,
    scale: Optional[float] = None,
    title_prefix: str = "Validation Sample",
):
    gt = y_true_complex[sample_idx, 0]
    pr = y_pred_complex[sample_idx, 0]

    gx_true = np.real(gt)
    gy_true = np.imag(gt)

    gx_pred = np.real(pr)
    gy_pred = np.imag(pr)

    H, W = gx_true.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    yy_s = yy[::stride, ::stride]
    xx_s = xx[::stride, ::stride]

    gx_true_s = gx_true[::stride, ::stride]
    gy_true_s = gy_true[::stride, ::stride]

    gx_pred_s = gx_pred[::stride, ::stride]
    gy_pred_s = gy_pred[::stride, ::stride]

    plt.figure(figsize=(10, 5))

    plt.quiver(
        xx_s, yy_s,
        gx_true_s, gy_true_s,
        angles='xy',
        scale_units='xy',
        scale=scale,
        color='blue',
        alpha=0.8,
        label='Actual'
    )

    plt.quiver(
        xx_s, yy_s,
        gx_pred_s, gy_pred_s,
        angles='xy',
        scale_units='xy',
        scale=scale,
        color='red',
        alpha=0.6,
        label='Estimated'
    )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.xlim(-0.5, W - 0.5)
    plt.ylim(H - 0.5, -0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{title_prefix} {sample_idx}: Actual vs Estimated Phase Gradient")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_validation_overlays(
    y_true_complex: np.ndarray,
    y_pred_complex: np.ndarray,
    sample_indices=(0, 1, 2, 3),
    stride: int = 1,
    scale: Optional[float] = None,
):
    n = len(sample_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        gt = y_true_complex[idx, 0]
        pr = y_pred_complex[idx, 0]

        gx_true = np.real(gt)
        gy_true = np.imag(gt)

        gx_pred = np.real(pr)
        gy_pred = np.imag(pr)

        H, W = gx_true.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        yy_s = yy[::stride, ::stride]
        xx_s = xx[::stride, ::stride]

        ax.quiver(
            xx_s, yy_s,
            gx_true[::stride, ::stride],
            gy_true[::stride, ::stride],
            angles='xy',
            scale_units='xy',
            scale=scale,
            color='blue',
            alpha=0.8,
            label='Actual'
        )

        ax.quiver(
            xx_s, yy_s,
            gx_pred[::stride, ::stride],
            gy_pred[::stride, ::stride],
            angles='xy',
            scale_units='xy',
            scale=scale,
            color='red',
            alpha=0.6,
            label='Estimated'
        )

        ax.invert_yaxis()
        ax.axis("equal")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_title(f"Val sample {idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. Main usage
# ============================================================

if __name__ == "__main__":
    print("Xtrain shape:", Xtrain.shape, Xtrain.dtype)
    print("Ytrain shape:", Ytrain.shape, Ytrain.dtype)
    print("Xval shape  :", Xval.shape, Xval.dtype)
    print("Yval shape  :", Yval.shape, Yval.dtype)

    if MODEL_TYPE == "complex":
        train_ds = ComplexGradientDataset(Xtrain, Ytrain)
        val_ds = ComplexGradientDataset(Xval, Yval)
        model = ComplexPhaseGradientCNN(hidden_channels=HIDDEN_CHANNELS)

    elif MODEL_TYPE == "real":
        train_ds = RealGradientDataset(Xtrain, Ytrain)
        val_ds = RealGradientDataset(Xval, Yval)
        model = RealPhaseGradientCNN(hidden_channels=HIDDEN_CHANNELS)

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = model.to(device)

    print(f"Model type: {MODEL_TYPE}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=5e-4,
        weight_decay=1e-4,
        max_epochs=20,
        patience=6,
        grad_clip=1.0,
        save_path=f"best_{MODEL_TYPE}_phase_gradient_cnn.pt",
    )

    Yval_pred = predict_complex(
        model=model,
        X=Xval,
        device=device,
        batch_size=128,
    )

    print("Validation prediction shape:", Yval_pred.shape)

    plot_quiver_overlay(
        y_true_complex=Yval,
        y_pred_complex=Yval_pred,
        sample_idx=100,
        stride=1,
        scale=None,
        title_prefix=f"Validation ({MODEL_TYPE})"
    )

    plot_multiple_validation_overlays(
        y_true_complex=Yval,
        y_pred_complex=Yval_pred,
        sample_indices=(100, 101, 102, 223),
        stride=1,
        scale=None,
    )

#%% perform classification based on mean and std of temporal consistency


