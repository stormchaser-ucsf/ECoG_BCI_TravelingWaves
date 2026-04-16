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

(
Xtrain,Xval,Xtest,
Ytrain,Yval,Ytest,
idx_train,idx_val,idx_test
) = training_test_split_Mu_PhaseGradient_equal(X,Y,0.7)
   
Xtrain = np.concatenate((Xtrain,Xtest))       
Ytrain = np.concatenate((Ytrain,Ytest))       
idx_train = np.concat((idx_train,idx_test))
    
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

#%% REAL / COMPLEX MODELS (MAIN)

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
HIDDEN_CHANNELS = 36


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
        lr=3e-4,
        weight_decay=1e-4,
        max_epochs=20,
        patience=6,
        grad_clip=0.75,
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

#%% (MAIN) perform classification based on mean and std of temporal consistency

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# ASSUMPTIONS
# ============================================================
# You already have:
#   X           -> list of trials
#                  each trial is either:
#                     np.ndarray of shape (T, 11, 23), complex
#                  or [np.ndarray of shape (T, 11, 23), complex]
#
#   idx_train   -> indices of training trials
#   idx_val     -> indices of validation trials
#   model       -> trained model
#   device      -> torch device
#   MODEL_TYPE  -> "complex" or "real"
#
# Trial labels:
#   odd trials = wave
#   even trials = non-wave
#
# If Python index 0 corresponds to "trial 1", then:
#   idx 0,2,4,... are wave trials
# and ONE_BASED_LABELS should be True.
# ============================================================

ONE_BASED_LABELS = True


# ============================================================
# 1. Helpers
# ============================================================

def unwrap_trial(trial):
    """
    Accepts either:
      - np.ndarray of shape (T, 11, 23)
      - [np.ndarray] or (np.ndarray,)
    Returns:
      np.ndarray of shape (T, 11, 23), complex
    """
    if isinstance(trial, np.ndarray):
        arr = trial
    elif isinstance(trial, (list, tuple)) and len(trial) == 1:
        arr = trial[0]
    else:
        raise ValueError(f"Unexpected trial format: type={type(trial)}")

    if not isinstance(arr, np.ndarray):
        raise ValueError("Unwrapped trial is not a numpy array")

    if arr.ndim != 3:
        raise ValueError(f"Expected trial shape (T,11,23), got {arr.shape}")

    if not np.iscomplexobj(arr):
        raise ValueError("Trial must be complex-valued")

    return arr


def is_wave_trial(trial_idx, one_based_labels=True):
    """
    If one_based_labels=True:
      Python idx 0 -> trial 1 -> odd -> wave
      So 0,2,4,... are wave.
    """
    if one_based_labels:
        return (trial_idx % 2 == 0)
    else:
        return (trial_idx % 2 == 1)


# ============================================================
# 2. Predict one trial
# ============================================================

@torch.no_grad()
def predict_trial_complex(model, trial_complex, device, model_type="complex", batch_size=256):
    """
    trial_complex: np.ndarray, shape (T, 11, 23), complex

    Returns:
      pred_complex: np.ndarray, shape (T, 11, 23), complex
    """
    model.eval()

    T, H, W = trial_complex.shape

    if model_type == "complex":
        x = trial_complex[:, None, :, :]   # (T,1,H,W)
        x_t = torch.from_numpy(x)
        if x_t.dtype == torch.complex128:
            x_t = x_t.to(torch.complex64)

    elif model_type == "real":
        xr = trial_complex.real[:, None, :, :]
        xi = trial_complex.imag[:, None, :, :]
        x = np.concatenate([xr, xi], axis=1)   # (T,2,H,W)
        x_t = torch.from_numpy(x).float()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    preds = []
    for start in range(0, T, batch_size):
        xb = x_t[start:start + batch_size].to(device)
        yb = model(xb).cpu()
        preds.append(yb)

    preds = torch.cat(preds, dim=0).numpy()

    if model_type == "complex":
        pred_complex = preds[:, 0, :, :]   # (T,H,W), complex
    elif model_type == "real":
        pred_complex = preds[:, 0, :, :] + 1j * preds[:, 1, :, :]

    return pred_complex


# ============================================================
# 3. Stability metric
# ============================================================

def compute_trial_stability_timeseries(pred_complex_trial):
    """
    pred_complex_trial: shape (T, 11, 23), complex

    Stability per time step:
      mean over space of |G[t+1] - G[t]|

    Returns:
      stability_t: shape (T-1,)
    """
    dG = pred_complex_trial[1:] - pred_complex_trial[:-1]   # (T-1,H,W), complex
    stability_t = np.abs(dG).mean(axis=(1, 2))              # (T-1,)
    return stability_t


def compute_trial_stability_features(pred_complex_trial):
    """
    Returns:
      mean_stability, std_stability, stability_t
    """
    stability_t = compute_trial_stability_timeseries(pred_complex_trial)
    mean_stability = stability_t.mean()
    std_stability = stability_t.std()
    return mean_stability, std_stability, stability_t


# ============================================================
# 4. Extract features for a set of trials
# ============================================================

def extract_stability_features_for_indices(
    X,
    indices,
    model,
    device,
    model_type="complex",
    one_based_labels=True,
    batch_size=256,
):
    """
    Returns dict with:
      trial_idx
      is_wave
      mean_stability
      std_stability
      stability_t
      pred_complex
    """
    results = {
        "trial_idx": [],
        "is_wave": [],
        "mean_stability": [],
        "std_stability": [],
        "stability_t": [],
        "pred_complex": [],
    }

    for trial_idx in indices:
        trial_complex = unwrap_trial(X[trial_idx])  # (T,11,23), complex

        pred_complex = predict_trial_complex(
            model=model,
            trial_complex=trial_complex,
            device=device,
            model_type=model_type,
            batch_size=batch_size,
        )

        mean_stab, std_stab, stability_t = compute_trial_stability_features(pred_complex)

        results["trial_idx"].append(trial_idx)
        results["is_wave"].append(is_wave_trial(trial_idx, one_based_labels=one_based_labels))
        results["mean_stability"].append(mean_stab)
        results["std_stability"].append(std_stab)
        results["stability_t"].append(stability_t)
        results["pred_complex"].append(pred_complex)

    results["trial_idx"] = np.array(results["trial_idx"])
    results["is_wave"] = np.array(results["is_wave"], dtype=bool)
    results["mean_stability"] = np.array(results["mean_stability"])
    results["std_stability"] = np.array(results["std_stability"])

    return results


# ============================================================
# 5. Plotting
# ============================================================

def plot_stability_feature_space(train_results, val_results):
    plt.figure(figsize=(8, 6))

    train_wave = train_results["is_wave"]
    train_nonwave = ~train_wave

    val_wave = val_results["is_wave"]
    val_nonwave = ~val_wave

    plt.scatter(
        train_results["mean_stability"][train_wave],
        train_results["std_stability"][train_wave],
        s=50,
        alpha=0.8,
        label="Train Wave"
    )

    plt.scatter(
        train_results["mean_stability"][train_nonwave],
        train_results["std_stability"][train_nonwave],
        s=50,
        alpha=0.8,
        label="Train Non-Wave"
    )

    plt.scatter(
        val_results["mean_stability"][val_wave],
        val_results["std_stability"][val_wave],
        s=90,
        marker='x',
        alpha=0.9,
        label="Val Wave"
    )

    plt.scatter(
        val_results["mean_stability"][val_nonwave],
        val_results["std_stability"][val_nonwave],
        s=90,
        marker='x',
        alpha=0.9,
        label="Val Non-Wave"
    )

    plt.xlabel("Mean Stability")
    plt.ylabel("Std Stability")
    plt.title("Trial-wise Stability Feature Space")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_distributions(train_results):
    wave = train_results["is_wave"]
    nonwave = ~wave

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(train_results["mean_stability"][wave], bins=20, alpha=0.7, label="Wave")
    axes[0].hist(train_results["mean_stability"][nonwave], bins=20, alpha=0.7, label="Non-Wave")
    axes[0].set_title("Mean Stability")
    axes[0].legend()

    axes[1].hist(train_results["std_stability"][wave], bins=20, alpha=0.7, label="Wave")
    axes[1].hist(train_results["std_stability"][nonwave], bins=20, alpha=0.7, label="Non-Wave")
    axes[1].set_title("Std Stability")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Simple centroid classifier in 2D feature space
# ============================================================

def compute_class_centroids(train_results):
    wave = train_results["is_wave"]
    nonwave = ~wave

    wave_centroid = np.array([
        train_results["mean_stability"][wave].mean(),
        train_results["std_stability"][wave].mean(),
    ])

    nonwave_centroid = np.array([
        train_results["mean_stability"][nonwave].mean(),
        train_results["std_stability"][nonwave].mean(),
    ])

    return wave_centroid, nonwave_centroid


def assign_by_centroid(results, wave_centroid, nonwave_centroid):
    pts = np.column_stack([results["mean_stability"], results["std_stability"]])

    d_wave = np.linalg.norm(pts - wave_centroid[None, :], axis=1)
    d_nonwave = np.linalg.norm(pts - nonwave_centroid[None, :], axis=1)

    pred_is_wave = d_wave < d_nonwave
    return pred_is_wave, d_wave, d_nonwave


# ============================================================
# 7. Run everything
# ============================================================

print("Extracting train stability features...")
train_results = extract_stability_features_for_indices(
    X=X,
    indices=idx_train,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
)

print("Extracting validation stability features...")
val_results = extract_stability_features_for_indices(
    X=X,
    indices=idx_val,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
)

print("Done.")
print("Train feature shape:", train_results["mean_stability"].shape)
print("Val feature shape  :", val_results["mean_stability"].shape)

# 2D plot
plot_stability_feature_space(train_results, val_results)

# Optional train histograms
plot_train_distributions(train_results)

# Optional centroid-based classification
wave_centroid, nonwave_centroid = compute_class_centroids(train_results)
pred_val_is_wave, d_wave, d_nonwave = assign_by_centroid(
    val_results,
    wave_centroid,
    nonwave_centroid
)

true_val_is_wave = val_results["is_wave"]
acc = (pred_val_is_wave == true_val_is_wave).mean()

print("Wave centroid     :", wave_centroid)
print("Non-wave centroid :", nonwave_centroid)
print(f"Validation centroid-based accuracy: {acc:.3f}")


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 8. SVM classifier on 2D stability features
# ============================================================

def make_feature_matrix(results):
    """
    Returns:
      X_feat: shape (n_trials, 2)
      y:      shape (n_trials,)
    """
    X_feat = np.column_stack([
        results["mean_stability"],
        results["std_stability"]
    ])
    y = results["is_wave"].astype(int)   # wave=1, non-wave=0
    return X_feat, y


# Build train/val feature matrices
X_train_feat, y_train = make_feature_matrix(train_results)
X_val_feat, y_val = make_feature_matrix(val_results)

# SVM pipeline: standardize -> RBF SVM
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# Fit
svm_clf.fit(X_train_feat, y_train)

# Predict
y_train_pred = svm_clf.predict(X_train_feat)
y_val_pred = svm_clf.predict(X_val_feat)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"SVM train accuracy: {train_acc:.3f}")
print(f"SVM val accuracy  : {val_acc:.3f}")

print("\nValidation confusion matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nValidation classification report:")
print(classification_report(
    y_val,
    y_val_pred,
    target_names=["Non-Wave", "Wave"]
))

#%% all in one with actual data also 


import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# ASSUMPTIONS
# ============================================================
# Already available in memory:
#   X           : list-like of trials, each trial complex array
#   Y           : list-like of true phase-gradient trials, each trial complex array
#   idx_train   : training trial indices
#   idx_val     : validation trial indices
#   model       : trained model
#   device      : torch.device
#   MODEL_TYPE  : "complex" or "real"
#
# Trial labels:
#   odd trials = wave
#   even trials = non-wave
#
# If Python index 0 corresponds to human trial 1, set:
#   ONE_BASED_LABELS = True
# Then Python indices 0,2,4,... are wave trials.
# ============================================================

ONE_BASED_LABELS = True


# ============================================================
# 1. Helpers
# ============================================================

def is_wave_trial(trial_idx, one_based_labels=True):
    if one_based_labels:
        return (trial_idx % 2 == 0)
    return (trial_idx % 2 == 1)


def unwrap_trial_item(trial):
    """
    Handles:
      - np.ndarray
      - [np.ndarray]
      - (np.ndarray,)
    """
    if isinstance(trial, np.ndarray):
        return trial
    if isinstance(trial, (list, tuple)) and len(trial) == 1:
        return np.asarray(trial[0])
    return np.asarray(trial)


def ensure_trial_TCHW(arr, name="array"):
    """
    Convert trial array to shape (T, 1, H, W).

    Accepts:
      (T, H, W)
      (T, 1, H, W)
      (1, T, H, W)

    Returns:
      arr with shape (T, 1, H, W)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        # (T,H,W) -> (T,1,H,W)
        return arr[:, None, :, :]

    if arr.ndim != 4:
        raise ValueError(f"{name}: expected 3D or 4D array, got shape {arr.shape}")

    if arr.shape[1] == 1:
        # already (T,1,H,W)
        return arr

    if arr.shape[0] == 1:
        # (1,T,H,W) -> (T,1,H,W)
        return np.transpose(arr, (1, 0, 2, 3))

    raise ValueError(
        f"{name}: could not interpret shape {arr.shape} as "
        f"(T,H,W), (T,1,H,W), or (1,T,H,W)"
    )


def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


# ============================================================
# 2. Model prediction for one trial
# ============================================================

@torch.no_grad()
def predict_from_array(model, X_complex, device, model_type="complex", batch_size=256):
    """
    X_complex: np.ndarray, shape (T,1,H,W), complex

    Returns:
      Y_pred_complex: np.ndarray, shape (T,1,H,W), complex
    """
    model.eval()

    if model_type == "complex":
        x_t = torch.from_numpy(X_complex)
        if x_t.dtype == torch.complex128:
            x_t = x_t.to(torch.complex64)

    elif model_type == "real":
        xr = X_complex.real
        xi = X_complex.imag
        x2 = np.concatenate([xr, xi], axis=1)   # (T,2,H,W)
        x_t = torch.from_numpy(x2).float()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    preds = []
    for start in range(0, x_t.shape[0], batch_size):
        xb = x_t[start:start + batch_size].to(device)
        yb = model(xb).cpu()
        preds.append(yb)

    preds = torch.cat(preds, dim=0).numpy()

    if model_type == "complex":
        return preds

    elif model_type == "real":
        return preds[:, 0:1] + 1j * preds[:, 1:2]


# ============================================================
# 3. Stability metric
# ============================================================

def stability_timeseries_from_complex_gradients(G_complex):
    """
    G_complex: np.ndarray, shape (T,1,H,W), complex

    Returns:
      stability_t: np.ndarray, shape (T-1,)
      where stability_t[t] = mean_{space} |G[t+1] - G[t]|
    """
    dG = G_complex[1:] - G_complex[:-1]              # (T-1,1,H,W), complex
    stability_t = np.abs(dG).mean(axis=(1, 2, 3))   # average over channel + space
    return stability_t


def trial_level_features_from_stability(stability_t):
    return stability_t.mean(), stability_t.std()


# ============================================================
# 4. Extract true vs predicted stability for trial indices
# ============================================================

def extract_true_vs_estimated_stability_for_indices(
    X_list,
    Y_list,
    indices,
    model,
    device,
    model_type="complex",
    one_based_labels=True,
    batch_size=256,
):
    """
    Returns dict with:
      trial_idx
      is_wave
      true_stability_t
      pred_stability_t
      mean_true
      std_true
      mean_pred
      std_pred
      corr_stability_t
      mse_stability_t
    """
    results = {
        "trial_idx": [],
        "is_wave": [],
        "true_stability_t": [],
        "pred_stability_t": [],
        "mean_true": [],
        "std_true": [],
        "mean_pred": [],
        "std_pred": [],
        "corr_stability_t": [],
        "mse_stability_t": [],
    }

    for idx in indices:
        X_trial = unwrap_trial_item(X_list[idx])
        Y_trial = unwrap_trial_item(Y_list[idx])

        X_trial = ensure_trial_TCHW(X_trial, name=f"X[{idx}]")
        Y_trial = ensure_trial_TCHW(Y_trial, name=f"Y[{idx}]")

        if not np.iscomplexobj(X_trial):
            raise ValueError(f"X[{idx}] must be complex-valued, got dtype {X_trial.dtype}")
        if not np.iscomplexobj(Y_trial):
            raise ValueError(f"Y[{idx}] must be complex-valued, got dtype {Y_trial.dtype}")

        Y_pred = predict_from_array(
            model=model,
            X_complex=X_trial,
            device=device,
            model_type=model_type,
            batch_size=batch_size,
        )

        true_stability_t = stability_timeseries_from_complex_gradients(Y_trial)
        pred_stability_t = stability_timeseries_from_complex_gradients(Y_pred)

        mean_true, std_true = trial_level_features_from_stability(true_stability_t)
        mean_pred, std_pred = trial_level_features_from_stability(pred_stability_t)

        corr_t = safe_corr(true_stability_t, pred_stability_t)
        mse_t = np.mean((true_stability_t - pred_stability_t) ** 2)

        results["trial_idx"].append(idx)
        results["is_wave"].append(is_wave_trial(idx, one_based_labels=one_based_labels))
        results["true_stability_t"].append(true_stability_t)
        results["pred_stability_t"].append(pred_stability_t)
        results["mean_true"].append(mean_true)
        results["std_true"].append(std_true)
        results["mean_pred"].append(mean_pred)
        results["std_pred"].append(std_pred)
        results["corr_stability_t"].append(corr_t)
        results["mse_stability_t"].append(mse_t)

    for k in [
        "trial_idx", "is_wave", "mean_true", "std_true",
        "mean_pred", "std_pred", "corr_stability_t", "mse_stability_t"
    ]:
        results[k] = np.array(results[k])

    results["is_wave"] = results["is_wave"].astype(bool)
    return results


# ============================================================
# 5. Summary metrics
# ============================================================

def summarize_true_vs_pred_results(name, results):
    mean_corr_time = np.nanmean(results["corr_stability_t"])
    mean_mse_time = np.mean(results["mse_stability_t"])

    corr_mean_feature = safe_corr(results["mean_true"], results["mean_pred"])
    corr_std_feature = safe_corr(results["std_true"], results["std_pred"])

    mse_mean_feature = np.mean((results["mean_true"] - results["mean_pred"]) ** 2)
    mse_std_feature = np.mean((results["std_true"] - results["std_pred"]) ** 2)

    print(f"\n{name} summary")
    print("-" * 60)
    print(f"Mean per-trial corr(true_stability_t, pred_stability_t): {mean_corr_time:.4f}")
    print(f"Mean per-trial MSE(true_stability_t, pred_stability_t):  {mean_mse_time:.6e}")
    print(f"Corr(mean_true, mean_pred): {corr_mean_feature:.4f}")
    print(f"Corr(std_true, std_pred):   {corr_std_feature:.4f}")
    print(f"MSE(mean_true, mean_pred):  {mse_mean_feature:.6e}")
    print(f"MSE(std_true, std_pred):    {mse_std_feature:.6e}")


# ============================================================
# 6. SVM on 2D features
# ============================================================

def make_feature_matrix(results, use_pred=True):
    if use_pred:
        X_feat = np.column_stack([results["mean_pred"], results["std_pred"]])
    else:
        X_feat = np.column_stack([results["mean_true"], results["std_true"]])

    y = results["is_wave"].astype(int)
    return X_feat, y


def fit_and_evaluate_svm(train_X, train_y, val_X, val_y, name="Predicted"):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    clf.fit(train_X, train_y)

    y_train_pred = clf.predict(train_X)
    y_val_pred = clf.predict(val_X)

    train_acc = accuracy_score(train_y, y_train_pred)
    val_acc = accuracy_score(val_y, y_val_pred)

    print(f"\n{name} feature SVM")
    print("-" * 60)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print("\nValidation confusion matrix:")
    print(confusion_matrix(val_y, y_val_pred))
    print("\nValidation classification report:")
    print(classification_report(val_y, y_val_pred, target_names=["Non-Wave", "Wave"]))

    return clf, train_acc, val_acc, y_val_pred


# ============================================================
# 7. Plotting
# ============================================================

def plot_true_vs_pred_feature_scatter(train_results, val_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tr_wave = train_results["is_wave"]
    va_wave = val_results["is_wave"]

    ax = axes[0]
    ax.scatter(train_results["mean_true"][tr_wave], train_results["std_true"][tr_wave],
               s=45, alpha=0.8, label="Train Wave")
    ax.scatter(train_results["mean_true"][~tr_wave], train_results["std_true"][~tr_wave],
               s=45, alpha=0.8, label="Train Non-Wave")
    ax.scatter(val_results["mean_true"][va_wave], val_results["std_true"][va_wave],
               s=80, marker='x', alpha=0.9, label="Val Wave")
    ax.scatter(val_results["mean_true"][~va_wave], val_results["std_true"][~va_wave],
               s=80, marker='x', alpha=0.9, label="Val Non-Wave")
    ax.set_title("TRUE stability features")
    ax.set_xlabel("Mean Stability")
    ax.set_ylabel("Std Stability")
    ax.legend()

    ax = axes[1]
    ax.scatter(train_results["mean_pred"][tr_wave], train_results["std_pred"][tr_wave],
               s=45, alpha=0.8, label="Train Wave")
    ax.scatter(train_results["mean_pred"][~tr_wave], train_results["std_pred"][~tr_wave],
               s=45, alpha=0.8, label="Train Non-Wave")
    ax.scatter(val_results["mean_pred"][va_wave], val_results["std_pred"][va_wave],
               s=80, marker='x', alpha=0.9, label="Val Wave")
    ax.scatter(val_results["mean_pred"][~va_wave], val_results["std_pred"][~va_wave],
               s=80, marker='x', alpha=0.9, label="Val Non-Wave")
    ax.set_title("PREDICTED stability features")
    ax.set_xlabel("Mean Stability")
    ax.set_ylabel("Std Stability")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_trial_stability_overlay(results, which="val", n_trials=5):
    n_trials = min(n_trials, len(results["trial_idx"]))
    fig, axes = plt.subplots(n_trials, 1, figsize=(10, 2.5 * n_trials), sharex=False)

    if n_trials == 1:
        axes = [axes]

    for ax, i in zip(axes, range(n_trials)):
        t_true = results["true_stability_t"][i]
        t_pred = results["pred_stability_t"][i]
        trial_idx = results["trial_idx"][i]
        label = "Wave" if results["is_wave"][i] else "Non-Wave"
        corr_t = results["corr_stability_t"][i]

        ax.plot(t_true, label="True")
        ax.plot(t_pred, label="Pred")
        ax.set_title(f"{which} trial idx={trial_idx} | {label} | corr={corr_t:.3f}")
        ax.set_ylabel("Stability")
        ax.legend()

    axes[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. RUN
# ============================================================

print("Extracting train true-vs-predicted stability...")
train_cmp = extract_true_vs_estimated_stability_for_indices(
    X_list=X,
    Y_list=Y,
    indices=idx_train,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
)

print("Extracting val true-vs-predicted stability...")
val_cmp = extract_true_vs_estimated_stability_for_indices(
    X_list=X,
    Y_list=Y,
    indices=idx_val,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
)

summarize_true_vs_pred_results("TRAIN", train_cmp)
summarize_true_vs_pred_results("VAL", val_cmp)

plot_true_vs_pred_feature_scatter(train_cmp, val_cmp)
plot_trial_stability_overlay(val_cmp, which="val", n_trials=5)

# SVM using predicted features
X_train_pred, y_train = make_feature_matrix(train_cmp, use_pred=True)
X_val_pred, y_val = make_feature_matrix(val_cmp, use_pred=True)

svm_pred, train_acc_pred, val_acc_pred, y_val_pred = fit_and_evaluate_svm(
    X_train_pred, y_train, X_val_pred, y_val, name="Predicted"
)

# SVM using true features
X_train_true, y_train_true = make_feature_matrix(train_cmp, use_pred=False)
X_val_true, y_val_true = make_feature_matrix(val_cmp, use_pred=False)

svm_true, train_acc_true, val_acc_true, y_val_true_pred = fit_and_evaluate_svm(
    X_train_true, y_train_true, X_val_true, y_val_true, name="True"
)

print("\nAccuracy comparison")
print("-" * 60)
print(f"Predicted-feature SVM val acc: {val_acc_pred:.4f}")
print(f"True-feature SVM val acc:      {val_acc_true:.4f}")

#%% MODIFIED CODE WITH 8 BIN WINDOWS FOR CLASSIFICATION

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# ASSUMPTIONS
# ============================================================
# You already have:
#   X           -> list of trials
#                  each trial may be:
#                     np.ndarray of shape (T, 11, 23), complex
#                  or np.ndarray of shape (1, T, 11, 23), complex
#                  or np.ndarray of shape (T, 1, 11, 23), complex
#                  or [np.ndarray of one of the above]
#
#   idx_train   -> indices of training trials
#   idx_val     -> indices of validation trials
#   model       -> trained model
#   device      -> torch device
#   MODEL_TYPE  -> "complex" or "real"
#
# Trial labels:
#   odd trials = wave
#   even trials = non-wave
#
# If Python index 0 corresponds to "trial 1", then:
#   idx 0,2,4,... are wave trials
# and ONE_BASED_LABELS should be True.
# ============================================================

ONE_BASED_LABELS = True

# chunk settings
CHUNK_LEN = 8
CHUNK_OVERLAP = 2
CHUNK_STEP = CHUNK_LEN - CHUNK_OVERLAP


# ============================================================
# 1. Helpers
# ============================================================

def unwrap_trial(trial):
    """
    Accepts:
      - np.ndarray
      - [np.ndarray] or (np.ndarray,)
    Returns:
      np.ndarray
    """
    if isinstance(trial, np.ndarray):
        arr = trial
    elif isinstance(trial, (list, tuple)) and len(trial) == 1:
        arr = trial[0]
    else:
        arr = np.asarray(trial)

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if not np.iscomplexobj(arr):
        raise ValueError("Trial must be complex-valued")

    return arr


def ensure_trial_THW(arr):
    """
    Convert a trial to shape (T, H, W), complex.

    Accepts:
      (T, H, W)
      (1, T, H, W)
      (T, 1, H, W)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            # (1, T, H, W) -> (T, H, W)
            return arr[0]
        if arr.shape[1] == 1:
            # (T, 1, H, W) -> (T, H, W)
            return arr[:, 0]

    raise ValueError(f"Expected trial shape (T,H,W), (1,T,H,W), or (T,1,H,W), got {arr.shape}")


def is_wave_trial(trial_idx, one_based_labels=True):
    """
    If one_based_labels=True:
      Python idx 0 -> trial 1 -> odd -> wave
      So 0,2,4,... are wave.
    """
    if one_based_labels:
        return (trial_idx % 2 == 0)
    else:
        return (trial_idx % 2 == 1)


def maybe_cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# 2. Chunking
# ============================================================

def make_time_chunks(trial_complex, chunk_len=8, chunk_overlap=2):
    """
    trial_complex: np.ndarray, shape (T, H, W), complex

    Returns:
      chunks: list of np.ndarray, each shape (t_chunk, H, W)
      starts: list of start indices
    """
    T = trial_complex.shape[0]

    if T <= chunk_len:
        return [trial_complex], [0]

    step = chunk_len - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_len must be > chunk_overlap")

    chunks = []
    starts = []

    start = 0
    while start + chunk_len <= T:
        chunks.append(trial_complex[start:start + chunk_len])
        starts.append(start)
        start += step

    return chunks, starts


# ============================================================
# 3. Predict one chunk/trial
# ============================================================

@torch.no_grad()
def predict_trial_complex(model, trial_complex, device, model_type="complex", batch_size=256):
    """
    trial_complex: np.ndarray, shape (T, H, W), complex

    Returns:
      pred_complex: np.ndarray, shape (T, H, W), complex
    """
    model.eval()

    T, H, W = trial_complex.shape

    if model_type == "complex":
        x = trial_complex[:, None, :, :]   # (T,1,H,W)
        x_t = torch.from_numpy(x)
        if x_t.dtype == torch.complex128:
            x_t = x_t.to(torch.complex64)

    elif model_type == "real":
        xr = trial_complex.real[:, None, :, :]
        xi = trial_complex.imag[:, None, :, :]
        x = np.concatenate([xr, xi], axis=1)   # (T,2,H,W)
        x_t = torch.from_numpy(x).float()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    preds = []
    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    for start in range(0, T, batch_size):
        xb = x_t[start:start + batch_size].to(device, non_blocking=True)
        yb = model(xb).cpu()
        preds.append(yb)

    maybe_cuda_sync(device)
    infer_time_sec = time.perf_counter() - t0

    preds = torch.cat(preds, dim=0).numpy()

    if model_type == "complex":
        pred_complex = preds[:, 0, :, :]   # (T,H,W), complex
    elif model_type == "real":
        pred_complex = preds[:, 0, :, :] + 1j * preds[:, 1, :, :]

    return pred_complex, infer_time_sec


# ============================================================
# 4. Stability metric
# ============================================================

def compute_chunk_stability_timeseries(pred_complex_chunk):
    """
    pred_complex_chunk: shape (Tchunk, H, W), complex

    Stability per time step:
      mean over space of |G[t+1] - G[t]|

    Returns:
      stability_t: shape (Tchunk-1,)
    """
    if pred_complex_chunk.shape[0] < 2:
        # cannot take a difference, define as zero-length/degenerate
        return np.array([0.0], dtype=np.float64)

    dG = pred_complex_chunk[1:] - pred_complex_chunk[:-1]   # (Tchunk-1,H,W), complex
    stability_t = np.abs(dG).mean(axis=(1, 2))              # (Tchunk-1,)
    return stability_t


def compute_chunk_stability_features(pred_complex_chunk):
    """
    Returns:
      mean_stability, std_stability, stability_t
    """
    stability_t = compute_chunk_stability_timeseries(pred_complex_chunk)
    mean_stability = stability_t.mean()
    std_stability = stability_t.std()
    return mean_stability, std_stability, stability_t


# ============================================================
# 5. Extract chunk-wise features for a set of trial indices
# ============================================================

def extract_stability_features_for_indices_chunked(
    X,
    indices,
    model,
    device,
    model_type="complex",
    one_based_labels=True,
    batch_size=256,
    chunk_len=8,
    chunk_overlap=2,
):
    """
    Each chunk becomes a new sample.

    Returns dict with:
      trial_idx            original trial index
      chunk_idx            chunk number within trial
      chunk_start          start index within trial
      is_wave
      mean_stability
      std_stability
      stability_t
      pred_complex
      inference_time_sec
      n_timepoints
    """
    results = {
        "trial_idx": [],
        "chunk_idx": [],
        "chunk_start": [],
        "is_wave": [],
        "mean_stability": [],
        "std_stability": [],
        "stability_t": [],
        "pred_complex": [],
        "inference_time_sec": [],
        "n_timepoints": [],
    }

    total_inference_time = 0.0
    total_chunks = 0

    maybe_cuda_sync(device)
    t_extract_start = time.perf_counter()

    for trial_idx in indices:
        trial_complex = unwrap_trial(X[trial_idx])
        trial_complex = ensure_trial_THW(trial_complex)  # (T,H,W), complex

        chunks, starts = make_time_chunks(
            trial_complex,
            chunk_len=chunk_len,
            chunk_overlap=chunk_overlap,
        )

        label = is_wave_trial(trial_idx, one_based_labels=one_based_labels)

        for j, (chunk, start_idx) in enumerate(zip(chunks, starts)):
            pred_complex, infer_time_sec = predict_trial_complex(
                model=model,
                trial_complex=chunk,
                device=device,
                model_type=model_type,
                batch_size=batch_size,
            )

            mean_stab, std_stab, stability_t = compute_chunk_stability_features(pred_complex)

            results["trial_idx"].append(trial_idx)
            results["chunk_idx"].append(j)
            results["chunk_start"].append(start_idx)
            results["is_wave"].append(label)
            results["mean_stability"].append(mean_stab)
            results["std_stability"].append(std_stab)
            results["stability_t"].append(stability_t)
            results["pred_complex"].append(pred_complex)
            results["inference_time_sec"].append(infer_time_sec)
            results["n_timepoints"].append(chunk.shape[0])

            total_inference_time += infer_time_sec
            total_chunks += 1

    maybe_cuda_sync(device)
    total_extract_time = time.perf_counter() - t_extract_start

    results["trial_idx"] = np.array(results["trial_idx"])
    results["chunk_idx"] = np.array(results["chunk_idx"])
    results["chunk_start"] = np.array(results["chunk_start"])
    results["is_wave"] = np.array(results["is_wave"], dtype=bool)
    results["mean_stability"] = np.array(results["mean_stability"])
    results["std_stability"] = np.array(results["std_stability"])
    results["inference_time_sec"] = np.array(results["inference_time_sec"])
    results["n_timepoints"] = np.array(results["n_timepoints"])

    results["total_inference_time_sec"] = total_inference_time
    results["total_extract_time_sec"] = total_extract_time
    results["n_chunks"] = total_chunks

    return results


# ============================================================
# 6. Plotting
# ============================================================

def plot_stability_feature_space(train_results, val_results):
    plt.figure(figsize=(8, 6))

    train_wave = train_results["is_wave"]
    train_nonwave = ~train_wave

    val_wave = val_results["is_wave"]
    val_nonwave = ~val_wave

    plt.scatter(
        train_results["mean_stability"][train_wave],
        train_results["std_stability"][train_wave],
        s=35,
        alpha=0.65,
        label="Train Wave Chunks"
    )

    plt.scatter(
        train_results["mean_stability"][train_nonwave],
        train_results["std_stability"][train_nonwave],
        s=35,
        alpha=0.65,
        label="Train Non-Wave Chunks"
    )

    plt.scatter(
        val_results["mean_stability"][val_wave],
        val_results["std_stability"][val_wave],
        s=70,
        marker='x',
        alpha=0.85,
        label="Val Wave Chunks"
    )

    plt.scatter(
        val_results["mean_stability"][val_nonwave],
        val_results["std_stability"][val_nonwave],
        s=70,
        marker='x',
        alpha=0.85,
        label="Val Non-Wave Chunks"
    )

    plt.xlabel("Mean Stability")
    plt.ylabel("Std Stability")
    plt.title("Chunk-wise Stability Feature Space")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_distributions(train_results):
    wave = train_results["is_wave"]
    nonwave = ~wave

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(train_results["mean_stability"][wave], bins=30, alpha=0.7, label="Wave")
    axes[0].hist(train_results["mean_stability"][nonwave], bins=30, alpha=0.7, label="Non-Wave")
    axes[0].set_title("Chunk Mean Stability")
    axes[0].legend()

    axes[1].hist(train_results["std_stability"][wave], bins=30, alpha=0.7, label="Wave")
    axes[1].hist(train_results["std_stability"][nonwave], bins=30, alpha=0.7, label="Non-Wave")
    axes[1].set_title("Chunk Std Stability")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Feature matrices and SVM
# ============================================================

def make_feature_matrix(results):
    """
    Returns:
      X_feat: shape (n_chunks, 2)
      y:      shape (n_chunks,)
    """
    X_feat = np.column_stack([
        results["mean_stability"],
        results["std_stability"]
    ])
    y = results["is_wave"].astype(int)   # wave=1, non-wave=0
    return X_feat, y


# ============================================================
# 8. Run everything
# ============================================================

print("Extracting chunk-wise train stability features...")
train_results = extract_stability_features_for_indices_chunked(
    X=X,
    indices=idx_train,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
    chunk_len=CHUNK_LEN,
    chunk_overlap=CHUNK_OVERLAP,
)

print("Extracting chunk-wise validation stability features...")
val_results = extract_stability_features_for_indices_chunked(
    X=X,
    indices=idx_val,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
    chunk_len=CHUNK_LEN,
    chunk_overlap=CHUNK_OVERLAP,
)

print("Done.")
print("Train chunk feature shape:", train_results["mean_stability"].shape)
print("Val chunk feature shape  :", val_results["mean_stability"].shape)

print("\nTiming summary for model inference / feature extraction")
print("-" * 60)
print(f"Train chunks: {train_results['n_chunks']}")
print(f"Val chunks  : {val_results['n_chunks']}")
print(f"Train total inference time (GPU): {train_results['total_inference_time_sec']:.3f} sec")
print(f"Val total inference time (GPU)  : {val_results['total_inference_time_sec']:.3f} sec")
print(f"Train total extraction time     : {train_results['total_extract_time_sec']:.3f} sec")
print(f"Val total extraction time       : {val_results['total_extract_time_sec']:.3f} sec")
print(f"Mean inference time per train chunk: {train_results['inference_time_sec'].mean():.6f} sec")
print(f"Mean inference time per val chunk  : {val_results['inference_time_sec'].mean():.6f} sec")

# 2D plot
plot_stability_feature_space(train_results, val_results)

# Optional train histograms
plot_train_distributions(train_results)

# ============================================================
# 9. SVM classifier on chunk-wise 2D stability features
# ============================================================

X_train_feat, y_train = make_feature_matrix(train_results)
X_val_feat, y_val = make_feature_matrix(val_results)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

t_svm_fit_start = time.perf_counter()
svm_clf.fit(X_train_feat, y_train)
t_svm_fit_sec = time.perf_counter() - t_svm_fit_start

t_svm_pred_start = time.perf_counter()
y_train_pred = svm_clf.predict(X_train_feat)
y_val_pred = svm_clf.predict(X_val_feat)
t_svm_pred_sec = time.perf_counter() - t_svm_pred_start

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print("\nSVM results on chunk-wise samples")
print("-" * 60)
print(f"SVM train accuracy: {train_acc:.3f}")
print(f"SVM val accuracy  : {val_acc:.3f}")

print("\nValidation confusion matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nValidation classification report:")
print(classification_report(
    y_val,
    y_val_pred,
    target_names=["Non-Wave", "Wave"]
))

print("\nTiming summary for classifier")
print("-" * 60)
print(f"SVM fit time    : {t_svm_fit_sec:.6f} sec")
print(f"SVM predict time: {t_svm_pred_sec:.6f} sec")

print("\nOverall approximate runtime")
print("-" * 60)
total_model_time = train_results['total_extract_time_sec'] + val_results['total_extract_time_sec']
total_classifier_time = t_svm_fit_sec + t_svm_pred_sec
print(f"Model + feature extraction total: {total_model_time:.3f} sec")
print(f"Classifier total               : {total_classifier_time:.3f} sec")
print(f"Grand total                    : {total_model_time + total_classifier_time:.3f} sec")


#%% SAME AS ABOVE BUT CHUNKING ONLY VALIDATION TRIALS

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# ASSUMPTIONS
# ============================================================
# You already have:
#   X           -> list of trials
#                  each trial may be:
#                     np.ndarray of shape (T, 11, 23), complex
#                  or np.ndarray of shape (1, T, 11, 23), complex
#                  or np.ndarray of shape (T, 1, 11, 23), complex
#                  or [np.ndarray of one of the above]
#
#   idx_train   -> indices of training trials
#   idx_val     -> indices of validation trials
#   model       -> trained model
#   device      -> torch device
#   MODEL_TYPE  -> "complex" or "real"
#
# Trial labels:
#   odd trials = wave
#   even trials = non-wave
#
# If Python index 0 corresponds to "trial 1", then:
#   idx 0,2,4,... are wave trials
# and ONE_BASED_LABELS should be True.
# ============================================================

ONE_BASED_LABELS = True

# Validation chunk settings only
CHUNK_LEN = 8
CHUNK_OVERLAP = 2
CHUNK_STEP = CHUNK_LEN - CHUNK_OVERLAP


# ============================================================
# 1. Helpers
# ============================================================

def unwrap_trial(trial):
    """
    Accepts:
      - np.ndarray
      - [np.ndarray] or (np.ndarray,)
    Returns:
      np.ndarray
    """
    if isinstance(trial, np.ndarray):
        arr = trial
    elif isinstance(trial, (list, tuple)) and len(trial) == 1:
        arr = trial[0]
    else:
        arr = np.asarray(trial)

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if not np.iscomplexobj(arr):
        raise ValueError("Trial must be complex-valued")

    return arr


def ensure_trial_THW(arr):
    """
    Convert a trial to shape (T, H, W), complex.

    Accepts:
      (T, H, W)
      (1, T, H, W)
      (T, 1, H, W)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            # (1, T, H, W) -> (T, H, W)
            return arr[0]
        if arr.shape[1] == 1:
            # (T, 1, H, W) -> (T, H, W)
            return arr[:, 0]

    raise ValueError(f"Expected trial shape (T,H,W), (1,T,H,W), or (T,1,H,W), got {arr.shape}")


def is_wave_trial(trial_idx, one_based_labels=True):
    """
    If one_based_labels=True:
      Python idx 0 -> trial 1 -> odd -> wave
      So 0,2,4,... are wave.
    """
    if one_based_labels:
        return (trial_idx % 2 == 0)
    else:
        return (trial_idx % 2 == 1)


def maybe_cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# 2. Validation chunking only
# ============================================================

def make_time_chunks(trial_complex, chunk_len=8, chunk_overlap=2):
    """
    trial_complex: np.ndarray, shape (T, H, W), complex

    Returns:
      chunks: list of np.ndarray, each shape (t_chunk, H, W)
      starts: list of start indices
    """
    T = trial_complex.shape[0]

    if T <= chunk_len:
        return [trial_complex], [0]

    step = chunk_len - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_len must be > chunk_overlap")

    chunks = []
    starts = []

    start = 0
    while start + chunk_len <= T:
        chunks.append(trial_complex[start:start + chunk_len])
        starts.append(start)
        start += step

    return chunks, starts


# ============================================================
# 3. Predict one trial/chunk
# ============================================================

@torch.no_grad()
def predict_trial_complex(model, trial_complex, device, model_type="complex", batch_size=256):
    """
    trial_complex: np.ndarray, shape (T, H, W), complex

    Returns:
      pred_complex: np.ndarray, shape (T, H, W), complex
      infer_time_sec: float
    """
    model.eval()

    T, H, W = trial_complex.shape

    if model_type == "complex":
        x = trial_complex[:, None, :, :]   # (T,1,H,W)
        x_t = torch.from_numpy(x)
        if x_t.dtype == torch.complex128:
            x_t = x_t.to(torch.complex64)

    elif model_type == "real":
        xr = trial_complex.real[:, None, :, :]
        xi = trial_complex.imag[:, None, :, :]
        x = np.concatenate([xr, xi], axis=1)   # (T,2,H,W)
        x_t = torch.from_numpy(x).float()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    preds = []
    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    for start in range(0, T, batch_size):
        xb = x_t[start:start + batch_size].to(device, non_blocking=True)
        yb = model(xb).cpu()
        preds.append(yb)

    maybe_cuda_sync(device)
    infer_time_sec = time.perf_counter() - t0

    preds = torch.cat(preds, dim=0).numpy()

    if model_type == "complex":
        pred_complex = preds[:, 0, :, :]   # (T,H,W), complex
    elif model_type == "real":
        pred_complex = preds[:, 0, :, :] + 1j * preds[:, 1, :, :]

    return pred_complex, infer_time_sec


# ============================================================
# 4. Stability metric
# ============================================================

def compute_stability_timeseries(pred_complex_trial):
    """
    pred_complex_trial: shape (T, H, W), complex

    Stability per time step:
      mean over space of |G[t+1] - G[t]|

    Returns:
      stability_t: shape (T-1,)
    """
    if pred_complex_trial.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)

    dG = pred_complex_trial[1:] - pred_complex_trial[:-1]   # (T-1,H,W), complex
    stability_t = np.abs(dG).mean(axis=(1, 2))              # (T-1,)
    return stability_t


def compute_stability_features(pred_complex_trial):
    """
    Returns:
      mean_stability, std_stability, stability_t
    """
    stability_t = compute_stability_timeseries(pred_complex_trial)
    mean_stability = stability_t.mean()
    std_stability = stability_t.std()
    return mean_stability, std_stability, stability_t


# ============================================================
# 5. TRAIN: one sample per full trial
# ============================================================

def extract_train_stability_features_full_trials(
    X,
    indices,
    model,
    device,
    model_type="complex",
    one_based_labels=True,
    batch_size=256,
):
    """
    Training set:
      each full trial -> one sample

    Returns dict with:
      trial_idx
      is_wave
      mean_stability
      std_stability
      stability_t
      pred_complex
      inference_time_sec
      n_timepoints
      total_inference_time_sec
      total_extract_time_sec
      n_samples
    """
    results = {
        "trial_idx": [],
        "is_wave": [],
        "mean_stability": [],
        "std_stability": [],
        "stability_t": [],
        "pred_complex": [],
        "inference_time_sec": [],
        "n_timepoints": [],
    }

    total_inference_time = 0.0

    maybe_cuda_sync(device)
    t_extract_start = time.perf_counter()

    for trial_idx in indices:
        trial_complex = unwrap_trial(X[trial_idx])
        trial_complex = ensure_trial_THW(trial_complex)  # (T,H,W), complex

        pred_complex, infer_time_sec = predict_trial_complex(
            model=model,
            trial_complex=trial_complex,
            device=device,
            model_type=model_type,
            batch_size=batch_size,
        )

        mean_stab, std_stab, stability_t = compute_stability_features(pred_complex)

        results["trial_idx"].append(trial_idx)
        results["is_wave"].append(is_wave_trial(trial_idx, one_based_labels=one_based_labels))
        results["mean_stability"].append(mean_stab)
        results["std_stability"].append(std_stab)
        results["stability_t"].append(stability_t)
        results["pred_complex"].append(pred_complex)
        results["inference_time_sec"].append(infer_time_sec)
        results["n_timepoints"].append(trial_complex.shape[0])

        total_inference_time += infer_time_sec

    maybe_cuda_sync(device)
    total_extract_time = time.perf_counter() - t_extract_start

    results["trial_idx"] = np.array(results["trial_idx"])
    results["is_wave"] = np.array(results["is_wave"], dtype=bool)
    results["mean_stability"] = np.array(results["mean_stability"])
    results["std_stability"] = np.array(results["std_stability"])
    results["inference_time_sec"] = np.array(results["inference_time_sec"])
    results["n_timepoints"] = np.array(results["n_timepoints"])

    results["total_inference_time_sec"] = total_inference_time
    results["total_extract_time_sec"] = total_extract_time
    results["n_samples"] = len(results["trial_idx"])

    return results


# ============================================================
# 6. VAL: one sample per chunk
# ============================================================

def extract_val_stability_features_chunked(
    X,
    indices,
    model,
    device,
    model_type="complex",
    one_based_labels=True,
    batch_size=256,
    chunk_len=8,
    chunk_overlap=2,
):
    """
    Validation set:
      each chunk -> one sample

    Returns dict with:
      trial_idx
      chunk_idx
      chunk_start
      is_wave
      mean_stability
      std_stability
      stability_t
      pred_complex
      inference_time_sec
      n_timepoints
      total_inference_time_sec
      total_extract_time_sec
      n_samples
    """
    results = {
        "trial_idx": [],
        "chunk_idx": [],
        "chunk_start": [],
        "is_wave": [],
        "mean_stability": [],
        "std_stability": [],
        "stability_t": [],
        "pred_complex": [],
        "inference_time_sec": [],
        "n_timepoints": [],
    }

    total_inference_time = 0.0

    maybe_cuda_sync(device)
    t_extract_start = time.perf_counter()

    for trial_idx in indices:
        trial_complex = unwrap_trial(X[trial_idx])
        trial_complex = ensure_trial_THW(trial_complex)  # (T,H,W), complex

        chunks, starts = make_time_chunks(
            trial_complex,
            chunk_len=chunk_len,
            chunk_overlap=chunk_overlap,
        )

        label = is_wave_trial(trial_idx, one_based_labels=one_based_labels)

        for chunk_idx, (chunk, start_idx) in enumerate(zip(chunks, starts)):
            pred_complex, infer_time_sec = predict_trial_complex(
                model=model,
                trial_complex=chunk,
                device=device,
                model_type=model_type,
                batch_size=batch_size,
            )

            mean_stab, std_stab, stability_t = compute_stability_features(pred_complex)

            results["trial_idx"].append(trial_idx)
            results["chunk_idx"].append(chunk_idx)
            results["chunk_start"].append(start_idx)
            results["is_wave"].append(label)
            results["mean_stability"].append(mean_stab)
            results["std_stability"].append(std_stab)
            results["stability_t"].append(stability_t)
            results["pred_complex"].append(pred_complex)
            results["inference_time_sec"].append(infer_time_sec)
            results["n_timepoints"].append(chunk.shape[0])

            total_inference_time += infer_time_sec

    maybe_cuda_sync(device)
    total_extract_time = time.perf_counter() - t_extract_start

    results["trial_idx"] = np.array(results["trial_idx"])
    results["chunk_idx"] = np.array(results["chunk_idx"])
    results["chunk_start"] = np.array(results["chunk_start"])
    results["is_wave"] = np.array(results["is_wave"], dtype=bool)
    results["mean_stability"] = np.array(results["mean_stability"])
    results["std_stability"] = np.array(results["std_stability"])
    results["inference_time_sec"] = np.array(results["inference_time_sec"])
    results["n_timepoints"] = np.array(results["n_timepoints"])

    results["total_inference_time_sec"] = total_inference_time
    results["total_extract_time_sec"] = total_extract_time
    results["n_samples"] = len(results["trial_idx"])

    return results


# ============================================================
# 7. Plotting
# ============================================================

def plot_stability_feature_space(train_results, val_results):
    plt.figure(figsize=(8, 6))

    train_wave = train_results["is_wave"]
    train_nonwave = ~train_wave

    val_wave = val_results["is_wave"]
    val_nonwave = ~val_wave

    plt.scatter(
        train_results["mean_stability"][train_wave],
        train_results["std_stability"][train_wave],
        s=50,
        alpha=0.8,
        label="Train Wave Trials"
    )

    plt.scatter(
        train_results["mean_stability"][train_nonwave],
        train_results["std_stability"][train_nonwave],
        s=50,
        alpha=0.8,
        label="Train Non-Wave Trials"
    )

    plt.scatter(
        val_results["mean_stability"][val_wave],
        val_results["std_stability"][val_wave],
        s=70,
        marker='x',
        alpha=0.9,
        label="Val Wave Chunks"
    )

    plt.scatter(
        val_results["mean_stability"][val_nonwave],
        val_results["std_stability"][val_nonwave],
        s=70,
        marker='x',
        alpha=0.9,
        label="Val Non-Wave Chunks"
    )

    plt.xlabel("Mean Stability")
    plt.ylabel("Std Stability")
    plt.title("Train Trial Features vs Validation Chunk Features")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_distributions(train_results):
    wave = train_results["is_wave"]
    nonwave = ~wave

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(train_results["mean_stability"][wave], bins=20, alpha=0.7, label="Wave")
    axes[0].hist(train_results["mean_stability"][nonwave], bins=20, alpha=0.7, label="Non-Wave")
    axes[0].set_title("Train Trial Mean Stability")
    axes[0].legend()

    axes[1].hist(train_results["std_stability"][wave], bins=20, alpha=0.7, label="Wave")
    axes[1].hist(train_results["std_stability"][nonwave], bins=20, alpha=0.7, label="Non-Wave")
    axes[1].set_title("Train Trial Std Stability")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Feature matrices and SVM
# ============================================================

def make_feature_matrix(results):
    """
    Returns:
      X_feat: shape (n_samples, 2)
      y:      shape (n_samples,)
    """
    X_feat = np.column_stack([
        results["mean_stability"],
        results["std_stability"]
    ])
    y = results["is_wave"].astype(int)   # wave=1, non-wave=0
    return X_feat, y


# ============================================================
# 9. Run everything
# ============================================================

print("Extracting TRAIN stability features from full trials...")
train_results = extract_train_stability_features_full_trials(
    X=X,
    indices=idx_train,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
)

print("Extracting VALIDATION stability features from chunks...")
val_results = extract_val_stability_features_chunked(
    X=X,
    indices=idx_val,
    model=model,
    device=device,
    model_type=MODEL_TYPE,
    one_based_labels=ONE_BASED_LABELS,
    batch_size=256,
    chunk_len=CHUNK_LEN,
    chunk_overlap=CHUNK_OVERLAP,
)

print("Done.")
print("Train feature shape:", train_results["mean_stability"].shape)
print("Val feature shape  :", val_results["mean_stability"].shape)

print("\nTiming summary for model inference / feature extraction")
print("-" * 60)
print(f"Train samples (full trials): {train_results['n_samples']}")
print(f"Val samples (chunks)       : {val_results['n_samples']}")
print(f"Train total inference time (GPU): {train_results['total_inference_time_sec']:.3f} sec")
print(f"Val total inference time (GPU)  : {val_results['total_inference_time_sec']:.3f} sec")
print(f"Train total extraction time     : {train_results['total_extract_time_sec']:.3f} sec")
print(f"Val total extraction time       : {val_results['total_extract_time_sec']:.3f} sec")
print(f"Mean inference time per train trial: {train_results['inference_time_sec'].mean():.6f} sec")
print(f"Mean inference time per val chunk   : {val_results['inference_time_sec'].mean():.6f} sec")

# Plot
plot_stability_feature_space(train_results, val_results)
plot_train_distributions(train_results)

# ============================================================
# 10. SVM classifier
# ============================================================

X_train_feat, y_train = make_feature_matrix(train_results)
X_val_feat, y_val = make_feature_matrix(val_results)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

t_svm_fit_start = time.perf_counter()
svm_clf.fit(X_train_feat, y_train)
t_svm_fit_sec = time.perf_counter() - t_svm_fit_start

t_svm_pred_start = time.perf_counter()
y_train_pred = svm_clf.predict(X_train_feat)
y_val_pred = svm_clf.predict(X_val_feat)
t_svm_pred_sec = time.perf_counter() - t_svm_pred_start

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print("\nSVM results")
print("-" * 60)
print(f"SVM train accuracy: {train_acc:.3f}")
print(f"SVM val accuracy  : {val_acc:.3f}")

print("\nValidation confusion matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nValidation classification report:")
print(classification_report(
    y_val,
    y_val_pred,
    target_names=["Non-Wave", "Wave"]
))

print("\nTiming summary for classifier")
print("-" * 60)
print(f"SVM fit time    : {t_svm_fit_sec:.6f} sec")
print(f"SVM predict time: {t_svm_pred_sec:.6f} sec")

print("\nOverall approximate runtime")
print("-" * 60)
total_model_time = train_results['total_extract_time_sec'] + val_results['total_extract_time_sec']
total_classifier_time = t_svm_fit_sec + t_svm_pred_sec
print(f"Model + feature extraction total: {total_model_time:.3f} sec")
print(f"Classifier total               : {total_classifier_time:.3f} sec")
print(f"Grand total                    : {total_model_time + total_classifier_time:.3f} sec")

#%% SVM WITH CHUNKING AND STABILITY OVER CHUNKS RATHER THAN MEAN AND STD
# with more features 

import time
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ============================================================
# SETTINGS
# ============================================================

ONE_BASED_LABELS = True
CHUNK_LEN = 8
CHUNK_OVERLAP = 2


# ============================================================
# HELPERS
# ============================================================

def unwrap_trial(trial):
    if isinstance(trial, np.ndarray):
        return trial
    if isinstance(trial, (list, tuple)) and len(trial) == 1:
        return trial[0]
    return np.asarray(trial)


def ensure_trial_THW(arr):
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[1] == 1:
            return arr[:, 0]

    raise ValueError(f"Bad shape: {arr.shape}")


def is_wave_trial(idx):
    return (idx % 2 == 0) if ONE_BASED_LABELS else (idx % 2 == 1)


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# CHUNKING
# ============================================================

def make_chunks(trial):
    T = trial.shape[0]

    if T <= CHUNK_LEN:
        return [trial]

    step = CHUNK_LEN - CHUNK_OVERLAP
    chunks = []

    for start in range(0, T - CHUNK_LEN + 1, step):
        chunks.append(trial[start:start + CHUNK_LEN])

    return chunks


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.no_grad()
def predict(model, trial, device, model_type="complex"):
    T = trial.shape[0]

    if model_type == "complex":
        x = trial[:, None]
        x = torch.from_numpy(x)
        if x.dtype == torch.complex128:
            x = x.to(torch.complex64)

    else:
        xr = trial.real[:, None]
        xi = trial.imag[:, None]
        x = torch.from_numpy(np.concatenate([xr, xi], axis=1)).float()

    maybe_sync(device)
    t0 = time.perf_counter()

    y = model(x.to(device)).cpu().numpy()

    maybe_sync(device)
    dt = time.perf_counter() - t0

    if model_type == "complex":
        return y[:, 0], dt
    else:
        return y[:, 0] + 1j * y[:, 1], dt


# ============================================================
# FEATURE EXTRACTION (KEY PART)
# ============================================================

def extract_features(G):
    """
    G: (T, H, W) complex gradient field
    Returns: 5D feature vector
    """

    # ---------- Stability ----------
    if G.shape[0] < 2:
        dG = np.zeros_like(G)
    else:
        dG = G[1:] - G[:-1]

    stability = np.abs(dG).mean(axis=(1, 2))

    mean_stability = stability.mean()
    std_stability = stability.std()

    # ---------- Gradient magnitude ----------
    grad_mag = np.abs(G)
    mean_grad_mag = grad_mag.mean()

    # ---------- Directional coherence ----------
    G_norm = G / (np.abs(G) + 1e-6)
    coherence = np.abs(G_norm.mean(axis=(1, 2))).mean()

    # ---------- Directional change ----------
    d_dir = np.abs(G_norm[1:] - G_norm[:-1]).mean(axis=(1, 2))
    dir_change = d_dir.mean()

    return np.array([
        mean_stability,
        std_stability,
        mean_grad_mag,
        coherence,
        dir_change
    ], dtype=np.float32)


# ============================================================
# DATASET BUILDING
# ============================================================

def build_dataset(X, indices, model, device, model_type):
    features = []
    labels = []
    total_time = 0.0

    maybe_sync(device)
    t_start = time.perf_counter()

    for idx in indices:
        trial = ensure_trial_THW(unwrap_trial(X[idx]))
        chunks = make_chunks(trial)

        label = int(is_wave_trial(idx))

        for chunk in chunks:
            G, dt = predict(model, chunk, device, model_type)
            feat = extract_features(G)

            features.append(feat)
            labels.append(label)
            total_time += dt

    maybe_sync(device)
    total_extract_time = time.perf_counter() - t_start

    return (
        np.vstack(features),
        np.array(labels),
        total_time,
        total_extract_time
    )


# ============================================================
# RUN PIPELINE
# ============================================================

print("Extracting TRAIN features...")
X_train, y_train, t_inf_train, t_ext_train = build_dataset(
    X, idx_train, model, device, MODEL_TYPE
)

print("Extracting VAL features...")
X_val, y_val, t_inf_val, t_ext_val = build_dataset(
    X, idx_val, model, device, MODEL_TYPE
)

print("Train shape:", X_train.shape)
print("Val shape  :", X_val.shape)

# ============================================================
# SVM
# ============================================================

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0))
])

t0 = time.perf_counter()
clf.fit(X_train, y_train)
t_fit = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred = clf.predict(X_val)
t_pred = time.perf_counter() - t0

acc = accuracy_score(y_val, y_pred)

print("\nRESULTS")
print("=" * 40)
print(f"Validation Accuracy: {acc:.3f}")

print("\nTIMING")
print("=" * 40)
print(f"Train inference time: {t_inf_train:.3f}s")
print(f"Val inference time  : {t_inf_val:.3f}s")
print(f"Feature extraction  : {t_ext_train + t_ext_val:.3f}s")
print(f"SVM fit time        : {t_fit:.6f}s")
print(f"SVM predict time    : {t_pred:.6f}s")
print(f"Total runtime       : {t_ext_train + t_ext_val + t_fit + t_pred:.3f}s")

#%% MU - CNN - PHASE GRADIENT - CNN - WAVE / NON WAVE CLASSIFICATION

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

ONE_BASED_LABELS = True

CHUNK_LEN = 8
CHUNK_OVERLAP = 2
CHUNK_STEP = CHUNK_LEN - CHUNK_OVERLAP

GRADIENT_SOURCE = "predicted"   # "predicted" or "true"

CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_LR = 1e-3
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_MAX_EPOCHS = 30
CLASSIFIER_PATIENCE = 6

# assumes these already exist in memory:
# X, Y, idx_train, idx_val, model, device, MODEL_TYPE


# ============================================================
# 1. Helpers
# ============================================================

def maybe_cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unwrap_trial(trial):
    if isinstance(trial, np.ndarray):
        arr = trial
    elif isinstance(trial, (list, tuple)) and len(trial) == 1:
        arr = trial[0]
    else:
        arr = np.asarray(trial)

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if not np.iscomplexobj(arr):
        raise ValueError("Trial must be complex-valued")

    return arr


def ensure_trial_THW(arr):
    """
    Convert to (T, H, W), complex.

    Accepts:
      (T,H,W)
      (1,T,H,W)
      (T,1,H,W)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[1] == 1:
            return arr[:, 0]

    raise ValueError(f"Expected (T,H,W), (1,T,H,W), or (T,1,H,W), got {arr.shape}")


def is_wave_trial(idx, one_based_labels=True):
    if one_based_labels:
        return (idx % 2 == 0)
    else:
        return (idx % 2 == 1)


def make_chunks(trial_complex, chunk_len=8, chunk_overlap=2):
    """
    trial_complex: (T,H,W), complex

    Returns list of chunks, each (t_chunk,H,W)
    """
    T = trial_complex.shape[0]

    if T <= chunk_len:
        return [trial_complex], [0]

    step = chunk_len - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_len must be > chunk_overlap")

    chunks = []
    starts = []

    for start in range(0, T - chunk_len + 1, step):
        chunks.append(trial_complex[start:start + chunk_len])
        starts.append(start)

    return chunks, starts


# ============================================================
# 2. Predict gradients from mu model
# ============================================================

@torch.no_grad()
def predict_trial_complex(model, trial_complex, device, model_type="complex", batch_size=256):
    """
    Input:
      trial_complex: (T,H,W), complex mu

    Returns:
      pred_complex: (T,H,W), complex gradient
      infer_time_sec
    """
    model.eval()

    T, H, W = trial_complex.shape

    if model_type == "complex":
        x = trial_complex[:, None, :, :]   # (T,1,H,W)
        x_t = torch.from_numpy(x)
        if x_t.dtype == torch.complex128:
            x_t = x_t.to(torch.complex64)

    elif model_type == "real":
        xr = trial_complex.real[:, None, :, :]
        xi = trial_complex.imag[:, None, :, :]
        x = np.concatenate([xr, xi], axis=1)   # (T,2,H,W)
        x_t = torch.from_numpy(x).float()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    preds = []

    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    for start in range(0, T, batch_size):
        xb = x_t[start:start + batch_size].to(device, non_blocking=True)
        yb = model(xb).cpu()
        preds.append(yb)

    maybe_cuda_sync(device)
    infer_time_sec = time.perf_counter() - t0

    preds = torch.cat(preds, dim=0).numpy()

    if model_type == "complex":
        pred_complex = preds[:, 0, :, :]
    elif model_type == "real":
        pred_complex = preds[:, 0, :, :] + 1j * preds[:, 1, :, :]

    return pred_complex, infer_time_sec


# ============================================================
# 3. Build chunk dataset for classifier
# ============================================================

def complex_gradient_to_channels(G):
    """
    G: (T,H,W), complex gradient
    Returns:
      (2,T,H,W), float32
      channel 0 = gx
      channel 1 = gy
    """
    gx = G.real.astype(np.float32)
    gy = G.imag.astype(np.float32)
    return np.stack([gx, gy], axis=0)


def build_chunk_gradient_dataset(
    X,
    Y,
    indices,
    gradient_source,
    mu_to_grad_model,
    device,
    model_type="complex",
    chunk_len=8,
    chunk_overlap=2,
    one_based_labels=True,
    pred_batch_size=256,
):
    """
    Returns:
      X_chunks: (N, 2, Tchunk, H, W)
      y_chunks: (N,)
      meta: dict
    """
    X_chunks = []
    y_chunks = []

    meta = {
        "trial_idx": [],
        "chunk_start": [],
        "chunk_idx": [],
        "is_wave": [],
        "grad_extract_time_sec": [],
    }

    total_grad_extract_time = 0.0

    for idx in indices:
        x_trial = ensure_trial_THW(unwrap_trial(X[idx]))   # mu data
        y_trial = ensure_trial_THW(unwrap_trial(Y[idx]))   # true gradients

        label = int(is_wave_trial(idx, one_based_labels=one_based_labels))

        # choose source
        if gradient_source == "predicted":
            G_full, dt = predict_trial_complex(
                model=mu_to_grad_model,
                trial_complex=x_trial,
                device=device,
                model_type=model_type,
                batch_size=pred_batch_size,
            )
        elif gradient_source == "true":
            G_full = y_trial
            dt = 0.0
        else:
            raise ValueError(f"Unknown gradient_source: {gradient_source}")

        total_grad_extract_time += dt

        chunks, starts = make_chunks(G_full, chunk_len=chunk_len, chunk_overlap=chunk_overlap)

        for j, (chunk, start_idx) in enumerate(zip(chunks, starts)):
            X_chunks.append(complex_gradient_to_channels(chunk))  # (2,T,H,W)
            y_chunks.append(label)

            meta["trial_idx"].append(idx)
            meta["chunk_start"].append(start_idx)
            meta["chunk_idx"].append(j)
            meta["is_wave"].append(label)
            meta["grad_extract_time_sec"].append(dt)

    X_chunks = np.stack(X_chunks, axis=0).astype(np.float32)
    y_chunks = np.array(y_chunks, dtype=np.int64)

    for k in meta:
        meta[k] = np.array(meta[k])

    meta["total_grad_extract_time_sec"] = total_grad_extract_time
    meta["n_samples"] = len(y_chunks)

    return X_chunks, y_chunks, meta


# ============================================================
# 4. Dataset
# ============================================================

class GradientChunkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 5. 3D CNN classifier on gradients
# ============================================================

class Gradient3DCNNClassifier(nn.Module):
    """
    Input:
      (B, 2, T, H, W)
    """
    def __init__(self, in_channels=2, base_channels=16, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * base_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 6. Training utilities
# ============================================================

class EarlyStopping:
    def __init__(self, patience=6, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None
        self.stop = False

    def step(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def run_epoch_classifier(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_loss += loss.item() * xb.shape[0]
        total_n += xb.shape[0]

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=30,
    patience=6,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch_classifier(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_acc = run_epoch_classifier(
            model, val_loader, optimizer, criterion, device, train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.6f} | val_acc={val_acc:.3f}"
        )

        early_stopper.step(val_loss, model)

        if early_stopper.stop:
            print(f"Early stopping at epoch {epoch}")
            break

    maybe_cuda_sync(device)
    train_time_sec = time.perf_counter() - t0

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    return history, train_time_sec


@torch.no_grad()
def evaluate_classifier_predictions(model, loader, device):
    model.eval()

    all_preds = []
    all_true = []

    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_true.append(yb.numpy())

    maybe_cuda_sync(device)
    infer_time_sec = time.perf_counter() - t0

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    return y_true, y_pred, infer_time_sec


# ============================================================
# 7. Build datasets
# ============================================================

print(f"Building gradient chunk datasets using source = {GRADIENT_SOURCE!r} ...")

t0 = time.perf_counter()

X_train_chunks, y_train_chunks, train_meta = build_chunk_gradient_dataset(
    X=X,
    Y=Y,
    indices=idx_train,
    gradient_source=GRADIENT_SOURCE,
    mu_to_grad_model=model,
    device=device,
    model_type=MODEL_TYPE,
    chunk_len=CHUNK_LEN,
    chunk_overlap=CHUNK_OVERLAP,
    one_based_labels=ONE_BASED_LABELS,
    pred_batch_size=256,
)

X_val_chunks, y_val_chunks, val_meta = build_chunk_gradient_dataset(
    X=X,
    Y=Y,
    indices=idx_val,
    gradient_source=GRADIENT_SOURCE,
    mu_to_grad_model=model,
    device=device,
    model_type=MODEL_TYPE,
    chunk_len=CHUNK_LEN,
    chunk_overlap=CHUNK_OVERLAP,
    one_based_labels=ONE_BASED_LABELS,
    pred_batch_size=256,
)

dataset_build_time_sec = time.perf_counter() - t0

print("Train chunk tensor shape:", X_train_chunks.shape)
print("Val chunk tensor shape  :", X_val_chunks.shape)

train_ds = GradientChunkDataset(X_train_chunks, y_train_chunks)
val_ds = GradientChunkDataset(X_val_chunks, y_val_chunks)

train_loader = DataLoader(
    train_ds,
    batch_size=CLASSIFIER_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=CLASSIFIER_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ============================================================
# 8. Train classifier
# ============================================================

clf_model = Gradient3DCNNClassifier(in_channels=2, base_channels=16, num_classes=2).to(device)

print(f"Classifier parameters: {sum(p.numel() for p in clf_model.parameters() if p.requires_grad):,}")

history, classifier_train_time_sec = train_classifier(
    model=clf_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    lr=CLASSIFIER_LR,
    weight_decay=CLASSIFIER_WEIGHT_DECAY,
    max_epochs=CLASSIFIER_MAX_EPOCHS,
    patience=CLASSIFIER_PATIENCE,
)


# ============================================================
# 9. Final evaluation
# ============================================================

y_true_train, y_pred_train, train_pred_time_sec = evaluate_classifier_predictions(
    clf_model, train_loader, device
)
y_true_val, y_pred_val, val_pred_time_sec = evaluate_classifier_predictions(
    clf_model, val_loader, device
)

train_acc = (y_true_train == y_pred_train).mean()
val_acc = (y_true_val == y_pred_val).mean()

print("\nFINAL RESULTS")
print("=" * 60)
print(f"Gradient source used : {GRADIENT_SOURCE}")
print(f"Train accuracy       : {train_acc:.3f}")
print(f"Validation accuracy  : {val_acc:.3f}")

# chunk-level confusion matrix
cm = np.zeros((2, 2), dtype=int)
for t, p in zip(y_true_val, y_pred_val):
    cm[t, p] += 1

print("\nValidation confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)


# ============================================================
# 10. Runtime summary
# ============================================================

print("\nRUNTIME SUMMARY")
print("=" * 60)
print(f"Gradient dataset build wall time     : {dataset_build_time_sec:.3f} sec")
print(f"Train gradient extraction GPU time   : {train_meta['total_grad_extract_time_sec']:.3f} sec")
print(f"Val gradient extraction GPU time     : {val_meta['total_grad_extract_time_sec']:.3f} sec")
print(f"Classifier training time             : {classifier_train_time_sec:.3f} sec")
print(f"Classifier train-set inference time  : {train_pred_time_sec:.3f} sec")
print(f"Classifier val-set inference time    : {val_pred_time_sec:.3f} sec")

total_gpu_related = (
    train_meta['total_grad_extract_time_sec']
    + val_meta['total_grad_extract_time_sec']
    + classifier_train_time_sec
    + train_pred_time_sec
    + val_pred_time_sec
)

print(f"Approx total GPU-related runtime     : {total_gpu_related:.3f} sec")

#%% CLASSFICATION WAVE VS NON WAVE USING COMPLEX VALUED MU

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
VAL_FRACTION = 0.2

ONE_BASED_LABELS = True   # idx 0 -> trial 1 -> odd -> wave

CHUNK_LEN = 8
CHUNK_OVERLAP = 6
CHUNK_STEP = CHUNK_LEN - CHUNK_OVERLAP  # 6

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 6
GRAD_CLIP = 1.0

BASE_CHANNELS = 28
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def maybe_cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unwrap_trial(trial):
    if isinstance(trial, np.ndarray):
        arr = trial
    elif isinstance(trial, (list, tuple)) and len(trial) == 1:
        arr = trial[0]
    else:
        arr = np.asarray(trial)

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if not np.iscomplexobj(arr):
        raise ValueError("Trial must be complex-valued")

    return arr


def ensure_trial_THW(arr):
    """
    Convert a trial to shape (T, H, W), complex.

    Accepts:
      (T, H, W)
      (1, T, H, W)
      (T, 1, H, W)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[1] == 1:
            return arr[:, 0]

    raise ValueError(f"Expected (T,H,W), (1,T,H,W), or (T,1,H,W), got {arr.shape}")


def is_wave_trial(idx, one_based_labels=True):
    if one_based_labels:
        return (idx % 2 == 0)
    else:
        return (idx % 2 == 1)


def get_trial_labels(n_trials, one_based_labels=True):
    return np.array(
        [1 if is_wave_trial(i, one_based_labels) else 0 for i in range(n_trials)],
        dtype=np.int64
    )


# ============================================================
# SPLIT TRAIN / VAL TRIALS
# ============================================================

def split_trial_indices(X, val_fraction=0.2, seed=42, one_based_labels=True):
    n_trials = len(X)
    all_indices = np.arange(n_trials)
    labels = get_trial_labels(n_trials, one_based_labels=one_based_labels)

    idx_train, idx_val = train_test_split(
        all_indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )

    return np.array(sorted(idx_train)), np.array(sorted(idx_val))


# ============================================================
# CHUNK EXTRACTION
# ============================================================

def pad_trial_to_min_length(trial, min_len=8):
    """
    If T < min_len, pad by repeating the last frame.
    """
    T = trial.shape[0]
    if T >= min_len:
        return trial

    n_pad = min_len - T
    last_frame = trial[-1:, :, :]
    pad = np.repeat(last_frame, repeats=n_pad, axis=0)
    return np.concatenate([trial, pad], axis=0)


def extract_overlapping_chunks(trial, chunk_len=8, overlap=2):
    """
    trial: (T,H,W), complex

    Returns list of chunks, each (chunk_len,H,W)

    If T < chunk_len, pad to chunk_len and return one chunk.
    """
    trial = pad_trial_to_min_length(trial, min_len=chunk_len)
    T = trial.shape[0]

    step = chunk_len - overlap
    if step <= 0:
        raise ValueError("chunk_len must be > overlap")

    chunks = []

    if T == chunk_len:
        chunks.append(trial)
        return chunks

    for start in range(0, T - chunk_len + 1, step):
        chunks.append(trial[start:start + chunk_len])

    return chunks


def complex_chunk_to_2channel(chunk):
    """
    chunk: (T,H,W), complex
    returns: (2,T,H,W), float32
    """
    xr = chunk.real.astype(np.float32)
    xi = chunk.imag.astype(np.float32)
    return np.stack([xr, xi], axis=0)


def build_chunk_dataset(X, indices, one_based_labels=True):
    """
    Returns:
      X_chunks: (N,2,T,H,W)
      y_chunks: (N,)
      meta: dict
    """
    X_chunks = []
    y_chunks = []

    meta = {
        "trial_idx": [],
        "chunk_idx": [],
        "is_wave": [],
        "orig_T": [],
    }

    for idx in indices:
        trial = ensure_trial_THW(unwrap_trial(X[idx]))
        label = 1 if is_wave_trial(idx, one_based_labels=one_based_labels) else 0

        chunks = extract_overlapping_chunks(
            trial,
            chunk_len=CHUNK_LEN,
            overlap=CHUNK_OVERLAP
        )

        for j, chunk in enumerate(chunks):
            X_chunks.append(complex_chunk_to_2channel(chunk))
            y_chunks.append(label)

            meta["trial_idx"].append(idx)
            meta["chunk_idx"].append(j)
            meta["is_wave"].append(label)
            meta["orig_T"].append(trial.shape[0])

    X_chunks = np.stack(X_chunks, axis=0).astype(np.float32)
    y_chunks = np.array(y_chunks, dtype=np.int64)

    for k in meta:
        meta[k] = np.array(meta[k])

    return X_chunks, y_chunks, meta


def balance_training_chunks(X_train, y_train, meta_train, seed=42):
    """
    Undersample majority class to match minority class exactly.
    """
    rng = np.random.default_rng(seed)

    idx_wave = np.where(y_train == 1)[0]
    idx_non = np.where(y_train == 0)[0]

    n_keep = min(len(idx_wave), len(idx_non))

    idx_wave_keep = rng.choice(idx_wave, size=n_keep, replace=False)
    idx_non_keep = rng.choice(idx_non, size=n_keep, replace=False)

    keep = np.concatenate([idx_wave_keep, idx_non_keep])
    rng.shuffle(keep)

    X_bal = X_train[keep]
    y_bal = y_train[keep]
    meta_bal = {k: v[keep] for k, v in meta_train.items()}

    return X_bal, y_bal, meta_bal


# ============================================================
# NORMALIZATION
# ============================================================

def compute_channel_stats(X_train):
    """
    X_train: (N,2,T,H,W)
    Compute mean/std from training data only.
    """
    mean = X_train.mean(axis=(0, 2, 3, 4), keepdims=True)
    std = X_train.std(axis=(0, 2, 3, 4), keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_chunks(X, mean, std):
    return (X - mean) / std


# ============================================================
# DATASET
# ============================================================

class MuChunkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL: 6 Conv3D layers, no pooling, MLP head
# ============================================================

class Mu3DCNNClassifier(nn.Module):
    """
    Input:
      (B, 2, 8, H, W)

    No max pooling, no average pooling.
    Dimensionality reduction is done using strided convolutions and MLP.
    """
    def __init__(self, input_shape=(2, 8, 11, 23), base_channels=24, num_classes=2):
        super().__init__()

        c_in, t_in, h_in, w_in = input_shape

        self.features = nn.Sequential(
            # 1
            nn.Conv3d(c_in, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 2
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 3: reduce dimensions
            nn.Conv3d(base_channels, 2 * base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 4
            nn.Conv3d(2 * base_channels, 2 * base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 5: reduce dimensions again
            nn.Conv3d(2 * base_channels, 4 * base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 6
            nn.Conv3d(4 * base_channels, 4 * base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # infer flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c_in, t_in, h_in, w_in)
            feat = self.features(dummy)
            self.flat_dim = feat.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# TRAINING UTILITIES
# ============================================================

class EarlyStopping:
    def __init__(self, patience=6, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None
        self.stop = False

    def step(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def run_epoch(model, loader, optimizer, criterion, device, train=True, grad_clip=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_loss += loss.item() * xb.shape[0]
        total_n += xb.shape[0]

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def train_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=PATIENCE)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device,
            train=True, grad_clip=GRAD_CLIP
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, device,
            train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.6f} | val_acc={val_acc:.3f}"
        )

        early_stopper.step(val_loss, model)

        if early_stopper.stop:
            print(f"Early stopping at epoch {epoch}")
            break

    maybe_cuda_sync(device)
    train_time_sec = time.perf_counter() - t0

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    return history, train_time_sec


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []

    maybe_cuda_sync(device)
    t0 = time.perf_counter()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_true.append(yb.numpy())
        all_pred.append(preds)

    maybe_cuda_sync(device)
    infer_time_sec = time.perf_counter() - t0

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return y_true, y_pred, infer_time_sec


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Split trials
    idx_train, idx_val = split_trial_indices(
        X,
        val_fraction=VAL_FRACTION,
        seed=SEED,
        one_based_labels=ONE_BASED_LABELS,
    )

    print("Number of train trials:", len(idx_train))
    print("Number of val trials  :", len(idx_val))

    # Build chunk datasets
    X_train_chunks, y_train_chunks, meta_train = build_chunk_dataset(
        X, idx_train, one_based_labels=ONE_BASED_LABELS
    )
    X_val_chunks, y_val_chunks, meta_val = build_chunk_dataset(
        X, idx_val, one_based_labels=ONE_BASED_LABELS
    )

    print("\nBefore balancing:")
    print("Train chunk shape:", X_train_chunks.shape)
    print("Val chunk shape  :", X_val_chunks.shape)
    print("Train class counts:", np.bincount(y_train_chunks))
    print("Val class counts  :", np.bincount(y_val_chunks))

    # Balance training chunks only
    X_train_bal, y_train_bal, meta_train_bal = balance_training_chunks(
        X_train_chunks, y_train_chunks, meta_train, seed=SEED
    )

    print("\nAfter balancing training chunks:")
    print("Balanced train chunk shape:", X_train_bal.shape)
    print("Balanced train class counts:", np.bincount(y_train_bal))

    # Normalize using training data only
    mean, std = compute_channel_stats(X_train_bal)
    X_train_bal = normalize_chunks(X_train_bal, mean, std)
    X_val_chunks = normalize_chunks(X_val_chunks, mean, std)

    # Datasets / loaders
    train_ds = MuChunkDataset(X_train_bal, y_train_bal)
    val_ds = MuChunkDataset(X_val_chunks, y_val_chunks)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = Mu3DCNNClassifier(
        input_shape=(2, CHUNK_LEN, X_train_bal.shape[-2], X_train_bal.shape[-1]),
        base_channels=BASE_CHANNELS,
        num_classes=2
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    # Train
    history, train_time_sec = train_model(model, train_loader, val_loader, DEVICE)

    # Evaluate
    y_train_true, y_train_pred, train_infer_time = predict_loader(model, train_loader, DEVICE)
    y_val_true, y_val_pred, val_infer_time = predict_loader(model, val_loader, DEVICE)

    train_acc = accuracy_score(y_train_true, y_train_pred)

    val_counts = np.bincount(y_val_true, minlength=2)
    val_is_balanced = (val_counts[0] == val_counts[1])

    if val_is_balanced:
        val_metric_name = "Accuracy"
        val_metric = accuracy_score(y_val_true, y_val_pred)
    else:
        val_metric_name = "Balanced Accuracy"
        val_metric = balanced_accuracy_score(y_val_true, y_val_pred)

    print("\nFINAL RESULTS")
    print("=" * 60)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Validation {val_metric_name.lower()}: {val_metric:.3f}")

    print("\nValidation class counts:")
    print(val_counts)

    print("\nValidation confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_val_true, y_val_pred))

    print("\nValidation classification report:")
    print(classification_report(
        y_val_true,
        y_val_pred,
        target_names=["Non-Wave", "Wave"]
    ))

    print("\nTIMING")
    print("=" * 60)
    print(f"Training time            : {train_time_sec:.3f} sec")
    print(f"Train inference time     : {train_infer_time:.3f} sec")
    print(f"Validation inference time: {val_infer_time:.3f} sec")

    print("\nCHUNKING CHOICE")
    print("=" * 60)
    print(
        f"Extracted {CHUNK_LEN}-bin chunks with overlap {CHUNK_OVERLAP} "
        f"(step {CHUNK_STEP})."
    )
    print(
        "If a trial had fewer than 8 time bins, it was padded by repeating the last frame "
        "until length 8, so every trial contributed at least one sample."
    )
    print(
        "If a trial had more time bins, all overlapping chunks were extracted, "
        "so longer trials contributed more samples."
    )
    print(
        "Training chunks were balanced across classes by random undersampling of the majority class. "
        "Validation chunks were left unchanged."
    )
    
#%% ANOTHER VERSION OF ABOVE
    

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
VAL_FRACTION = 0.2

ONE_BASED_LABELS = True   # idx 0 -> trial 1 -> wave
WINDOW = 8
STEP = 2                  # stride 2 => overlap 6

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def unwrap_item(x):
    """
    Handle nested list items.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], np.ndarray):
        return x[0]
    return np.asarray(x)


def ensure_THW(arr):
    """
    Convert to (T,H,W), complex.
    Accepts:
      (T,H,W)
      (1,T,H,W)
      (T,1,H,W)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[1] == 1:
            return arr[:, 0]

    raise ValueError(f"Expected (T,H,W), (1,T,H,W), or (T,1,H,W), got {arr.shape}")


def is_wave_trial(idx, one_based_labels=True):
    """
    odd trials = wave, even trials = non-wave
    If idx 0 is human trial 1, then 0,2,4,... are wave.
    """
    if one_based_labels:
        return (idx % 2 == 0)
    else:
        return (idx % 2 == 1)


def phase_normalize(z):
    return z / (np.abs(z) + 1e-8)


def sliding_windows_last_included(tmp, win=8, step=2):
    """
    tmp: (T,H,W), complex

    Returns:
      windows: (Nwin, win, H, W)
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if tmp.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got {tmp.shape}")

    T = tmp.shape[0]
    if T < win:
        return None

    # shape: (T-win+1, H, W, win)
    w = sliding_window_view(tmp, window_shape=win, axis=0)

    # -> (num_windows, win, H, W)
    w = np.transpose(w, (0, 3, 1, 2))

    # stride by STEP
    w_step = w[::step]

    # include last window if not already included
    last_window = w[-1:]
    if len(w_step) == 0:
        w_step = last_window
    else:
        if not np.array_equal(w_step[-1], last_window[0]):
            w_step = np.concatenate([w_step, last_window], axis=0)

    return w_step


def equal_sample_count(X_data, y_data):
    """
    Balance classes by undersampling majority class.
    """
    rng = np.random.default_rng(SEED)

    y_data = np.asarray(y_data).reshape(-1)
    idx0 = np.where(y_data == 0)[0]
    idx1 = np.where(y_data == 1)[0]

    n_keep = min(len(idx0), len(idx1))

    idx0_keep = rng.choice(idx0, size=n_keep, replace=False)
    idx1_keep = rng.choice(idx1, size=n_keep, replace=False)

    keep = np.concatenate([idx0_keep, idx1_keep])
    rng.shuffle(keep)

    return X_data[keep], y_data[keep]


# ============================================================
# SPLIT TRIALS
# ============================================================

def split_trial_indices_from_X(X, val_fraction=0.2, seed=42, one_based_labels=True):
    """
    Stratified split based on odd/even trial identity.
    """
    n_trials = len(X)
    indices = np.arange(n_trials)
    labels = np.array(
        [1 if is_wave_trial(i, one_based_labels=one_based_labels) else 0 for i in indices],
        dtype=np.int64
    )

    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )

    return np.array(sorted(idx_train)), np.array(sorted(idx_val))


# ============================================================
# BUILD DATASET FROM LIST-OF-LISTS
# ============================================================

def build_chunk_dataset_from_X(X, trial_indices, balance=False, one_based_labels=True,
                               phase_only=True, win=8, step=2):
    """
    X is list of trials.
    Each X[idx] is a list of arrays.
    Each array is (T,H,W) complex (or convertible to that).

    Returns:
      X_chunks: (N, win, H, W) complex
      y_chunks: (N,)
    """
    X_chunks = []
    y_chunks = []

    for idx in trial_indices:
        label = 1 if is_wave_trial(idx, one_based_labels=one_based_labels) else 0

        trial_list = X[idx]
        if not isinstance(trial_list, (list, tuple)):
            trial_list = [trial_list]

        for item in trial_list:
            arr = unwrap_item(item)
            arr = ensure_THW(arr)

            if arr.shape[0] < win:
                continue

            if phase_only:
                arr = phase_normalize(arr)

            windows = sliding_windows_last_included(arr, win=win, step=step)
            if windows is None:
                continue

            X_chunks.extend(list(windows))
            y_chunks.extend([label] * windows.shape[0])

    X_chunks = np.array(X_chunks)
    y_chunks = np.array(y_chunks, dtype=np.float32)

    if balance:
        X_chunks, y_chunks = equal_sample_count(X_chunks, y_chunks)

    return X_chunks, y_chunks


# ============================================================
# DATASET
# ============================================================

class ComplexChunkDataset(Dataset):
    """
    Input:
      X_complex: (N, T, H, W) complex
    Converted to:
      (N, 2, T, H, W) real
    """
    def __init__(self, X_complex, y):
        xr = X_complex.real.astype(np.float32)
        xi = X_complex.imag.astype(np.float32)

        self.X = np.stack([xr, xi], axis=1)  # (N,2,T,H,W)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ============================================================
# MODEL
# ============================================================

class PhaseWaveCNN3D_Stronger(nn.Module):
    """
    6 Conv3D layers, no pooling, MLP head.
    """
    def __init__(self, input_shape=(2, 8, 11, 23), dropout=0.30):
        super().__init__()

        c, t, h, w = input_shape

        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 24, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(24, 32, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv4 = nn.Conv3d(32, 40, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(40, 48, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(48, 24, kernel_size=1, stride=1, padding=0)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, c, t, h, w)
            z = self._forward_conv(dummy)
            feat_dim = z.numel()

        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def _forward_conv(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ============================================================
# TRAINING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=6):
        self.patience = patience
        self.best_val_acc = -1.0
        self.counter = 0
        self.best_state = None

    def step(self, val_acc, model):
        improved = val_acc > self.best_val_acc
        if improved:
            self.best_val_acc = val_acc
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


def evaluate_binary_model(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0
    all_y = []
    all_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            preds = (torch.sigmoid(logits) >= 0.5).float()

            total_loss += loss.item() * x.shape[0]
            total_n += x.shape[0]

            all_y.append(y.cpu().numpy().reshape(-1))
            all_pred.append(preds.cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    counts = np.bincount(y_true.astype(int), minlength=2)
    if counts[0] == counts[1]:
        metric_name = "accuracy"
        metric_value = accuracy_score(y_true, y_pred)
    else:
        metric_name = "balanced_accuracy"
        metric_value = balanced_accuracy_score(y_true, y_pred)

    return total_loss / total_n, metric_name, metric_value, y_true, y_pred


def train_phase_model(model, Xtrain, labels_train, Xval, labels_val,
                      epochs=20, batch_size=128, lr=1e-3, device="cuda"):

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    train_ds = ComplexChunkDataset(Xtrain, labels_train)
    val_ds = ComplexChunkDataset(Xval, labels_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    early_stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()

            train_loss_sum += loss.item() * x.shape[0]
            train_correct += (preds == y).sum().item()
            train_n += x.shape[0]

        train_loss = train_loss_sum / train_n
        train_acc = train_correct / train_n

        val_loss, val_metric_name, val_metric, _, _ = evaluate_binary_model(model, val_loader, device)

        stop = early_stopper.step(val_metric, model)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val {val_metric_name}: {val_metric:.4f}"
        )

        if stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(early_stopper.best_state)
    return model, early_stopper.best_val_acc


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Split trials from X directly
    # --------------------------------------------------------
    idx_train, idx_val = split_trial_indices_from_X(
        X,
        val_fraction=VAL_FRACTION,
        seed=SEED,
        one_based_labels=ONE_BASED_LABELS
    )

    print("Train trials:", len(idx_train))
    print("Val trials  :", len(idx_val))

    # --------------------------------------------------------
    # Build chunk datasets from list-of-lists X
    # --------------------------------------------------------
    Xtrain_phase, labels_train = build_chunk_dataset_from_X(
        X,
        idx_train,
        balance=True,
        one_based_labels=ONE_BASED_LABELS,
        phase_only=True,
        win=WINDOW,
        step=STEP
    )

    Xval_phase, labels_val = build_chunk_dataset_from_X(
        X,
        idx_val,
        balance=True,
        one_based_labels=ONE_BASED_LABELS,
        phase_only=True,
        win=WINDOW,
        step=STEP
    )

    print("Xtrain_phase shape:", Xtrain_phase.shape)
    print("Xval_phase shape  :", Xval_phase.shape)
    print("Train class counts:", np.bincount(labels_train.astype(int)))
    print("Val class counts  :", np.bincount(labels_val.astype(int)))

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = PhaseWaveCNN3D_Stronger(
        input_shape=(2, Xtrain_phase.shape[1], Xtrain_phase.shape[2], Xtrain_phase.shape[3]),
        dropout=0.30
    )

    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    model, best_val_metric = train_phase_model(
        model,
        Xtrain_phase,
        labels_train,
        Xval_phase,
        labels_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE
    )

    print("Best validation metric:", best_val_metric)

    # --------------------------------------------------------
    # Final validation report
    # --------------------------------------------------------
    val_ds = ComplexChunkDataset(Xval_phase, labels_val)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    val_loss, val_metric_name, val_metric, y_true, y_pred = evaluate_binary_model(model, val_loader, DEVICE)

    print("\nFINAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val {val_metric_name}: {val_metric:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Wave", "Wave"]))


