

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



# %% 1. CONFIGURATION — edit paths/settings; run this cell once (Ctrl+Enter)
# For Linux, e.g. MAT_PATH = "/home/username/data/B3.mat"
MAT_PATH = "/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/mu_wave_trial_data_labeled_B3.mat"
MAT_KEY = "mu_wave_trial_data_labeled_B3"
OUTPUT_DIR = "/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/CNN3D/CNN3D_Res"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
N_PLOTS = 5

# %% 2. COMPLETE PIPELINE — run this cell (Ctrl+Enter)
"""Trial-level 3D CNN classifier for complex mu traveling-wave data.

Dependencies: numpy scipy scikit-learn matplotlib torch
Optional for MATLAB v7.3 files: mat73

Example:
    python train_mu_wave_3dcnn.py --mat B3.mat
"""
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

FS = 50
WINDOW = 6                 # 120 ms at 50 Hz
TRAIN_STEP = WINDOW        # nonoverlapping within contiguous runs
TEST_STEP = 1              # 20 ms progression (100 ms overlap)
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matlab_field(obj, name):
    if isinstance(obj, dict):
        return obj[name]
    if isinstance(obj, np.void):
        return obj[name]
    return getattr(obj, name)


def load_trials(path, key):

    import mat73
    import numpy as np

    print("Loading MATLAB v7.3 file...")

    data = mat73.loadmat(path)

    raw = data[key]

    trials = []

    # MATLAB 1 x N struct arrays are typically represented
    # by mat73 as a dictionary of field lists.
    #
    # raw['mu']     -> list of T x 11 x 23 arrays
    # raw['labels'] -> list of T x 1 arrays
    # raw['CL']     -> list of scalar values

    if isinstance(raw, dict):

        mu_all = raw['mu']
        labels_all = raw['labels']
        cl_all = raw['CL']

        n_trials = len(mu_all)

        print(f"Found {n_trials} trials.")

        for i in range(n_trials):

            mu = np.asarray(mu_all[i])
            labels = np.asarray(labels_all[i]).reshape(-1)
            cl = int(np.asarray(cl_all[i]).reshape(-1)[0])

            # Verify dimensions
            assert mu.ndim == 3, (
                f"Trial {i}: Expected 3D mu, got {mu.shape}"
            )

            assert mu.shape[1:] == (11, 23), (
                f"Trial {i}: Unexpected electrode grid {mu.shape}"
            )

            assert mu.shape[0] == len(labels), (
                f"Trial {i}: Temporal dimension mismatch."
            )

            assert np.iscomplexobj(mu), (
                f"Trial {i}: Mu data is not complex."
            )

            trials.append({
                "mu": mu.astype(np.complex64),
                "labels": labels.astype(np.int64),
                "CL": cl
            })

    elif isinstance(raw, list):

        # Some mat73 versions return a list of dictionaries
        for i, item in enumerate(raw):

            mu = np.asarray(item['mu'])
            labels = np.asarray(item['labels']).reshape(-1)
            cl = int(np.asarray(item['CL']).reshape(-1)[0])

            assert mu.shape == (len(labels), 11, 23)

            trials.append({
                "mu": mu.astype(np.complex64),
                "labels": labels.astype(np.int64),
                "CL": cl
            })

    else:
        raise ValueError(
            f"Unexpected MATLAB structure type: {type(raw)}"
        )

    print(f"Successfully loaded {len(trials)} trials.")
    print("First trial mu shape:", trials[0]['mu'].shape)
    print("First trial labels shape:", trials[0]['labels'].shape)
    print("First trial CL:", trials[0]['CL'])

    return trials


def split_trials(trials, seed):
    idx = np.arange(len(trials))
    strat = np.array([t['CL'] for t in trials])

    def split(a, test_size, stratify):
        try:
            return train_test_split(a, test_size=test_size, random_state=seed,
                                    stratify=stratify)
        except ValueError:
            print('Warning: insufficient trials for CL stratification; using random split')
            return train_test_split(a, test_size=test_size, random_state=seed)

    tr, other = split(idx, 0.30, strat)
    val, test = split(other, 0.50, strat[other])
    return np.asarray(tr), np.asarray(val), np.asarray(test)


def contiguous_run_windows(trial):
    """Extract disjoint, pure-state (6,11,23) windows inside each label run."""
    y, x = trial['labels'], trial['mu']
    edges = np.r_[0, np.flatnonzero(np.diff(y.astype(int)) != 0) + 1, len(y)]
    X, Y = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        for start in range(a, b - WINDOW + 1, TRAIN_STEP):
            X.append(x[start:start + WINDOW])
            Y.append(int(y[a]))
    if not X:
        return np.empty((0, WINDOW, 11, 23), dtype=np.complex64), np.empty(0, dtype=np.int64)
    return np.stack(X), np.asarray(Y, dtype=np.int64)


def collect_run_windows(trials, indices):
    pairs = [contiguous_run_windows(trials[i]) for i in indices]
    xs = [x for x, _ in pairs if len(x)]
    ys = [y for _, y in pairs if len(y)]
    if not xs:
        raise ValueError('No 120 ms contiguous windows in selected trials')
    return np.concatenate(xs), np.concatenate(ys)


def balance_training(X, y, rng):
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n = min(len(idx0), len(idx1))
    if n == 0:
        raise ValueError('Training must contain both classes')
    chosen = np.r_[rng.choice(idx0, n, replace=False), rng.choice(idx1, n, replace=False)]
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def sliding_windows(trial):
    """1-sample test stride. Mixed windows are labeled by strict majority."""
    from numpy.lib.stride_tricks import sliding_window_view
    x, y = trial['mu'], trial['labels']
    starts = np.arange(len(y) - WINDOW + 1, step=TEST_STEP)
    # Views are indexed as (start, row, col, time); rearrange to (start,time,row,col).
    X = sliding_window_view(x, WINDOW, axis=0)[starts].transpose(0, 3, 1, 2)
    counts = np.convolve(y.astype(int), np.ones(WINDOW, dtype=int), mode='valid')[starts]
    Y = (counts > WINDOW / 2).astype(np.int64)  # 3/3 tie => nonwave
    pure = (counts == 0) | (counts == WINDOW)
    return X, Y, starts, pure


class Conv3DBranch(nn.Module):
    """True spatiotemporal convolutions over (time, electrode row, electrode col)."""
    def __init__(self):
        super().__init__()
        def block(ni, no):
            return nn.Sequential(
                nn.Conv3d(ni, no, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(no), nn.GELU()
            )
        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            block(128, 176),
            nn.AdaptiveAvgPool3d((1, 2, 2)),
            nn.Flatten(),
            nn.Linear(176 * 2 * 2, 384), nn.GELU(), nn.Dropout(0.30),
        )

    def forward(self, x):
        return self.features(x)


class ComplexMu3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.real_net = Conv3DBranch()
        self.imag_net = Conv3DBranch()
        self.classifier = nn.Sequential(
            nn.Linear(768, 128), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (batch, 2 [real/imag], 1 [conv3d channel], time, height, width)
        a = self.real_net(x[:, 0])
        b = self.imag_net(x[:, 1])
        return self.classifier(torch.cat([a, b], dim=1)).squeeze(-1)


def compute_scale(X):
    """Scalar RMS of complex amplitude; fit on training windows only."""
    scale = float(np.sqrt(np.mean(np.abs(X) ** 2)))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError('Invalid/near-zero training RMS')
    return scale


def to_tensor(X, scale, height=11, width=23):
    """(N,6,11,23) complex -> (N,2,1,6,11,23) real torch tensor."""
    if X.ndim != 4 or X.shape[1:] != (WINDOW, height, width):
        raise ValueError(f'Unexpected batch shape: {X.shape}')
    z = X / scale
    paired = np.stack([z.real, z.imag], axis=1).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(paired[:, :, None]))


def make_loader(X, y, scale, batch_size, shuffle):
    ds = TensorDataset(to_tensor(X, scale), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            total_correct += ((logits >= 0).float() == y).sum().item()
            count += len(y)
    return total_loss / count, total_correct / count


def predict_probabilities(model, X, scale, device, batch_size):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for p in range(0, len(X), batch_size):
            xx = to_tensor(X[p:p + batch_size], scale).to(device)
            probabilities.append(torch.sigmoid(model(xx)).cpu().numpy())
    return np.concatenate(probabilities)


def train_model(model, train_loader, val_loader, device, epochs, lr, patience):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    best_state = None
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f'Epoch {epoch:03d} | train BCE {total/n:.4f} | '
              f'val BCE {val_loss:.4f} | val acc {val_acc:.3%}')
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch}; best val BCE {best_loss:.4f}')
                break
    model.load_state_dict(best_state)
    return model


def plot_test_trial(trial, starts, y_true, y_pred, probs, trial_id, destination):
    """Window decisions are indexed by their center time (one decision per 20 ms)."""
    time = np.arange(len(trial['labels'])) / FS
    center_time = (starts + (WINDOW - 1) / 2) / FS
    fig, axs = plt.subplots(2, 1, figsize=(15, 5.5), sharex=True,
                            gridspec_kw={'height_ratios': [1.3, 1]})
    axs[0].step(time, trial['labels'], where='post', label='Ground truth (sample-level)',
                linewidth=1.6, color='black', alpha=0.65)
    axs[0].step(center_time, y_pred, where='mid', label='CNN prediction (120 ms)',
                linewidth=1.3, color='tab:orange')
    axs[0].set_ylabel('Wave label')
    axs[0].set_ylim(-0.15, 1.15)
    axs[0].legend(loc='upper right')
    axs[0].set_title(f'Test trial {trial_id}, CL={trial["CL"]}: '
                     f'window accuracy = {np.mean(y_true == y_pred):.1%}')
    axs[1].plot(center_time, probs, color='tab:blue', label='P(wave)')
    axs[1].axhline(0.5, color='black', ls='--', lw=1)
    axs[1].set_ylabel('P(wave)')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main(args):
    set_seed(SEED)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trials = load_trials(args.mat, args.key)
    tr, val, test = split_trials(trials, SEED)
    print(f'Trials total={len(trials)} train={len(tr)} val={len(val)} test={len(test)}')
    print(f'Channels={11 * 23}, device=',
          'cuda' if torch.cuda.is_available() else 'cpu')

    Xtrain, Ytrain = collect_run_windows(trials, tr)
    print('Training windows before balancing:', np.bincount(Ytrain, minlength=2))
    Xtrain, Ytrain = balance_training(Xtrain, Ytrain, np.random.default_rng(SEED))
    Xval, Yval = collect_run_windows(trials, val)
    print('Training windows after balancing:', np.bincount(Ytrain, minlength=2))
    print('Validation windows (natural class counts):', np.bincount(Yval, minlength=2))

    scale = compute_scale(Xtrain)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = make_loader(Xtrain, Ytrain, scale, args.batch_size, True)
    val_loader = make_loader(Xval, Yval, scale, args.batch_size, False)
    model = ComplexMu3DCNN()
    print(f'Parameter count: {sum(p.numel() for p in model.parameters()):,}')
    model = train_model(model, train_loader, val_loader, device,
                        args.epochs, args.lr, patience=6)

    results = []
    for trial_id in test:
        t = trials[trial_id]
        if len(t['labels']) < WINDOW:
            continue
        X, Y, starts, pure = sliding_windows(t)
        probs = predict_probabilities(model, X, scale, device, args.batch_size)
        pred = (probs >= 0.5).astype(np.int64)
        acc = np.mean(pred == Y)
        results.append(dict(trial_id=int(trial_id), CL=t['CL'], accuracy=float(acc),
                            n_windows=len(Y), truth=Y, pred=pred, pure=pure,
                            probability=probs, starts=starts))
        print(f'Test trial {trial_id:3d} CL={t["CL"]} windows={len(Y):5d} acc={acc:.2%}')

    correct = sum(np.sum(r['truth'] == r['pred']) for r in results)
    total = sum(r['n_windows'] for r in results)
    print(f'\nTEST WINDOW ACCURACY (all trials pooled): {correct / total:.2%}')
    all_truth = np.concatenate([r['truth'] for r in results])
    all_pred = np.concatenate([r['pred'] for r in results])
    all_pure = np.concatenate([r['pure'] for r in results])
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_truth, all_pred, average='binary', zero_division=0)
    print(f'Balanced accuracy={balanced_accuracy_score(all_truth, all_pred):.2%}; '
          f'wave precision={precision:.2%}, recall={recall:.2%}, F1={f1:.2%}')
    for mask, kind in [(all_pure, 'pure-state'), (~all_pure, 'transition')]:
        if mask.any():
            print(f'{kind} windows: accuracy={np.mean(all_truth[mask] == all_pred[mask]):.2%}, n={mask.sum()}')
    print(f'MEAN PER-TRIAL ACCURACY: {np.mean([r["accuracy"] for r in results]):.2%}')
    for cl in (0, 1):
        rs = [r for r in results if r['CL'] == cl]
        if rs:
            ncorrect = sum(np.sum(r['truth'] == r['pred']) for r in rs)
            n = sum(r['n_windows'] for r in rs)
            print(f'CL={cl}: {ncorrect / n:.2%} across {len(rs)} test trials')

    for r in results[:args.n_plots]:
        path = output / f'test_trial_{r["trial_id"]:03d}.png'
        plot_test_trial(trials[r['trial_id']], r['starts'], r['truth'], r['pred'],
                        r['probability'], r['trial_id'], path)
    torch.save(dict(model=model.state_dict(), scale=scale,
                    n_channels=11 * 23, height=11, width=23, fs=FS, window=WINDOW,
                    train_trials=tr, validation_trials=val, test_trials=test),
               output / 'best_mu_wave_cnn.pt')
    np.savez_compressed(output / 'test_predictions.npz', results=np.asarray(results, dtype=object))
    print(f'Saved plots, model and test predictions in {output}')


# Run this entire second cell with Ctrl+Enter after running the configuration cell.
from types import SimpleNamespace

args = SimpleNamespace(
    mat=MAT_PATH,
    key=MAT_KEY,
    output=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    n_plots=N_PLOTS,
)
main(args)
