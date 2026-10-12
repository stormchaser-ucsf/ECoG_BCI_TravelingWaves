# %% 1. CONFIGURATION — Ctrl+Enter
# CPU ONLY; no CUDA tensors or CUDA operations are used.
import os
try:
    import psutil
    PHYSICAL_CORES = psutil.cpu_count(logical=False) or 8
except ImportError:
    PHYSICAL_CORES = max(1, (os.cpu_count() or 8)//2)

MAT_PATH = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/mu_wave_trial_data_labeled_B3.mat'
MAT_KEY = 'mu_wave_trial_data_labeled_B3'
OUTPUT_DIR = '/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/CNN3D/CNN3D_Res_augmented'

# Train each model for at most 5 epochs. For one model only, e.g. ('complex',).
MODEL_TYPES = ('joint', 'complex')  # optionally include 'dual' (original two independent branches)
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 6
CPU_THREADS = PHYSICAL_CORES
CPU_INTEROP_THREADS = 2
NUM_WORKERS = 0             # Spyder-safe; PyTorch kernels parallelize over CPU cores
SEED = 42

WINDOW_MS = 120            # try 120, 160, 200 in separate runs; 50 Hz -> 6,8,10 samples
TEST_STEP = 1              # 20 ms stride, overlapping test windows
TRAIN_STEP = None         # None => WINDOW samples (nonoverlapping training windows)
INCLUDE_TRANSITIONS = True  # majority-vote labels for mixed training windows
USE_WEIGHTED_LOSS = True    # keep *all* nonwave training data; pos_weight = negatives/positives
# If False, training is globally balanced by undersampling; not recommended with weighted loss.

AUGMENT = True
GLOBAL_PHASE_ROTATION = True
AMPLITUDE_RANGE = (0.8, 1.2)
COMPLEX_NOISE_REL_RMS = 0.02
# Phase is rotated uniformly for the entire window, preserving phase gradients.

THRESHOLD = 0.2
N_PLOTS = 10
PLOT_SHOW = False         # True: open figures in Spyder; False: save PNGs only

# %% 2. COMPLETE PIPELINE — Ctrl+Enter after running configuration
import copy
import csv
import random
from pathlib import Path

import mat73
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

FS = 50
WINDOW = int(round(WINDOW_MS * FS / 1000))
assert WINDOW >= 3 and abs(WINDOW * 1000/FS - WINDOW_MS) < 1e-8
TRAIN_STEP = WINDOW if TRAIN_STEP is None else TRAIN_STEP
DEVICE = torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_trials(path, key):
    print('Loading MATLAB v7.3 struct array...')
    raw = mat73.loadmat(path)[key]
    if isinstance(raw, dict):
        entries = zip(raw['mu'], raw['labels'], raw['CL'])
    elif isinstance(raw, list):
        entries = ((x['mu'], x['labels'], x['CL']) for x in raw)
    else:
        raise TypeError(f'Unexpected mat73 representation: {type(raw)}')
    trials = []
    for i, (mu, y, cl) in enumerate(entries):
        mu = np.asarray(mu)
        y = np.asarray(y).reshape(-1).astype(np.int64)
        cl = int(np.asarray(cl).reshape(-1)[0])
        if mu.ndim != 3 or mu.shape[1:] != (11, 23) or mu.shape[0] != len(y):
            raise ValueError(f'Trial {i}: mu={mu.shape}, labels={y.shape}; expected (T,11,23)/(T,)')
        if not np.iscomplexobj(mu) or not np.isin(y, [0, 1]).all():
            raise ValueError(f'Trial {i}: expected complex mu and binary labels')
        trials.append({'mu': np.asarray(mu, dtype=np.complex64), 'labels': y, 'CL': cl})
    print(f'Loaded {len(trials)} trials, first shape={trials[0]["mu"].shape}')
    return trials


def split_trials(trials):
    ids = np.arange(len(trials))
    cl = np.asarray([t['CL'] for t in trials])
    train, rem = train_test_split(ids, test_size=.30, random_state=SEED, stratify=cl)
    valid, test = train_test_split(rem, test_size=.50, random_state=SEED, stratify=cl[rem])
    print(f'Trial split: training={len(train)}, validation={len(valid)}, test={len(test)}')
    return train, valid, test


def index_windows(trials, trial_ids, stride, transitions):
    """Rows: (trial_id, start, label), without materializing huge window arrays.

    Training: nonoverlap, optionally include mixed windows.
    Validation: same labeling rule as training, without augmentation.
    """
    records = []
    for idx in trial_ids:
        labels = trials[int(idx)]['labels']
        if transitions:
            starts = range(0, len(labels)-WINDOW+1, stride)
            for s in starts:
                n = int(labels[s:s+WINDOW].sum())
                records.append((int(idx), s, int(n > WINDOW/2)))
        else:
            # Align to each contiguous run start, like the user's original implementation.
            edges = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
            for a, b in zip(edges[:-1], edges[1:]):
                for s in range(int(a), int(b)-WINDOW+1, stride):
                    records.append((int(idx), s, int(labels[a])))
    return np.asarray(records, dtype=np.int64).reshape(-1, 3)


def estimate_scale(trials, records, max_windows=10000):
    # Training-only global RMS; fixed for validation/test. Estimate on a random subset.
    rng = np.random.default_rng(SEED)
    ids = rng.choice(len(records), size=min(max_windows, len(records)), replace=False)
    ss, n = 0.0, 0
    for i in ids:
        tid, start, _ = records[int(i)]
        z = trials[int(tid)]['mu'][int(start):int(start)+WINDOW]
        ss += np.sum(np.abs(z.astype(np.complex128))**2)
        n += z.size
    scale = float(np.sqrt(ss/n))
    if not np.isfinite(scale) or scale < 1e-10:
        raise ValueError('Invalid training amplitude scale')
    return scale


class MuWindowDataset(Dataset):
    def __init__(self, trials, records, scale, augment=False):
        self.trials, self.records, self.scale, self.augment = trials, records, scale, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        tid, start, label = self.records[index]
        z = self.trials[int(tid)]['mu'][int(start):int(start)+WINDOW] / self.scale
        if self.augment:
            if GLOBAL_PHASE_ROTATION:
                theta = np.random.uniform(-np.pi, np.pi)
                z = z * np.complex64(np.exp(1j*theta))
            if AMPLITUDE_RANGE is not None:
                z = z * np.random.uniform(*AMPLITUDE_RANGE)
            if COMPLEX_NOISE_REL_RMS > 0:
                sigma = COMPLEX_NOISE_REL_RMS * np.sqrt(np.mean(np.abs(z)**2))
                noise = (np.random.normal(size=z.shape) + 1j*np.random.normal(size=z.shape)) / np.sqrt(2)
                z = z + sigma*noise
        x = np.stack([z.real, z.imag], axis=0).astype(np.float32)  # (2,T,11,23)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(float(label), dtype=torch.float32)


def block(ni, no):
    return nn.Sequential(nn.Conv3d(ni, no, 3, padding=1), nn.BatchNorm3d(no), nn.GELU())


class RealBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 24), block(24, 48), nn.MaxPool3d((1,2,2)),
            block(48, 88), block(88, 112), nn.AdaptiveAvgPool3d((1,2,2)),
            nn.Flatten(), nn.Linear(448,192), nn.GELU(), nn.Dropout(.3))

    def forward(self, x):
        return self.features(x)


class DualBranchCNN(nn.Module):
    """Original independent real and imaginary spatial branches."""
    def __init__(self):
        super().__init__()
        self.real_net, self.imag_net = RealBranch(), RealBranch()
        self.head = nn.Sequential(nn.Linear(384,96),nn.GELU(),nn.Dropout(.3),nn.Linear(96,1))

    def forward(self, x):
        a, b = self.real_net(x[:,0:1]), self.imag_net(x[:,1:2])
        return self.head(torch.cat([a,b],1)).squeeze(1)


class JointRealImagCNN(nn.Module):
    """Real-valued CNN that mixes real/imaginary components from first layer."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            block(2,32),block(32,64),nn.MaxPool3d((1,2,2)),
            block(64,112),block(112,144),nn.AdaptiveAvgPool3d((1,2,2)),
            nn.Flatten(),nn.Linear(576,192),nn.GELU(),nn.Dropout(.3))
        self.head = nn.Sequential(nn.Linear(192,96),nn.GELU(),nn.Linear(96,1))

    def forward(self,x):
        return self.head(self.features(x)).squeeze(1)


class ComplexConv3d(nn.Module):
    """True complex convolution with tied real/imaginary kernels:
    (Wr+iWi)*(xr+ixi) = (Wr*xr-Wi*xi)+i(Wr*xi+Wi*xr).
    Input/output are stacked real and imaginary tensors (B,2,C,T,H,W).
    No convolution biases => global phase equivariance.
    """
    def __init__(self, ni, no):
        super().__init__()
        self.wr = nn.Conv3d(ni,no,3,padding=1,bias=False)
        self.wi = nn.Conv3d(ni,no,3,padding=1,bias=False)

    def forward(self, x):
        xr, xi = x[:,0],x[:,1]
        yr = self.wr(xr)-self.wi(xi)
        yi = self.wr(xi)+self.wi(xr)
        return torch.stack((yr,yi),dim=1)


class ComplexModReLU(nn.Module):
    """Magnitude-gated complex activation; preserves the complex phase."""
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1,channels,1,1,1))

    def forward(self, z):
        mag = torch.sqrt(z[:,0].square()+z[:,1].square()+1e-8)
        gain = F.relu(mag+self.bias)/(mag+1e-6)
        return z * gain.unsqueeze(1)


class ComplexBlock(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.conv = ComplexConv3d(ni,no)
        self.act = ComplexModReLU(no)

    def forward(self,x):
        return self.act(self.conv(x))


class ComplexWaveCNN(nn.Module):
    """Actual complex-valued 3D convolutions and complex nonlinearities.
    Magnitude of complex pooled output -> real-valued classification head.
    This yields global phase invariance in exact arithmetic.
    """
    def __init__(self):
        super().__init__()
        self.c1 = ComplexBlock(1,12)
        self.c2 = ComplexBlock(12,24)
        self.c3 = ComplexBlock(24,40)
        self.c4 = ComplexBlock(40,56)
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(56*2*2,128),nn.GELU(),
                                  nn.Dropout(.3),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,x):
        # (B,2,T,H,W) -> (B,2,1,T,H,W)
        z = x.unsqueeze(2)
        z = self.c1(z)
        z = self.c2(z)
        z = torch.stack([F.avg_pool3d(z[:,j],(1,2,2)) for j in (0,1)],dim=1)
        z = self.c3(z)
        z = self.c4(z)
        z = torch.stack([F.adaptive_avg_pool3d(z[:,j],(1,2,2)) for j in (0,1)],dim=1)
        magnitude = torch.sqrt(z[:,0].square()+z[:,1].square()+1e-8)
        return self.head(magnitude).squeeze(1)


ARCHITECTURES = {'dual':DualBranchCNN,'joint':JointRealImagCNN,'complex':ComplexWaveCNN}


def fit_model(model, train_loader, valid_loader, pos_weight, tag):
    model.to(DEVICE)
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    criterion_val = nn.BCEWithLogitsLoss()  # validation natural distribution; unweighted BCE
    opt = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-3)
    best, best_state, waiting = float('inf'), None, 0
    for epoch in range(1,EPOCHS+1):
        model.train()
        running, count = 0.,0
        for x,y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion_train(model(x),y)
            loss.backward()
            opt.step()
            running += loss.item()*len(y)
            count += len(y)
        model.eval()
        vl, vc, vn = 0.,0,0
        with torch.no_grad():
            for x,y in valid_loader:
                logits = model(x)
                vl += criterion_val(logits,y).item()*len(y)
                vc += ((logits>=0).float()==y).sum().item()
                vn += len(y)
        vl /= vn
        print(f'{tag:8s} epoch {epoch:02d}/{EPOCHS} | weighted train BCE={running/count:.4f} '
              f'| val BCE={vl:.4f} | val accuracy@0.5={vc/vn:.2%}', flush=True)
        if vl < best - 1e-5:
            best,best_state,waiting = vl,copy.deepcopy(model.state_dict()),0
        else:
            waiting += 1
            if waiting >= PATIENCE:
                print(f'Patience={PATIENCE}; early stop at epoch {epoch}')
                break
    model.load_state_dict(best_state)
    return model


def test_one_trial(model, trial, scale):
    y = trial['labels']
    starts = np.arange(0,len(y)-WINDOW+1,TEST_STEP,dtype=int)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0,len(starts),BATCH_SIZE):
            s_batch = starts[i:i+BATCH_SIZE]
            Z = np.stack([trial['mu'][s:s+WINDOW] for s in s_batch])/scale
            x = np.stack((Z.real,Z.imag),axis=1).astype(np.float32)
            logits = model(torch.from_numpy(np.ascontiguousarray(x)))
            probs.extend(torch.sigmoid(logits).numpy().tolist())
    counts = np.convolve(y,np.ones(WINDOW,dtype=int),mode='valid')[starts]
    truth = (counts > WINDOW/2).astype(np.int64)
    pure = (counts==0)|(counts==WINDOW)
    pred = (np.asarray(probs)>=THRESHOLD).astype(np.int64)
    balanced = balanced_accuracy_score(truth,pred) if np.unique(truth).size == 2 else float('nan')
    return dict(truth=truth,pred=pred,probability=np.asarray(probs),starts=starts,pure=pure,
                accuracy=float((pred==truth).mean()),balanced_accuracy=float(balanced))


def plot_trial(trial,result,model_name,trial_id,path):
    t = np.arange(len(trial['labels']))/FS
    centers = (result['starts']+(WINDOW-1)/2)/FS
    fig,axs = plt.subplots(2,1,figsize=(14,5.5),sharex=True)
    axs[0].step(t,trial['labels'],where='post',c='k',lw=1.6,label='Original sample ground truth')
    axs[0].step(centers,result['pred'],where='mid',c='tab:red',lw=1.2,label='CNN prediction')
    axs[0].set_ylim(-.1,1.1);axs[0].set_ylabel('Wave state');axs[0].legend(loc='upper right')
    axs[1].plot(centers,result['probability'],c='tab:blue',lw=1.2)
    axs[1].axhline(THRESHOLD,c='tab:red',ls='--',label=f'Threshold {THRESHOLD}')
    axs[1].set_ylim(-.05,1.05);axs[1].set_ylabel('P(wave)');axs[1].set_xlabel('Time (s)')
    axs[1].legend(loc='upper right')
    fig.suptitle(f'{model_name} | trial {trial_id} | CL={trial["CL"]} | '
                 f'acc={result["accuracy"]:.1%}, balanced acc={result["balanced_accuracy"]:.1%}')
    fig.tight_layout()
    fig.savefig(path,dpi=170)
    if PLOT_SHOW: plt.show()
    plt.close(fig)


def main():
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(CPU_INTEROP_THREADS)
    except RuntimeError:
        print('Spyder inter-op CPU threads already initialized; continuing.')
    seed_everything(SEED)
    print(f'CPU only; intra-op threads={torch.get_num_threads()}, window={WINDOW} samples ({WINDOW_MS}ms)')
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True,exist_ok=True)
    trials = load_trials(MAT_PATH,MAT_KEY)
    train_ids, valid_ids, test_ids = split_trials(trials)
    train_records = index_windows(trials,train_ids,TRAIN_STEP,INCLUDE_TRANSITIONS)
    valid_records = index_windows(trials,valid_ids,WINDOW,INCLUDE_TRANSITIONS)
    if not len(train_records) or not len(valid_records):
        raise ValueError('No training or validation windows')
    labels = train_records[:,2]
    counts = np.bincount(labels,minlength=2)
    print('Train labels before sampling:',counts)
    if np.any(counts==0): raise ValueError('Training data must contain both classes')
    if not USE_WEIGHTED_LOSS:
        rng = np.random.default_rng(SEED)
        n = min(counts)
        selected = np.r_[rng.choice(np.flatnonzero(labels==0),n,replace=False),
                         rng.choice(np.flatnonzero(labels==1),n,replace=False)]
        rng.shuffle(selected)
        train_records = train_records[selected]
        counts = np.bincount(train_records[:,2],minlength=2)
    pos_weight = float(counts[0]/counts[1]) if USE_WEIGHTED_LOSS else 1.0
    print('Training counts:',counts,'| training positive weight:',pos_weight)
    print('Validation windows:',len(valid_records),'counts=',np.bincount(valid_records[:,2],minlength=2))
    scale = estimate_scale(trials,train_records)
    train_loader = DataLoader(MuWindowDataset(trials,train_records,scale,augment=AUGMENT),
                              batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    valid_loader = DataLoader(MuWindowDataset(trials,valid_records,scale,augment=False),
                              batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

    all_results = {}
    all_models = {}
    summary_rows = []
    # Fixed visualization trials across architectures.
    viz = np.random.default_rng(SEED).choice(test_ids,size=min(N_PLOTS,len(test_ids)),replace=False)
    for model_name in MODEL_TYPES:
        if model_name not in ARCHITECTURES:
            raise ValueError(f'Unknown model {model_name}; options={tuple(ARCHITECTURES)}')
        seed_everything(SEED)
        model = ARCHITECTURES[model_name]()
        nparams = sum(p.numel() for p in model.parameters())
        print(f'\n=== {model_name.upper()} CNN | trainable parameters = {nparams:,} ===',flush=True)
        model = fit_model(model,train_loader,valid_loader,pos_weight,model_name)
        all_models[model_name] = model
        model_dir = out/model_name
        model_dir.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.state_dict(),'architecture':model_name,'window':WINDOW,
                    'scale':scale,'train_ids':train_ids,'valid_ids':valid_ids,
                    'test_ids':test_ids,'threshold':THRESHOLD},model_dir/'best_model.pt')
        results = []
        for j, tid in enumerate(test_ids):
            tid = int(tid)
            if len(trials[tid]['labels']) < WINDOW: continue
            r = test_one_trial(model,trials[tid],scale)
            r.update(trial_id=tid,CL=int(trials[tid]['CL']))
            results.append(r)
            summary_rows.append({'model':model_name,'trial_id':tid,'CL':r['CL'],
                                 'accuracy':r['accuracy'],'balanced_accuracy':r['balanced_accuracy']})
            if (j+1)%50 == 0:
                print(f'{model_name}: tested {j+1}/{len(test_ids)} trials',flush=True)
            if tid in viz:
                plot_trial(trials[tid],r,model_name,tid,model_dir/f'test_trial_{tid:04d}.png')
        all_results[model_name]=results
        np.savez_compressed(model_dir/'test_predictions.npz',results=np.asarray(results,dtype=object))
        yt = np.concatenate([r['truth'] for r in results])
        yp = np.concatenate([r['pred'] for r in results])
        pure = np.concatenate([r['pure'] for r in results])
        prec,rec,f1,_ = precision_recall_fscore_support(yt,yp,average='binary',zero_division=0)
        print(f'\n{model_name} @ threshold={THRESHOLD}: accuracy={(yt==yp).mean():.2%}; '
              f'balanced_accuracy={balanced_accuracy_score(yt,yp):.2%}; '
              f'precision={prec:.2%}; recall={rec:.2%}; F1={f1:.2%}')
        print('Pure-window accuracy:',f'{(yt[pure]==yp[pure]).mean():.2%}' if pure.any() else 'n/a')
        print('Transition-window accuracy:',f'{(yt[~pure]==yp[~pure]).mean():.2%}' if (~pure).any() else 'n/a')
        for cl in (0,1):
            rs = [r for r in results if r['CL']==cl]
            if rs:
                yy = np.concatenate([r['truth'] for r in rs]); pp=np.concatenate([r['pred'] for r in rs])
                print(f'CL={cl}: acc={(yy==pp).mean():.2%}, balanced_acc={balanced_accuracy_score(yy,pp):.2%}')
    with open(out/'per_trial_metrics.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','trial_id','CL','accuracy','balanced_accuracy'])
        writer.writeheader();writer.writerows(summary_rows)
    print('\nSaved models, trials, plots, and accuracy CSV to',out)
    return all_models, trials, test_ids, scale, DEVICE, all_results

# Globals exposed for subsequent Spyder cells (e.g. changing the probability threshold).
models, trials, test, scale, device, results_by_model = main()

# %% 3. RE-PLOT / RE-THRESHOLD EXISTING TEST PROBABILITIES (no retraining)
# Run this AFTER Cell 2. Changing TEST_THRESHOLD requires neither training nor inference.
TEST_THRESHOLD = 0.2
for model_name, model_results in results_by_model.items():
    plot_dir = Path(OUTPUT_DIR)/model_name/f'rethreshold_{TEST_THRESHOLD:.2f}'
    plot_dir.mkdir(parents=True,exist_ok=True)
    for r in model_results[:N_PLOTS]:
        r2 = dict(r)
        r2['pred'] = (r['probability']>=TEST_THRESHOLD).astype(np.int64)
        r2['accuracy'] = float((r2['pred']==r2['truth']).mean())
        r2['balanced_accuracy'] = (float(balanced_accuracy_score(r2['truth'],r2['pred']))
                                   if np.unique(r2['truth']).size==2 else float('nan'))
        # plot_trial uses global THRESHOLD for reference line. Temporarily set it for plotting.
        old_threshold=THRESHOLD
        THRESHOLD=TEST_THRESHOLD
        plot_trial(trials[r2['trial_id']],r2,model_name,r2['trial_id'],
                   plot_dir/f'trial_{r2["trial_id"]:04d}.png')
        THRESHOLD=old_threshold
