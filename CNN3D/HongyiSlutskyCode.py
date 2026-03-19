
#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:06:52 2026

@author: user
"""

import os, numpy as np, pandas as pd
import scipy.io, scipy.signal as sp_signal
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 9, 'axes.titlesize': 10,
    'figure.dpi': 120, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── paths ──
#RAW_DIR = '/mnt/DataDrive/RobertGraspData/ECoG_force_isometric_share/raw'
RAW_DIR = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/SlutzkyData/raw/'
os.chdir(RAW_DIR)
OUT_DIR = os.path.join(os.path.dirname(os.getcwd()), 'results', 'mu_analysis')
os.makedirs(OUT_DIR, exist_ok=True)
print(f'Output: {OUT_DIR}')

# ── recordings ──
RECORDINGS = {
    'ED_R28': {'file': 'ED_forceTaskS001R28.mat',        'patient': 'ED', 'ecog_key': 'signal',           'ecog_cols': slice(0,16), 'force_col': 39},
    'ED_R29': {'file': 'ED_forceTaskS001R29.mat',        'patient': 'ED', 'ecog_key': 'signal',           'ecog_cols': slice(0,16), 'force_col': 39},
    'ED_R69': {'file': 'ED_forceTaskS001R69.mat',        'patient': 'ED', 'ecog_key': 'signal',           'ecog_cols': slice(0,16), 'force_col': 39},
    'ED_R70': {'file': 'ED_forceTaskS001R70.mat',        'patient': 'ED', 'ecog_key': 'signal',           'ecog_cols': slice(0,16), 'force_col': 39},
    'KC_R02': {'file': 'KC_ForceTaskS001R02_with_NK.mat','patient': 'KC', 'ecog_key': 'N_KsectionInterp', 'ecog_cols': slice(0,16), 'force_col': 39},
    'RM_R01': {'file': 'RM_ForceTaskS001R01_with_NK.mat','patient': 'RM', 'ecog_key': 'N_KsectionInterp', 'ecog_cols': slice(0,16), 'force_col': 39},
}

# ── mu sub-bands + beta for comparison ──
MU_BANDS = {
    'mu_7_9':   (7, 9),
    'mu_8_10':  (8, 10),
    'mu_10_13': (10, 13),
    'mu_8_13':  (8, 13),
}

# Beta uses 2 sub-bands per STANDARD_PIPELINE (13-30 Hz)
BETA_SUB = [(13, 19), (19, 30)]

# All bands for plotting (mu + beta)
ALL_BANDS = {**MU_BANDS, 'beta_13_30': (13, 30)}

# ── parameters ──
THRESH_FACTOR = 0.35
MIN_EVENT_SEC = 0.30
LONG_THR_S    = 2.5
FILTER_ORDER  = 4
SMOOTH_MS     = 100
N_BEST_CH     = 8

print('Config OK — 4 mu bands + beta:', list(ALL_BANDS.keys()))

#%%
# ── utility functions ──

from scipy.signal import hilbert

def estimate_fs(t, default=1000.0):
    if t is None or len(t) < 3 or not np.all(np.isfinite(t)):
        return default
    dt = np.median(np.diff(t))
    if dt <= 0:
        return default
    fs_est = float(np.round(1.0 / dt))
    return fs_est if np.isfinite(fs_est) and fs_est >= 400 else default

def load_recording(key, info):
    path = os.path.join(RAW_DIR, info['file'])
    mat = scipy.io.loadmat(path)
    signal = mat['signal'].astype(float)
    force_raw = signal[:, info['force_col']]
    if info['ecog_key'] == 'signal':
        subdural = signal[:, info['ecog_cols']]
    else:
        subdural = mat[info['ecog_key']][:, info['ecog_cols']].astype(float)
    t = np.ravel(mat.get('t', mat.get('t2', [None])))
    fs = estimate_fs(t if t[0] is not None else None)
    return {'key': key, 'patient': info['patient'], 'fs': fs,
            'n_samples': len(force_raw), 'force_raw': force_raw,
            'subdural': subdural}

def detect_events(force_raw, fs):
    force_norm = (force_raw - np.min(force_raw)) / (np.max(force_raw) - np.min(force_raw) + 1e-12)
    force_inv = 1.0 - force_norm
    w = max(3, int(round(fs * 0.10)))
    force_sm = np.convolve(force_inv, np.ones(w)/w, mode='same')
    p50, p95 = np.percentile(force_sm, [50, 95])
    th = p50 + THRESH_FACTOR * (p95 - p50)
    is_grasp = force_sm > th
    trans = np.diff(is_grasp.astype(int))
    starts = np.where(trans == 1)[0] + 1
    ends   = np.where(trans == -1)[0] + 1
    if is_grasp[0]:  starts = np.insert(starts, 0, 0)
    if is_grasp[-1]: ends = np.append(ends, len(is_grasp) - 1)
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    dur = (ends - starts) / fs
    k = dur >= MIN_EVENT_SEC
    return force_sm, th, starts[k], ends[k], dur[k]

def bandpower_perchannel(x, fs, band):
    """Bandpass → Hilbert envelope → smooth → log10, per channel.

    Pipeline:
    1. Butterworth bandpass (order 4, zero-phase)
    2. Hilbert transform → analytic signal → |analytic| = instantaneous amplitude
    3. Square amplitude → instantaneous power
    4. Smooth with 100 ms moving average
    5. log10 transform
    """
    f1, f2 = band
    if f2 >= fs / 2:
        return np.zeros((x.shape[0], x.shape[1]))
    sos = sp_signal.butter(FILTER_ORDER, [f1, f2], btype='bandpass', fs=fs, output='sos')
    xf = sp_signal.sosfiltfilt(sos, x, axis=0)
    # Hilbert envelope: analytic signal → absolute value = instantaneous amplitude
    analytic = hilbert(xf, axis=0)
    envelope = np.abs(analytic)           # instantaneous amplitude
    power = envelope ** 2                 # instantaneous power
    smooth_win = max(3, int(round(fs * SMOOTH_MS / 1000)))
    kernel = np.ones(smooth_win) / smooth_win
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=power)
    return np.log10(smoothed + 1e-12)

def subband_logpow_perchannel(x, fs, subbands):
    """Average Hilbert log-power across multiple sub-bands per channel (for beta)."""
    band_logs = []
    smooth_win = max(3, int(round(fs * SMOOTH_MS / 1000)))
    kernel = np.ones(smooth_win) / smooth_win
    for f1, f2 in subbands:
        if f2 >= fs / 2:
            continue
        sos = sp_signal.butter(FILTER_ORDER, [f1, f2], btype='bandpass', fs=fs, output='sos')
        xf = sp_signal.sosfiltfilt(sos, x, axis=0)
        analytic = hilbert(xf, axis=0)
        envelope = np.abs(analytic)
        power = envelope ** 2
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=power)
        band_logs.append(np.log10(smoothed + 1e-12))
    if not band_logs:
        return np.zeros((x.shape[0], x.shape[1]))
    return np.mean(np.array(band_logs), axis=0)

def select_channels_by_contrast(feature_perch, starts, ends, n_best):
    n_samples, n_channels = feature_perch.shape
    n_best = min(n_best, n_channels)
    grasp_mask = np.zeros(n_samples, dtype=bool)
    for s, e in zip(starts, ends):
        grasp_mask[int(s):int(e)] = True
    contrasts = np.array([
        np.mean(feature_perch[grasp_mask, ch]) - np.mean(feature_perch[~grasp_mask, ch])
        for ch in range(n_channels)
    ])
    best_idx = np.argsort(-np.abs(contrasts))[:n_best]
    return np.sort(best_idx), contrasts

def average_selected_channels(feature_perch, channel_idx):
    y = feature_perch[:, channel_idx].mean(axis=1)
    return (y - y.mean()) / (y.std() + 1e-12)


from scipy.signal import welch, find_peaks
from scipy.stats import zscore
import statsmodels.api as sm


def compute_spectral_peaks_1f(data, fs=1000):
    """
    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        Input time-series data.
    fs : float
        Sampling frequency.

    Returns
    -------
    spectral_peaks : list of dict
        One dict per channel, with keys:
            'freqs' : array of detected peak frequencies
            'pow'   : array of z-scored peak powers
        Empty entries are omitted for channels with len < 1024.
    stats_tmp : list
        p-value of slope term from the robust fit, for each processed channel.
    pow_freq : ndarray
        Welch PSD for each processed channel.
    ffreq : ndarray
        Frequency vector for each processed channel.
    osc_clus_tmp : ndarray or None
        Counts of peaks across channels for frequency bins 2..40 Hz
        using ±1 Hz neighborhood.
    """

    spectral_peaks = []
    stats_tmp = []
    pow_freq = []
    ffreq = []

    n_channels = data.shape[1]

    for ii in range(n_channels):
        x = data[:, ii]

        if len(x) >= 1024:
            # MATLAB: [Pxx,F] = pwelch(x,1024,512,1024,1e3);
            # In scipy, noverlap is directly overlap, so 512 matches MATLAB here.
            F, Pxx = welch(
                x,
                fs=fs,
                window='hamming',
                nperseg=1024,
                noverlap=512,
                nfft=1024
            )

            pow_freq.append(Pxx)
            ffreq.append(F)

            idx = (F > 0) & (F <= 40)
            F1 = F[idx]
            F1 = np.log2(F1)

            power_spect = Pxx[idx]
            power_spect = np.log2(power_spect)

            # Robust linear fit
            X = sm.add_constant(F1)
            rlm_model = sm.RLM(power_spect, X, M=sm.robust.norms.HuberT())
            rlm_results = rlm_model.fit()

            # Approximate p-values are not always as directly defined for RLM
            # as in MATLAB fitlm. If you specifically need p-values, OLS is simpler,
            # but this keeps the robust fit.
            try:
                stats_tmp.append(rlm_results.pvalues[1])
            except Exception:
                stats_tmp.append(np.nan)

            bhat = rlm_results.params
            yhat = X @ bhat
            
            # plt.figure();plt.plot(2**F1,power_spect)
            # plt.plot(2**F1,yhat)
            

            # Residual spectrum, z-scored
            power_spect = zscore(power_spect - yhat)
            
            # plt.figure();plt.plot(2**F1,power_spect)
            # plt.axhline(y=1,color='r')

            aa,bb = find_peaks(power_spect)     
            ff=2**F1
            # plt.vlines(ff[aa],-4,2,'r')
            
            peak_loc = aa[power_spect[aa] > 1]

            freqs = 2 ** F1[peak_loc]
            pow_vals = power_spect[peak_loc]

            spectral_peaks.append({
                "freqs": freqs,
                "pow": pow_vals
            })

    pow_freq = np.array(pow_freq, dtype=object)
    ffreq = np.array(ffreq, dtype=object)

    # getting oscillation clusters
    osc_clus_tmp = None
    if len(spectral_peaks) > 0:
        osc_clus_tmp = []
        for f in range(2, 41):
            ff = [f - 1, f + 1]
            tmp = 0
            ch_tmp = []

            for j in range(len(spectral_peaks)):
                freqs = spectral_peaks[j]["freqs"]
                for k in range(len(freqs)):
                    if ff[0] <= freqs[k] <= ff[1]:
                        tmp += 1
                        ch_tmp.append(j)

            osc_clus_tmp.append(tmp)

        osc_clus_tmp = np.array(osc_clus_tmp)

    return spectral_peaks, stats_tmp, pow_freq, ffreq, osc_clus_tmp

print('Functions defined — using Hilbert transform for power envelope.')

#%%

# ── load all recordings, detect events, compute mu + beta power ──

data = {}
for key, info in RECORDINGS.items():
    print(f'Loading {key} ... ', end='')
    d = load_recording(key, info)

    # detect events
    force_sm, th, starts, ends, dur = detect_events(d['force_raw'], d['fs'])
    d['force_smooth'] = force_sm
    d['threshold'] = th
    d['starts'], d['ends'], d['durations'] = starts, ends, dur
    long_mask = dur >= LONG_THR_S
    d['long_mask'] = long_mask

    # channel selection: use full mu (8-13) for picking best channels
    mu_full_perch = bandpower_perchannel(d['subdural'], d['fs'], MU_BANDS['mu_8_13'])
    best_ch, contrasts = select_channels_by_contrast(mu_full_perch, starts, ends, N_BEST_CH)
    d['best_channels'] = best_ch
    d['mu_contrasts'] = contrasts

    # compute all 4 mu sub-band powers (averaged across best channels, z-scored)
    for bname, band in MU_BANDS.items():
        perch = bandpower_perchannel(d['subdural'], d['fs'], band)
        d[bname] = average_selected_channels(perch, best_ch)

    # compute beta (13-30 Hz) using sub-band averaging per STANDARD_PIPELINE
    beta_perch = subband_logpow_perchannel(d['subdural'], d['fs'], BETA_SUB)
    d['beta_13_30'] = average_selected_channels(beta_perch, best_ch)
    
    # compute 1/f power spectrum across each channels during force trials
    data_raw = d['subdural']
    fs = d['fs']
    # data shape: (samples, channels)
    spectral_peaks, stats_tmp, pow_freq, ffreq, osc_clus_tmp = compute_spectral_peaks_1f(data_raw, fs)
    plt.figure()
    plt.plot(np.arange(2,41),osc_clus_tmp)

    n_long = int(np.sum(long_mask))
    print(f'fs={d["fs"]:.0f}, {len(dur)} events, {n_long} long (≥3s), best_ch={list(best_ch)}')
    data[key] = d

# summary
total_long = sum(int(np.sum(data[k]['long_mask'])) for k in data)
print(f'\nTotal long grasp events (≥3s): {total_long}')


#%% COMPUTE 1/F DURING EPOCHS WHERE FORCE DATA CAPTURED



#%%

# ── time-series: force + 4 mu bands + beta per recording ──

BAND_COLORS = {
    'mu_7_9':     '#1f77b4',   # blue
    'mu_8_10':    '#ff7f0e',   # orange
    'mu_10_13':   '#2ca02c',   # green
    'mu_8_13':    '#d62728',   # red
    'beta_13_30': '#9467bd',   # purple
}
BAND_LABELS = {
    'mu_7_9':     'Mu 7–9 Hz',
    'mu_8_10':    'Mu 8–10 Hz',
    'mu_10_13':   'Mu 10–13 Hz',
    'mu_8_13':    'Mu 8–13 Hz',
    'beta_13_30': 'Beta 13–30 Hz',
}

for key in RECORDINGS:
    d = data[key]
    fs = d['fs']
    t_sec = np.arange(d['n_samples']) / fs

    fig, axes = plt.subplots(6, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1]})
    fig.suptitle(f'{key}  (fs={fs:.0f} Hz, {int(np.sum(d["long_mask"]))} long events)', fontsize=12)

    # row 0: force
    ax = axes[0]
    ax.plot(t_sec, d['force_smooth'], color='k', lw=0.5)
    ax.axhline(d['threshold'], color='gray', ls='--', lw=0.5)
    ax.set_ylabel('Force\n(inv. norm.)')

    # shade events on all rows
    for ax_i in axes:
        for s, e, dur_i in zip(d['starts'], d['ends'], d['durations']):
            ts, te = s / fs, e / fs
            if dur_i >= LONG_THR_S:
                ax_i.axvspan(ts, te, alpha=0.25, color='red', zorder=0)
            else:
                ax_i.axvspan(ts, te, alpha=0.08, color='gray', zorder=0)

    # rows 1-5: mu bands + beta
    for i, bname in enumerate(ALL_BANDS):
        ax = axes[i + 1]
        ax.plot(t_sec, d[bname], color=BAND_COLORS[bname], lw=0.4)
        ax.set_ylabel(BAND_LABELS[bname])

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'timeseries_{key}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'timeseries_{key}.svg'))
    plt.show()
    print(f'Saved timeseries_{key}.{{pdf,svg}}')
    
 #%%
 
 # ── example single long grasp events: one per patient, pick longest event ──

# select one representative recording per patient (the one with most long events)
patient_recs = {}
for key in RECORDINGS:
    d = data[key]
    pat = d['patient']
    n_long = int(np.sum(d['long_mask']))
    if pat not in patient_recs or n_long > patient_recs[pat][1]:
        patient_recs[pat] = (key, n_long)

# for each patient, pick the longest event from the best recording
examples = []
for pat, (key, _) in patient_recs.items():
    d = data[key]
    long_idx = np.where(d['long_mask'])[0]
    longest_i = long_idx[np.argmax(d['durations'][long_idx])]
    examples.append((key, pat, longest_i))

print('Selected examples:')
for key, pat, ei in examples:
    d = data[key]
    print(f'  {pat}: {key} event #{ei}, dur={d["durations"][ei]:.1f}s')

# ── plot: one figure per example (6 rows: force + 5 bands) ──
PAD_S = 1.0  # padding before onset and after offset

for key, pat, ei in examples:
    d = data[key]
    fs = d['fs']
    s_samp = int(d['starts'][ei])
    e_samp = int(d['ends'][ei])
    dur_s  = d['durations'][ei]

    # window with padding
    w_start = max(0, s_samp - int(PAD_S * fs))
    w_end   = min(d['n_samples'], e_samp + int(PAD_S * fs))
    t_win = (np.arange(w_start, w_end) - s_samp) / fs  # time relative to onset

    fig, axes = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Example: {key} (Patient {pat}) — Event #{ei}, duration={dur_s:.1f}s', fontsize=12)

    # grasp shading on all axes
    for ax in axes:
        ax.axvspan(0, dur_s, alpha=0.2, color='red', zorder=0, label='Grasp')
        ax.axvline(0, color='k', ls='--', lw=0.6, alpha=0.5)
        ax.axvline(dur_s, color='k', ls='--', lw=0.6, alpha=0.5)

    # row 0: force
    axes[0].plot(t_win, d['force_smooth'][w_start:w_end], color='k', lw=0.8)
    axes[0].axhline(d['threshold'], color='gray', ls=':', lw=0.5)
    axes[0].set_ylabel('Force\n(inv. norm.)')

    # rows 1-5: bands
    for i, bname in enumerate(ALL_BANDS):
        ax = axes[i + 1]
        sig = d[bname][w_start:w_end]
        ax.plot(t_win, sig, color=BAND_COLORS[bname], lw=0.8)
        ax.axhline(0, color='gray', ls=':', lw=0.4)
        ax.set_ylabel(BAND_LABELS[bname])

        # annotate mean during grasp vs pre-grasp baseline
        pre_mask = (t_win >= -PAD_S) & (t_win < 0)
        grasp_mask = (t_win >= 0) & (t_win <= dur_s)
        if np.any(pre_mask) and np.any(grasp_mask):
            pre_mean = np.mean(sig[pre_mask])
            grasp_mean = np.mean(sig[grasp_mask])
            delta = grasp_mean - pre_mean
            ax.text(0.98, 0.92, f'Δ={delta:+.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    axes[-1].set_xlabel('Time from grasp onset (s)')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'example_event_{key}_e{ei}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'example_event_{key}_e{ei}.svg'))
    plt.show()
    print(f'Saved example_event_{key}_e{ei}.{{pdf,svg}}')

# ── also show 3 more examples from different recordings (variety) ──
print('\n--- Additional examples (one per recording with ≥3 long events) ---')
extra_examples = []
for key in RECORDINGS:
    d = data[key]
    long_idx = np.where(d['long_mask'])[0]
    # skip if already used or too few events
    if key in [ex[0] for ex in examples] or len(long_idx) < 2:
        continue
    # pick 2nd longest (variety — not the same one if same recording)
    sorted_by_dur = long_idx[np.argsort(-d['durations'][long_idx])]
    extra_examples.append((key, d['patient'], sorted_by_dur[0]))

for key, pat, ei in extra_examples:
    d = data[key]
    fs = d['fs']
    s_samp = int(d['starts'][ei])
    e_samp = int(d['ends'][ei])
    dur_s  = d['durations'][ei]
    w_start = max(0, s_samp - int(PAD_S * fs))
    w_end   = min(d['n_samples'], e_samp + int(PAD_S * fs))
    t_win = (np.arange(w_start, w_end) - s_samp) / fs

    fig, axes = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Example: {key} (Patient {pat}) — Event #{ei}, duration={dur_s:.1f}s', fontsize=12)

    for ax in axes:
        ax.axvspan(0, dur_s, alpha=0.2, color='red', zorder=0)
        ax.axvline(0, color='k', ls='--', lw=0.6, alpha=0.5)
        ax.axvline(dur_s, color='k', ls='--', lw=0.6, alpha=0.5)

    axes[0].plot(t_win, d['force_smooth'][w_start:w_end], color='k', lw=0.8)
    axes[0].axhline(d['threshold'], color='gray', ls=':', lw=0.5)
    axes[0].set_ylabel('Force\n(inv. norm.)')

    for i, bname in enumerate(ALL_BANDS):
        ax = axes[i + 1]
        sig = d[bname][w_start:w_end]
        ax.plot(t_win, sig, color=BAND_COLORS[bname], lw=0.8)
        ax.axhline(0, color='gray', ls=':', lw=0.4)
        ax.set_ylabel(BAND_LABELS[bname])
        pre_mask = (t_win >= -PAD_S) & (t_win < 0)
        grasp_mask = (t_win >= 0) & (t_win <= dur_s)
        if np.any(pre_mask) and np.any(grasp_mask):
            delta = np.mean(sig[grasp_mask]) - np.mean(sig[pre_mask])
            ax.text(0.98, 0.92, f'Δ={delta:+.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    axes[-1].set_xlabel('Time from grasp onset (s)')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'example_event_{key}_e{ei}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'example_event_{key}_e{ei}.svg'))
    plt.show()
    print(f'Saved example_event_{key}_e{ei}.{{pdf,svg}}')
    
    