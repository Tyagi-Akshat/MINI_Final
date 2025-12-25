# /mnt/data/features_paper.py
import numpy as np
import pywt
from scipy import stats
from scipy.signal import welch
from math import log2
from typing import Tuple

def dwt_beta_band(signal: np.ndarray, wavelet='db4', level=6, beta_level=4):
    """
    Decompose `signal` with DWT up to `level` and return the detail coefficients
    corresponding to the paper's beta band (detail at beta_level).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs: [cA_n, cD_n, cD_n-1, ..., cD1]
    # detail index for level `beta_level` => coeffs[-beta_level]
    idx = -beta_level
    beta = coeffs[idx]
    # Upsample/interpolate beta to original length for feature calculation (simple approach)
    beta_up = pywt.upcoef('d', beta, wavelet, level=beta_level, take=len(signal))
    return beta_up

def energy(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def scale_variance(x: np.ndarray) -> float:
    v = np.var(x)
    if v <= 0:
        return 0.0
    return float(log2(v) if v > 0 else 0.0)

def rms_value(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))

def roll_off(x: np.ndarray, fraction=0.85):
    # rolloff frequency estimation requires PSD; for single-channel epoch with uniform sampling
    # We approximate roll-off in sample-domain: find index where cumulative energy reaches fraction
    total = np.sum(np.abs(x))
    if total == 0:
        return 0.0
    cum = np.cumsum(np.abs(x))
    idx = np.searchsorted(cum, fraction * total)
    return float(idx / len(x))

def variance(x: np.ndarray) -> float:
    return float(np.var(x))

def approximate_entropy(x: np.ndarray, m=2, r_factor=0.15) -> float:
    # ApEn(m,r,N) as in paper; r = r_factor * std(x)
    x = np.asarray(x)
    N = len(x)
    if N <= m + 1:
        return 0.0
    r = r_factor * np.std(x)
    def _phi(m_):
        patterns = np.array([x[i:i+m_] for i in range(N - m_ + 1)])
        C = []
        for i in range(len(patterns)):
            d = np.max(np.abs(patterns - patterns[i]), axis=1)
            C.append(np.sum(d <= r) / (N - m_ + 1.0))
        C = np.array(C)
        return np.sum(np.log(C + 1e-12)) / (N - m_ + 1.0)
    return float(_phi(m) - _phi(m+1))

def zero_crossings(x: np.ndarray, threshold=0.0) -> int:
    x = np.asarray(x)
    crossings = np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) > threshold))
    return int(crossings)

def modified_mav(x: np.ndarray):
    N = len(x)
    w = np.ones(N) * 0.5
    n1 = int(0.25 * N)
    n2 = int(0.75 * N)
    w[n1:n2] = 1.0
    mmav = np.sum(w * np.abs(x)) / N
    return float(mmav)

def features_from_epoch(epoch_signal: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    epoch_signal: 1D array (single channel) or 2D array (channels x samples)
    returns vector of 8 features (computed by averaging across channels)
    """
    arr = np.array(epoch_signal)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    feats = []
    for ch in arr:
        beta = dwt_beta_band(ch, wavelet='db4', level=6, beta_level=4)
        feats.append([
            energy(beta),
            scale_variance(beta),
            rms_value(beta),
            roll_off(beta),
            variance(beta),
            approximate_entropy(beta, m=2, r_factor=0.15),
            zero_crossings(beta, threshold=0.0),
            modified_mav(beta)
        ])
    feats = np.array(feats)
    # average across channels (paper used features from beta band across electrodes)
    feat_mean = np.mean(feats, axis=0)
    return feat_mean  # length-8 vector
