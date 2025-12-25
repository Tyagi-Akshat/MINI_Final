"""
Utility functions for EEG signal processing and visualization.
Includes safe filtering, plotting helpers, and data validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Tuple, Optional, Union, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_filter(
    raw: mne.io.Raw,
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freq: Optional[float] = None,
    method: str = 'auto'
) -> mne.io.Raw:
    """
    Apply filtering with automatic FIR/IIR selection to avoid kernel length issues.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float, optional
        Low cutoff frequency (highpass)
    h_freq : float, optional
        High cutoff frequency (lowpass)
    notch_freq : float, optional
        Notch filter frequency (e.g., 50 or 60 Hz)
    method : str
        'auto', 'fir', or 'iir'. If 'auto', will try FIR first and fallback to IIR
    
    Returns
    -------
    raw : mne.io.Raw
        Filtered raw data
    """
    sfreq = raw.info['sfreq']
    nyquist = sfreq / 2.0
    
    # Clamp high frequency to below Nyquist
    if h_freq is not None and h_freq >= nyquist:
        original_h_freq = h_freq
        h_freq = nyquist * 0.95  # 95% of Nyquist
        logger.warning(f"High cutoff {original_h_freq} Hz >= Nyquist ({nyquist} Hz). "
                      f"Clamping to {h_freq:.2f} Hz")
    
    # Bandpass/highpass/lowpass filter
    if l_freq is not None or h_freq is not None:
        if method == 'auto':
            try:
                # Try FIR first with relaxed parameters
                raw = raw.copy().filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method='fir',
                    fir_design='firwin',
                    phase='zero',
                    fir_window='hamming',
                    verbose=False
                )
                logger.info(f"Applied FIR filter: {l_freq}-{h_freq} Hz")
            except Exception as e:
                logger.warning(f"FIR filtering failed: {e}. Falling back to IIR.")
                # Fallback to IIR (Butterworth 4th order)
                raw = raw.copy().filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method='iir',
                    iir_params={'order': 4, 'ftype': 'butter'},
                    verbose=False
                )
                logger.info(f"Applied IIR filter: {l_freq}-{h_freq} Hz")
        elif method == 'fir':
            raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
            logger.info(f"Applied FIR filter: {l_freq}-{h_freq} Hz")
        elif method == 'iir':
            raw = raw.copy().filter(
                l_freq=l_freq,
                h_freq=h_freq,
                method='iir',
                iir_params={'order': 4, 'ftype': 'butter'},
                verbose=False
            )
            logger.info(f"Applied IIR filter: {l_freq}-{h_freq} Hz")
    
    # Notch filter
    if notch_freq is not None:
        raw = raw.copy().notch_filter(freqs=notch_freq, verbose=False)
        logger.info(f"Applied notch filter at {notch_freq} Hz")
    
    return raw


def validate_epochs(epochs: mne.Epochs) -> dict:
    """
    Validate and summarize epoch data.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    
    Returns
    -------
    summary : dict
        Summary statistics
    """
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    n_times = len(epochs.times)
    events = epochs.events[:, -1]
    unique_events, counts = np.unique(events, return_counts=True)
    
    summary = {
        'n_epochs': n_epochs,
        'n_channels': n_channels,
        'n_times': n_times,
        'duration': epochs.times[-1] - epochs.times[0],
        'sfreq': epochs.info['sfreq'],
        'events': dict(zip(unique_events, counts))
    }
    
    logger.info(f"Epochs summary: {n_epochs} epochs, {n_channels} channels, "
                f"{n_times} time points, events: {summary['events']}")
    
    return summary


def plot_psd_topomap(epochs: mne.Epochs, freq_bands: dict, save_path: Optional[Path] = None):
    """
    Plot power spectral density topomaps for different frequency bands.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    freq_bands : dict
        Dictionary of frequency bands, e.g., {'alpha': (8, 12), 'beta': (13, 30)}
    save_path : Path, optional
        Path to save the figure
    """
    n_bands = len(freq_bands)
    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 4))
    
    if n_bands == 1:
        axes = [axes]
    
    for ax, (band_name, (fmin, fmax)) in zip(axes, freq_bands.items()):
        epochs.plot_psd_topomap(
            bands=[(fmin, fmax, band_name)],
            axes=ax,
            show=False,
            normalize=True
        )
        ax.set_title(f'{band_name.capitalize()} ({fmin}-{fmax} Hz)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PSD topomap to {save_path}")
    
    plt.close()


def plot_csp_patterns(
    csp_patterns: np.ndarray,
    info: mne.Info,
    n_components: int = 6,
    save_path: Optional[Path] = None
):
    """
    Plot CSP spatial patterns as topomaps.
    
    Parameters
    ----------
    csp_patterns : np.ndarray
        CSP patterns (channels × components)
    info : mne.Info
        MNE info object with channel locations
    n_components : int
        Number of components to plot
    save_path : Path, optional
        Path to save the figure
    """
    n_plot = min(n_components, csp_patterns.shape[1])
    fig, axes = plt.subplots(1, n_plot, figsize=(3 * n_plot, 3))
    
    if n_plot == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            csp_patterns[:, i],
            info,
            axes=ax,
            show=False
        )
        ax.set_title(f'CSP Component {i+1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved CSP patterns to {save_path}")
    
    plt.close()


def plot_event_timeline(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    save_path: Optional[Path] = None
):
    """
    Plot event timeline for a recording.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    events : np.ndarray
        Events array (n_events × 3)
    event_id : dict
        Event ID mapping
    save_path : Path, optional
        Path to save the figure
    """
    fig = mne.viz.plot_events(
        events,
        sfreq=raw.info['sfreq'],
        first_samp=raw.first_samp,
        event_id=event_id,
        show=False
    )
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved event timeline to {save_path}")
    
    plt.close(fig)


def create_output_directories(base_dir: Path):
    """
    Create necessary output directories.
    
    Parameters
    ----------
    base_dir : Path
        Base project directory
    """
    dirs = [
        base_dir / 'outputs',
        base_dir / 'outputs' / 'plots',
        base_dir / 'outputs' / 'confusion_matrices',
        base_dir / 'outputs' / 'logs',
        base_dir / 'models',
        base_dir / 'models' / 'scalers',
        base_dir / 'data'
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")


def normalize_features(
    X: np.ndarray,
    feature_range: Tuple[float, float] = (0.1, 0.9),
    scaler: Optional[object] = None
) -> Tuple[np.ndarray, object]:
    """
    Normalize features to specified range (as per paper: 0.1 to 0.9).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples × n_features)
    feature_range : tuple
        Target range for normalization
    scaler : object, optional
        Pre-fitted scaler (for transform only)
    
    Returns
    -------
    X_scaled : np.ndarray
        Normalized features
    scaler : object
        Fitted scaler object
    """
    from sklearn.preprocessing import MinMaxScaler
    
    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def check_data_balance(y: np.ndarray) -> dict:
    """
    Check class balance in labels.
    
    Parameters
    ----------
    y : np.ndarray
        Label array
    
    Returns
    -------
    balance : dict
        Class distribution
    """
    unique, counts = np.unique(y, return_counts=True)
    balance = dict(zip(unique, counts))
    
    total = len(y)
    logger.info(f"Class balance: {balance} (total: {total})")
    
    for label, count in balance.items():
        percentage = 100 * count / total
        logger.info(f"  Class {label}: {count} samples ({percentage:.1f}%)")
    
    return balance


def print_shapes(data_dict: dict):
    """
    Print shapes of data arrays for debugging.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of arrays with descriptive keys
    """
    logger.info("=" * 60)
    logger.info("Data Shapes:")
    for name, data in data_dict.items():
        if hasattr(data, 'shape'):
            logger.info(f"  {name}: {data.shape}")
        else:
            logger.info(f"  {name}: {type(data)}")
    logger.info("=" * 60)
