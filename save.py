"""
test_model.py

Usage examples:
  # Predict from a .npy epoch (channels x time):
  python test_model.py --npy path/to/epoch.npy --model outputs/csp_best_model.pkl

  # Predict from an EDF segment (start time seconds):
  python test_model.py --edf path/to/file.edf --start 10.0 --model outputs/csp_best_model.pkl

Notes:
- The saved model bundle should contain at least: 'csp', 'scaler', and either 'best_clf' or one of 'lda'/'svm'/'mlp'.
- .npy epoch must be 2D (channels, time) or (time, channels) (this script tries to auto-detect).
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import mne

BAND_LO = 8.0
BAND_HI = 30.0
RESAMPLE_SF = 160
EPOCH_SEC = 2.0  # training epoch length: tmax - tmin in seconds

def load_model(model_path):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    # Accept multiple bundle formats
    csp = bundle.get("csp") or bundle.get("CSP") or bundle.get("csp_obj")
    scaler = bundle.get("scaler")
    # classifier could be best_clf or 'classifier' or named classifiers
    clf = bundle.get("best_clf") or bundle.get("classifier") or bundle.get("lda") or bundle.get("svm") or bundle.get("mlp")
    channels = bundle.get("channels")
    n_csp = bundle.get("n_csp") or (getattr(csp, "n_components", None) if csp is not None else None)
    info = {"best_name": bundle.get("best_name"), "test_accuracy": bundle.get("test_accuracy")}
    return {"csp": csp, "scaler": scaler, "clf": clf, "channels": channels, "n_csp": n_csp, "meta": info}

def preprocess_npy(np_path, model_channels):
    arr = np.load(np_path)
    if arr.ndim != 2:
        raise ValueError("Numpy file must be 2D (channels x time) or (time x channels).")
    # auto-detect orientation
    if arr.shape[0] == len(model_channels) and arr.shape[1] != len(model_channels):
        epoch = arr.copy()
    elif arr.shape[1] == len(model_channels) and arr.shape[0] != len(model_channels):
        epoch = arr.T.copy()
    elif arr.shape[0] == len(model_channels) and arr.shape[1] == len(model_channels):
        epoch = arr.copy()  # ambiguous; assume channels x time
    else:
        # best-effort: if channels > expected, take first N; if shorter, error
        if arr.shape[0] >= len(model_channels):
            epoch = arr[:len(model_channels), :]
        elif arr.shape[1] >= len(model_channels):
            epoch = arr[:, :len(model_channels)].T
        else:
            raise ValueError(f"Numpy epoch channels ({arr.shape[0]}) < model channels ({len(model_channels)}).")
    # Resample/interpolate to desired length if needed
    desired_len = int(RESAMPLE_SF * EPOCH_SEC)
    cur_len = epoch.shape[1]
    if cur_len != desired_len:
        x_old = np.linspace(0, 1, cur_len)
        x_new = np.linspace(0, 1, desired_len)
        epoch = np.array([np.interp(x_new, x_old, epoch[ch, :]) for ch in range(epoch.shape[0])])
    # normalize per-channel same as training
    epoch = (epoch - epoch.mean(axis=1, keepdims=True)) / (epoch.std(axis=1, keepdims=True) + 1e-12)
    return epoch

def preprocess_edf(edf_path, start_sec, model_channels):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    # try to match model channels exactly
    picks = [ch for ch in model_channels if ch in raw.ch_names] if model_channels is not None else []
    # if not enough exact matches, try case-insensitive matching
    if len(picks) < len(model_channels or []):
        lower = {c.lower(): c for c in raw.ch_names}
        picks = []
        for mch in (model_channels or []):
            if mch is None:
                continue
            key = mch.lower()
            if key in lower:
                picks.append(lower[key])
    # fallback: use first N channels
    if not picks:
        picks = raw.ch_names[: (len(model_channels) if model_channels else 2)]
    raw.pick_channels(picks)
    try:
        raw.filter(BAND_LO, BAND_HI, fir_design='firwin', verbose=False)
    except Exception:
        pass
    if raw.info['sfreq'] != RESAMPLE_SF:
        raw.resample(RESAMPLE_SF, npad='auto')
    start = float(start_sec)
    stop = start + EPOCH_SEC
    # get_data expects sample indexes or use tmin/tmax with .copy().crop? simpler:
    data = raw.copy().get_data(start=int(start*RESAMPLE_SF), stop=int(stop*RESAMPLE_SF))
    # data shape channels x samples
    # if data length shorter than desired, pad or trim
    desired_len = int(RESAMPLE_SF * EPOCH_SEC)
    if data.shape[1] < desired_len:
        pad_width = desired_len - data.shape[1]
        data = np.pad(data, ((0,0),(0,pad_width)), mode='constant')
    elif data.shape[1] > desired_len:
        data = data[:, :desired_len]
    # normalize per-channel
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-12)
    return data

def predict_epoch(model_bundle, epoch_array):
    """
    epoch_array: channels x time
    returns (pred_label, probs_or_None)
    """
    csp = model_bundle["csp"]
    scaler = model_bundle["scaler"]
    clf = model_bundle["clf"]
    if csp is None or scaler is None or clf is None:
        raise ValueError("Model bundle missing required keys (csp/scaler/clf).")

    Xcsp = csp.transform(epoch_array[np.newaxis, :, :])  # shape (1, n_csp)
    Xs = scaler.transform(Xcsp)
    pred = clf.predict(Xs)[0]
    probs = None
    if hasattr(clf, "predict_proba"):
        try:
            probs = clf.predict_proba(Xs)[0]
        except Exception:
            probs = None
    return int(pred), probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/csp_best_model.pkl", help="path to saved model pickle")
    parser.add_argument("--npy", type=str, help="path to .npy epoch (channels x time)")
    parser.add_argument("--edf", type=str, help="path to EDF file")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds for EDF segment")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)
    print("Loaded model metadata:", model.get("meta"))

    if args.npy is None and args.edf is None:
        raise ValueError("Specify either --npy or --edf to provide an input epoch.")

    # If model channels unknown, accept any; but many pipelines saved channels
    model_channels = model.get("channels")
    if model_channels is None:
        print("[WARN] Model has no 'channels' key. The script will use EDF first channels or .npy shape.")

    if args.npy:
        epoch = preprocess_npy(args.npy, model_channels or [])
    else:
        epoch = preprocess_edf(args.edf, args.start, model_channels or [])

    # if epoch has more channels than model expects, slice
    if model_channels:
        if epoch.shape[0] > len(model_channels):
            epoch = epoch[:len(model_channels), :]
        elif epoch.shape[0] < len(model_channels):
            raise ValueError(f"Epoch channels ({epoch.shape[0]}) less than model channels ({len(model_channels)})")

    pred, probs = predict_epoch(model, epoch)
    print("Prediction:", pred)
    if probs is not None:
        print("Probabilities:", probs)

if __name__ == "__main__":
    main()
