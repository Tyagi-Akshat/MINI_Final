# ================================================================
# HYBRID PIPELINE (DWT + CSP) — FIXED CSP dimension / dynamic names
# ================================================================

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

# local DWT feature extractor (must exist)
from features_paper import features_from_epoch

# optional PNN
try:
    from models_paper import ProbabilisticNeuralNetwork
    pnn_available = True
except:
    pnn_available = False

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")
RUNS       = [3, 7, 11]
# support common PhysioNet channel name variants
CHANNELS   = ["C3..", "C4..", "C3.", "C4.", "C3", "C4", "Cz"]
EVENT_TMIN = -2.0
EVENT_TMAX = 0.0
RESAMPLE_SF = 160

OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# SYNTHETIC EVENTS (fallback)
# -------------------------------------------------------
def create_synthetic_events(raw, tstart=1.0, interval=8.0, epoch_len=2.0):
    sf = raw.info["sfreq"]
    dur = raw.n_times / sf
    onsets = np.arange(tstart, max(tstart, dur - epoch_len), interval)

    events = []
    for i, t in enumerate(onsets):
        sample = int(t * sf)
        # use codes 2 and 3 to be consistent with T1/T2 in many files
        code = 2 if (i % 2 == 0) else 3
        events.append([sample, 0, code])

    if len(events) == 0:
        return np.zeros((0, 3), dtype=int)
    return np.array(events, dtype=int)


# -------------------------------------------------------
# EXTRACT ALL EPOCHS FOR ONE SUBJECT
# -------------------------------------------------------
def extract_subject(subject_id):
    subj = f"S{subject_id:03d}"
    rows = []
    epochs_CSP = []
    labels_CSP = []
    groups = []

    for run in RUNS:
        f = DATA_DIR / f"{subj}R{run:02d}.edf"
        if not f.exists():
            continue

        try:
            raw = mne.io.read_raw_edf(str(f), preload=True, verbose=False)
        except Exception as e:
            print(f"[WARN] cannot read {f.name}: {e}")
            continue

        # pick channels that exist
        keep = [c for c in CHANNELS if c in raw.ch_names]
        if len(keep) == 0:
            continue
        raw.pick_channels(keep)

        # filtering & notch
        try:
            raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)
            raw.notch_filter(50.0, verbose=False)
        except Exception:
            pass

        # normalize channels
        raw._data = (raw._data - raw._data.mean(axis=1, keepdims=True)) / \
                    (raw._data.std(axis=1, keepdims=True) + 1e-12)

        # try to get events & event_id
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        except Exception:
            events = None
            event_id = None

        # build mapping for T1/T2 if present
        event_id_map = {}
        if events is not None and len(events) > 0 and event_id is not None:
            if "T1" in event_id and "T2" in event_id:
                event_id_map = {"T1": event_id["T1"], "T2": event_id["T2"]}

        # fallback to synthetic if T1/T2 not found
        if not event_id_map:
            events = create_synthetic_events(raw, tstart=1.0, interval=8.0, epoch_len=(EVENT_TMAX - EVENT_TMIN))
            if events.shape[0] == 0:
                continue
            event_id_map = {"T1": 2, "T2": 3}

        # create epochs using only T1 and T2 mapping
        try:
            epochs = mne.Epochs(raw, events, event_id=event_id_map,
                                tmin=EVENT_TMIN, tmax=EVENT_TMAX,
                                baseline=None, preload=True, verbose=False)
        except Exception as e:
            print(f"[WARN] Epochs creation failed for {f.name}: {e}")
            continue

        # resample
        if epochs.info["sfreq"] != RESAMPLE_SF:
            epochs = epochs.copy().resample(RESAMPLE_SF, npad="auto")

        # numeric code -> label (T1->0, T2->1)
        code_to_label = {event_id_map["T1"]: 0, event_id_map["T2"]: 1}

        X = epochs.get_data()  # n_epochs x n_chan x n_time
        codes = epochs.events[:, -1]

        # map codes to labels; filter out any not in code_to_label
        labels = []
        X_valid = []
        for ep, c in zip(X, codes):
            if int(c) in code_to_label:
                labels.append(code_to_label[int(c)])
                X_valid.append(ep)
        if len(X_valid) == 0:
            continue

        X_valid = np.array(X_valid)
        labels = np.array(labels, dtype=int)

        # compute DWT features per epoch and collect CSP raw epochs
        for ep, lb in zip(X_valid, labels):
            feat = features_from_epoch(ep, sfreq=RESAMPLE_SF)
            rows.append({
                "subject": subject_id,
                "run": run,
                "label": int(lb),
                **{f"f{k}": float(v) for k, v in enumerate(feat)}
            })

        for ep, lb in zip(X_valid, labels):
            epochs_CSP.append(ep)
            labels_CSP.append(lb)
            groups.append(subject_id)

    if len(rows) == 0:
        return None

    return {
        "rows": rows,
        "epochs": np.array(epochs_CSP) if len(epochs_CSP) > 0 else np.zeros((0, len(keep), 1)),
        "labels": np.array(labels_CSP) if len(labels_CSP) > 0 else np.zeros((0,)),
        "groups": np.array(groups) if len(groups) > 0 else np.zeros((0,))
    }


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main(subjects):
    print("\n=== HYBRID PIPELINE (DWT + CSP) ===\n")

    all_rows = []
    X_list = []
    y_list = []
    g_list = []

    for s in subjects:
        print(f"Processing subject {s} ...")
        res = extract_subject(s)
        if res is None:
            print(f"[INFO] Subject {s}: no epochs found.")
            continue
        all_rows.extend(res["rows"])
        if res["epochs"].size > 0:
            X_list.extend(list(res["epochs"]))
            y_list.extend(list(res["labels"]))
            g_list.extend(list(res["groups"]))

    if len(all_rows) == 0:
        print("ERROR: No data found. Check EDF files and annotations (T1/T2).")
        return

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)
    groups = np.array(g_list, dtype=int)

    print(f"Collected epochs: {len(y)} from subjects: {np.unique(groups)}")

    # Save DWT features CSV with paper names
    df = pd.DataFrame(all_rows)
    dwt_names = [
        "Energy", "Scale_Variance", "RMS", "Roll_Off",
        "Variance", "Approximate_Entropy", "Zero_Crossing", "Modified_MAV"
    ]
    rename_map = {f"f{i}": dwt_names[i] for i in range(len(dwt_names))}
    df = df.rename(columns=rename_map)
    df.to_csv(OUTPUT_DIR / "DWT_features_named.csv", index=False)
    print("[Saved] DWT_features_named.csv")

    # SAFE SPLIT
    print("\nPerforming safe train/test split...")
    n_samples = X.shape[0]
    n_groups = len(np.unique(groups))

    if n_samples < 2:
        raise RuntimeError("Not enough epochs to split (need >=2).")

    if n_groups < 2:
        print("[WARNING] Only one subject present -> using stratified split.")
        strat = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X, y, groups))
        except Exception as e:
            print("[WARN] GroupShuffleSplit failed:", e)
            unique = np.unique(groups)
            test_group = unique[:1]
            mask = np.isin(groups, test_group)
            test_idx = np.where(mask)[0]
            train_idx = np.where(~mask)[0]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train epochs: {len(y_train)}, Test epochs: {len(y_test)}")

    # CSP: choose n_components dynamically (<= n_channels)
    n_channels = X.shape[1] if X.ndim >= 3 else 1
    n_csp = min(8, n_channels)
    from mne.decoding import CSP
    print(f"Computing CSP with n_components={n_csp} (n_channels={n_channels})")
    try:
        csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)
        all_csp = csp.transform(X)
    except Exception as e:
        print("[WARN] CSP failed:", e)
        X_train_csp = np.zeros((len(y_train), n_csp))
        X_test_csp = np.zeros((len(y_test), n_csp))
        all_csp = np.zeros((len(y), n_csp))

    # DWT features for train/test/full
    dwt_train = np.array([features_from_epoch(ep, sfreq=RESAMPLE_SF) for ep in X_train])
    dwt_test  = np.array([features_from_epoch(ep, sfreq=RESAMPLE_SF) for ep in X_test])
    dwt_full  = np.array([features_from_epoch(ep, sfreq=RESAMPLE_SF) for ep in X])

    # Build hybrid matrices
    X_train_h = np.hstack([dwt_train, X_train_csp])
    X_test_h  = np.hstack([dwt_test,  X_test_csp])
    X_full_h  = np.hstack([dwt_full,  all_csp])

    # Names: use actual all_csp shape to create CSP column names
    csp_cols_count = all_csp.shape[1] if all_csp.ndim == 2 else 0
    csp_names = [f"CSP_{i+1}" for i in range(csp_cols_count)]
    hybrid_names = dwt_names + csp_names

    # Sanity: ensure column counts match
    if X_full_h.shape[1] != len(hybrid_names):
        print(f"[INFO] feature column count mismatch: X_full_h has {X_full_h.shape[1]} cols, hybrid_names has {len(hybrid_names)} names.")
        # adjust hybrid_names to match actual columns
        hybrid_names = [f"f_{i+1}" for i in range(X_full_h.shape[1])]
        print("[INFO] fallback: using generic feature names.")

    # scale (fit on train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_h)
    X_test_s  = scaler.transform(X_test_h)

    # Train classifiers
    results = {}

    if pnn_available:
        try:
            pnn = ProbabilisticNeuralNetwork()
            pnn.fit(X_train_s, y_train)
            y_pnn = pnn.predict(X_test_s)
            results["PNN"] = (y_pnn, accuracy_score(y_test, y_pnn), confusion_matrix(y_test, y_pnn))
        except Exception as e:
            print("[WARN] PNN failed:", e)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_s, y_train)
    y_lda = lda.predict(X_test_s)
    results["LDA"] = (y_lda, accuracy_score(y_test, y_lda), confusion_matrix(y_test, y_lda))

    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train_s, y_train)
    y_svm = svm.predict(X_test_s)
    results["SVM"] = (y_svm, accuracy_score(y_test, y_svm), confusion_matrix(y_test, y_svm))

    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1500, activation='relu')
    mlp.fit(X_train_s, y_train)
    y_mlp = mlp.predict(X_test_s)
    results["MLP"] = (y_mlp, accuracy_score(y_test, y_mlp), confusion_matrix(y_test, y_mlp))

    # Save features CSV (use hybrid_names which now matches X_full_h)
    df_h = pd.DataFrame({"subject": groups, "label": y})
    for i, nm in enumerate(hybrid_names):
        df_h[nm] = X_full_h[:, i]
    df_h.to_csv(OUTPUT_DIR / "all_DWT_CSP_features.csv", index=False)
    print("[Saved] all_DWT_CSP_features.csv")

    # Plots
    print("Saving plots...")
    model_list = list(results.keys())
    accs = [results[m][1] for m in model_list]

    plt.figure(figsize=(7,5))
    sns.barplot(x=model_list, y=accs)
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy (Hybrid)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Accuracy_Comparison.png", dpi=300)
    plt.close()

    for m in model_list:
        cm = results[m][2]
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{m} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"CM_{m}.png", dpi=300)
        plt.close()

    print("Hybrid pipeline completed successfully.")


# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[95], help="subject list")
    args = parser.parse_args()
    main(args.subjects)
