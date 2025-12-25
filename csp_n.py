# ===========================================
#      CSP MOTOR IMAGERY PIPELINE (FINAL)
#      — with model saving (best classifier)
# ===========================================

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from mne.decoding import CSP
from utils_py import safe_filter, create_output_directories
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ---------------------------------
# CONFIGURATION
# ---------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

RUNS = [3, 7, 11]               # executed movement runs
CHANNELS = ["C3", "C4"]
TMIN, TMAX = -2.0, 0.0          # ERD extraction window
EVENT_MAPPING = {"T1": 1, "T2": 2}   # Left / Right

# ensure outputs
create_output_directories(Path("."))


# ---------------------------------
# Load & Epoch Subject
# ---------------------------------
def extract_subject_epochs(subj_id):
    """
    Returns:
        X (np.ndarray): epochs stacked (N, n_channels, time)
        y (np.ndarray): labels (N,)
        ch_names (list or None): channel names used (first file where channels found)
    """
    subj = f"S{subj_id:03d}"
    epochs_list = []
    labels_list = []
    ch_names = None

    for run in RUNS:
        f = DATA_DIR / f"{subj}R{run:02d}.edf"
        if not f.exists():
            continue

        try:
            raw = mne.io.read_raw_edf(str(f), preload=True, verbose=False)
        except Exception as e:
            print(f"[WARN] cannot read {f.name}: {e}")
            continue

        # Pick requested channels that exist in file
        pick_chs = [ch for ch in CHANNELS if ch in raw.ch_names]
        if len(pick_chs) == 0:
            # if strict channels not present, try to pick sensible C-area channels (best-effort)
            pick_chs = [ch for ch in raw.ch_names if ch.upper().startswith("C")][:8]

        if len(pick_chs) < 2:
            # not enough channels -> skip run
            continue

        raw.pick_channels(pick_chs)

        if ch_names is None:
            ch_names = raw.ch_names.copy()

        # Filtering
        try:
            raw = safe_filter(raw, l_freq=8, h_freq=30)   # μ + β band
        except Exception:
            try:
                raw.filter(8, 30, fir_design="firwin", verbose=False)
            except Exception:
                pass

        # Subject normalization
        raw._data = (raw._data - raw._data.mean(axis=1, keepdims=True)) / \
                    (raw._data.std(axis=1, keepdims=True) + 1e-12)

        try:
            events, _ = mne.events_from_annotations(raw, verbose=False)
        except Exception:
            events = np.zeros((0, 3), dtype=int)

        if len(events) == 0:
            continue

        try:
            epochs = mne.Epochs(
                raw,
                events,
                event_id=EVENT_MAPPING,
                tmin=TMIN,
                tmax=TMAX,
                baseline=None,
                preload=True,
                verbose=False
            )
        except Exception:
            # Try creating epochs without event_id mapping and then filter labels
            try:
                epochs = mne.Epochs(raw, events, tmin=TMIN, tmax=TMAX, baseline=None, preload=True, verbose=False)
            except Exception:
                continue

        # Resample to 160 Hz
        if epochs.info["sfreq"] != 160:
            epochs = epochs.copy().resample(160, npad="auto")

        labels = epochs.events[:, -1] - 1  # map typical {1,2} -> {0,1}
        if len(labels) == 0:
            continue

        epochs_list.append(epochs.get_data())   # shape (N, ch, time)
        labels_list.append(labels)

    if len(epochs_list) == 0:
        return None, None, None

    X = np.vstack(epochs_list)
    y = np.hstack(labels_list)

    return X, y, ch_names


def ensure_plot_dirs(base: Path):
    plots_dir = base / "plots"
    cm_dir = base / "confusion_matrices"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, cm_dir


def plot_accuracy_bar(scores: dict, save_path: Path):
    models = list(scores.keys())
    accs = [scores[m]['accuracy'] for m in models]
    plt.figure(figsize=(7, 4))
    sns.barplot(x=models, y=accs)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracies (CSP Pipeline)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_csp_components(csp_obj: CSP, channel_names: list, save_dir: Path, n_components: int = None):
    patterns = getattr(csp_obj, "patterns_", None)
    if patterns is None:
        return

    n_comp = patterns.shape[1]
    if n_components is None:
        n_components = n_comp

    for i in range(min(n_comp, n_components)):
        plt.figure(figsize=(6, 3))
        vals = patterns[:, i]
        x = np.arange(len(vals))
        labels = channel_names if channel_names is not None else [f"ch{j}" for j in x]
        plt.bar(x, vals)
        plt.xticks(x, labels, rotation=45)
        plt.title(f"CSP Pattern Component {i+1}")
        plt.ylabel("Pattern weight")
        plt.tight_layout()
        plt.savefig(save_dir / f"csp_component_{i+1}.png", dpi=300)
        plt.close()


# ---------------------------------
# MAIN PIPELINE
# ---------------------------------
def main(subjects):

    create_output_directories(Path("."))

    print("\n=== Extracting all subjects data ===")

    all_X = []
    all_y = []
    all_groups = []
    ch_names_global = None

    for s in subjects:
        Xs, ys, ch_names = extract_subject_epochs(s)
        if Xs is None:
            print(f"Subject {s}: no epochs or insufficient channels - skipped")
            continue

        all_X.append(Xs)
        all_y.append(ys)
        all_groups.append(np.ones(len(ys)) * s)

        print(f"Subject {s}: {Xs.shape[0]} epochs")
        if ch_names_global is None and ch_names is not None:
            ch_names_global = ch_names

    if len(all_X) == 0:
        print("No data found. Exiting.")
        return

    X = np.vstack(all_X)       # (N_epochs, n_channels, time)
    y = np.hstack(all_y)
    groups = np.hstack(all_groups)

    print("Total epochs:", len(y))

    # -------------------------------
    # Group-safe train/test split
    # -------------------------------
    try:
        gss = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    except Exception:
        # fallback simple stratified split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y if len(np.unique(y)) > 1 else None))

    # -------------------------------
    # CSP (n_components safe)
    # -------------------------------
    n_channels = X.shape[1] if X.ndim >= 3 else 1
    n_csp = min(8, n_channels)
    csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)

    # Fit CSP on training EEG only
    try:
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)
    except Exception as e:
        print("[WARN] CSP fit/transform failed:", e)
        # fallback with zeros
        X_train_csp = np.zeros((X_train.shape[0], n_csp))
        X_test_csp = np.zeros((X_test.shape[0], n_csp))

    # -------------------------------
    # StandardScaler for classifier
    # -------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_csp)
    X_test_s = scaler.transform(X_test_csp)

    # -------------------------------
    # CLASSIFIERS
    # -------------------------------

    # 1) LDA (best for CSP)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_s, y_train)
    y_pred_lda = lda.predict(X_test_s)
    acc_lda = accuracy_score(y_test, y_pred_lda)
    cm_lda = confusion_matrix(y_test, y_pred_lda)

    # 2) SVM (optional)
    svm = SVC(kernel='rbf', C=1, gamma='scale', probability=False)
    svm.fit(X_train_s, y_train)
    y_pred_svm = svm.predict(X_test_s)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)

    # 3) BP-MLP (optional)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=1500,
        learning_rate_init=0.001,
        alpha=0.001
    )
    mlp.fit(X_train_s, y_train)
    y_pred_mlp = mlp.predict(X_test_s)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)

    # -------------------------------
    # METRICS
    # -------------------------------
    def metrics(cm):
        if cm.size == 1:
            acc = float(np.trace(cm)) / float(np.sum(cm))
            return acc, 0.0, 0.0
        TN, FP, FN, TP = cm.ravel()
        acc = (TP + TN) / (TP + TN + FP + FN)
        sen = TP / (TP + FN + 1e-9)
        spe = TN / (TN + FP + 1e-9)
        return acc, sen, spe

    acc_lda_m, sen_lda, spe_lda = metrics(cm_lda)
    acc_svm_m, sen_svm, spe_svm = metrics(cm_svm)
    acc_mlp_m, sen_mlp, spe_mlp = metrics(cm_mlp)

    print("\n=== RESULTS (CSP Pipeline) ===")

    print("\nLDA RESULTS:")
    print(cm_lda)
    print(f"Accuracy={acc_lda_m:.3f}, Sensitivity={sen_lda:.3f}, Specificity={spe_lda:.3f}")

    print("\nSVM RESULTS:")
    print(cm_svm)
    print(f"Accuracy={acc_svm_m:.3f}, Sensitivity={sen_svm:.3f}, Specificity={spe_svm:.3f}")

    print("\nMLP RESULTS:")
    print(cm_mlp)
    print(f"Accuracy={acc_mlp_m:.3f}, Sensitivity={sen_mlp:.3f}, Specificity={spe_mlp:.3f}")

    # save metrics CSV
    pd.DataFrame([{
        "acc_lda": float(acc_lda_m), "sen_lda": float(sen_lda), "spe_lda": float(spe_lda),
        "acc_svm": float(acc_svm_m), "sen_svm": float(sen_svm), "spe_svm": float(spe_svm),
        "acc_mlp": float(acc_mlp_m), "sen_mlp": float(sen_mlp), "spe_mlp": float(spe_mlp),
    }]).to_csv(OUTPUT_DIR / "csp_metrics.csv", index=False)

    plots_dir, cm_dir = ensure_plot_dirs(OUTPUT_DIR)

    scores = {
        "LDA": {"accuracy": acc_lda_m, "cm": cm_lda},
        "SVM": {"accuracy": acc_svm_m, "cm": cm_svm},
        "MLP": {"accuracy": acc_mlp_m, "cm": cm_mlp}
    }

    # Accuracy bar
    plot_accuracy_bar(scores, plots_dir / "accuracy_bar.png")

    # Confusion matrices
    for model_name, val in scores.items():
        plot_confusion_matrix(val["cm"], model_name, cm_dir / f"cm_{model_name}.png")

    # CSP component plots (bar-plot per component)
    try:
        plot_csp_components(csp, ch_names_global, plots_dir, n_components=min(6, n_csp))
    except Exception as e:
        print("Warning: CSP plotting failed:", e)

    print(f"\nPlots saved to: {plots_dir} and {cm_dir}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'csp_metrics.csv'}")

    # -------------------------------
    # SAVE BEST MODEL (highest accuracy)
    # -------------------------------
    model_scores = {
        "LDA": float(acc_lda_m),
        "SVM": float(acc_svm_m),
        "MLP": float(acc_mlp_m)
    }
    # pick best (ties broken by dictionary order)
    best_name = max(model_scores, key=model_scores.get)
    if best_name == "LDA":
        best_clf = lda
        best_acc = acc_lda_m
    elif best_name == "SVM":
        best_clf = svm
        best_acc = acc_svm_m
    else:
        best_clf = mlp
        best_acc = acc_mlp_m

    model_bundle = {
        "best_name": best_name,
        "best_clf": best_clf,
        "csp": csp,
        "scaler": scaler,
        "channels": ch_names_global,
        "n_csp": n_csp,
        "test_accuracy": float(best_acc)
    }

    model_path = OUTPUT_DIR / "csp_best_model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model_bundle, f)
        print(f"\n[Saved] Best model ({best_name}) -> {model_path}")
    except Exception as e:
        print("[ERROR] Failed to save model:", e)

    # final accuracy print
    print("\n=== FINAL ACCURACY SCORES ===")
    print(f"LDA: {acc_lda_m:.4f}")
    print(f"SVM: {acc_svm_m:.4f}")
    print(f"MLP: {acc_mlp_m:.4f}")
    print("==============================\n")


# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    main(args.subjects)
