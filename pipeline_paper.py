# ===========================================
#      CSP MOTOR IMAGERY PIPELINE (FINAL)
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

# ---------------------------------
# CONFIGURATION
# ---------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

RUNS = [3, 7, 11]               # executed movement runs
CHANNELS = ["C3", "C4"]
TMIN, TMAX = -2.0, 0.0          # ERD extraction window
EVENT_MAPPING = {"T1": 1, "T2": 2}   # Left / Right


# ---------------------------------
# Load & Epoch Subject
# ---------------------------------
def extract_subject_epochs(subj_id):

    subj = f"S{subj_id:03d}"
    epochs_list = []
    labels_list = []

    for run in RUNS:
        f = DATA_DIR / f"{subj}R{run:02d}.edf"
        if not f.exists():
            continue

        raw = mne.io.read_raw_edf(str(f), preload=True, verbose=False)

        # Pick C3/C4
        raw.pick_channels([ch for ch in CHANNELS if ch in raw.ch_names])

        # Filtering
        raw = safe_filter(raw, l_freq=8, h_freq=30)   # μ + β band

        # Subject normalization
        raw._data = (raw._data - raw._data.mean(axis=1, keepdims=True)) / \
                    (raw._data.std(axis=1, keepdims=True) + 1e-12)

        events, _ = mne.events_from_annotations(raw, verbose=False)

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
        except:
            continue

        labels = epochs.events[:, -1] - 1
        epochs = epochs.copy().resample(160, npad="auto")

        epochs_list.append(epochs.get_data())   # shape (N, ch, time)
        labels_list.append(labels)

    if len(epochs_list) == 0:
        return None, None

    X = np.vstack(epochs_list)
    y = np.hstack(labels_list)

    return X, y

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

    for s in subjects:
        Xs, ys = extract_subject_epochs(s)
        if Xs is None:
            continue

        all_X.append(Xs)
        all_y.append(ys)
        all_groups.append(np.ones(len(ys)) * s)

        print(f"Subject {s}: {Xs.shape[0]} epochs")

    X = np.vstack(all_X)       # (N_epochs, 2 channels, time)
    y = np.hstack(all_y)
    groups = np.hstack(all_groups)

    print("Total epochs:", len(y))

    # -------------------------------
    # Group-safe train/test split
    # -------------------------------
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # -------------------------------
    # CSP (8 filters)
    # -------------------------------
    csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)

    # Fit CSP on training EEG only
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

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

    # 2) SVM (optional)
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_train_s, y_train)
    y_pred_svm = svm.predict(X_test_s)

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

    # -------------------------------
    # METRICS
    # -------------------------------
    def metrics(cm):
        TN, FP, FN, TP = cm.ravel()
        acc = (TP + TN) / (TP + TN + FP + FN)
        sen = TP / (TP + FN + 1e-9)
        spe = TN / (TN + FP + 1e-9)
        return acc, sen, spe

    cm_lda = confusion_matrix(y_test, y_pred_lda)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)

    acc_lda, sen_lda, spe_lda = metrics(cm_lda)
    acc_svm, sen_svm, spe_svm = metrics(cm_svm)
    acc_mlp, sen_mlp, spe_mlp = metrics(cm_mlp)

    print("\n=== RESULTS (CSP Pipeline) ===")

    print("\nLDA RESULTS:")
    print(cm_lda)
    print(f"Accuracy={acc_lda:.3f}, Sensitivity={sen_lda:.3f}, Specificity={spe_lda:.3f}")

    print("\nSVM RESULTS:")
    print(cm_svm)
    print(f"Accuracy={acc_svm:.3f}, Sensitivity={sen_svm:.3f}, Specificity={spe_svm:.3f}")

    print("\nMLP RESULTS:")
    print(cm_mlp)
    print(f"Accuracy={acc_mlp:.3f}, Sensitivity={sen_mlp:.3f}, Specificity={spe_mlp:.3f}")

    # save
    pd.DataFrame([{
        "acc_lda": acc_lda, "sen_lda": sen_lda, "spe_lda": spe_lda,
        "acc_svm": acc_svm, "sen_svm": sen_svm, "spe_svm": spe_svm,
        "acc_mlp": acc_mlp, "sen_mlp": sen_mlp, "spe_mlp": spe_mlp,
    }]).to_csv(OUTPUT_DIR / "csp_metrics.csv", index=False)

    plots_dir, cm_dir = ensure_plot_dirs(OUTPUT_DIR)

    scores = {
        "LDA": {"accuracy": acc_lda, "cm": cm_lda},
        "SVM": {"accuracy": acc_svm, "cm": cm_svm},
        "MLP": {"accuracy": acc_mlp, "cm": cm_mlp}
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
        
        # ------------------------------------------------


    print(f"\nPlots saved to: {plots_dir} and {cm_dir}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'csp_metrics.csv'}")

# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    main(args.subjects)
