EEG Motor Imagery Classification

Left / Right Hand Movement Detection using Machine Learning & Deep Learning

📌 Project Overview

This project focuses on the classification of left and right hand motor movements using EEG (Electroencephalography) signals. The goal is to build an end-to-end EEG signal processing and classification pipeline suitable for Brain–Computer Interface (BCI) applications.

The system preprocesses raw EEG data, extracts discriminative neural features, and applies multiple machine learning and deep learning models to accurately classify motor imagery signals.

🧠 Dataset

Source: PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)

Subjects: Up to 109 healthy subjects

Channels: Motor cortex channels (C3, C4, Cz) and ICA-specific channels

Tasks: Executed left/right hand (fist) movements

Sampling Rate: 160 Hz

⚙️ Pipeline Overview

Data Download & Loading

Automated download of EEG .edf files from PhysioNet

Preprocessing

Bandpass filtering (0.5–90 Hz)

Notch filtering (50 Hz)

Epoch extraction (ERD, ERS, MRCP windows)

ICA-based artifact removal

Feature Extraction

Time-domain features (mean, variance, energy, etc.)

Frequency-domain features (band powers, spectral entropy)

CSP (Common Spatial Patterns)

Optional Riemannian features

Model Training & Evaluation

CSP + LDA

SVM (RBF kernel with hyperparameter tuning)

EEGNet (CNN-based deep learning model)

Subject-wise and cross-validation-based evaluation

🧪 EEG Concepts Used

ERD (Event-Related Desynchronization): Power decrease during motor activation

ERS (Event-Related Synchronization): Power increase after movement

MRCP (Movement-Related Cortical Potential): Low-frequency motor preparation signal

CSP: Spatial filtering technique to maximize class separability

📊 Results

Achieved high classification accuracy for left/right hand movement detection

SVM and CSP-based models showed strong performance

Subject-independent validation ensures robustness

🛠 Tech Stack

Programming: Python

EEG Processing: MNE

ML/DL: scikit-learn, PyTorch

Signal Processing: NumPy, SciPy

Data Handling: Pandas

Visualization: Matplotlib, Seaborn
🚀
