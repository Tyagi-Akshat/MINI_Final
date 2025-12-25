# /mnt/data/models_paper.py
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist

class ProbabilisticNeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    Simple PNN using Gaussian kernels (Parzen). Store all training points and compute
    per-class kernel sums. Predictions: argmax of summed densities.
    spread: sigma for Gaussian kernel.
    """
    def __init__(self, spread=0.1):
        self.spread = float(spread)
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_data_ = {}
        for c in self.classes_:
            self.class_data_[c] = X[y == c]
        return self
    def predict_proba(self, X):
        # compute per-class pdf sum for each sample
        X = np.asarray(X)
        n, _ = X.shape
        probs = np.zeros((n, len(self.classes_)))
        for i, c in enumerate(self.classes_):
            D = cdist(X, self.class_data_[c], metric='euclidean')  # (n_samples, n_c)
            # Gaussian kernel
            K = np.exp(-(D ** 2) / (2.0 * (self.spread ** 2)))
            probs[:, i] = np.sum(K, axis=1) / ( (np.sqrt(2*np.pi) * self.spread) * self.class_data_[c].shape[0] + 1e-12)
        # normalize
        s = probs.sum(axis=1, keepdims=True) + 1e-12
        return probs / s
    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

def build_bp(hidden_neurons=10, lr=0.05, max_iter=3500, random_state=42):
    """
    Backpropagation neural net wrapper matching paper's choice:
    - 8 inputs -> hidden layer(s) -> 1 output (binary)
    We use sigmoid activation ('logistic') and cross-entropy for classification.
    """
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_neurons,),
                        activation='logistic',
                        solver='adam',
                        learning_rate_init=lr,
                        max_iter=max_iter,
                        random_state=random_state,
                        verbose=False)
    return mlp
