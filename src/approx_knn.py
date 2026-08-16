import numpy as np
import faiss

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder


class ApproxKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=10, weights="distance", M=32, eps=1e-8):
        """
        Custom Scikit-Learn wrapper for FAISS Approximate Nearest Neighbors.

        Args:
            n_neighbors (int): Number of neighbors to use.
            weights (str): 'uniform' for majority vote, 'distance' for inverse distance weighting.
            M (int): HNSW graph parameter. Higher means more accurate but slower (default: 32).
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.M = M
        self.eps = eps

    def fit(self, X, y):
        # Validate inputs and store classes
        X, y = check_X_y(X, y)

        self.n_samples_fit_ = X.shape[0]

        # Safely encode labels to contiguous integers (0 to C-1)
        self.le_ = LabelEncoder()
        self.y_train_ = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        X_np = np.ascontiguousarray(X, dtype=np.float32)
        d = X_np.shape[1]

        # Initialize and populate the HNSW index
        self.index_ = faiss.IndexHNSWFlat(d, self.M)
        self.index_.add(X_np)

        return self

    def predict_proba(self, X):
        """Returns class probabilities for each sample."""
        check_is_fitted(self)
        X = check_array(X)
        X_np = np.ascontiguousarray(X, dtype=np.float32)

        k = min(self.n_neighbors, self.n_samples_fit_)

        # Search the index
        distances, indices = self.index_.search(X_np, k)

        # Handle FAISS -1 padding for missing neighbors
        invalid_mask = (indices == -1)
        safe_indices = np.maximum(indices, 0)
        neighbor_labels = self.y_train_[safe_indices]

        N, k_actual = neighbor_labels.shape
        num_classes = len(self.classes_)

        if self.weights == "distance":
            # FAISS returns squared L2. Take sqrt to match sklearn's standard inverse distance.
            safe_distances = np.maximum(distances, 0.0)
            weights = 1.0 / (np.sqrt(safe_distances) + self.eps)
        else:
            # Standard uniform majority vote
            weights = np.ones_like(distances, dtype=np.float32)

        # Zero out the weights of invalid (padded) neighbors
        weights[invalid_mask] = 0.0

        # Vectorized Weighted Voting
        row_offsets = np.arange(N) * num_classes
        flat_labels = (neighbor_labels + row_offsets[:, None]).ravel()
        flat_weights = weights.ravel()

        # Single C-level bincount for the entire dataset
        flat_counts = np.bincount(flat_labels, weights=flat_weights, minlength=N * num_classes)

        # Reshape back to rows (N, num_classes)
        counts_2d = flat_counts.reshape(N, num_classes)

        # Normalize rows to create probabilities
        row_sums = counts_2d.sum(axis=1, keepdims=True)
        # Avoid division by zero if all neighbors were invalid
        row_sums[row_sums == 0.0] = 1.0 

        return counts_2d / row_sums

    def predict(self, X):
        # Predict uses the argmax of predict_proba
        proba = self.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        return self.classes_[preds]