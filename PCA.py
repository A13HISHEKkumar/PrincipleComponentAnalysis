import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance matrix
        cov = np.cov(X.T)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort by eigenvalues (descending)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idxs]

        # select top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)

    def inverse_transform(self, X_projected):
        return np.dot(X_projected, self.components.T) + self.mean
