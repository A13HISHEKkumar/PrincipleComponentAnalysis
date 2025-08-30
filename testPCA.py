import numpy as np
import matplotlib.pyplot as plt
from PCA import PCA

# Generate simple 2D dataset
np.random.seed(0)
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# Fit PCA
pca = PCA(n_components=1)
pca.fit(X)
X_projected = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_projected)

print("Original shape:", X.shape)
print("Projected shape:", X_projected.shape)
print("Reconstructed shape:", X_reconstructed.shape)

# Plot original and reconstructed data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, label="Original data")
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.8, label="Reconstructed data", color="red")
plt.plot(
    [0, pca.components[0,0]*3],
    [0, pca.components[1,0]*3],
    color="blue", linewidth=2, label="First principal component"
)
plt.axis("equal")
plt.legend()
plt.show()
