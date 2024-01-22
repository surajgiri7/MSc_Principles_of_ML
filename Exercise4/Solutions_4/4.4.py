# -----------       task 4.4.1      -----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def FW_update_Z(X, M, tmax=1):
    """
    Frank-Wolfe update procedure for the Z matrix in k-means clustering.
    X: Data matrix (m x n)
    M: Centroid matrix (m x k)
    tmax: Number of iterations for the update
    """
    n = X.shape[1]  # Number of data points
    k = M.shape[1]  # Number of clusters
    Z = np.zeros((k, n))  # Initialize Z matrix

    for t in range(tmax):
        GZ = 2 * np.dot(np.dot(M.T, M), Z) - 2 * np.dot(M.T, X)
        row_indices = np.argmin(GZ, axis=0)
        Z = np.zeros_like(Z)  # Reset Z
        Z[row_indices, np.arange(n)] = 1  # Assign each data point to the closest centroid

    return Z

def FW_kMeans_Version1(X, k, Tmax=100):
    """
    k-means clustering using the Frank-Wolfe optimization method.
    X: Data matrix (m x n)
    k: Number of clusters
    Tmax: Maximum number of iterations
    """
    m, n = X.shape
    # Randomly initialize centroids by selecting k data points
    indices = np.random.choice(n, k, replace=False)
    M = X[:, indices]

    for T in range(Tmax):
        Z = FW_update_Z(X, M)
        M = np.dot(X, Z.T) @ np.linalg.pinv(Z @ Z.T)  # Update centroids

    return M, Z

# Load data 
data_file_path = './threeBlobs.csv'
data = pd.read_csv(data_file_path, header=None)

# Transpose the dataset so each row is a data point and each column is a feature 
data_transposed = data.T

# Convert data to NumPy array 
X = data_transposed.values.T  

# Perform k-means clustering
M, Z = FW_kMeans_Version1(X, k=3)

# Extracting the cluster assignments for each data point
cluster_assignments = np.argmax(Z, axis=0)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(data_transposed[0], data_transposed[1], c=cluster_assignments, cmap='viridis', marker='o', alpha=0.7, label='Data Points')
plt.scatter(M[0, :], M[1, :], c='red', marker='x', s=100, label='Centroids')

plt.title('K-Means Clustering Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


# please run this code multiple times, sometimes it doesnt yield the right center points



# -----------       task 4.4.2      -----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# update the cluster assignments
def FW_update_Z(X, centers):
    distances = np.sqrt(((X - centers[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

# k-means clustering
def FW_kMeans_Version1(X, n_clusters, n_init=10, max_iter=300):
    best_inertia = np.inf
    best_centers = None
    best_labels = None

    for _ in range(n_init):
        initial_centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        centers = initial_centers.copy()
        labels = None

        for _ in range(max_iter):
            labels = FW_update_Z(X, centers)
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(n_clusters)])
            if np.all(centers == new_centers):
                break
            centers = new_centers

        inertia = sum(((X[labels == j] - centers[j]) ** 2).sum() for j in range(n_clusters))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

    return best_centers, best_labels

# Load data
face_data_path = './faceMatrix.npy'
face_data = np.load(face_data_path)

k_faces = 16

cluster_centers_faces, _ = FW_kMeans_Version1(face_data.T, k_faces)

# Results
image_side = int(np.sqrt(cluster_centers_faces.shape[1]))
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    mean_face = cluster_centers_faces[i].reshape(image_side, image_side)
    ax.imshow(mean_face, cmap='gray')
    ax.axis('off')

plt.show()
