import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def FW_UPDATE_Z(X, centers):
    distances = np.sqrt(((X - centers[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

"""
procedure FW_UPDATE_Y(X, Y = [y_1, . . . , y_k], Z, t_max)
for t = 0, . . . , t_max - 1
    G_Y = 2 [XᵀXY ZZᵀ−XᵀXZᵀ]
    for i = 1, . . . , k
        o = argmin_j[G_Y]_ji
        y_i = y_i + (2/(t+2))[e_o −y_i]
return Y

where M = XZ^T(ZZ^T)^(-1) and Y = Z^T(ZZ^T)^(-1)
"""
"""
procedure FW_kMEANS_VERSION2(X ∈Rm×n, k, Tmax)
“randomly” initialize matrix M ∈R^(m×k)
for T = 0, . . . , T_max −1
    Z = (1/k)1_k×n
    Z = FW_UPDATE_Z(X, M, Z, t_max = 1)
    Y = (1/n) 1_n×k
    Y = FW_UPDATE_Y(X, Y , Z, t_max = 100)
    M = XY
return M, Y , Z
"""
def FW_UPDATE_Y(X, Y, Z, t_max):
    for t in range(t_max):
        GX = 2 * (X @ Y @ Z @ Z.T - X @ Z.T)
        for i in range(Y.shape[0]):
            o = np.argmin(GX[:, i])
            Y[i] = Y[i] + (t + 2) * (np.eye(Y.shape[0])[o] - Y[i])
    return Y

def FW_kMeans_Version2(X, n_clusters, n_init=10, max_iter=300, t_Z=1, t_Y=100):
    best_inertia = np.inf
    best_centers = None
    best_labels = None

    for _ in range(n_init):
        initial_centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        centers = initial_centers.copy()
        labels = None

        for _ in range(max_iter):
            labels = FW_UPDATE_Z(X, centers)
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(n_clusters)])
            if np.all(centers == new_centers):
                break
            centers = new_centers

        Z = np.zeros((n_clusters, X.shape[0]))
        Z[np.arange(len(labels)), labels] = 1
        Z = Z.T

        Y = np.ones((X.shape[0], n_clusters)) / n_clusters
        Y = FW_UPDATE_Y(X, Y, Z, t_Y)

        centers = X.T @ Y

        inertia = sum(((X[labels == j] - centers.T[j]) ** 2).sum() for j in range(n_clusters))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.T
            best_labels = labels

    return best_centers, best_labels

file_path = './threeBlobs.csv'
data = pd.read_csv(file_path, header=None, delimiter=',')
data_2d = data.values.T

centers, labels = FW_kMeans_Version2(data_2d, n_clusters=3)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', s=100)
plt.title('Clustering Result - threeBlobs.csv')
plt.show()

# face_data = np.load('faceMatrix.npy')

# face_centers, face_labels = FW_kMeans_Version2(face_data, n_clusters=10, t_Y=300)
# fig, axs = plt.subplots(2, 5, figsize=(10, 5))
# for i, ax in enumerate(axs.flatten()):
#     ax.imshow(face_centers[i].reshape(64, 64).T, cmap='gray')
#     ax.axis('off')
# plt.suptitle('Mean Faces - faceMatrix.npy')
# plt.show()


