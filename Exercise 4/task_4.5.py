import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def FW_UPDATE_Z(X, M, Z, t_max):
    for t in range(t_max):
        G_z = 2 * (M.T @ M @ Z - M.T @ X)
        for i in range(Z.shape[0]):  
            o = np.argmin(G_z[i, :]) 
            Z[i, :] = Z[i, :] + (2 / (t + 2)) * (np.eye(Z.shape[1])[o] - Z[i, :])  # update the i-th row
    return Z

def FW_UPDATE_Y(X, Y, Z, t_max):
    for t in range(t_max):
        G_y = 2 * (X.T @ X @ Y @ Z @ Z.T - X.T @ X @ Z.T)
        for i in range(Y.shape[1]):
            o = np.argmin(G_y[:, i])
            Y[:, i] = Y[:, i] + (2 / (t + 2)) * (np.eye(Y.shape[0])[o] - Y[:, i])
    return Y

def FW_kMEANS_VERSION2(X, k, t_max):
    m, n = X.shape
    M = np.random.rand(m, k)
    for _ in range(t_max):
        Z = np.ones((k, n)) / k
        Z = FW_UPDATE_Z(X, M, Z, t_max=1)
        Y = np.ones((n, k)) / n
        Y = FW_UPDATE_Y(X, Y, Z, t_max=100)
        M = X @ Y
    return M, Y, Z

if __name__ == "__main__":
    # load data
    data = pd.read_csv("threeBlobs.csv", header=None)
    k = 3
    if data.shape[0] < data.shape[1]:
        X_blob = data.T.values  # Transpose if more columns than rows
    else:
        X_blob = data.values 

    print(X_blob.shape)
    print(X_blob)

    # run k-means
    M, Y, Z = FW_kMEANS_VERSION2(X_blob, k, 100)
    # print(M.shape) # Centroids
    # print(M)
    # print(Y.shape) # Assignment matrix
    # print(Y)
    # print(Z.shape) # Clusters
    # print(Z)

    # plot the result of k-means M, Y and Z
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']  # Define colors for each cluster
    for i in range(k):  # k is the number of clusters
        plt.scatter(M[i, 0], M[i, 1], color=colors[i], marker='x', s=200, label=f'Centroid {i+1}')
        plt.scatter(Y[:, i], Z[i, :], color=colors[i], marker='o', s=200, label=f'Cluster {i+1}')
        plt.scatter(X_blob[:, 0], X_blob[:, 1], color=colors[i], label='Data points')

    # plt.scatter(M[:, 0], M[:, 1], color='black', label='Data points')
    plt.legend()
    plt.title(f'k Means Clustering - unconventional k-means clustering (part 2)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

