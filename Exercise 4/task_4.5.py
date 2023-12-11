import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def FW_UPDATE_Y(X, Y, Z, t_max):
    for t in range(t_max):
        G_y = 2 * (X.T @ X @ Y @ Z @ Z.T - X.T @ X @ Z.T)
        for i in range(Y.shape[1]):
            o = np.argmin(G_y[:, i])
            Y[:, i] = Y[:, i] + (2 / (t + 2)) * (np.eye(Y.shape[0])[o] - Y[:, i])
    return Y

def FW_UPDATE_Z(X, M, Z, t_max):
    for t in range(t_max):
        G_z = 2 * (M.T @ M @ Z - M.T @ X)
        for j in range(Z.shape[1]):
            o = np.argmin(G_z[:, j])
            Z[:, j] = Z[:, j] + (2 / (t + 2)) * (np.eye(Z.shape[0])[o] - Z[:, j])
    return Z

def FW_kMEANS_VERSION2(X, k, t_max):
    m, n = X.shape
    M = np.random.rand(m, k)
    for _ in range(t_max):
        Z = np.ones((k, n)) / k
        Z = FW_UPDATE_Z(X, M, Z, t_max=1)
        Y = np.ones((n, k)) / n
        Y = FW_UPDATE_Y(X, Y, Z, t_max=100)
        M = np.dot(X, Y)
    return M, Y, Z

if __name__ == "__main__":
    # load data
    data = pd.read_csv("threeBlobs.csv", header=None)
    k = 3
    if data.shape[0] < data.shape[1]:
        X_blob = data.T.values  # Transpose if more columns than rows
        print("0 < 1")
    else:
        X_blob = data.values 
        print("0 > 1")
    
    # plot data
    plt.figure(figsize=(10, 6))

    # run k-means
    M, Y, Z = FW_kMEANS_VERSION2(X_blob, k, 100)
    plt.scatter(X_blob[:, 0], X_blob[:, 1], color='black', label='Data points')
    plt.legend()
    plt.title(f'k={3}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()

    plt.show()

