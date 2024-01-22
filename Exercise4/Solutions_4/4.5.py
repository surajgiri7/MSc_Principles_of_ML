import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def FW_UPDATE_Y(X, Y, Z, t_max):
  for t in range(t_max):
    G_y = 2 * (X @ X.T @ Y @ Z @ Z.T - X @ X.T @ Z.T)
    for i in range(Y.shape[1]):
      o = np.argmin(G_y[:, i])
      Y[:, i] = Y[:, i] + (2 / (t + 2)) * (np.eye(Y.shape[0])[o] - Y[:, i])
  return Y


def FW_UPDATE_Z(X, M, Z, t_max):
  for t in range(t_max):
    G_z = 2 * (M.T @ M @ Z - M.T @ X.T)
    for j in range(Z.shape[1]):
      o = np.argmin(G_z[:, j])
      Z[:, j] = Z[:, j] + (2 / (t + 2)) * (np.eye(Z.shape[0])[o] - Z[:, j])
  return Z


def FW_kMEANS_VERSION1(X, k, t_max):
  m, n = X.shape
  M = np.random.rand(k, m)
  for _ in range(t_max):
    Z = np.ones((k, n)) / k
    Z = FW_UPDATE_Z(X, M, Z, t_max=1)
    M = X @ Z.T @ (Z @ Z.T)
  return M, Z


def FW_kMEANS_VERSION2(X, k, t_max):
  n, m = X.shape
  M = np.random.rand(m, k)
  for _ in range(t_max):
    Z = np.ones((k, n)) / k
    Z = FW_UPDATE_Z(X, M, Z, t_max=1)
    Y = np.ones((n, k)) / n
    Y = FW_UPDATE_Y(X, Y, Z, t_max=100)
    M = np.dot(X.T, Y)
  return M, Y, Z


if __name__ == "__main__":
  # Load data
  data = pd.read_csv("threeBlobs.csv", header=None)
  k = 3
  if data.shape[0] < data.shape[1]:
    X_blob = data.T.values  # Transpose if more columns than rows
  else:
    X_blob = data.values

  # Run k-means
  M, Y, Z = FW_kMEANS_VERSION2(X_blob, k, 100)
  M = M.T
  print("Centroids:")
  print(M)
  print("Cluster Assignments:")
  print(Z)

  # Plot data and centroids
  plt.figure(figsize=(10, 6))
  plt.scatter(X_blob[:, 0], X_blob[:, 1], color='black', label='Data points')
  plt.scatter(M[:, 0],
              M[:, 1],
              color='red',
              marker='X',
              s=100,
              label='Centroids')
  plt.legend()
  plt.title(f'k={3}')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.tight_layout()

  # Get the cluster indices for each data point
  cluster_indices = np.argmax(Z, axis=0)

  # Plot the clusters
  plt.scatter(X_blob[:, 0],
              X_blob[:, 1],
              c=cluster_indices,
              cmap='viridis',
              marker='o',
              edgecolors='k')

  plt.title('Cluster Plot with Centroids')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.savefig('result.png')
  plt.show()

  face_matrix = np.load('./faceMatrix.npy')
  print(face_matrix.shape)
  # Reshape the face matrix if needed (assuming each row is a flattened face)
  # face_matrix = face_matrix.reshape((num_faces, height, width))

  # Number of clusters (mean faces) you want to find
  k = 16

  # Run k-meansß
  M, Y, Z = FW_kMEANS_VERSION2(face_matrix, k, t_max=100)

  """
    for the faces the matrix (361,2429) which is really computallinally expensive 
  """
