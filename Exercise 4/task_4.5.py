




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# update the cluster assignments
def FW_update_Z(X, centers):
    distances = np.sqrt(((X - centers[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

def FW_update_Y(X, centers, labels, t_max):
    for i in range(t_max):
        for j in range(n_clusters):
            Y[i,j] = np.sum(X[i] - centers[j]) ** 2
    return Y

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
            new_labels = np.zeros_like(labels)
            for i in range(n_clusters):
                
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

# Load file
file_path = './threeBlobs.csv'
data = pd.read_csv(file_path, header=None, delimiter=',')

# Reshape data into 2D
data_2d = data.values.T

# Perform k-means clustering
n_clusters = 3
cluster_centers_2d, labels_2d = FW_kMeans_Version1(data_2d, n_clusters)

# Setting up the plot
cluster_colors = ['orange','blue', 'green']
data_markers = ['o'] * n_clusters  
cluster_markers = ['s', 's', 's']  

# Plot the results 
plt.figure(figsize=(10, 8))

for i in range(n_clusters):
    plt.scatter(data_2d[labels_2d == i, 0], data_2d[labels_2d == i, 1], 
                c=cluster_colors[i], marker=data_markers[i], label=f'Cluster {i}')

for i, center in enumerate(cluster_centers_2d):
    plt.scatter(center[0], center[1], 
                c=cluster_colors[i], marker=cluster_markers[i], 
                s=100, edgecolors='black', label=f'Center {i}')

plt.title('2D K-Means Clustering with Custom Colors and Markers')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.show()



