import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to select maximally different points using a greedy algorithm
def select_diverse_points(X, k):
    if k >= X.shape[0]:
        raise ValueError("k must be smaller than the number of data points")
    selected_indices = []
    for _ in range(k):
        max_dist = 0
        max_idx = -1
        for i in range(X.shape[0]):
            if i not in selected_indices:
                dist_sum = sum(np.linalg.norm(X[i] - X[j]) for j in selected_indices)
                if dist_sum > max_dist:
                    max_dist = dist_sum
                    max_idx = i
        selected_indices.append(max_idx)
    
    return selected_indices

#  task 4.3.1
# Load and prepare the dataset
file_path = 'threeBlobs.csv'
data = pd.read_csv(file_path, header=None)
if data.shape[0] < data.shape[1]:
    X_blob = data.T.values  # Transpose if more columns than rows
else:
    X_blob = data.values

plt.figure(figsize=(16, 6))

# Create a subplot for each value of k
for i, k_value in enumerate([3, 4, 5], start=1):
    plt.subplot(1, 3, i)
    selected_points_indices = select_diverse_points(X_blob, k_value)
    plt.scatter(X_blob[:, 0], X_blob[:, 1], color='black', label='Data points')
    plt.scatter(X_blob[selected_points_indices, 0], X_blob[selected_points_indices, 1], 
                color='red', marker='s', label=f'Selected points (k={k_value})')
    plt.legend()
    plt.title(f'k={k_value}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


#  task 4.3.2
def plot_faces(faces, num_faces, image_shape=(19, 19)):
    selected_indices = select_diverse_points(faces.T, num_faces)
    selected_faces = faces[:, selected_indices]
    grid_size = int(np.ceil(np.sqrt(num_faces)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    axes_flat = axes.flatten()
    # Plot each of the faces on the grid
    for i in range(grid_size**2):
        ax = axes_flat[i]
        ax.axis('off')  # Turn off the axis
        if i < num_faces:
            face_image = selected_faces[:, i].reshape(image_shape)
            ax.imshow(face_image, cmap='gray', interpolation='nearest')
    plt.show()

# Now we will visualize the faces for the specified values of k
k_values = [4, 9, 16, 25, 49, 100]
# Load the new dataset from the provided .npy file
face_matrix_path = 'faceMatrix.npy'
face_matrix = np.load(face_matrix_path).astype('float')

for k in k_values:
    plot_faces(face_matrix, k)


