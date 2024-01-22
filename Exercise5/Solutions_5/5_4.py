import pandas as pd
import numpy as np
from task5plot import compBBox, plot2dDataFnct

# Load data
X_train = pd.read_csv('twoMoons-X-trn.csv', header=None).values
y_train_str = pd.read_csv('twoMoons-y-trn.csv', header=None).values.flatten()
y_train = np.array([1 if label == '+1' else -1 for label in y_train_str])

# Custom kernelized L2 SVM with polynomial kernel
def polynomial_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y) + coef0) ** degree

def compute_kernel_matrix(X, X_samples, degree=3, coef0=1):
    n_samples = X.shape[0]
    n_samples_samples = X_samples.shape[0]
    K = np.zeros((n_samples, n_samples_samples))

    for i in range(n_samples):
        for j in range(n_samples_samples):
            K[i, j] = polynomial_kernel(X[i], X_samples[j], degree=degree, coef0=coef0)

    return K


def smo(kernel_matrix, y, C, tol=1e-4, max_passes=10, epsilon=1e-8):
    n_samples = kernel_matrix.shape[0]
    alpha = np.zeros(n_samples)
    b = 0

    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(n_samples):
            E_i = svm_decision_function(alpha, y, kernel_matrix, i, b) - y[i]
            if (y[i] * E_i < -tol and alpha[i] < C) or (y[i] * E_i > tol and alpha[i] > 0):
                j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                E_j = svm_decision_function(alpha, y, kernel_matrix, j, b) - y[j]

                alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                # Update alpha values
                denominator = kernel_matrix[i, i] + kernel_matrix[j, j] - 2 * kernel_matrix[i, j]
                if abs(denominator) > epsilon:
                    delta_alpha_i = (y[i] * (E_j - E_i)) / denominator
                    delta_alpha_j = y[j] * delta_alpha_i

                    alpha[i] += delta_alpha_i
                    alpha[j] -= delta_alpha_j

                    # Clip alpha values
                    alpha[i] = np.clip(alpha[i], 0, C)
                    alpha[j] = np.clip(alpha[j], 0, C)

                    # Update b
                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i, i] - y[j] * (
                                alpha[j] - alpha_j_old) * kernel_matrix[i, j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i, j] - y[j] * (
                                alpha[j] - alpha_j_old) * kernel_matrix[j, j]

                    if 0 < alpha[i] < C and 0 < alpha[j] < C:
                        b = (b1 + b2) / 2
                    elif np.isfinite(b1) and np.isfinite(b2):
                        b = (b1 + b2) / 2
                    elif np.isfinite(b1):
                        b = b1
                    elif np.isfinite(b2):
                        b = b2
                    else:
                        b = 0  # Default value

                    num_changed_alphas += 1

        passes = passes + 1 if num_changed_alphas == 0 else 0

    # Set non-finite alpha values to 0
    alpha[~np.isfinite(alpha)] = 0

    return alpha, b




def svm_decision_function(alpha, y, kernel_matrix, i, b):
    return np.sum(alpha * y * kernel_matrix[:, i]) - b

# Visualize decision functions with different degrees
bbox = compBBox(X_train.T)
matXlist = [X_train.T]

for degree in range(3, 6):
    kernel_matrix = compute_kernel_matrix(X_train.T, X_train.T, degree=degree)
    C = 0.1  # You can adjust the value of C if needed
    alpha, b = smo(kernel_matrix, y_train, C=C)  # Assuming you have a function smo for training

    print(f'Degree {degree}: Alpha values: {alpha}, b: {b}')

    # Calculate decision function values for visualization
    xs, ys = np.meshgrid(np.linspace(bbox['xmin'], bbox['xmax'], 400),
                         np.linspace(bbox['ymin'], bbox['ymax'], 400))

    fs = np.zeros(xs.shape)  # Initialize with zeros

    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            x_sample = np.array([xs[i, j], ys[i, j]]).reshape(1, -1)
            kernel_values = compute_kernel_matrix(X_train.T, x_sample, degree=degree)
            fs[i, j] = svm_decision_function(alpha, y_train, kernel_values, 0, b)

    # Visualize the decision function
    plot2dDataFnct(matXlist,
                   bbox,
                   fctF=(xs, ys, fs),
                   showAxes=True,
                   showCont=True,
                   showFnct=True,
                   filename=f'decision_function_degree_{degree}.png')
