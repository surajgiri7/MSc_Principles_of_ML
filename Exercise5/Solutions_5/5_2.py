import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def polynomial_kernel(x, y, b=1, d=3):
    return (np.dot(x, y) + b) ** d

# Regularization parameter
lambda_value = 0.5

data = pd.read_csv("noisyCubicPoly.csv", header=None)

X = data.iloc[0,:].values
y = data.iloc[1,:].values

# Kernel matrix
K = np.zeros((len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        K[i, j] = polynomial_kernel(X[i], X[j])

# Dual solution
alpha = np.linalg.inv(K + lambda_value * np.identity(len(X))) @ y

# Kernel vector
def polynomial_kernel_vector(x, X, b=1, d=3):
    return np.array([(polynomial_kernel(x, xi, b, d)) for xi in X])

# Predictions
predictions = np.array([np.dot(alpha, polynomial_kernel_vector(x, X)) for x in X])

cubic_interpolation = interp1d(X, predictions, kind='cubic')
X_ = np.linspace(X.min(), X.max(), 500)
Y_ = cubic_interpolation(X_)

# Plot the training data and the fitted model
plt.scatter(X, y, label='Training Data')
plt.plot(X_, Y_, label='Fitted Model', color='red')
plt.scatter(X, predictions, color='red', marker='o')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
