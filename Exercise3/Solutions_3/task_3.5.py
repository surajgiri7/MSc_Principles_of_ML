import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as optimize

# Path and reading data
file_path = './whDatadat.sec'
data = np.genfromtxt(
    file_path, dtype=[('weight', 'f8'), ('height', 'f8'), ('gender', 'U1')])

# Remove outliers (where weight is -1)
data = data[data['weight'] != -1]

# Extract height and weight
x = data['height']
y = data['weight']

# Normalize y
y_mean = np.mean(y)
y_normalized = y - y_mean

# kernel function

def kernel(x, theta):
    theta1, theta2, theta3, theta4 = theta
    n = len(x)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = (theta1 * np.exp(-((x[i] - x[j])**2) / theta2**2) +
                       theta3 * x[i] * x[j] +
                       (1 if i == j else 0) * theta4)
    return K

# negative likelihood function

def negLikelihood(theta, x, y):
    C = kernel(x, theta) + theta[3] * np.eye(len(x))
    return 0.5 * np.log(np.linalg.det(C)) + 0.5 * np.dot(y, np.linalg.solve(C, y))


# Optimize parameters
initial_theta = [1.0, 20.0, 0.5, 1.0]
bounds = [(0, None), (0, None), (0, None),
          (0, None)]  # Non-negative parameters
result = optimize.minimize(
    negLikelihood, initial_theta, args=(x, y_normalized), bounds=bounds)


def sample_from_gaussian(x, theta):
    C = kernel(x, theta) + theta[3] * np.eye(len(x))
    # Sample a vector
    y_prime = np.random.multivariate_normal(np.zeros(len(C)), C)
    # De-normalize y prime
    y_denormalized = y_prime + y_mean
    return y_denormalized


def cholesky_factorization(x, theta):
    C = kernel(x, theta) + theta[3] * np.eye(len(x))
    # Cholesky factor
    L = np.linalg.cholesky(C)
    w = np.random.multivariate_normal(np.zeros(len(x)), np.eye(len(x)))
    # Compute y prime
    y_prime = L @ w
    # De-normalize y prime
    y_denormalized = y_prime + y_mean
    return y_denormalized


y_sample = sample_from_gaussian(x, result.x)
y_cholesky = cholesky_factorization(x, result.x)

# output plots
fig, ax = plt.subplots(1, 2)
ax[0].scatter(x, y_sample, color='red', label='Xj,Yj')
ax[0].scatter(x, y, color='black', label="Xj,Y'j")
ax[1].scatter(x, y_cholesky, color='red', label='Xj,Yj')
ax[1].scatter(x, y, color='black', label="Xj,Y'j")
ax[0].set_title("Sampling from N(0, C)")
ax[1].set_title("Cholesky factorization")
fig.suptitle("Sampling a fitted Gaussian processes model")
plt.legend()
plt.show()
