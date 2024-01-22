import numpy as np
import scipy.optimize as optimize

# Path and reading data
file_path = './whDatadat.sec'
data = np.genfromtxt(file_path, dtype=[('weight', 'f8'), ('height', 'f8'), ('gender', 'U1')])

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
bounds = [(0, None), (0, None), (0, None), (0, None)]  # Non-negative parameters
result = optimize.minimize(negLikelihood, initial_theta, args=(x, y_normalized), bounds=bounds)

# Print the resulting vector theta
print(result.x)

