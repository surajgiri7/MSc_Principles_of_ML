import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# path and read data
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
def kernel(x, theta, x2=None):
    if x2 is None:
        x2 = x
    theta1, theta2, theta3 = theta
    n = len(x)
    m = len(x2)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = (theta1 * np.exp(-((x[i] - x2[j])**2) / theta2**2) +
                       theta3 * x[i] * x2[j])
    return K

# negative likelihood function
def negLikelihood(theta, x, y):
    C = kernel(x, theta[:3]) + theta[3] * np.eye(len(x))
    return 0.5 * np.log(np.linalg.det(C)) + 0.5 * np.dot(y, np.linalg.solve(C, y))

# Optimize parameters
initial_theta = [1.0, 20.0, 0.5, 1.0]
bounds = [(0, None), (0, None), (0, None), (0, None)]  # Non-negative parameters
result = optimize.minimize(negLikelihood, initial_theta, args=(x, y_normalized), bounds=bounds)

# New input values for prediction
x_star = np.linspace(140, 210, 200)

# Compute Kernel Matrices for predictions
K_xx = kernel(x, result.x[:3])
K_xstar_x = kernel(x_star, result.x[:3], x)
K_xxstar = kernel(x, result.x[:3], x_star)
K_xstar_xstar = kernel(x_star, result.x[:3])

# Covariance matrix C for predictions
C = K_xx + result.x[3] * np.eye(len(x))

# Predictions
mu_star = K_xstar_x @ np.linalg.inv(C) @ y_normalized
Sigma_star = K_xstar_xstar - K_xstar_x @ np.linalg.inv(C) @ K_xxstar
sigma_star = np.sqrt(np.diag(Sigma_star))

# De-normalize predictions
mu_star_denorm = mu_star + y_mean

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Original Data')
plt.plot(x_star, mu_star_denorm, color='red', label='Prediction')
plt.fill_between(x_star, mu_star_denorm - sigma_star, mu_star_denorm + sigma_star, color='red', alpha=0.2, label='Confidence Interval')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Gaussian Process Regression Predictions')
plt.legend()
plt.show()
