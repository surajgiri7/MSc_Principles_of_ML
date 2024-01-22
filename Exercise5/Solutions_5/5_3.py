import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.genfromtxt('noisyCubicPoly.csv', delimiter=',')
x = data[0, :]
y = data[1, :]

# Experiment with different parameters
parameter_sets = [
    (0.1, 1, 2),  # Lower C, same b and d
    (10, 1, 2),  # Higher C, same b and d
    (2, 0.5, 2),  # Same C, lower b, same d
    (2, 1, 5),  # Same C and b, higher d
]

# Plot the original data
plt.scatter(x, y, label='Training Data')

# Plot fitted models for different parameter sets
for i, params in enumerate(parameter_sets):
    C, b, d = params

    # Polynomial kernel matrix
    K = (b + np.outer(x, x)) ** d

    # Solve for lambda and b
    A = np.block([[K + (1 / C) * np.identity(len(x)), np.ones((len(x), 1))],
                  [np.ones((1, len(x))), np.zeros((1, 1))]])

    b_y = np.concatenate([y, np.zeros(1)])
    lambda_b = np.linalg.solve(A, b_y)

    # Extract lambda and b
    lambda_hat = lambda_b[:-1]
    b_hat = lambda_b[-1]


    # Predictions
    def predict(x_pred):
        k_x = (b + x * x_pred) ** d
        return np.dot(k_x, lambda_hat) + b_hat


    # Plot the fitted models
    x_pred = np.linspace(min(x), max(x), 1000)
    y_pred = [predict(x_) for x_ in x_pred]
    plt.plot(x_pred, y_pred, label=f'Model {i + 1}')

plt.title('Least Squares SVM Regression with Polynomial Kernel - Parameter Experiments')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
