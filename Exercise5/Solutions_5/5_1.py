import numpy as np


# task 5.1.1 
# Define the function phi which creates the feature map for the polynomial model
def phi(x, d):
    """
    Create a feature map for polynomial regression.
    
    Parameters:
    x (float): The input value.
    d (int): The degree of the polynomial.
    
    Returns:
    numpy.ndarray: The feature map vector.
    """
    # We use np.vander with increasing=False to create the feature map
    # such that the powers are in the order x^0, x^1, ..., x^d.
    # We only need the first column (x^0 to x^d), hence `[:, :d+1]`.
    return np.vander([x], d+1, increasing=True).flatten()

# Test the function with an example
example_x = 2.0
example_d = 3
feature_map = phi(example_x, example_d)
feature_map



# task 5.1.2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv

# Function to create a feature map for polynomial regression for a vector of input values
def phi_vectorized(x, d):
    """
    Create a feature map for polynomial regression for a vector of input values.
    
    Parameters:
    x (numpy.ndarray): The input values.
    d (int): The degree of the polynomial.
    
    Returns:
    numpy.ndarray: The feature matrix.
    """
    # The input x is now a vector, so we create a feature matrix
    # where each row is the feature map for a single value of x.
    return np.vander(x, d+1, increasing=True)

# Function to define a polynomial function that can accept an array of x values
def f_polynomial(x_values, weights, degree):
    # Apply the feature map to each x value and compute the polynomial's output
    return np.dot(phi_vectorized(x_values, degree), weights)

# Load the data from a CSV file
data_path = './noisyCubicPoly.csv'  # Replace with your CSV file path
data = pd.read_csv(data_path, header=None)

# Assuming that the first row of the data is x and the second row is y
x = data.iloc[0, :].values
y = data.iloc[1, :].values

# Degree of the polynomial to fit
d = 3

# Compute the feature matrix using the vectorized feature map function
Phi = phi_vectorized(x, d)

# Compute the weights using the pseudo-inverse for numerical stability
w_hat = pinv(Phi).dot(y)

# Create a range of x values for plotting the fitted curve
x_plot = np.linspace(min(x), max(x), 200)

# Plot the training data and the fitted polynomial curve
plt.scatter(x, y, color='blue', label='Training data')
plt.plot(x_plot, f_polynomial(x_plot, w_hat, d), color='red', label=f'Fitted polynomial of degree {d}')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Fit')
plt.legend()

# Show the plot
plt.show()




# task 5.1.3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv

# Function to create a feature map for polynomial regression for a vector of input values
def phi_vectorized(x, d):
    """
    Create a feature map for polynomial regression for a vector of input values.
    
    Parameters:
    x (numpy.ndarray): The input values.
    d (int): The degree of the polynomial.
    
    Returns:
    numpy.ndarray: The feature matrix.
    """
    return np.vander(x, d+1, increasing=True)

# Function to define a polynomial function that can accept an array of x values
def f_polynomial(x_values, weights, degree):
    """
    Apply the polynomial model to a set of x values.
    
    Parameters:
    x_values (numpy.ndarray): The input values.
    weights (numpy.ndarray): The weights of the model.
    degree (int): The degree of the polynomial.
    
    Returns:
    numpy.ndarray: The predicted values.
    """
    return np.dot(phi_vectorized(x_values, degree), weights)

# Load the data from a CSV file
data_path = './noisyCubicPoly.csv'  # Replace with your actual CSV file path
data = pd.read_csv(data_path, header=None)

# Assuming that the first row of the data is x and the second row is y
x = data.iloc[0, :].values
y = data.iloc[1, :].values

# Degree of the polynomial to fit
d = 9

# Compute the feature matrix using the vectorized feature map function
Phi = phi_vectorized(x, d)

# Compute the weights using the pseudo-inverse for numerical stability (Non-regularized)
w_hat = pinv(Phi).dot(y)

# Regularization parameter
lambda_reg = 0.5

# Compute the weights using the pseudo-inverse for numerical stability (Regularized)
I = np.eye(Phi.shape[1])
w_hat_reg = pinv(Phi.T.dot(Phi) + lambda_reg * I).dot(Phi.T).dot(y)

# Create a range of x values for plotting the fitted curves
x_plot = np.linspace(min(x), max(x), 200)

# Plot the training data and the fitted polynomial curve (Non-regularized)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Training data')
plt.plot(x_plot, f_polynomial(x_plot, w_hat, d), color='red', label=f'Fitted polynomial of degree {d}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Non-Regularized Polynomial Regression Fit')
plt.legend()

# Plot the training data and the regularized fitted polynomial curve
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='blue', label='Training data')
plt.plot(x_plot, f_polynomial(x_plot, w_hat_reg, d), color='green', label=f'Regularized polynomial of degree {d}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regularized Polynomial Regression Fit')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
