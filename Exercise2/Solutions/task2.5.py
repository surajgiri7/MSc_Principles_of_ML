import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load and read the data
file_path = 'myspace.csv'
myspace_data = pd.read_csv(file_path)

# Remove the zeros 
myspace_data = myspace_data[myspace_data.iloc[:, 1] > 0].reset_index(drop=True)

# Create the histogram 
h = myspace_data.iloc[:, 1].values

# Normalize h to create q
q = h * 0.98 / np.sum(h)

# Weibull distribution function
def weibull_distribution(t, alpha, beta):
    return (alpha / beta) * (t / beta) ** (alpha - 1) * np.exp(-(t / beta) ** alpha)

# Function to calculate Kullback-Leibler divergence
def kullback_leibler_divergence(params, q, t):
    alpha, beta = params
    f = weibull_distribution(t, alpha, beta)
    # Normalize the Weibull distribution so it sums to 0.98, like q
    f /= np.sum(f)
    f *= 0.98
    # avoid case for div/0 and log(0) by adding a small epsilon 
    epsilon = 1e-10
    kl_divergence = np.sum(f * np.log((f + epsilon) / (q + epsilon)))
    return kl_divergence

# Time points (t) starting from 1
t = np.arange(1, len(h) + 1)

# alpha and beta initial guesses 
initial_guess = [1.5, 100] 

# Minimize the Kullback-Leibler divergence
result = minimize(kullback_leibler_divergence, initial_guess, args=(q, t), bounds=[(1e-10, None), (1e-10, None)])

# Calculate the fitted Weibull distribution with the parameters
alpha_est, beta_est = result.x
print(alpha_est,beta_est)
fitted_weibull = weibull_distribution(t, alpha_est, beta_est)
fitted_weibull /= np.sum(fitted_weibull)
fitted_weibull *= 0.98

# Plot the distributions
plt.figure(figsize=(10, 5))
plt.plot(t, q, label='distribution qj', color='grey')
plt.plot(t, fitted_weibull, label='fitted model fj', color='red')
plt.title('Comparison of Discrete Distribution qj and Fitted Weibull Model fj')
plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.ylim(0, 0.006)
plt.yticks(np.arange(0, 0.007, 0.002))
plt.legend()
plt.show()