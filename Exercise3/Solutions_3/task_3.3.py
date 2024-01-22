import numpy as np
import matplotlib.pyplot as plt

# ----- the following values were used by all three sub-tasks ------

# number of samples to work with 
num_samples = 5

# input vector and zero vector
vecX = np.linspace(-5.0, 15.0, 55)
vec0 = np.zeros_like(vecX)

# ------------------------------------------------------------------


# task 3.3.1

# Value for the kernel parameter 
alphaL = 1.0

# Kernel matrix KL
KL = alphaL * np.outer(vecX, vecX)

# Sample 5 vectors from the Gaussian process using linear kernel matrix
samples_linear = [np.random.multivariate_normal(vec0, KL) for _ in range(num_samples)]

# Plot results
plt.figure(figsize=(15, 4))
for y in samples_linear:
    plt.plot(vecX, y, '-o')
plt.title('Linear Samples from Gaussian Process')
plt.xlabel('Input x')
plt.ylabel('Sampled y')
plt.grid(True)
plt.show()



# task 3.3.2

# Parameters for Gaussian kernel
alpha_G = 6.0
sigma_G = 1.5

# Gaussian kernel matrix KG
KG = alpha_G * np.exp(-np.subtract.outer(vecX, vecX)**2 / (2 * sigma_G**2))

# Sample 5 vectors from the Gaussian process using the Gaussian kernel matrix
samples_gaussian = [np.random.multivariate_normal(vec0, KG) for _ in range(num_samples)]

# Plot results
plt.figure(figsize=(15, 4))
for y in samples_gaussian:
    plt.plot(vecX, y, '-o')
plt.title('Samples from Gaussian Process with Gaussian Kernel')
plt.xlabel('Input x')
plt.ylabel('Sampled y')
plt.grid(True)
plt.show()



# task 3.3.3

#Kernel parameters for the combined kernel
alpha_L = 2.0
alpha_G = 6.0
sigma_G = 1.5

# Create linear kernel matrix KL and KG and combine them
KL = alpha_L * np.outer(vecX, vecX)
KG = alpha_G * np.exp(-np.subtract.outer(vecX, vecX)**2 / (2 * sigma_G**2))
KLG = KL + KG

# Sample 5 vectors from the Gaussian process using the combined kernel matrix
samples_combined = [np.random.multivariate_normal(vec0, KLG) for _ in range(num_samples)]

# Plot results
plt.figure(figsize=(10, 6))
for y in samples_combined:
    plt.plot(vecX, y, '-o')
plt.title('Samples from Gaussian Process with Combined Kernel')
plt.xlabel('Input x')
plt.ylabel('Sampled y')
plt.grid(True)
plt.show()
