import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def remove_outliers():
    """
    Returns the data array X without outliers
    """
    data = np.loadtxt('whDatadat.sec', dtype=object, comments='#', delimiter=None)
    w = data[:, 0].astype(float)
    h = data[:, 1].astype(float)   
    print(len(w))
    max_dev = 2 # most common max deviation from the mean
    mean = np.mean(w) 
    standard_dev = np.std(w) 
    outliers_mask = np.abs((w - mean) / standard_dev) > max_dev 
    X = np.column_stack((h[~outliers_mask],w[~outliers_mask]))
    return X

def likelihood(X): 
    """
    Returns the maximum likelihood parameters for a given data matrix X
    """
    mean_value = np.mean(X, axis=0) # axis=0 for column-wise mean
    cov_matrix = np.cov(X, rowvar=False)  # Set rowvar=False for variables in columns
    # Print the mean vector and covariance matrix
    print("Mean Vector:")
    print(mean_value)
    print("\nCovariance Matrix:")
    print(cov_matrix) 
    return mean_value,cov_matrix

def pred(X, mean_values, cov_matrix, h):
    """
    Returns the conditional expectation E[w|h] for a given height h 
    and the maximum likelihood parameters mean_values and cov_matrix using the formula:
    E[w|h] = E[w] + cov(hw) * (h - E[h]) / cov(hh)
    """
    # Extracting mean height and weight values
    mean_height = mean_values[0]
    mean_weight = mean_values[1]
    
    # Finding the indices of height and weight in the dataset
    h_index = 0
    w_index = 1
    
    # Calculating conditional expectation E[w|h]
    cov_hw = cov_matrix[h_index][w_index]
    cov_hh = cov_matrix[h_index][h_index]
    
    conditional_expectation = mean_weight + cov_hw * (h - mean_height) / cov_hh
    
    return conditional_expectation


if __name__ == "__main__":    
    data_wthout_outliers = remove_outliers() 
    mean_value,cov_matrix = likelihood(data_wthout_outliers) 
    # Predicting the weight for a height of 140, 150, 160, 170, 180, 190, 200, 210
    h = [140, 150, 160, 170, 180, 190, 200, 210]
    for i in h:
        conditional_mean_w = pred(data_wthout_outliers, mean_value, cov_matrix, i)
        print("\nPredicted weight for a height of ", i, " cm: ", conditional_mean_w, " kg.")

