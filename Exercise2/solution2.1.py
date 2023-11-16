import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def remove_outliers():
    data = np.loadtxt('whDatadat.sec', dtype=object, comments='#', delimiter=None)
    w = data[:, 0].astype(float)
    h = data[:, 1].astype(float)   
    print(len(w))
    max_dev = 2 #most common max dev
    mean = np.mean(w) 
    standard_dev = np.std(w) 
    outliers_mask = np.abs((w - mean) / standard_dev) > max_dev 
    X = np.column_stack((h[~outliers_mask],w[~outliers_mask]))
    return X

def likelihood(X): 

    mean_value = np.mean(X) 
    cov_matrix = np.cov(X, rowvar=False)  # Set rowvar=False for variables in columns
    # Print the mean vector and covariance matrix
    print("Mean Vector:")
    print(mean_value)
    print("\nCovariance Matrix:")
    print(cov_matrix) 
    return mean_value,cov_matrix

def pred(X,mean_values,cov_matrix,h): 
    """
    E[w|h] = mean_w + delta * ((std_w)/(std_h))*(h-mean_h) 

    where delta = cov(w,h)/std_w*std_h
    """
    print(X.shape)
    h = X[:,:1]  
    w = X[:,1] 
    print(w)
    h_mean = np.mean(h)  
    h_std = np.std(h) 
    w_mean = np.mean(w) 
    w_std = np.std(w)  
    corr = cov_matrix/h_std*w_std 
    Exp = w_mean + corr * (w_std/h_std)*(h-h_mean) 








data_wthout_outliers = remove_outliers() 
mean_value,cov_matrix = likelihood(data_wthout_outliers) 
print(mean_value)
pred(data_wthout_outliers,mean_value,cov_matrix)


