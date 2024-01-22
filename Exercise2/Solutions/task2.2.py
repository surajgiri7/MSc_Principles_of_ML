import numpy as np
import pandas as pd
from numpy import log
from scipy.optimize import curve_fit, fsolve
import matplotlib.pyplot as plt
import csv
from scipy.optimize import minimize


def task2_2():
    # Read the CSV file
    df = pd.read_csv('myspace.csv', header=None)
    # Get the 'date' column
    date_array = df[0].values
    # Get the 'value' column
    h = df[1].values
    # Remove leading zeros
    # Convert arrays to numpy arrays
    date_array = np.array(date_array)
    h = np.array(h)
    # Find the first non-zero element in h
    first_non_zero_index = next((i for i, x in enumerate(h) if x), None)
    # If all values are zero, first_non_zero_index will be None
    if first_non_zero_index is not None:
        # Remove the leading zeros from h and corresponding entries in date_array
        date_array = date_array[first_non_zero_index:]
        h = h[first_non_zero_index:]

    shape_guess = 1
    scale_guess = 1
    params_guess = (shape_guess, scale_guess)
    t = np.arange(1, len(h) + 1)
    # Solve the system of equations using Newton's method
    result = minimize(log_likelihood, params_guess, args=(t, h))
    shape, scale = result.x
    print(shape, scale)
    # shape, scale = fsolve(newtons_method, params_guess, args=(t,h))
    # Calculate the PDF for these x values using the estimated shape and scale parameters
    pdf = weibull_pdf(t, shape, scale)
    # Plot the histogram of your data
    plt.bar(np.arange(0, len(h)), h)
    # plt.hist(h, bins=len(h), density=True, alpha=0.6, color='g')

    # Plot the Weibull PDF
    plt.plot(t, pdf * np.sum(h), 'r-')

    plt.show()


def weibull_pdf(x, shape, scale):
    if shape <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be positive.")
    return (shape / scale) * ((x / scale) ** (shape - 1)) * np.exp(-((x / scale) ** shape))


def log_likelihood(params, t, h):
    alpha, beta = params
    N = np.sum(h)
    return -(N * (log(alpha) - alpha * log(beta)) + (alpha - 1) * np.sum(h * np.log(t)) - np.sum(
        h * ((t / beta) ** alpha)))


def newtons_method(params, observation_values, counts):
    shape_alpha,scale_beta = params
    N_size = np.sum(counts)
    # Hessian matrix & gradient vector
    d2L_dA2 = -(N_size / (shape_alpha ** 2)) - np.sum(
        counts * ((observation_values / scale_beta) ** shape_alpha) * counts * (
            (np.log(observation_values / scale_beta) ** 2)))
    d2L_dB2 = (shape_alpha / (scale_beta ** 2)) * (
            N_size - (shape_alpha + 1) * np.sum(counts * ((observation_values / scale_beta) ** shape_alpha)))
    d2L_dAdB = (1 / scale_beta) * np.sum(counts * ((observation_values / scale_beta) ** shape_alpha)) + (
            shape_alpha / scale_beta) * np.sum(counts * ((observation_values / scale_beta) ** shape_alpha) * np.log(
        observation_values / scale_beta) - N_size / scale_beta)
    dL_da = (N_size / shape_alpha) - N_size * np.log(scale_beta) + np.sum(counts * np.log(observation_values)) - np.sum(
        counts * ((observation_values / scale_beta) ** shape_alpha) * np.log(observation_values / scale_beta))
    dL_db = (shape_alpha / scale_beta) * (np.sum(counts * (observation_values / scale_beta) ** shape_alpha - N_size))
    hessian_matrix = np.linalg.inv(np.array([[d2L_dA2, d2L_dAdB], [d2L_dAdB, d2L_dB2]]))
    gradient_vector = np.array([[-dL_da], [-dL_db]])
    multiplication_result = np.dot(hessian_matrix, gradient_vector)
    result = [[shape_alpha], [scale_beta]] + multiplication_result
    return np.squeeze(result)


task2_2()
