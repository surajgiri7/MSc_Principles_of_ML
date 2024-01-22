import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def weibull_pdf(x, shape, scale, amplitude):
    """
    Compute the Weibull distribution probability density function.

    Parameters:
    x : float or array_like
        The point(s) at which to evaluate the pdf.
    shape : float
        A.k.a. Alpha in the task
        The shape parameter of the Weibull distribution. Must be positive.
    scale : float
        A.k.a. Beta in the task
        The scale parameter of the Weibull distribution. Must be positive.
    amplitude: float
        A.k.a. A in the task
        The amplitude of the Weibull distribution.

    Returns:
    pdf : float or array_like
        The value(s) of the pdf at x.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be positive.")
    return amplitude * (shape / scale) * ((x / scale) ** (shape - 1)) * np.exp(-(x / scale) ** shape)

df = pd.read_csv('myspace.csv', header=None)
date_array = df[0].values
value_array = df[1].values
date_array = np.array(date_array)
h = np.array(value_array)

t = np.arange(1,len(value_array)+1)

popt, pcov = curve_fit(weibull_pdf, t, h, p0=[1, 1, 1000], bounds=([0, 0, 100], [1000., 1000., np.inf]))
for i in range(20):    
    popt, pcov = curve_fit(weibull_pdf, t, h, p0=popt, bounds=([0, 0, 100], [1000., 1000., np.inf]))
plt.plot(t, h, 'b-', label='data')
plt.plot(t, weibull_pdf(t, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.show()