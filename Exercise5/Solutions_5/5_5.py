import csv
from scipy.optimize import minimize
import numpy as np

def load_data(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in zip(*csvreader):

            data.append(list(map(float, row)))
    return np.array(data)


def compute_z(X):
    XtX = np.dot(X.T, X)
    z = np.diagonal(XtX)
    return z


def compute_u(X, z):
    n = X.shape[1]
    def objective(u):
        return np.dot(u.T, np.dot(X.T, X).dot(u)) - np.dot(u.T, z)
    u_initial = np.random.rand(n)  # Initial guess for u
    result = minimize(objective, u_initial)
    u = result.x
    return u


def center_radius(X, u):
    c = np.dot(X, u)
    XtX = np.dot(X.T, X)
    z = np.diagonal(XtX)
    r = np.sqrt(np.dot(u.T, z) - np.dot(u.T, np.dot(XtX, u)))
    return c, r

data = load_data('./threeBlobs.csv')

z = compute_z(data.T)
u = compute_u(data.T,z)

c, r = center_radius(data.T, u)

print(f'center: {c}, radius:{r}')


