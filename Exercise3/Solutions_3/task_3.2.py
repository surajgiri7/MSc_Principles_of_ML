import numpy as np

# linearKernelMatrix which takes two inputs u and v and a parameter alpha
# and returns a matrix of size len(u) x len(v) where the i,jth entry is
# alpha*u[i]*v[j] and the matrix K = K (u, v | alpha) is the linear kernel

def diffMatrix(u, v):
    u = np.array(u)[:, None]
    v = np.array(v)[None, :]
    return u - v

def prodMatrix(u, v):
    u = np.array(u)[:, None]
    v = np.array(v)[None, :]
    return u * v

def linearKernelMatrix(u, v, alpha):
    K = alpha * prodMatrix(u, v)
    return K

def gaussianKernelMatrix(u, v, alpha, sigma):
    K = alpha * np.exp((-1) * (diffMatrix(u, v) ** 2) / (2 * (sigma ** 2)))
    return K

if __name__ == '__main__': 

    u_array = 3 
    v_array = 6
    u = np.arange(u_array) 
    v = np.arange(v_array) 
    print(u)
    print(v)
    alpha = 0.5
    sigma = 1
    print(linearKernelMatrix(u, v, alpha))
    print(gaussianKernelMatrix(u, v, alpha, sigma))