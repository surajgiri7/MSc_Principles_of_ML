import numpy as np

def diffMatrix(u, v):
    u = np.array(u)[:, None]
    v = np.array(v)
    return u - v

def prodMatrix(u, v):
    u = np.array(u)[:, None]
    v = np.array(v)
    return u * v


if __name__ == '__main__':
    u = [1, 2, 3]
    v = [4, 5, 6, 7]
    print(diffMatrix(u, v))
    print(prodMatrix(u, v))