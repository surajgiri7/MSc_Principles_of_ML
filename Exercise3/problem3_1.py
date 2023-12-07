import numpy as np

def diffMatrix(u, v):
    u = np.array(u)[:, None]
    v = np.array(v)[None, :]
    return u - v

def prodMatrix(u, v):
    u = np.array(u)[:, None]
    v = np.array(v)[None, :]
    return u * v


if __name__ == '__main__': 

    num_uarray = 3 
    num_varray = 6
    u = np.arange(num_uarray) 
    v = np.arange(num_varray) 
    print(u)
    print(v)
    print(diffMatrix(u, v))
    print(prodMatrix(u, v))
