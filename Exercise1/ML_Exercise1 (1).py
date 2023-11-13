#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Richard Restetzki
# Exercise 1: Principles of Machine Learning

import numpy as np
import imageio.v3 as iio
import numpy.linalg as la
import scipy.ndimage as img
import itertools as iter


# In[2]:


# Task 1.1 (Beware of numerical instabilities)
############################################################################################################################


# In[3]:


# Task 1.1.1

matX = np.array([[ 1.00000, 0.00000, 0.00000],
[-1.00000, 0.00001, 0.00000]])
vecY = np.array( [ 0.00000, 0.00001, 0.00000] )
vecW = la.inv(matX @ matX.T) @ matX @ vecY
print (vecW)

# The solution is off by a slight bit because the matrix inverse is cannot be computed exactly with the given precision


# In[4]:


# Task 1.1.2

Xt_Q,Xt_R = la.qr(matX.T)
vecW = la.inv(Xt_R) @ Xt_Q.T @ vecY
print(vecW)

# The solution is accurate because we only need to invert an upper triange matrix which can be done exactly


# In[5]:


# Task 1.1.3

vecW = la.lstsq(matX.T, vecY, rcond=-1)[0]
print(vecW)

# The method definitely does not use the eqution (3) directly as the parameter "rcond=-1" enforces the la.lstsq() function to use machine precision.
# As we also used machine precision for Task 1.1.1 where (3) is solved directly, this should output in the same inaccurate result which it did not.


# Task 1.1.4
# 
# Of course, rounding errors will emerge when using machine precision floats as representation for real numbers. Therefore, one should always think about the condition of the input arguments (in this case the input-matrix matX is not very well-conditioned) and the error propagation within our calculations. Inverting just any matrix should always be avoided where possible. As seen above, the replacement of a matrix inversion by inverting an upper triangle matrix is sufficient in this case.

# In[6]:


# Task 1.2 (cellular automata, the Boolean Fourier transform, and LSQ)
############################################################################################################################


# In[7]:


# Task 1.2.1 [5 points]

matX = np.array([[ 1,  1,  1,  1, -1, -1, -1, -1],
 [ 1,  1, -1, -1,  1,  1, -1, -1],
 [ 1, -1,  1, -1,  1, -1,  1, -1]])

vecY110 = np.array([ 1,-1,-1,-1, 1,-1,-1, 1])
vecW110 = la.lstsq(matX.T, vecY110, rcond=None)[0]
vecYhat110 = matX.T @ vecW110
print("vecY110: ", vecY110, "; vecYhat110: ", vecYhat110)
print("Residual for rule 110: ", vecY110-vecYhat110)

vecY126 = np.array([ 1,-1,-1,-1,-1,-1,-1, 1])
vecW126 = la.lstsq(matX.T, vecY126, rcond=None)[0]
vecYhat126 = matX.T @ vecW126
print("vecY126: ", vecY126, "; vecYhat126: ", vecYhat126)
print("Residual for rule 126: ", vecY126-vecYhat126)

# The least squares problem cannot be fitted appropriately by linear models. In a way, we are prescribing a random 8-dimensional target vector
# to our 3-dimensional feature map and expecting the emerging problem to be of linear nature which it is unsurprisingly not


# In[8]:


# Task 1.2.2 [5 points]

def powerset(iterable):
    # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return iter.chain.from_iterable(iter.combinations(s, r) for r in range(len(s)+1))

def phi(vecX):
    # This function realizes the transformation phi by taking any vector x and returning the vector phi(x) with size 2^n
    n = vecX.shape[0]
    vecPhiX = np.zeros(pow(2,n))
    iteration = 0
    for S in powerset(range(n)):
        # This loop goes over every set S in the powerset of the index set I {0,1,...,n-1} (note that here the first index is 0 for obv. reasons)
        # The powerset of the index set is used instead of directly creating the powerset of {x_i} as the x_i's might get large leading to a high memory usage
        entry = 1
        for index in S:
            # This loop realizes the multiplication needed for computing phi_S(x)
            entry *= vecX[index]
        vecPhiX[iteration] = entry
        iteration += 1
    return vecPhiX

# print (phi(np.array([2,3,5])))
# print (phi(np.array([2,3,5,7])))


# In[9]:


# Task 1.2.3 [10 points]

def Phi(matX):
    n, size = matX.shape
    matPhiXt = np.zeros( (size, pow(2,n)) )
    for column in range(size):
        # We replace each of the vectors x_i.T (rows of matX.T) with the respective lifted vectors phi_i.T (rows of matPhiXt)
        matPhiXt[column] = phi(matX.T[column])
    return matPhiXt.T

matPhiX = Phi(matX)

vecW110 = la.lstsq(matPhiX.T, vecY110, rcond=None)[0]
vecYhat110 = matPhiX.T @ vecW110
print("vecY110: ", vecY110, "; vecYhat110: ", vecYhat110)
print("Residual for rule 110: ", vecY110-vecYhat110)

vecW126 = la.lstsq(matPhiX.T, vecY126, rcond=None)[0]
vecYhat126 = matPhiX.T @ vecW126
print("vecY126: ", vecY126, "; vecYhat126: ", vecYhat126)
print("Residual for rule 126: ", vecY126-vecYhat126)

# The errors are negligible and we get a way better fit with a residual close to 0 at machine precision level
# We had to pay for the good fit by increasing the dimension of the feature space to 2^n (=8 in this case)
# This always allows a perfect fit for a 2^n dimensional target vector (assuming all functions in the feature space are independent)


# In[10]:


# Task 1.3 (Estimating the fractal dimension of objects in pictures)
############################################################################################################################


# In[11]:


# Task 1.3.1 [20 points]

def binarize(imgF):
    # This does give DeprecationWarnings for the gaussian_filter and the binary_closing.
    # The functions were not changed because this code was explicitly given within the exercise
    imgD = np.abs(img.filters.gaussian_filter(imgF, sigma=0.50) - \
                    img.filters.gaussian_filter(imgF, sigma=1.00))
    
    return img.morphology.binary_closing(np.where(imgD < 0.1*imgD.max(), 0, 1))

def linregression(vecX, vecY):
    if vecX.shape==vecY.shape:
        # Phi is the feature matrix for the linear regression with the data vector x
        Phi = np.concatenate( ([np.ones(vecX.shape[0])], [vecX]), axis=0)
        return la.lstsq(Phi.T, vecY, rcond=-1)[0]
    else:
        print("Input error: lin regression")
        return -1

def fractaldim(imgF):
    imgBin = binarize(imgF)
    L = round(np.log2(imgBin.shape[0]))
    # We store the l's in the scalings vector because the scaling of each iteration can be computed separately and then we don't need to compute the log
    scalings = np.array(range(2,L))
    counts = np.array(range(2,L))
    for exponent in scalings:
        # pow(2,L-exponent) is exactly the number of frames each with width pow(2,exponent) that can be fitted into the total width (same for height)
        count = 0
        for i in range(pow(2,L-exponent)):
            for j in range(pow(2,L-exponent)):
                # The condition of the if statement is only true if at least one pixel of the block in the i-th row and the j-th column is "True"
                if np.any(imgBin[ (i*pow(2,exponent)):((i+1)*pow(2,exponent)), (j*pow(2,exponent)):((j+1)*pow(2,exponent))]):
                    count += 1
        counts[exponent-2] = count
    scalings *= -1
    scalings += L
    # Return only the second of the optimal model parameters which corresponds to the slope of the linear function and thus the estimate D of the dimension
    return linregression(scalings, np.log2(counts))[1]

imgTree = iio.imread('tree.png', mode='L').astype(float)
imgLightning = iio.imread('lightning.png', mode='L').astype(float)
print("Fractal dimension of the Tree: ", fractaldim(imgTree))
print("Fractal dimension of the Lightning: ", fractaldim(imgLightning))

