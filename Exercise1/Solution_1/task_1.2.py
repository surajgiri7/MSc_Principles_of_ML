import numpy as np
import imageio.v3 as iio
import numpy.linalg as la
import scipy.ndimage as img
import itertools as iter

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
