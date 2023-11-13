import numpy as np
import imageio.v3 as iio
import numpy.linalg as la
import scipy.ndimage as img
import itertools as iter

# Task 1.1.1

matX = np.array([[ 1.00000, 0.00000, 0.00000],
[-1.00000, 0.00001, 0.00000]])
vecY = np.array( [ 0.00000, 0.00001, 0.00000] )
vecW = la.inv(matX @ matX.T) @ matX @ vecY
print (vecW)

# The solution is off by a slight bit because the matrix inverse is cannot be computed exactly with the given precision



# Task 1.1.2

Xt_Q,Xt_R = la.qr(matX.T)
vecW = la.inv(Xt_R) @ Xt_Q.T @ vecY
print(vecW)

# The solution is accurate because we only need to invert an upper triange matrix which can be done exactly


# Task 1.1.3

vecW = la.lstsq(matX.T, vecY, rcond=-1)[0]
print(vecW)

# The method definitely does not use the eqution (3) directly as the parameter "rcond=-1" enforces the la.lstsq() function to use machine precision.
# As we also used machine precision for Task 1.1.1 where (3) is solved directly, this should output in the same inaccurate result which it did not.


# Task 1.1.4
#
# Of course, rounding errors will emerge when using machine precision floats as representation for real numbers. Therefore, one should always think about the condition of the input arguments (in this case the input-matrix matX is not very well-conditioned) and the error propagation within our calculations. Inverting just any matrix should always be avoided where possible. As seen above, the replacement of a matrix inversion by inverting an upper triangle matrix is sufficient in this case.

