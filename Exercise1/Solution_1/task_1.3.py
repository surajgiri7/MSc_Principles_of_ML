import numpy as np
import imageio.v3 as iio
import numpy.linalg as la
import scipy.ndimage as img
import itertools as iter
# Task 1.3 (Estimating the fractal dimension of objects in pictures)
############################################################################################################################


# In[11]:


# Task 1.3.1 [20 points]

def binarize(imgF):
    # This does give DeprecationWarnings for the gaussian_filter and the binary_closing.
    # The functions were not changed because this code was explicitly given within the exercise
    imgD = np.abs(img.filters.gaussian_filter(imgF, sigma=0.50) - \
                  img.filters.gaussian_filter(imgF, sigma=1.00))

    return img.morphology.binary_closing(np.where(imgD < 0.1 * imgD.max(), 0, 1))


def linregression(vecX, vecY):
    if vecX.shape == vecY.shape:
        # Phi is the feature matrix for the linear regression with the data vector x
        Phi = np.concatenate(([np.ones(vecX.shape[0])], [vecX]), axis=0)
        return la.lstsq(Phi.T, vecY, rcond=-1)[0]
    else:
        print("Input error: lin regression")
        return -1


def fractaldim(imgF):
    imgBin = binarize(imgF)
    L = round(np.log2(imgBin.shape[0]))
    # We store the l's in the scalings vector because the scaling of each iteration can be computed separately and then we don't need to compute the log
    scalings = np.array(range(2, L))
    counts = np.array(range(2, L))
    for exponent in scalings:
        # pow(2,L-exponent) is exactly the number of frames each with width pow(2,exponent) that can be fitted into the total width (same for height)
        count = 0
        for i in range(pow(2, L - exponent)):
            for j in range(pow(2, L - exponent)):
                # The condition of the if statement is only true if at least one pixel of the block in the i-th row and the j-th column is "True"
                if np.any(imgBin[(i * pow(2, exponent)):((i + 1) * pow(2, exponent)),
                          (j * pow(2, exponent)):((j + 1) * pow(2, exponent))]):
                    count += 1
        counts[exponent - 2] = count
    scalings *= -1
    scalings += L
    # Return only the second of the optimal model parameters which corresponds to the slope of the linear function and thus the estimate D of the dimension
    return linregression(scalings, np.log2(counts))[1]


imgTree = iio.imread('tree.png', mode='L').astype(float)
imgLightning = iio.imread('lightning.png', mode='L').astype(float)
print("Fractal dimension of the Tree: ", fractaldim(imgTree))
print("Fractal dimension of the Lightning: ", fractaldim(imgLightning))

