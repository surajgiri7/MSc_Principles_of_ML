import numpy as np
import pandas as pd
from task5plot import compBBox, plot2dDataFnct

X_train = pd.read_csv('twoMoons-X-trn.csv', header=None)
y_train = pd.read_csv('twoMoons-y-trn.csv', header=None)

print(X_train.shape)
print(y_train.shape)

import scipy.spatial as spt

def SQEDMAB(A,B): 
    return spt.distance.cdist(A.T, B.T, 'sqeuclidean')

def gaussKernelVector(matXtst, matXtrn, **kPars):
    sigm = kPars['sigm'] if 'sigm' in kPars else 1
    if matXtst.ndim == 1:
        dist = np.sum((matXtrn.T-matXtst)**2, axis=1)
    else:
        dist = SQEDMAB(matXtst, matXtrn)
    return np.exp(-0.5/sigm**2 * dist)

def gaussKernelMatrix(matXtrn, **kPars):
    sigm = kPars['sigm'] if 'sigm' in kPars else 1
    return np.exp(-0.5/sigm**2 * SQEDMAB(matXtrn, matXtrn))

def gaussKernelVector(matXtst, matXtrn, **kPars):
    sigm = kPars['sigm'] if 'sigm' in kPars else 1
    if matXtst.ndim == 1:
        dist = np.sum((matXtrn.T-matXtst)**2, axis=1)
    else:
        dist = SQEDMAB(matXtst, matXtrn)
    return np.exp(-0.5/sigm**2 * dist)


### training data points and labels
matX = X_train
vecY = y_train
### kernel parameters and functions
kParams = {'sigm' : 2.}
kMatFct = gaussKernelMatrix
kVecFct = gaussKernelVector
### compute support vectors, their labels, and Lagrange multipliers

def trainKernelL2SVM(matX, vecY, kFct, kPars, C=1., T=10_000):
    matK = kFct(matX, **kPars)
    _, n = matK.shape
    matI = np.eye(n)
    matY = np.outer(vecY, vecY)
    matM = matK * matY + matY + matI / C
    vecM = np.ones(n) / n
    for t in range(T):
        beta = 2 / (t+2)
        grad = matM @ vecM
        vecM += beta * (matI[np.argmin(grad)] - vecM)
    ### return support vectors, their labels and Lagarange multipliers
    return matX[:,vecM>0], vecY[vecM>0], vecM[vecM>0]

matXs, vecYs, vecMs = trainKernelL2SVM(matX, vecY, kMatFct, kParams, C=100.)

### extract class specific training data for plotting
X1 = matX[:,vecY<0]
X2 = matX[:,vecY>0]
### compute bounding box of training data for plotting
bbox = compBBox(matX)
### compute decision function for plotting
decFnct = compDecFnct(matXs, vecYs, vecMs, kVecFct, kParams, bbox)

def compDecFnct(matX, vecY, vecM, kFct, kPars, bbox, nx=512):
    w = bbox['xmax'] - bbox['xmin']
    h = bbox['ymax'] - bbox['ymin']
    ny = int(nx*h/w)
    xs, ys = np.meshgrid(np.linspace(bbox['xmin'], bbox['xmax'], nx),
                         np.linspace(bbox['ymin'], bbox['ymax'], ny))
    matXtst = np.vstack((xs.flatten(), ys.flatten()))
    vecKtst = kFct(matXtst, matX, **kPars)
    vecYtst = np.sign(np.sum(vecKtst * vecY * vecM, axis=1) + vecY @ vecM)
    ### return a triple of arrays of pixel x- and y-coordinates ### and of the correspponding function values f(x,y)
    return xs, ys, vecYtst.reshape(ny, nx)

plot2dDataFnct([X1, X2], bbox, showAxes=True, filename='xmpl1.pdf')
plot2dDataFnct([X1, X2], bbox, fctF=decFnct, showAxes=True, filename='xmpl2.pdf') 
plot2dDataFnct([X1, X2], bbox, fctF=decFnct, showAxes=False, showCont=True, filename='xmpl3.pdf')

















# def trainKernelL2SVM(matX, vecY, kFct, kPars, C=1., T=10_000):
#     matK = kFct(matX, **kPars)
#     _, n = matK.shape
#     matI = np.eye(n)
#     matY = np.outer(vecY, vecY)
#     matM = matK * matY + matY + matI / C
#     vecM = np.ones(n) / n
#     for t in range(T):
#         beta = 2 / (t+2)
#         grad = matM @ vecM
#         vecM += beta * (matI[np.argmin(grad)] - vecM)
#     ### return support vectors, their labels and Lagarange multipliers
#     return matX[:,vecM>0], vecY[vecM>0], vecM[vecM>0]

# def polynomial_kernel(matXtst, matXtrn, **kPars):
#     d = kPars['d'] if 'd' in kPars else 3
#     b = kPars['b'] if 'b' in kPars else 1
#     return (b + np.dot(matXtst.T, matXtrn)) ** d

# for d in range(3, 6):
#     kParams = {'d': d}
#     kMatFct = polynomial_kernel
#     kVecFct = polynomial_kernel
#     matXs, vecYs, vecMs = trainKernelL2SVM(X_train, y_train, kMatFct, kParams, C=100.)
#     bbox = compBBox(X_train)
#     decFnct = compBBox(matXs, vecYs, vecMs, kVecFct, kParams, bbox)
#     plot2dDataFnct([X_train[y_train.flatten()<0], X_train[y_train.flatten()>0]], bbox, fctF=decFnct, showAxes=True)