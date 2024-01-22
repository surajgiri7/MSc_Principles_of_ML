import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from task5plot import compBBox, plot2dDataFnct

# Load the twoMoons dataset from CSV files
X = pd.read_csv('twoMoons-X-trn.csv',header=None)
y = pd.read_csv('twoMoons-y-trn.csv',header=None)

print("the X data") 
print(X.shape) 
print(y.shape) 


def clip(x, threshold):
  """Clips the values in x to a maximum value of threshold.

  Args:
    x: A numpy array.
    threshold: The maximum value to clip x to.

  Returns:
    A new numpy array with the clipped values.
  """

  return np.clip(x, 0, threshold)



# Function to train kernelized L2 SVM
def trainKernelL2SVM(matX, vecY, kFct, kPars, C=1., T=100):
    matK = kFct(matX,matX, **kPars)  # Pass both matX and matX to kFct 
    print(matK.shape)
    n = matK.shape[0]   
    print("value n")
    print(n)
    matI = np.eye(n) 
    print(matI)
    matY = np.outer(vecY, vecY) 
    print(matY.shape)
    matM = matK * matY + matY + matI / C 
    # matM = clip(matM, 10000)
    vecM = np.ones(n) / n 
    print("vECm") 
    print(vecM.shape) 
    print(type(vecM)) 
    print("mATm") 
    print(type(matM))

    for t in range(T): 
        print(t)
        beta = 2 / (t+2)
        grad = np.matmul(matM,vecM)
        vecM += beta * (matI[np.argmin(grad)] - vecM)
        print("vector M")
        print(vecM)
        print("grad")
        print(grad)
        print("matY")
        print(matY)
        print("matM")
        print(matM)

    print("outside loop")
    # print(matX.loc[:,vecM>0])
    # print(vecM[vecM>0])

    return matX.loc[:, vecM > 0], vecY[vecM > 0], vecM[vecM > 0]



# Define polynomial kernel function
def polynomial_kernel(X, Y, degree):   
     
    return (1 + np.dot(X.T, Y)) ** degree

# Visualize the decision boundaries for different degrees
degrees = [3, 4, 5]
plt.figure(figsize=(12, 4))

for i, degree in enumerate(degrees, 1):
    # Train L2 SVM with polynomial kernel   
    print("i:")
    print(i)
    print("BEFORE HAI")
    kFct = lambda X, Y: polynomial_kernel(X, Y, degree=degree) 
    print("after kFct")
    kPars = {}
    C = 1.0
    # learned_weights = trainKernelL2SVM(X, y, kFct, kPars, C, T=100)
    matXs, vecYs, vecMs = trainKernelL2SVM(X, y, kFct, kPars, C=100., T=100)

print("||||||||||||||||||")
print(matXs)
print(vecYs)
print(vecMs)

### extract class specific training data for plotting
X1 = X[:,y<0]
X2 = X[:,y>0]

print("X1")
print(X1)
print("X2")
print(X2)

### compute bounding box of training data for plotting
# bbox = compBBox(X)
# print("bbox")
# print(bbox)
# plot2dDataFnct([X1, X2], bbox, showAxes=True, filename='xmpl1.pdf')



# print(type(learned_weights))
# print(learned_weights.shape)

    # # Compute decision function
    # bbox = compBBox(X)
    # xs, ys, fs = compDecFnct(X, y, learned_weights, kFct, kPars, bbox)

# X1 = matX[:,vecY<0]
# X2 = matX[:,vecY>0]
# bbox = compBBox(X)
# plot2dDataFnct([X1, X2], bbox, showAxes=True, filename='xmpl1.pdf')
