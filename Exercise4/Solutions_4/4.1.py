import numpy as np
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt


def gradientF(vector: np.ndarray, n: int, data: np.ndarray) -> np.ndarray:
    return 2 * data.transpose() @ data @ (vector - np.ones(n) / n)


def frank_wolfe(T:int, n: int, data: np.ndarray) -> np.ndarray:
    vecW = np.ones(n) / n  # center of the standard simplex
    for t in tqdm(range(T)):
        beta = 2 / (t + 2)
        vecG = gradientF(vecW, n, data)  # depends on the problem at hand
        imin = np.argmin(vecG)
        vecW *= (1 - beta)
        vecW[imin] += beta
    return data @ vecW


if __name__ == '__main__':
    with open('threeBlobs.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array(list(reader), dtype=float)
    # Groud truth of mean values for data
    gr = np.array([np.mean(data[0]), np.mean(data[1])])
    print(gr)
    n = data.shape[1]
    T = [10,100,1000,10000]
    vecWhat = []
    for i in T:
        vecWhat.append(frank_wolfe(i,n,data))
    print(vecWhat)
    plt.scatter(data[0],data[1])
    plt.scatter(gr[0], gr[1], marker='s', label=f"ground")
    for i in range(len(vecWhat)):
        plt.scatter(vecWhat[i][0],vecWhat[i][1],marker='s',label=f"{T[i]}")
    plt.legend()
    plt.show()
