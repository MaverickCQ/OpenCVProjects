import numpy as np
import numpy.linalg as la
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# --- this are all the imports you are supposed to use!
# --- please do not add more imports!

n_views = 101
n_features = 215

# --- add your code here ---
def remove_trans(x):
    row = x.shape[0]
    col = x.shape[1]

    for i in range(col):
        sumX = 0
        sumY = 0
        for j in range(row):
            if j%2 == 0:
                sumX += x[j][i]
            else:
                sumY += x[j][i]
        
        avgX = sumX / (row/2)
        avgY = sumY / (row/2)

        for j in range(row):
            if j%2 == 0:
                x[j, i] -= avgX
            else:
                x[j, i] -= avgY
    return x

def computeShape(x):
    # remove the translation by substract the average
    x = remove_trans(x.copy()) 
    
    # compute SVD
    U, sigma, Vt = np.linalg.svd(x)

    D_sqrt = np.array([[pow(sigma[0],0.5), 0, 0],
    [0, pow(sigma[1], 0.5), 0],
    [0, 0, pow(sigma[2], 0.5)]    
    ])
    
    # U dot D^(0.5)
    motion = np.dot(U[:,:3],D_sqrt)
    # D^(0.5) dot Vt(3xn)
    S = np.dot(D_sqrt, Vt[:3,:])



    return motion, S


def task2():
    data_matrix = np.genfromtxt('./data/data_matrix.txt', dtype=float, skip_header=0)
    
    # (3xn)
    motion, S = computeShape(data_matrix.copy())


    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(S.T[:, 0], S.T[:, 1], S.T[:, 2])
    plt.show()

task2()




# --- how to plot 3d ---
"""pts3d = np.random.randint(0, 10, (100, 3))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])

plt.show()"""
