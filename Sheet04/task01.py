import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here
def getGradient(Im, V, nodesSize):
    nodes = np.zeros(nodesSize) # 9x20
    row = nodes.shape[0]
    col = nodes.shape[1]

    for i in range(row):
        # 9x1 --> 3x3
        r = i % 3  - 1# which row in 3x3 
        c = i // 3  - 1# which column in 3x3
        for j in range(col):            
            x = V[j][0]+r
            y = V[j][1]+c
            gx = Im[x+1][y] - Im[x][y] # Gx
            gy = Im[x][y-1] - Im[x][y] # Gy           
            nodes[i][j] = -(gx*gx+gy*gy)# -|| G^2 ||
    return nodes

def getDistance(pointA, pointB):
    d2 = pow(pointA[0]-pointB[0],2) + pow(pointA[1]-pointB[1],2)
    return pow(d2, 0.5)

def averageDistance(V):
    totalDis = 0
    n = len(V)
    for i in range(n):        
        if i == len(V)-1:# Vn and V1
            totalDis += getDistance(V[0], V[i])
        else:
            totalDis += getDistance(V[i+1], V[i])
    
    return totalDis/n

def updateNodes(nodes, V):
    alpha = 10
    row = nodes.shape[0]
    col = nodes.shape[1]
    newNodes = np.copy(nodes)
    lastnodes = np.zeros((row,col))
    avrDis = averageDistance(V)

    for i in range(row):
        currentR = i % 3  - 1# which row in 3x3 
        currentC = i // 3  - 1# which column in 3x3
        
        for j in range(col):
            currentX = V[j][0] + currentR # x of the point that we are computing
            currentY = V[j][1] + currentC # y of the point that we are computing
            minimum = 9999
            last = 0
            for k in range(row):
                r = k % 3  - 1# which row in 3x3 
                c = k // 3  - 1# which column in 3x3
                if j != 0:                    
                    x = V[j-1][0] + r
                    y = V[j-1][1] + c
                    elastic = alpha * pow(getDistance([currentX,currentY], [x, y])-avrDis,2) 
                    cost = newNodes[k][j-1] + elastic + newNodes[i][j]
                    if cost < minimum:
                        minimum = cost
                        lastnodes[i][j] = k
                else: # Vn & V1
                    x = V[col-1][0]+r
                    y = V[col-1][1]+c
                    elastic = alpha * pow(getDistance([currentX,currentY], [x, y])-avrDis,2) 
                    cost = newNodes[k][col-1] + elastic + newNodes[i][j]
                    if cost < minimum:
                        minimum = cost
                        lastnodes[i][j] = k
            newNodes[i][j] = minimum
    
    return newNodes, lastnodes

def updateV(nodes, V, lastnodes):
    newV = np.copy(V)
    row = nodes.shape[0]
    col = nodes.shape[1]
    row_of_minimum = 0
    minValue = nodes[0][col-1]

    # find the minimum of the last v
    for i in range(1, row):
        if nodes[i][col-1] < minValue:
            minValue = nodes[i][col-1]
            row_of_minimum = i

    r = row_of_minimum % 3  - 1# which row in 3x3 
    c = row_of_minimum // 3  - 1# which column in 3x3
    # assign a new position to the last v
    newV[col-1][0] = newV[col-1][0] + r # x
    newV[col-1][1] = newV[col-1][1] + c # y
    
    # sice we get the position of the last v, keep finding all the v
    for j in range(col-2,-1,-1): 
        last = int(lastnodes[row_of_minimum][j+1])
        r = last % 3  - 1# which row in 3x3 
        c = last // 3  - 1# which column in 3x3
        newV[j][0] = newV[j][0] + r
        newV[j][1] = newV[j][1] + c
        row_of_minimum = last
    return newV

# ------------------------


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200

    # ------------------------
    # your implementation here    
    nodes = getGradient(Im, V, (9,20))  
    # ------------------------

    for t in range(n_steps):
        # ------------------------
        # your implementation here 
        tmp = nodes
        nodes, lastnodes = updateNodes(tmp, V)
        V = updateV(nodes, V, lastnodes)
        nodes = getGradient(Im, V, (9,20))
        # ------------------------

        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    #run('images/coffee.png', radius=100)
