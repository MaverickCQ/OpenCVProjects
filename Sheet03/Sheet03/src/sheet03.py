import numpy as np
import cv2 as cv
import random


##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray,30,250,apertureSize=3) # detect the edges
    theta_res = 2 # set the resolution of theta
    d_res = 1 # set the distance resolution
    threshold = 50 # threshold of the accumulator
    '''
    ...
    your code ...
    ...
    '''
    lines = cv.HoughLines(edges,d_res,theta_res,threshold)
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv.imshow("processed",edges) 
    cv.waitKey(0) 
    cv.destroyAllWindows()
    print(edges)
    cv.imshow("processed",img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    '''
    ...
    your code ...
    ...
    '''
	# Go through every points
    for x in range(img_edges.shape[0]):
        for y in range(img_edges.shape[1]):
			# if this is a edge point
            if img_edges[x][y] > 0:
                for theta in range(int(180 / theta_step_sz)):
                    d = x * np.sin(theta) + y * np.cos(theta)
                    accumulator[int(theta), int(d)] += 1
	# Get the detec lines
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
			# only choose the line whuch larger than threshold
            if accumulator[i][j] >= threshold:
                detected_lines.append([[j, i]])

    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray,30,250,apertureSize=3) # detect the edges    
    detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    
	# put the lines we got on the img
    for line in detected_lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv.imshow("processed",img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray,30,250,apertureSize=3) # detect the edges
    theta_res = 1 # set the resolution of theta
    d_res = 1 # set the distance resolution
    threshold = 25
    lines, accumulator = myHoughLines(edges, d_res, theta_res, threshold)
    '''
    ...
    your code ...
    ...
    '''
    #print(accumulator)
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)    
    
    cv.imshow("processed",img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()

    data = np.array([[0,0]])
    # filter accumulator so that only the point which hughr than threshold are white
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i][j] > threshold:
                accumulator[i][j] = 255
                data = np.append(data,[[j,i]],axis=0)
            else:
                accumulator[i][j] = 0
    data = np.delete(data,0,axis = 0)
    cv.imshow("processed",accumulator) 
    cv.waitKey(0) 
    cv.destroyAllWindows()
    

    radius = 10
    centroids = {}
    # initially all the points are centroid
    for i in range(len(data)):
        centroids[i] = data[i]
        
    while True:
        # Each iteration every windowss move once
        new_centroids = []
        # Go through all the certers
        for i in centroids:
            in_bandwidth = []
            centroid = centroids[i]
            # put all the point in the window together
            for featureset in data:
                if np.linalg.norm(featureset-centroid) < radius:
                    in_bandwidth.append(featureset)
            # compute the mean value in the window
            new_centroid = np.average(in_bandwidth,axis=0)
            # update centroid list
            new_centroids.append(tuple(new_centroid))
        # eliminate duplicate centroid
        uniques = sorted(list(set(new_centroids)))
        
        prev_centroids = dict(centroids)

        # update centroids for this iteration
        centroids = {}
        for i in range(len(uniques)):
            centroids[i] = np.array(uniques[i])

        optimized = True

        for i in centroids:
            # if the centroid moved, keep doing
            if not np.array_equal(centroids[i], prev_centroids[i]):
                optimized = False
            if not optimized:
                break
                
        if optimized:
            break
    print(centroids)

    # draw the line 
    for centroid in centroids.values():
        a = np.cos(centroid[1])
        b = np.sin(centroid[1])
        x0 = a*centroid[0]
        y0 = b*centroid[0]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),5)    
    
    cv.imshow("gg",img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()
    


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...

        # update clusters' centers and check for convergence
        # ...

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################

if __name__ == "__main__":
    #task_1_a()
    #task_1_b()
    #task_2()
    #task_3_a()
    #task_3_b()
    #task_3_c()
    task_4_a()

