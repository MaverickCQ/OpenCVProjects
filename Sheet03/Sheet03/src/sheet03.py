import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt


##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    #img_blur = cv.medianBlur(img,5)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray, 50, 150, apertureSize  = 3)
    #minLineLength = 100
    #maxLineGap = 10
    lines = cv.HoughLines(edges,1,np.pi/180,50)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        
    circles = cv.HoughCircles(img_gray,cv.HOUGH_GRADIENT,1,20, param1=50,param2=30,
                              minRadius=0, maxRadius=0)
    cv.imwrite("../images/houghLine.png",img)

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
            in_circle = []
            centroid = centroids[i]
            # put all the point in the window together
            for featureset in data:
                if np.linalg.norm(featureset-centroid) < radius:
                    in_circle.append(featureset)
            # compute the mean value in the window
            new_centroid = np.average(in_circle,axis=0)
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

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    '''
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    #clusters = [[] for i in range(k)]\
    clusters = np.zeros(len(data))
    
    #print (centers.shape)
    # initialize centers using some random points from data
    # ....
    rand_indices = np.random.choice(data.shape[0], size = k)
    centroids = data[rand_indices]
        
        
    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...
        for i in range(len(data)):
            dist_to_centroids = dist(data[i], centroids)
            cluster = np.argmin(dist_to_centroids)
            clusters[i] = cluster

        # update clusters' centers and check for convergence
        # ...
        for i in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == i]
            centroids[i] = np.mean(points, axis=0)

        iterationNo += 1
        print('iterationNo = ', iterationNo)
        '''
    attempts=10
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers = cv.kmeans(data,k,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    res = centers[label.flatten()]

    return res, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    img_float = np.float32(img)
    img_gray = cv.cvtColor(img_float, cv.COLOR_BGR2GRAY)    
    
    result_image, centers = myKmeans(img_gray, 2)
  #  result_image = result_image.reshape((img_gray.shape))
    cv.imwrite("../images/intensity_2.png",result_image)
    result_image, centers = myKmeans(img_gray, 4)
  #  result_image = result_image.reshape((img_gray.shape))
    cv.imwrite("../images/intensity_4.png",result_image)
    result_image, centers = myKmeans(img_gray, 6)
   # result_image = result_image.reshape((img_gray.shape))
    cv.imwrite("../images/intensity_6.png",result_image)


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    img_float = np.float32(img) 
    img_reshape = img_float.reshape((-1,3))
    
    result_image, centers = myKmeans(img_reshape, 2)
    result_image = result_image.reshape((img.shape))
    cv.imwrite("../images/color_2.png",result_image)
    result_image, centers = myKmeans(img_reshape, 4)
    result_image = result_image.reshape((img.shape))
    cv.imwrite("../images/color_4.png",result_image)
    result_image, centers = myKmeans(img_reshape, 6)
    result_image = result_image.reshape((img.shape))
    cv.imwrite("../images/color_6.png",result_image)
    


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    img_float = np.float32(img) 
    img_gray  = cv.cvtColor(img_float, cv.COLOR_BGR2GRAY)
    
    img_reshape = img_gray.reshape((-1,2))
    
    result_image, centers = myKmeans(img_reshape, 2)
    result_image = result_image.reshape((img_gray.shape))
    cv.imwrite("../images/intensity_sclaed_2.png",result_image)
    result_image, centers = myKmeans(img_reshape,4)
    result_image = result_image.reshape((img_gray.shape))
    cv.imwrite("../images/intensity_sclaed_4.png",result_image)
    result_image, centers = myKmeans(img_reshape, 6)
    result_image = result_image.reshape((img_gray.shape))
    cv.imwrite("../images/intensity_sclaed_6.png",result_image)
    
    


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
	# construct the D matrix
    D = np.array([
	[2.2,0,0,0,0,0,0,0],
	[0,2.1,0,0,0,0,0,0],
	[0,0,2.6,0,0,0,0,0],
	[0,0,0,3,0,0,0,0],
	[0,0,0,0,3,0,0,0],
	[0,0,0,0,0,3,0,0],
	[0,0,0,0,0,0,3,0],
	[0,0,0,0,0,0,0,2]
	] )
	# construct the W matrix
    W = np.array([
	[0,1,0.2,1,0,0,0,0],
	[1,0,0.1,0,1,0,0,0],
	[0.2,0.1,0,1,0,1,0.3,0],
	[1,0,1,0,0,1,0,0],
	[0,1,0,0,0,0,1,1],
	[0,0,1,1,0,0,1,0],
	[0,0,0.3,0,1,1,0,1],
	[0,0,0,0,1,0,1,0]
	] )
    '''
    ...
    your code ...
    ...
    '''
    # compute D^(1/2)
    D_rs_inv = np.sqrt(D)  
    # Get D^(-1/2)   
    for i in range(D_rs_inv.shape[0]):
        for j in range(D_rs_inv.shape[1]):
            if D_rs_inv[i][j] != 0:
                D_rs_inv[i][j] = 1./D_rs_inv[i][j]
    
    # A = D^(-1/2) * (D-W) * D^(-1/2)
    A = np.dot(D_rs_inv,np.dot(D-W,D_rs_inv))    
    
    # Get the eigen value, eigen vector of z
    bool, eigenValues_z, eigenVectors_z = cv.eigen(A)
    eigenVectors_y = np.dot(D_rs_inv, eigenVectors_z)    
    
    # 0 = (z1^T)(Z0) = (y1^T)D1, after print out the result, we get the vector we want    
    print("second smallest: " + str(eigenVectors_y[6]))
    
    

def task_4_b():
    print("c1: ACDF")
    print("C2: BEGH")
    cost = ((1+0.1+0.3+1)/(2.2+2.6+3+3)) + ((1+0.1+0.3+1)/(2.1+3+3.3+2))
    print("cost: "+ str(cost))
##############################################
##############################################
##############################################

if __name__ == "__main__":
    #task_1_a()
	#task_1_b()
	task_2()
    #task_3_a()
    #task_3_b()
    #task_3_c()
    #task_4_a()
    #task_4_b()


