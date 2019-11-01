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
    lines = cv.HoughLines(edges,1,np.pi/180,200)
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
    print(circles)
    
    '''for i in circles[0:]:
        cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        cv.circle(img,(i[0],i[1]),2,(0,0,255),3)/'''
   # cv.imwrite("../images/blur.png", img_blur)
    cv.imwrite("../images/gray.png", img_gray)
    cv.imwrite("../images/canny.png", edges)
    #cv.imwrite("../images/houghLine.png",img)
    


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
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    #detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
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
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


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
   # task_1_b()
   # task_2()
    task_3_a()
    task_3_b()
    task_3_c()
   # task_4_a()

