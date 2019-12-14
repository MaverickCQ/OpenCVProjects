import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import NearestNeighbors


def main():
    # Load the images
    img1 = cv.imread("data/exercise3/mountain1.png")
    img2 = cv.imread("data/exercise3/mountain2.png")
    #print(img1.shape)
    #print(img2.shape)

    # extract sift keypoints and descriptors
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1 , descriptors1 = sift.detectAndCompute(img1, None)
    #print(keypoints1)
    print(descriptors1.shape)
    keypoints2 , descriptors2 = sift.detectAndCompute(img2, None)
    
    #print(len(keypoints1))
    # your own implementation of matching
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(descriptors1)
    #k = neigh.kneighbors(descriptors2)
    #print(k[0].shape)
    points = []
    for i in range(len(keypoints1)) :
        dis, index = neigh.kneighbors(descriptors2, 2, return_distance=True)
        #print(dis, index)
        ratio = dis[0][0] / dis[0][1]
        #print(ratio)
        if (ratio<0.4) :
          match = cv.DMatch(i,index[0][0], dis[0][0])
          points.append(match)
    
    #print(points)
    points = sorted(points, key = lambda x:x.distance)
    img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, points[:50], img2, flags=2)
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    cv.imwrite("data/exercise3/mountain3.png",img3)
    plt.show()
    # display the matches

    pass


if __name__ == '__main__':
    main()
