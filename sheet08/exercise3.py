import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import NearestNeighbors

def showImg(img, name="Image"):
    cv.imshow(name,img) 
    cv.waitKey(0) 
    cv.destroyAllWindows()

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
    

    KPImg_1=cv.drawKeypoints(img1.copy(),keypoints1,img1.copy())
    KPImg_2=cv.drawKeypoints(img2.copy(),keypoints2,img2.copy())
    
    cv.imwrite("data/exercise3/keyPoints_1.png", KPImg_1)
    cv.imwrite("data/exercise3/keyPoints_2.png", KPImg_2)
    
    # your own implementation of matching
    neigh = NearestNeighbors(2, 0.4)
    neigh.fit(descriptors2)   

    newImg = img1.copy()
    draw_params = dict(matchColor = (0,255,0))
    matches = []
    for i in range(len(keypoints1)):
        distance, index = neigh.kneighbors([descriptors1[i]], 2, return_distance=True)
        ratio = distance[0][0] / distance[0][1]
        # accept the first match
        if ratio < 0.4:
            match = cv.DMatch(i, index[0][0], distance[0][0])
            matches.append(match)            

    # display the matches
    newImg = cv.drawMatches(newImg,keypoints1,img2,keypoints2,matches, None, **draw_params)
    #showImg(newImg)
    cv.imwrite("data/exercise3/mountainMatches.png", newImg)
    


if __name__ == '__main__':
    main()
