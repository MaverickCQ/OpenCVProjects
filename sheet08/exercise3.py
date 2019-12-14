import cv2 as cv
import numpy as np
import matplotlib.pylab as plt


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
    #print(descriptors1)
    keypoints2 , descriptors2 = sift.detectAndCompute(img2, None)
    
    # your own implementation of matching

    # display the matches

    pass


if __name__ == '__main__':
    main()
