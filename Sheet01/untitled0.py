
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:28:12 2019

@author: subbu
"""
from numpy import linalg as LA
import numpy as np
import cv2 as cv

def gaussianblur(bonn_gray,kernel):
    cv.GaussianBlur(bonn_gray,(5,5),kernel)

#Task 2


#print(max)


##Task 4

def GaussianKernel(v1, v2, sigma):
    return np.exp(-LA.norm(v1-v2, axis=None)**2/(2.*sigma**2))



sigma=2*np.sqrt(2)
kernel=GaussianKernel(0,0,sigma)
print("kernel", kernel)

'''blur=gaussianblur(bonn_gray,kernel)
blur = cv.GaussianBlur(bonn_gray,(5,5),kernel)
cv.imwrite("/home/subbu/Downloads/Sheet00/task4_1.png",blur)


dst = cv.filter2D(bonn_gray,5,kernel)
cv.imwrite("/home/subbu/Downloads/Sheet00/task4_2.png",dst)

sepfilter=cv.sepFilter2D(bonn_gray,5, kernel, kernel)
cv.imwrite("/home/subbu/Downloads/Sheet00/task4_3.png",sepfilter)



###Task 5
blur1 = cv.GaussianBlur(bonn_gray,(5,5),2)
blur2 = cv.GaussianBlur(blur1,(5,5),2)
blur3=cv.GaussianBlur(bonn_gray,(5,5),sigma)

cv.imwrite("/home/subbu/Downloads/Sheet00/sig1.png",blur2)
cv.imwrite("/home/subbu/Downloads/Sheet00/sig2.png",blur3)
diff=cv.absdiff(blur2, blur3)

min, max, min_l, max_l = cv.minMaxLoc(diff)
print(max)

'''
