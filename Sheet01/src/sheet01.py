import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys

def gaussianblur(bonn_gray,kernel):
    cv.GaussianBlur(bonn_gray,(5,5),kernel)


def custom_integral_def(img_gray):
    rows, columns = img_gray.shape
    
    custom_integral = np.empty([rows+1, columns+1], dtype=int)
    
    for i in range (0, rows+1):
        custom_integral[i][0] = 0
        
    for j in range (0, columns+1):
        custom_integral[0][j] = 0
            
    for i in range (1, rows+1):
        for j in range (1, columns+1):
            custom_integral[i][j] = img_gray[i-1][j-1]
      
    rows, columns = custom_integral.shape
    
    for i in range (1, rows):
        for j in range (1, columns):
            custom_integral[i][j] = custom_integral[i][j] + custom_integral[i-1][j] + custom_integral[i][j-1] - custom_integral[i-1][j-1]
    
    return custom_integral

def s_p_noise (img,prob):
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noise = img
    noise[rnd < prob ] = 0
    noise[rnd > 1 - prob] = 255
    return noise 

if __name__ == '__main__':
    img_path = sys.argv[1]
    
#    =========================================================================    
#    ==================== Task 1 =================================
#    The Integral Image is used as a quick and effective way of calculating the sum of values
#    =========================================================================    
    print('Task 1:');
    
#   ============== a =========================
    img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
    print(custom_integral_def(img_gray))

#   ============== b =========================
#   ============== i =========================
    rows, columns = img_gray.shape
    
    sumOfPixels = 0
    
    for i in range (0, rows):
        for j in range (0, columns):
            sumOfPixels += img_gray[i][j]
            
    mean_gray = sumOfPixels/img_gray.size
    print("Mean Gray of summed up pixels of img : ", mean_gray)
#   ============== ii =========================    
    img_integral = cv.integral(img_gray)
    
    rows, columns = img_integral.shape
    
    sumOfPixels = 0
    
    for i in range (0, rows):
        for j in range (0, columns):
            sumOfPixels += int(img_integral[i][j])
            
    mean_gray = sumOfPixels/img_integral.size
    print("Mean Gray of summed up pixels of integral : ", mean_gray)
#   ============== iii =========================
    
    custom_integral = custom_integral_def(img_gray)
    
    sumOfPixels = 0
    
    for i in range (0, rows):
        for j in range (0, columns):
            sumOfPixels += int(custom_integral[i][j])
    mean_gray = sumOfPixels/custom_integral.size
    print("Mean Gray of summed up pixels of custom integral : ", mean_gray)
    
#   ============== c =========================
    rows, columns = img_gray.shape
    
    start = cv.getTickCount()
    
    for k in range (0, 10):
        rand_coord = np.random.randint(0, rows/2 , 2)
        end_x = rand_coord[0] + 100
        end_y = rand_coord[1] + 100
        
        new_img = img_gray[rand_coord[0]:end_x, rand_coord[1]:end_y]
        
        rows_new, columns_new = new_img.shape
        for i in range (0, rows_new):
            for j in range (0, columns_new):
                sumOfPixels += new_img[i][j]
            
            mean_gray = sumOfPixels/img_gray.size
    end = cv.getTickCount()
    print("run-time of summing up each pixel:", end - start)

#   ============== ii =========================
    start = cv.getTickCount()
    for k in range (0, 10):
        rand_coord = np.random.randint(0, rows/2 , 2)
        end_x = rand_coord[0] + 100
        end_y = rand_coord[1] + 100
        
        new_img = img_gray[rand_coord[0]:end_x, rand_coord[1]:end_y]
        new_img_integral = cv.integral(new_img)
        rows_new, columns_new = new_img_integral.shape
        
        sumOfPixels = 0
    
        for i in range (0, rows_new):
            for j in range (0, columns_new):
                sumOfPixels += int(new_img_integral[i][j])
        mean_gray = sumOfPixels/img_integral.size
    
    end = cv.getTickCount()
    print("run-time of cv.integral:", end - start)

#   ============== iii =========================    
    start = cv.getTickCount()
    for k in range (0, 10):
        rand_coord = np.random.randint(0, rows/2 , 2)
        end_x = rand_coord[0] + 100
        end_y = rand_coord[1] + 100
        
        new_img = img_gray[rand_coord[0]:end_x, rand_coord[1]:end_y]
        new_custom_integral = custom_integral_def(new_img)
        rows_new, columns_new = new_custom_integral.shape
        
        sumOfPixels = 0
    
        for i in range (0, rows_new):
            for j in range (0, columns_new):
                sumOfPixels += int(new_img_integral[i][j])
        mean_gray = sumOfPixels/img_integral.size
    
    end = cv.getTickCount()
    print("run-time of custom integral:", end - start)

#    =========================================================================    
#    ==================== Task 2 =================================
#    =========================================================================    
    print('Task 2:');

#   ============== a =========================
    img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    hist = cv.equalizeHist(img_gray)
    cv.imwrite('task2_1.png',hist)
    
#   ============== b =========================
    orig = np.array(img_gray)

    hist = np.copy(orig)

    res, nek = np.unique(orig, return_counts=True)
    pk = nek/img_gray.size
    pk_length = len(pk)

    sk = np.cumsum(pk)
    mul = sk*np.max(orig)
    roundVal = np.round(mul)
    for i in range(len(orig)):
        for j in range(len(orig[0])):
            hist[i][j] = roundVal[np.where(res == orig[i][j])]

    cv.imwrite("task2_2.png",hist)

    diff=cv.absdiff(orig, hist)
    print(diff)
    min, max, min_l, max_l = cv.minMaxLoc(diff)

    print(max)

    
    


#    =========================================================================    
#    ==================== Task 4 =================================
#    =========================================================================    
    print('Task 4:');

    def GaussianKernel(v1, v2, sigma):
        return np.exp(-np.linalg.norm(v1-v2, axis=None)**2/(2.*sigma**2))



    sigma=2*np.sqrt(2)
    kernel=GaussianKernel(0,0,sigma)
#   ============== i =========================
    
    blur=gaussianblur(img_gray,kernel)
    blur = cv.GaussianBlur(img_gray,(5,5),kernel)
    cv.imwrite("task4_1.png",blur)
#   ============== ii =========================


    dst = cv.filter2D(img_gray,5,kernel)
    cv.imwrite("task4_2.png",dst)
#   ============== iii =========================

    sepfilter=cv.sepFilter2D(img_gray,5, kernel, kernel)
    cv.imwrite("task4_3.png",sepfilter)



#    =========================================================================    
#    ==================== Task 5 =================================
#    =========================================================================    
    print('Task 5:');
#   ============== i =========================

    blur1 = cv.GaussianBlur(img_gray,(5,5),2)
    blur2 = cv.GaussianBlur(blur1,(5,5),2)
#   ============== ii =========================

    blur3=cv.GaussianBlur(img_gray,(5,5),sigma)

    cv.imwrite("sig1.png",blur2)
    cv.imwrite("sig2.png",blur3)
    diff=cv.absdiff(blur2, blur3)

    min, max, min_l, max_l = cv.minMaxLoc(diff)
    print(max)




#    =========================================================================    
#    ==================== Task 7 =================================
#    =========================================================================    
    print('Task 7:');

    s_p_img = s_p_noise(img_gray, 0.2)
    
    cv.imwrite("snp.png", s_p_img) 
 
#   ============== i =========================
    
    blur = cv.GaussianBlur(s_p_img,(3,3),2)
    
    cv.imwrite("FilterGaussian.png", blur) 

#   ============== ii =========================
    
    median = cv.medianBlur(s_p_img,3)
    
    cv.imwrite("FilterMedianBlur.png", median) 
    
#   ============== iii =========================
    
    bilateral = cv.bilateralFilter(s_p_img,9,70,70)
    
    cv.imwrite("FilterBilateralFilter.png", bilateral) 
    


#    =========================================================================    
#    ==================== Task 8 =================================
#    =========================================================================    
    print('Task 8:');

#   ============== i =========================    
    
    kernel1 = np.array([[0.0113, 0.0838, 0.0113],
                        [0.0838, 0.6193, 0.0838],
                        [0.0113, 0.0838, 0.0113]])
    kernel2 = np.array([[-0.8984, 0.1472, 1.1410],
                        [-1.9075, 0.1566, 2.1359],
                        [-0.8659, 0.0573, 1.0337]])
    
    filter1 = cv.filter2D(img_gray, -1, kernel1)
    filter2 = cv.filter2D(img_gray, -1, kernel2)
    
    cv.imwrite("kernel1.png", filter1) 
    cv.imwrite("kernel2.png", filter2) 

#   ============== ii =========================
    
    w1, u1, vt1 = cv.SVDecomp(kernel1)
    w2, u2, vt2 = cv.SVDecomp(kernel2)
    
    kernel1_1d = u1[:,0, None]*np.sqrt(w1[0])
    kernel2_1d = u2[:,0, None]*np.sqrt(w2[0])
    
    filter1_1d = cv.filter2D(img_gray, -1, kernel1_1d)
    
    cv.imwrite("kernel1_1d.png", filter1_1d)
    
    filter2_1d = cv.filter2D(img_gray, -1, kernel2_1d)
    
    cv.imwrite("kernel2_1d.png", filter2_1d)
    
#   ============== iii =========================
    
    diff1 = cv.absdiff(filter1, filter1_1d)
    diff2 = cv.absdiff(filter2, filter2_1d)
    
    max1 = np.max(diff1)
    max2 = np.max(diff2)
    
    print("Kernel 1 2dD vs 1D:", max1)
    print("Kernel 2 2dD vs 1D:", max2)
    
    
    
    
    
    
    
    

