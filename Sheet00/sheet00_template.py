import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png' 

    # 2a: read and display the image 
    img = cv.imread(img_path)

    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    
    img_cpy = img
    img_rgb = cv.cvtColor(img_cpy, cv.COLOR_BGR2RGB)
    img_gray_rgb = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)
    
    
    width = img_rgb.shape[1]
    height = img_rgb.shape[0]
    channel = img_rgb.shape[2]    
    
    
   # print (img_gray)
    intensity = img_gray_rgb * 0.5
    #print("---------------------------------")
   # print (intensity)
    
    for i in range (0, 1):
        for j in range (0, 1):
            img_rgb[i][j] = - intensity[i][j]
            if(img_rgb[i][j][0] < 0):
                img_rgb[i][j][0] = 0
            if(img_rgb[i][j][1] < 0):
                img_rgb[i][j][1] = 0
            if(img_rgb[i][j][2] < 0):
                img_rgb[i][j][2] = 0
    
    img_cpy = img_rgb
       
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above
    
    img_cpy = img 

    img_cpy =-intensity         
    
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    

    # 2e: Extract the center patch and place randomly in the image
    center_x = int(img.shape[0]/2)
    center_y = int(img.shape[1]/2)
    
    img_patch = img[center_x-8:center_x+7, center_y-8:center_y+7]
    
    display_image('2 - e - Center Patch', img_patch)  
    
    # Random location of the patch for placement
        
    rand_coord = np.random.randint(0,img.shape[0],2)
    
    print (rand_coord[0])
    print (rand_coord[1])
    
    img_cpy = img 
        
    img_cpy[rand_coord[1]:rand_coord[1]+img_patch.shape[0], rand_coord[0]:rand_coord[0]+img_patch.shape[1]] =  img_patch
    
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)  

    # 2f: Draw random rectangles and ellipses
    
    img_cpy = img
    
    for i in range (0, 10):
        rand_coord = np.random.randint(0, 300, 2)
        rand_size = np.random.randint(0, 20, 2)
        cv.rectangle(img_cpy, (rand_coord[0], rand_coord[1]), (rand_coord[0]+rand_size[0], rand_coord[1]+rand_size[1]), (255, 165, 00), 2)
        rand_coord_ellipse = np.random.randint(0, 300, 2)
        rand_size_ellipse_axis_M = np.random.randint(0, 30, 1)
        rand_size_ellipse_axis = np.random.randint(0, 10, 1)
        rand_size_ellipse_angle = np.random.randint(0, 360, 1)
        cv.ellipse(img_cpy,(rand_coord_ellipse[0], rand_coord_ellipse[1]),(rand_size_ellipse_axis_M,rand_size_ellipse_axis),0,0,360,(220, 20, 60),2)
        
    display_image('2 - f - Rectangles and Ellipses', img_cpy)
       
    # destroy all windows
    cv.destroyAllWindows()
