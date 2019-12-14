import cv2
import numpy as np
import matplotlib.pylab as plt


def get_tensor(img):
    gradient = np.gradient(img)
    Ix = gradient[1]
    Iy = gradient[0]
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Iy * Ix    

    return Ixx, Iyy, Ixy

def Harris(img, win_size, k, thr, Ixx, Iyy, Ixy):
    row = img.shape[0]
    col = img.shape[1]
    offset = int(win_size / 2)
    
    color_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)

    #Loop through image and find our corners
    for i in range(offset, row-offset):
        for j in range(offset, col-offset):
            #Calculate sum of squares
            windowIxx = Ixx[i-offset:i+offset+1, j-offset:j+offset+1]
            windowIxy = Ixy[i-offset:i+offset+1, j-offset:j+offset+1]
            windowIyy = Iyy[i-offset:i+offset+1, j-offset:j+offset+1]
            
            # determine of windows
            Sxx = np.linalg.det(windowIxx) 
            Sxy = np.linalg.det(windowIxy) 
            Syy = np.linalg.det(windowIyy) 

            # determinant of M and trace,
            # |M| - k(Tr(M))^2
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #If corner response is over threshold, color the point and add to corner list
            if r > thr:
                """print(r)"""
                color_img.itemset((i, j, 0), 0)
                color_img.itemset((i, j, 1), 0)
                color_img.itemset((i, j, 2), 255)
    return color_img


def Forstner(img, win_size, k, thr, Ixx, Iyy, Ixy):
    row = img.shape[0]
    col = img.shape[1]
    offset = int(win_size / 2)
    
    color_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)

    #Loop through image and find our corners
    for i in range(offset, row-offset):
        for j in range(offset, col-offset):
            #Calculate sum of squares
            windowIxx = Ixx[i-offset:i+offset+1, j-offset:j+offset+1]
            windowIxy = Ixy[i-offset:i+offset+1, j-offset:j+offset+1]
            windowIyy = Iyy[i-offset:i+offset+1, j-offset:j+offset+1]
            
            # determine of windows
            Sxx = np.linalg.det(windowIxx) 
            Sxy = np.linalg.det(windowIxy) 
            Syy = np.linalg.det(windowIyy) 

            # determinant of M and trace,            
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy

            # w = |M| / tr(M)
            # q = 4|M| / tr(M)^2
            if trace != 0:
                w = det / trace
                q = (4*det) / (trace**2)

                #If corner response is over threshold, color the point and add to corner list
                if w > thr[0] and q > thr[1]:
                    color_img.itemset((i, j, 0), 0)
                    color_img.itemset((i, j, 1), 0)
                    color_img.itemset((i, j, 2), 255)
    return color_img

def showImg(img, name="Image"):
    cv2.imshow(name,img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def main():
    # Load the image
    img = cv2.imread('./data/exercise2/building.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    """showImg(img)"""

    # Compute Structural Tensor
    Ixx, Iyy, Ixy = get_tensor(img)
    # Harris Corner Detection
    win_size = 3
    k = 0.04
    thr = 2000000000
    
    finalImg = Harris(img, win_size, k, thr, Ixx, Iyy, Ixy)
    if finalImg is not None:
        showImg(finalImg, "Harris")
    # Forstner Corner Detection
    win_size = 3
    k = 0.04
    thr = (4000,-0.1) # threshold of w and q
    
    finalImg = Forstner(img, win_size, k, thr, Ixx, Iyy, Ixy)
    if finalImg is not None:
        showImg(finalImg, "Forstner")


if __name__ == '__main__':
    main()
