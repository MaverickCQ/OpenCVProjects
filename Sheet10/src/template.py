import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

NUM_IMAGES=14
NUM_Boards = NUM_IMAGES
image_prefix = "../images/"
image_suffix = ".png"
images_files_list = [osp.join(image_prefix, f) for f in os.listdir(image_prefix)
                     if osp.isfile(osp.join(image_prefix, f)) and f.endswith(image_suffix)]
board_w = 10
board_h = 7
board_size = (board_w, board_h)
board_n = board_w * board_h
img_shape = (0,0)
obj = []
for ptIdx in range(0, board_n):
    obj.append(np.array([[ptIdx/board_w, ptIdx%board_w, 0.0]],np.float32))
obj = np.vstack(obj)

def showImg(img, name="Image"):
    cv2.imshow(name,img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def task1():
    #implement your solution
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    imagePoints = [] # 2d points in image plane.
    objectPoints = [] # 3d points in real world space

    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    
    for img_file in images_files_list:
        print(img_file)
        img = cv2.imread(img_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(img_gray, board_size, None)

        if ret == True:
            # Refining corners position with sub-pixels based algorithm
            cv2.cornerSubPix(img_gray, corners, (3, 3), (-1, -1), subpix_criteria)
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            imagePoints.append(corners)
            objectPoints.append(objp)            
        else:
            print('No chessboard')

        showImg(img, img_file)
    return imagePoints, objectPoints

def task2(imagePoints, objectPoints):
    #implement your solution
    img = cv2.imread(images_files_list[0])
    img_size = (img.shape[0], img.shape[1])
    ret, CM, D, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, img_size, None, None)
    
    return CM, D, rvecs, tvecs

def task3(imagePoints, objectPoints, CM, D, rvecs, tvecs):
    #implement your solution
    pass

def task4(CM, D):
    #implement your solution
    pass

def task5(CM, rvecs, tvecs):
    #implement your solution
    pass

def main():
    #Showing images
    """for img_file in images_files_list:
        print(img_file)
        img = cv2.imread(img_file)
        cv2.imshow("Task1", img)
        cv2.waitKey(10)"""
    
    imagePoints, objectPoints = task1() #Calling Task 1
        
    CM, D, rvecs, tvecs = task2(imagePoints, objectPoints) #Calling Task 2

    task3(imagePoints, objectPoints, CM, D, rvecs, tvecs)  # Calling Task 3

    #task4(CM, D) # Calling Task 4

    #task5(CM, rvecs, tvecs) # Calling Task 5
    
    print("FINISH!")

main()