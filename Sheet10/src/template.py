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
    cv2.imwrite(name,img)
    #cv2.imshow(name,img) 
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()

def task1():
    #implement your solution
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    imagePoints = [] # 2d points in image plane.
    objectPoints = [] # 3d points in real world space

    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    i = 0
    for img_file in images_files_list:
        #print(img_file)
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

        showImg(img, str(i)+".png")
        i+=1
        #cv2.imwrite(str(i)+".png",img)
    return imagePoints, objectPoints

def task2(imagePoints, objectPoints):
    #implement your solution
    img = cv2.imread(images_files_list[0])
    img_size = (img.shape[0], img.shape[1])
    ret, CM, D, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, img_size, None, None)
    print("CM: ", CM)
    print("D:", D)
    print("rotation :", rvecs)
    print("translation:", tvecs)
    return CM, D, rvecs, tvecs

def task3(imagePoints, objectPoints, CM, D, rvecs, tvecs):    
    #implement your solution
    #print("task3")
    #newImagePoints = np.vstack(imagePoints)
    #print(newImagePoints.shape)
    step1 = 0
    step2 = 0
    k = 0 
    i=0
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for img_file in images_files_list:
        img = cv2.imread(img_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img_gray, board_size, None)
        newCorners = cv2.cornerSubPix(img_gray, corners, (3, 3), (-1, -1), subpix_criteria)
        corner_a = np.vstack(corners)
        newCorner_a = np.vstack(newCorners)
        step1 += np.sum(corner_a[:,0] - newCorner_a[:,0])
        step2 += np.sum(corner_a[:,1] - newCorner_a[:,1])
        k = corner_a.shape[0]
    #print (step1)    
    ex = step1 / (NUM_IMAGES * k)
    ey = step2 / (NUM_IMAGES * k)
    print ("Ex , Ey", ex, ey)
    
def task4(CM, D):
    #implement your solution
    print("task4")
    i = 0
    NCM, roi = cv2.getOptimalNewCameraMatrix(CM, D, (board_w, board_h), 1, (board_w, board_h))
    for img_file in images_files_list:
        img = cv2.imread(img_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.undistort(img_gray, CM, D, None, NCM)
        showImg((np.abs(img_gray,dst)), "task4_"+str(i)+".png")
        i+=1
        print (np.sum(np.abs(img_gray,dst)))
            

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

    task4(CM, D) # Calling Task 4

    task5(CM, rvecs, tvecs) # Calling Task 5
    
    print("FINISH!")

main()