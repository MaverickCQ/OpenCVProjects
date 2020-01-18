import cv2
import numpy as np

def drawEpipolar(im1,im2,corr1,corr2,fundMat):

    x1 = np.zeros((corr1.shape[0],corr1.shape[1]+1))# Pad 1 ==> homogenious
    for i in range(len(corr1)):
        x1[i] = np.append(corr1[i],[[1]])
    line = np.dot(fundMat, x1.T).T
    
    for i in range(len(line)):
        cv2.line(im1, (int(-line[i][2]/line[i][0]),20), (0,int(-line[i][2]/line[i][1])+20), (0,0,255),2)
        cv2.circle(im2, (int(corr2[i][0]),int(corr2[i][1])), 3, (0,0,255), -1)
        
    ## Insert epipolar lines
    print("Drawing epipolar lines")
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def display_correspondences(im1,im2,corr1,corr2):

    ## Insert correspondences
    img1 = im1.copy()
    img2 = im2.copy()

    for i in range(len(corr1)):
        x1 = int(corr1[i][0])
        y1 = int(corr1[i][1])
        x2 = int(corr2[i][0])
        y2 = int(corr2[i][1])

        cv2.circle(img1, (x1,y1), 3, (0,0,255), -1)
        cv2.circle(img2, (x2,y2), 3, (0,0,255), -1)
    print("Display correspondences")
    cv2.imshow('Image 1', img1), \
    cv2.imshow('Image 2', img2), cv2.waitKey(0), cv2.destroyAllWindows()
    #, cv2.waitKey(0), cv2.destroyAllWindows()
    return
    
def computeFundMat(im1,im2,corr1,corr2):
    fundMat = np.zeros((3,3))

    # Normalize corr1 
    x1 = np.zeros((corr1.shape[0],corr1.shape[1]+1))# Pad 1 ==> homogenious
    for i in range(len(corr1)):
        x1[i] = np.append(corr1[i],[[1]])
    
    mean = np.mean(x1[:2], axis=1)# Compute the mean because later we would move the centroid to origin

    for i in range(len(x1)):# Translate so that the origin is on the origin
        x1[i][0] = x1[i][0] - mean[0]
        x1[i][1] = x1[i][1] - mean[1]
    
    sum = 0# Calculate the average distance from origin
    for i in range(len(x1)):
        sum += np.sqrt(x1[i][0]*x1[i][0] + x1[i][1]*x1[i][1])
    sum = sum/len(x1)   

    for i in range(len(x1)):# Translate so that the origin is on the origin
        x1[i][0] = x1[i][0] + mean[0]
        x1[i][1] = x1[i][1] + mean[1] 
    
    S1 = np.sqrt(2) / sum# Scale the point so the average distance from the origin is equal to sqrt(2)
    T1 = np.array([
        [S1, 0, -S1 * mean[0]],
        [0, S1, -S1 * mean[1]],
        [0, 0, 1]
    ])

    x1 = np.dot(T1, x1.T)
    
    # Normalize corr2    
    x2 = np.zeros((corr2.shape[0],corr2.shape[1]+1))# Pad 1 ==> homogenious
    for i in range(len(corr2)):
        x2[i] = np.append(corr2[i],[[1]])
    
    mean = np.mean(x2[:2], axis=1)# Compute the mean because later we would move the centroid to origin

    for i in range(len(x2)):# Translate so that the origin is on the origin
        x2[i][0] = x2[i][0] - mean[0]
        x2[i][1] = x2[i][1] - mean[1]
    
    sum = 0# Calculate the average distance from origin
    for i in range(len(x2)):
        sum += np.sqrt(x2[i][0]*x2[i][0] + x2[i][1]*x2[i][1])
    sum = sum/len(x2)    

    for i in range(len(x2)):# Translate so that the origin is on the origin
        x2[i][0] = x2[i][0] + mean[0]
        x2[i][1] = x2[i][1] + mean[1]
    
    S2 = np.sqrt(2) / sum# Scale the point so the average distance from the origin is equal to sqrt(2)
    T2 = np.array([
        [S2, 0, -S2 * mean[0]],
        [0, S2, -S2 * mean[1]],
        [0, 0, 1]
    ])

    x2 = np.dot(T2, x2.T)


    n = x2.shape[1] # number of the points, it should be the same for x1 and x2   
    # Compute Fundamental matrix
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    fundMat = V[-1].reshape(3,3)
        
    # constrain fundMat
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(fundMat)
    S[2] = 0
    fundMat = np.dot(U,np.dot(np.diag(S),V))

    # reverse normalization
    fundMat = np.dot(T1.T,np.dot(fundMat,T2))
    

    return fundMat

def question_q1_q2(im1,im2,correspondences):
    ## Compute and print Fundamental Matrix using the normalized corresponding points method.
    ## Display corresponding points and Epipolar lines
    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]

    #print(corr1)

    print("Compute Fundamental Matrix : ")
    fundMat = computeFundMat(im1.copy(),im2.copy(),corr1,corr2)
    
    display_correspondences(im1.copy(),im2.copy(),corr1,corr2)
    drawEpipolar(im1.copy(),im2.copy(),corr1,corr2,fundMat)
    return


def question_q3(im1, im2):
    
    ## compute disparity map
    print("Compute Disparity Map (2~3 minutes)")
    #cv2.imwrite('disparity1.png', dispar)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    dispar = np.zeros_like(im1)

    R = im1.shape[0]
    C = im1.shape[1]

    # match every 7x7 patch from im2 to im1, so we can get the position of same template in two pics
    for i in range(3, im1.shape[0]-3):
        for j in range(3, im1.shape[1]-3): 
            # since its too time consuming, i sampling here
            if i%3==0 and j%3==0:
                temp = im2[i-3:i+3, j-3:j+3]
                result = cv2.matchTemplate(im1,temp, cv2.TM_SQDIFF)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                topLeft = minLoc
                # Disparity = X1 - X2
                dispar[i][j] = abs(abs(R/2 - (topLeft[0]+3)) - abs(R/2 - j))
            else:
                dispar[i][j] = dispar[i//3*3][j//3*3]

    # convert Disparity above to 0~255
    maxVal = np.amax(dispar)
    minVal = np.amin(dispar)
    for i in range(dispar.shape[0]):
        for j in range(dispar.shape[1]):
            dispar[i][j] = 255 * (dispar[i][j]-minVal) / (maxVal - minVal)
   
    """cv2.imshow("img",dispar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    
    
    ## Display disparity Map
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), \
    cv2.imshow('Disparity Map', dispar), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_q4(im1, im2, correspondences):
    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]
    ## Perform Image rectification

    ### usage of either one is permitted
    print ("Fundamental Matrix")
    #fundMat = np.asmatrix([[]]) ## Insert the given matrix
    fundMat = computeFundMat(im1.copy(),im2.copy(),corr1,corr2)
    
    ## Compute Rectification or Homography
    print("Compute Rectification")
    ## Apply Homography
    h1, status = cv2.findHomography(corr1, corr2)
    im1 = cv2.warpPerspective(im1, h1, (im2.shape[1],im2.shape[0]))
    h2, status = cv2.findHomography(corr2, corr1)
    im2 = cv2.warpPerspective(im2, h2, (im1.shape[1],im1.shape[0]))    
    print("Display Warped Images")
    cv2.imwrite('Warped Image_1.png', im1)
    cv2.imwrite('Warped Image_2.png', im2)
    return

def main():

    apt1 = cv2.imread('../images/apt1.jpg')
    apt2 = cv2.imread('../images/apt2.jpg')
    aloe1 = cv2.imread('../images/aloe1.png')
    aloe2 = cv2.imread('../images/aloe2.png')
    correspondences = np.genfromtxt('../images/corresp.txt', dtype=float, skip_header=1)
    question_q1_q2(apt1,apt2,correspondences)
    question_q3(aloe1,aloe2)
    question_q4(apt1,apt2,correspondences)

if __name__ == '__main__':
    main()
