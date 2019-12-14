import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random
import matplotlib.pylab as plt


def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Compute the PCA
    #k= 100
    pca = PCA(100)
    pca.fit(X_train)
    

    # Visualize Eigen Faces
    #print(pca.singular_values_)
    #fig = plt.figure()
    eigen_faces = pca.components_[:10]
    for i in range (eigen_faces.shape[0]) :
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(eigen_faces[i].reshape(h,w), cmap=plt.cm.gray)  
    plt.title("Visualize Eigen Faces")
    plt.show()   

    # Compute reconstruction error
    #face = []
    face1 = cv.imread('data/exercise1/detect/face/boris.jpg', cv.IMREAD_GRAYSCALE)
    face1 = cv.resize(face1, (w,h))
    face2 = cv.imread('data/exercise1/detect/face/merkel.jpg', cv.IMREAD_GRAYSCALE)
    face2 = cv.resize(face2, (w,h))
    face3 = cv.imread('data/exercise1/detect/face/obama.jpg', cv.IMREAD_GRAYSCALE)
    face3 = cv.resize(face3, (w,h))
    face4 = cv.imread('data/exercise1/detect/face/putin.jpg', cv.IMREAD_GRAYSCALE)
    face4 = cv.resize(face4, (w,h))
    face5 = cv.imread('data/exercise1/detect/face/trump.jpg', cv.IMREAD_GRAYSCALE)
    face5 = cv.resize(face5, (w,h))
    other1 = cv.imread('data/exercise1/detect/other/cat.jpg', cv.IMREAD_GRAYSCALE)
    other1 = cv.resize(other1, (w,h))
    other2 = cv.imread('data/exercise1/detect/other/dog.jpg',cv.IMREAD_GRAYSCALE)
    other2 = cv.resize(other2, (w,h))
    other3 = cv.imread('data/exercise1/detect/other/flag.jpg',cv.IMREAD_GRAYSCALE)
    other3 = cv.resize(other3, (w,h))
    other4 = cv.imread('data/exercise1/detect/other/flower.jpg',cv.IMREAD_GRAYSCALE)
    other4 = cv.resize(other4, (w,h))
    other5 = cv.imread('data/exercise1/detect/other/monkey.jpg',cv.IMREAD_GRAYSCALE)
    other5 = cv.resize(other5, (w,h))
    face = np.array([face1, face2, face3, face4, face5]).astype(np.float32)
    others = np.array((other1, other2, other3, other4, other5)).astype(np.float32)
    
    eigenfaces = pca.components_.reshape((100, h, w))
    mean = pca.mean_
    print(mean)
    eigenvectors = pca.components_
    #print(eigenvectors.shape)
    threshold = 0
    print (face.shape)
    for i in range(face.shape[0]):
        img = mean
        k = np.zeros((face.shape[0]))
        j = 0
        coefficient = 0
        f = 1
        #while(f!=0) :
        img = np.subtract(face[i].flatten(),mean)
            #print(img.shape)
        coefficients = eigenvectors.dot(img)
        #j += 1
        #if(np.absolute(img,coefficient)<10):
        #        f = 0
        x_mean = (mean + (eigenvectors.T * coefficients).sum(axis=1))
        #print(x_mean.shape)
        newThreshold = np.abs(face[i].flatten() - x_mean)
        #print(newThreshold.shape)
        threshold = newThreshold
        #k[i] = j
        print(threshold)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(x_mean.reshape(h,w), cmap='gray')
        plt.show()     
    

    # Perform face detection
    for i in range(face.shape[0]):
        img = np.subtract(face[i].flatten(),mean)
        coefficients = eigenvectors.dot(img)
        x_mean = (mean + (eigenvectors.T * coefficients).sum(axis=1))
        newThreshold = np.abs(face[i].flatten() - x_mean)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if((newThreshold<threshold).any()):
            print("face")
        else : 
            print("object")
        ax.imshow(x_mean.reshape(h,w), cmap='gray')
        plt.show()
        
    for i in range(others.shape[0]):
        img = np.subtract(others[i].flatten(),mean)
        coefficients = eigenvectors.dot(img)
        x_mean = (mean + (eigenvectors.T * coefficients).sum(axis=1))
        newThreshold = np.abs(others[i].flatten() - x_mean)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if((newThreshold<threshold).all()):
            print("face")
        else : 
            print("object")
        ax.imshow(x_mean.reshape(h,w), cmap='gray')
        plt.show()
       

    # Perform face recognition
    classifier = KNeighborsClassifier(10)
    classifier.fit(X_train, y_train)
    accuracy = classifier.predict(X_test)
    print(classification_report(y_test, accuracy))   
    

if __name__ == '__main__':
    main()
