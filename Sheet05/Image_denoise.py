import cv2
import numpy as np
import maxflow

def question_3(I,rho=0.7,pairwise_cost_same=0.005,pairwise_cost_diff=0.2):


    ### 1) Define Graph
    g = maxflow.Graph[float]()

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)



    ### 6) Maxflow
    g.maxflow()

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    ### Use Alpha expansion binary image for each label

    ### 1) Define Graph
    def getGraph(img):                
        g = maxflow.Graph[float]()
        return g
        
    ### 2) Add pixels as nodes
    def getNodes(img, g):
        row =img.shape[0]
        col = img.shape[1]
        nodes = g.add_nodes(row*col)
        return nodes
    ### 3) Compute Unary cost
    ### 4) Add terminal edges
    def addTedges(g,  nodes, a, img):
        row =img.shape[0]
        col = img.shape[1]

        for i in range(row):
            for j in range(col):
                if img[i][j] == a:
                    g.add_tedge(nodes[i*col + j], 0.8, float("inf"))
                else:
                    g.add_tedge(nodes[i*col + j], 0.8, 0.1)                
        return g

    ### 5) Add Node edges
    def Potts(A, B):
            return(0 if A == B else 1) 

    def addEdges(g, nodes, a, img): # need row/col because nodes is 1D
        g1 =addTedges(g,  nodes, a, img)
        g2 = addVEdges(g1,  nodes, a, img)
        g3 = addHEdges(g2,  nodes, a, img)
        return g3
    ### Vertical Edges
    def addHEdges(g,  nodes, a, img):
        row =img.shape[0]
        col = img.shape[1]

        for i in range(row):
            for j in range(col-1):
                P = Potts(img[i][j], img[i][j+1])
                g.add_edge(nodes[i*col + j], nodes[i*col + j + 1], P, P)
        return g
    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)
    def addVEdges(g,  nodes, a, img):
        row =img.shape[0]
        col = img.shape[1]

        for j in range(col):
            for i in range(row-1):
                P = Potts(img[i][j], img[i+1][j])
                g.add_edge(nodes[i*col + j], nodes[(i+1)*col + j ], P, P)
        return g

    def updateImg(g, img, nodes, a):
        newImg = img.copy()
        row =img.shape[0]
        col = img.shape[1]
        for i in range(row):
            for j in range(col):
                if g.get_segment(nodes[i*col+j]) == 1:
                    newImg[i][j] = a
        return newImg

    ### 6) Maxflow
    img = I
    flows = [0,0,0]
    temp = [0,0,0]

    while True:
        for i in range(len(labels)):
            g = getGraph(img)
            nodes = getNodes(img, g)
            g = addEdges(g, nodes, labels[i], img)
            flows[i] = g.maxflow()
            img = updateImg(g, img, nodes, labels[i])
            
        if flows == temp:
            break
        else: # yes, idk how to do deep copy :)
            temp[0] = flows[0]
            temp[1] = flows[1]
            temp[2] = flows[2]


    Denoised_I = img
    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4
    question_4(image_q4, rho=0.8)    
    return

if __name__ == "__main__":
    main()



