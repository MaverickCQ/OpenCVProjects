import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def task_2():   
   #data = open('data/hands_aligned_train.txt.new', 'r')
   #magic_nr, size = struct.unpack(">II", data.read(8))
   #content = data.read()
   #x = 
   #print(content[2])
   with open('./data/hands_aligned_train.txt.new') as newData:
        lines = [line.rstrip('\n') for line in newData]
        data = [np.fromstring(line, dtype=int, sep=' ') for line in lines[1:]]
  # data = np.loadtxt('data/hands_aligned_train.txt.new', dtype=int, comments='#', delimiter=None)
   x = np.array(np.array(data))
   x = x.transpose()
   #print(x)   
   x_mean = np.sum(x,axis=0)/x.shape[0]
   #print(x_mean)
   X = x[:] - x_mean
   #print(X)
   XXT =  X @ np.transpose(X) 
   #print(XXT)
   u, e, v = np.linalg.svd(XXT)
   print(e)
   #e.sort()
   eCollection = 0
   k = 1
   for i in range (len(e)):
       eCollection += e[i]
       minimum_energy = eCollection / e.sum()
       if (minimum_energy > 0.9):
           k = i
           break
   print("k------------",k)
   #print(u.shape)
   l2jj = e[k:].sum() 
   sigma_sq = l2jj / (e.shape[0] - k)   
   uk =  u[:, :k]
   l2k = e[:k]
   print(sigma_sq)
   print(l2k)
   phi = uk * np.sqrt(l2k-sigma_sq)
   print (phi)
   return x_mean, phi

def task_3():
    pass

task_2()

task_3()


