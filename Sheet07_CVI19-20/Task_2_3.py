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
   XXT =  np.transpose(X) @ X
   print(XXT.shape)
   #print(XXT)
   u, e, v = np.linalg.svd(XXT)
   print(u.shape)
   #print(e)
   #e.sort()
   eCollection = 0
   N = 1
   for i in range (len(e)):
       eCollection += e[i]
       minimum_energy = eCollection / e.sum()
       if (minimum_energy > 0.9):
           N = i
           break
  # print("N------------",N)
   #print(u.shape)
   l2jj = e[N:].sum() 
   sigma_sq = l2jj / (e.shape[0] - N)   
   uk =  u[:,:N]
   l2k = e[:N]
  # print(sigma_sq)
  # print(l2k)
   #print(uk.shape)
   phi = uk * np.sqrt(l2k-sigma_sq)
   #print(x_mean)
   #print (phi)
   hi = np.array([-0.4,-0.2,0.0,0.2,0.4])
   #print(x_mean.shape)
   #print(phi.shape)
   wi = x_mean + (phi * hi.reshape(1,N)).sum(axis=1)
   fig = plt.figure()
   ax = fig.add_subplot()
   wi = np.transpose(wi.reshape(wi.shape[0] // 2, 2, order='F'))
   ax.plot(wi[0,:],wi[1,:])
   plt.show()

def task_3():
    pass

task_2()

task_3()


