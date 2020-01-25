import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os
import os.path as osp

# --- this are all the imports you are supposed to use!
# --- please do not add more imports!

n_views = 101
n_features = 215
data_matrix = np.genfromtxt('./data/data_matrix.txt', dtype=float, skip_header=0)    
fig = plt.figure()
image_prefix = "./data/"
image_suffix = ".jpg"
images_files_list = [osp.join(image_prefix, f) for f in os.listdir(image_prefix)
                     if osp.isfile(osp.join(image_prefix, f)) and f.endswith(image_suffix)]
image_list = []
red = [0,0,255]
#print(data_matrix.shape)
M, N = int(data_matrix.shape[0] / 2), data_matrix.shape[1]
data = np.zeros((M, N, 2), dtype=int)
for i in range(2*M):
    data[int(i / 2), :, i%2] = data_matrix[i, :]
k = 0
for img_file in images_files_list: #assuming gif
    im = cv2.imread(img_file)
    for i in range (N):
        cv2.circle(im, (data[k, i , 0],data[k, i , 1]), 1, red, 5)
    img = plt.imshow(im, animated=True)
    image_list.append([img])
    k+=1
    #plt.show()
    
ani = animation.ArtistAnimation(fig, image_list, interval = 50, blit=False, repeat_delay = 1000)
#writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save('basic_animation.mp4', writer='ffmpeg')

#print(data_matrix.shape)
#print(data_matrix[0][:])
plt.show()


# --- add your code here ---