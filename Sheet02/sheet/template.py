import cv2
import numpy as np
import time
from scipy.signal import fftconvolve

def get_convolution_using_fourier_transform(image, kernel):
    f = np.fft.fft2(image)
    return f

def task1():
    image = cv2.imread("data/einstein.jpeg", 0)
    kernel = None  # calculate kernel
    #image = cv2.imread('/home/subbu/Downloads/Sheet02/sheet/data/einstein.jpeg', 0)
    x=cv2.getGaussianKernel(ksize=7, sigma=1)
    kernel = x*x.T #calculate kernel
    conv_result = cv2.filter2D(image,-1,kernel)
    fft_result = get_convolution_using_fourier_transform(image, kernel)   
    
    #print()
    
    cv2.imwrite("data/conv_result.png",conv_result)
    cv2.imwrite("data/np.uint8(fft_result.real).png",np.uint8(fft_result.real))
    print(np.mean(np.abs(conv_result - fft_result)))
	
def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
	image = cv2.imread('data/traffic.jpg', 0)
	
	result_images=[]
	for i in range(len(pyramid_image)):
		image=pyramid_image[i]
		tpl=pyramid_template[i]
		if i==0:
			result=norm_cross_corr(image,tpl)
            
		
		T, threshimg = cv2.threshold(result, threshold, 1., cv2.THRESH_TOZERO)
		result_images.append(threshimg)

	return threshimg
def window_sum_2d(image, window_shape):

    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])

    return window_sum

def norm_cross_corr(image, template):
	pad_input=False
	imageShape = image.shape
	image = np.array(image, dtype=np.float64, copy=False)
	padWidth = tuple((width, width) for width in template.shape)
	mode='constant'
	constant_values=0
	if mode == 'constant':
		image = np.pad(image, pad_width=padWidth, mode=mode,
                       constant_values=constant_values)
	else:
		image = np.pad(image, pad_width=padWidth, mode=mode)
	imageWindowSum = window_sum_2d(image, template.shape)
	imageWindowSum2 = window_sum_2d(image ** 2, template.shape)
	templateMean = template.mean()
	volumeTemplate = np.prod(template.shape)
	sSdTemplate = np.sum((template - templateMean) ** 2)
	xcorr = fftconvolve(image, template[::-1, ::-1],
                            mode="valid")[1:-1, 1:-1]
	num = xcorr - imageWindowSum * templateMean
	den = imageWindowSum2
	np.multiply(imageWindowSum, imageWindowSum, out=imageWindowSum)
	np.divide(imageWindowSum, volumeTemplate, out=imageWindowSum)
	den -= imageWindowSum
	den *= sSdTemplate
	np.maximum(den, 0, out=den)  
	np.sqrt(den, out=den)
	result = np.zeros_like(xcorr, dtype=np.float64)
	mask = den > np.finfo(np.float64).eps
	result[mask] = num[mask] / den[mask]
	parts = []
	for i in range(template.ndim):
		if pad_input:
			d0 = (template.shape[i] - 1) // 2
			d1 = d0 + imageShape[i]
		else:
			d0 = template.shape[i] - 1
			d1 = d0 + imageShape[i] - template.shape[i] + 1
		parts.append(slice(d0, d1))

	return result[tuple(parts)]

def task2():
   
    image = cv2.imread('data/lena.png', 0)
    template = cv2.imread('data/eye.png', 0)

    result_ncc = norm_cross_corr(image, template)
    loc = np.where( result_ncc >= 0.7)
    w, h = template.shape[::-1]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        
    cv2.imwrite("data/task2.png",image)
    
    

    
def build_gaussian_pyramid_opencv(image, num_levels):
	imgpyr = [image]
	tempImg = image
	for i in range(0,num_levels):
		tempImg = cv2.pyrDown(tempImg)
		imgpyr.append(tempImg)
	imgpyr.reverse()
	return imgpyr

def build_gaussian_pyramid(image, num_levels, sigma):
	selfGaussianPyramid=[image]
	out = image.copy()
	for i in range(0,num_levels):
		kernel = cv2.getGaussianKernel(5, sigma)
		kernel = kernel*kernel.T
		conv_result = cv2.filter2D(out,-1,kernel)
		out = conv_result[::2,::2]
		selfGaussianPyramid.append(out)
		cv2.imwrite("data/build_gaussian_pyramid.png",out)

	selfGaussianPyramid.reverse()	
	return selfGaussianPyramid




def task3():

    image = cv2.imread('data/traffic.jpg', 0)
    
    template = cv2.imread('data/traffic-template.png', 0)
    pyramid_opencv = build_gaussian_pyramid_opencv(image, 4)
    print(pyramid_opencv)
    
    pyramid_per = build_gaussian_pyramid(image, 4, 1)

    pyramid_template = build_gaussian_pyramid(template, 4,1)
    start=time.clock()
    template_matching_ncc = norm_cross_corr(image,template)	
    print(template_matching_ncc)
    
    threshold=0.7
    print("template matching by using custom implementation of normalized cross-correlation",time.clock()-start)
    startFast = time.clock()
    result = template_matching_multiple_scales(pyramid_per, pyramid_template, threshold)
    print(result)
    print("template matching by using pyramids",time.clock()-startFast)


def get_derivative_of_gaussian_kernel(size, sigma):
    
    interval = (2*sigma+1.)/(size)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., size+1)
   # x = np.linspace(- (size // 2), size // 2)
    y = np.transpose(x)
    x2 = x**2
    y2 = y**2
    numerator = (x2+y2)/2*(sigma**2)
    x = - x/2*np.pi*sigma**4
    kernel_x = x * np.exp(-numerator)
    y = - y/2*np.pi*sigma**4
    kernel_y =  y * np.exp(-numerator)
       
    return kernel_x/kernel_x.sum(), kernel_y/kernel_y.sum()


def task4():
    image = cv2.imread("data/einstein.jpeg", 0)
    
    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)
    
    #print (kernel_x, kernel_y)

    edges_x = cv2.filter2D(image, 5, kernel_x)  # convolve with kernel_x
    edges_y = cv2.filter2D(image, 5, kernel_y)  # convolve with kernel_y
    
    cv2.imwrite("data/edges_x.png", edges_x)
    cv2.imwrite("data/edges_y.png", edges_y)
   
    magnitude = np.sqrt(kernel_x**2+kernel_y**2)  # compute edge magnitude
    direction = np.arctan2(kernel_y, kernel_x)  # compute edge direction

    cv2.imwrite("data/Magnitude.png", magnitude)
    cv2.imwrite("data/Direction.png", direction)

def l2_distance_transform_1D(dataInput, n):
    output = np.zeros(dataInput.shape)
    k = 0
    v = np.zeros((n,))
    z = np.zeros((n + 1,))
    v[0] = int(0)
    z[0] = -np.inf
    z[1] = +np.inf
    for q in range(1, n):
        s = ((dataInput[q] + q * q) - (dataInput[q-1] + v[k] * v[k])) 
        s = s / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = (((dataInput[q] + q * q) - (dataInput[q-1] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = +np.inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        value = ((q - v[k]) * (q - v[k]) + dataInput[q-1])
        if value > 255: value = 255
        if value < 0: value = 0
        output[q] = value
    #print output
    return output

def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    height, width = edge_function.shape
    f = np.zeros(max(height, width))
    # transform along columns
    for x in range(width):
        f = edge_function[:,x]
        edge_function[:,x] = l2_distance_transform_1D(f, height)
    # transform along rows
    for y in range(height):
        f = edge_function[y,:]
        edge_function[y,:] = l2_distance_transform_1D(f, width)
    return edge_function


def task5():
    image = cv2.imread("data/traffic.jpg", 0)

    edges = cv2.Canny(image,100,200) # compute edges
    cv2.imwrite("data/traffic1.png", edges)
    edge_function = image  # prepare edges for distance transform    
    positive_inf =  +np.inf
    negative_inf = -np.inf
   # print (edge_function)
    dist_transfom_mine = l2_distance_transform_2D(
        edge_function, positive_inf, negative_inf
    )
    
    #print (dist_transfom_mine)
    cv2.imwrite("data/traffic4.png", dist_transfom_mine)
    
    dist_transfom_cv = cv2.distanceTransform(image,cv2.DIST_L2, 3)
    cv2.imwrite("data/traffic3.png", dist_transfom_cv)
    #mean = cv2.absdiff(dist_transfom_mine, dist_transfom_cv)
    
   # print ("mean absolute difference", mean)
    print("Euclidean distance transform mean absolute difference ",np.mean(np.abs(dist_transfom_mine - dist_transfom_cv)))
    
    # compare and print mean absolute difference


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
