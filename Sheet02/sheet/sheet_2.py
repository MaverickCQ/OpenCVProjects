import cv2
import numpy as np
import time

from scipy.signal import fftconvolve

def get_convolution_using_fourier_transform(image, kernel):
    f = np.fft.fft2(image)
    return f

def task1():
    image = cv2.imread("../data/einstein.jpeg", 0)
    kernel = None  # calculate kernel
    image = cv2.imread('/home/subbu/Downloads/Sheet02/sheet/data/einstein.jpeg', 0)
    x=cv2.getGaussianKernel(ksize=7, sigma=1)
    kernel = x*x.T #calculate kernel
    conv_result = cv2.filter2D(image,-1,kernel) #calculate convolution of image and kernel
    fft_result = get_convolution_using_fourier_transform(image, kernel)
    
    
    #print()
    
    cv2.imwrite("/home/subbu/Downloads/Sheet02/sheet/data/task1_1.png",conv_result)
    cv2.imwrite("/home/subbu/Downloads/Sheet02/sheet/data/task1_2.png",np.uint8(fft_result.real))
    print(np.mean(np.abs(conv_result - fft_result)))
	
def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
	image = cv2.imread('/home/subbu/Downloads/Sheet02/sheet/data/traffic.jpg', 0)
	
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
   
    image = cv2.imread('/home/subbu/Downloads/Sheet02/sheet/data/lena.png', 0)
    template = cv2.imread('/home/subbu/Downloads/Sheet02/sheet/data/eye.png', 0)

    result_ncc = norm_cross_corr(image, template)
    loc = np.where( result_ncc >= 0.7)
    w, h = template.shape[::-1]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        
    cv2.imwrite('/home/subbu/Downloads/Sheet02/sheet/data/image_normalized_correlation.png',image)
    
    

    
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
		cv2.imwrite('/home/subbu/Downloads/Sheet02/sheet/data/pyramid_images.png',out)

	selfGaussianPyramid.reverse()	
	return selfGaussianPyramid




def task3():

    image = cv2.imread('/home/subbu/Downloads/Sheet02/sheet/data/traffic.jpg', 0)
    
    template = cv2.imread('/home/subbu/Downloads/Sheet02/sheet/data/traffic-template.png', 0)
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





if __name__ == "__main__":
    task1()
    task2()
    task3()
#    task4()
#    task5()