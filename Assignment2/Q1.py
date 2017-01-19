import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage


# univ = sp.ndimage.imread('/mnt/home/jayaraj/Study/CS_CVIP/hw4/univ.jpeg',flatten=True )
univ = sp.ndimage.imread('univ.jpeg',flatten=True )

# plt.imshow(univ,cmap='Greys_r')
# plt.show()

dog_mask = [[0,0,-1,-1,-1,0,0],[0,-2,-3,-3,-3,-2,0],[-1,-3,5,5,5,-3,-1],[-1,-3,5,16,5,-3,-1],[-1,-3,5,5,5,-3,-1],[0,-2,-3,-3,-3,-2,0],[0,0,-1,-1,-1,0,0]]
univ_dog_mask = signal.convolve2d(univ, dog_mask)
plt.subplot(331)
plt.title('Q1.a After applying DOG mask')
plt.imshow(univ_dog_mask,cmap='Greys_r')
# plt.show()

log_mask = [[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]]
univ_log_mask = signal.convolve2d(univ, log_mask)
plt.subplot(332)
plt.title('After applying LOG mask')
plt.imshow(univ_log_mask,cmap='Greys_r')
# plt.show()

def zero_crossing(arr):
	# print arr
	arr_zero_crossing = np.zeros((arr.shape[0],arr.shape[1]))#<-----confirm 
	for i in range(arr.shape[0]-1):
		for j in range(arr.shape[1]-1):
			cur_sign = 0
			left_sign = 0
			up_sign = 0
			down_sign = 0
			right_sign = 0

			if arr[i][j]>=0:
				cur_sign=1

			if arr[i-1][j]>=0:
				left_sign=1

			if arr[i][j-1]>=0:
				up_sign=1
		
			# if i!=arr.shape[0]:
			# 	right_sign=1

			# if j!=arr.shape[1]:
			# 	down_sign=1

			if arr[i][j+1]>=0:
				right_sign=1
			if arr[i+1][j]>=0:
				down_sign=1

			if ((cur_sign!=left_sign) or (cur_sign!=up_sign) or (cur_sign!=down_sign) or (cur_sign!=right_sign)):
				arr_zero_crossing[i][j]=255

	# print arr_zero_crossing
	return arr_zero_crossing

univ_dog_zero_crossed = zero_crossing(univ_dog_mask)
plt.subplot(334)
plt.title('Q1.b DOG - zero crossed')
plt.imshow(univ_dog_zero_crossed,cmap='Greys_r')
# plt.show()

univ_log_zero_crossed = zero_crossing(univ_log_mask)
plt.subplot(335)
plt.title('LOG - zero crossed')
plt.imshow(univ_log_zero_crossed,cmap='Greys_r')
# plt.show()

dx = ndimage.sobel(univ, 1)
# plt.subplot(331)
# plt.title('dx')
# plt.imshow(dx,cmap='Greys_r')
# plt.show()

dy = ndimage.sobel(univ, 0)
# plt.subplot(331)
# plt.title('dy')
# plt.imshow(dx,cmap='Greys_r')
# plt.show()


mag = np.hypot(dx, dy)
mag *= 255.0 / np.max(mag)
# plt.imshow(mag,cmap='Greys_r')
# plt.show()

mag_thresholded = mag
threshhold = 40
low_values_indices = mag < threshhold
mag_thresholded[low_values_indices]=0
plt.subplot(336)
title_for_sobel = "SOBEL - threshold="+str(threshhold)
plt.title(title_for_sobel)
plt.imshow(mag_thresholded,cmap='Greys_r')
# plt.show()

def anding_array(a1,a2):
	m = a1.shape[0]
	n = a1.shape[1]

 	result = np.zeros((m,n))
	for i in range(m):
		for j in range(n):
			if ( np.logical_and(a1[i][j], a2[i][j]) ):
				result[i][j] = 0
			else :
				result[i][j] = 255
	# print result
	return result

univ_dog_anded_result = anding_array(mag_thresholded,univ_dog_zero_crossed)
plt.subplot(337)
plt.title('Q1.c DoG ANDed with sobel')
plt.imshow(univ_dog_anded_result,cmap='Greys_r')
# plt.show()

univ_log_anded_result = anding_array(mag_thresholded,univ_log_zero_crossed)
plt.subplot(338)
plt.title('Q1.d LoG ANDed sobel')
plt.imshow(univ_log_anded_result,cmap='Greys_r')
plt.show()






