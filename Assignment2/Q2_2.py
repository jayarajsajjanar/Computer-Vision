import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage

def threshold_trial(a,t):
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			if i%2==0 and j%2!=0:
				if abs(a[i][j])>t:
					(a[i][j])=1
				else:
					(a[i][j])=0
			if i%2!=0 and j%2==0:
				if abs(a[i][j])>t:
					(a[i][j])=1
				else:
					a[i][j]=0
	return a

def super_grid2(a):

	length_col_pad = a.shape[0]
	col_pad = np.zeros((1,length_col_pad))
	for i in range(a.shape[1]):
		a = np.insert(a,i*2, col_pad,1) 

	length_row_pad = a.shape[1]
	row_pad = np.zeros((1,length_row_pad))
	for i in range(a.shape[0]):
		a = np.insert(a,i*2,row_pad ,0)

	b=a
	b=b*2
	b=b/2

	for i in range(a.shape[0]-1):
		if i%2==0:
			b[i,:]=a[i+1,:]-a[i-1,:]
	for j in range(a.shape[1]-1):
		if j%2==0:
			b[:,j]=a[:,j+1]-a[:,j-1]

	return b

veg = sp.ndimage.imread('veg.jpeg',flatten=True )

super_grid_veg = super_grid2(veg)

plt.subplot(121)
plt.title('Super grid of veg image')
plt.imshow(super_grid_veg, cmap='Greys_r')

thresholded_veg = threshold_trial(super_grid_veg,80)
plt.subplot(122)
plt.title('Veg after applying threshold=200')
plt.imshow(thresholded_veg, cmap='Greys_r')
plt.show()
