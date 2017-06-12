# This script sums all the DFXRM images collected at the same angle
# Input: five-dimensional array written by getdata.py (mu, gamma, omega,
# pixel_x, pixel_y); output: three dimensional array (omega, pixel_x, pixel_y)

import numpy as np

# Load the npy file from getdata.py (on panda2)
A = np.load('/home/gpu/astra_data/April_2017_sundaynight/dataarray.npy')

A_3d = np.zeros([A.shape[2], A.shape[3],  A.shape[4]])
# Sum images recorded at the same omega
for oo in range(A.shape[2]):
	A_oo = np.zeros([A.shape[3], A.shape[4]])
	A_oo[:,:] = A[:, :, oo, :, :].sum()
	A_3d[oo, :, :] = A_oo[:,:]

# Save the result
np.save('../astra_data/April_2017_sundaynight/A_3d.npy', A_3d)
