# This script sums all the DFXRM images collected at the same angle
# Input: five-dimensional array written by getdata.py (mu, gamma, omega,
# pixel_x, pixel_y); output: three dimensional array (omega, pixel_x, pixel_y)

import numpy as np
import sys

# Load the npy file from getdata.py (on panda2)
A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray.npy')

A_3d = np.zeros([A.shape[3], A.shape[2],  A.shape[4]])	# (x, omega, y)
# Sum images recorded at the same omega
for oo in range(A.shape[2]):
    A_oo = np.zeros([A.shape[3], A.shape[4]])
    for ii in range(A.shape[0]):
        for jj in range(A.shape[1]):
	    A_oo[:,:] += A[ii, jj, oo, :, :]

    for kk in range(A.shape[3]):
        for ll in range(A.shape[4]):
            A_3d[kk, oo, ll] = A_oo[kk,ll]

# Save the result
np.save('A_3d.npy', A_3d)
