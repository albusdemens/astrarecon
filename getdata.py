# This script sums all the DFXRM images collected at the same angle
# Input: five-dimensional array written by getdata.py (mu, gamma, omega,
# pixel_x, pixel_y); output: three dimensional array (omega, pixel_x, pixel_y)

import numpy as np
import sys
import matplotlib.pyplot as plt

threshold = raw_input("Threshold value for the summed images: ")
print "You entered", threshold
# Sample 1: 300

# Load the npy file from getdata.py (on panda2)
A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray_final.npy')

A_3d = np.zeros([A.shape[3], A.shape[2],  A.shape[4]])	# (x, omega, y)
# Sum images recorded at the same omega
for oo in range(A.shape[2]):
    A_oo = np.zeros([A.shape[3], A.shape[4]])
    A_oo_th = np.zeros([A.shape[3], A.shape[4]])
    for ii in range(A.shape[0]):
        for jj in range(A.shape[1]):
	    A_oo[:,:] += A[ii, jj, oo, :, :]

    A_oo_th = A_oo
    A_oo_th[A_oo < int(threshold[0])] = 0
    for kk in range(A.shape[3]):
        for ll in range(A.shape[4]):
            A_3d[kk, oo, ll] = A_oo[kk,ll]

Path = raw_input("Where shoudl I store the summed images?: ")

np.save(Path + '/summed_data_astra.npy', A_3d)
