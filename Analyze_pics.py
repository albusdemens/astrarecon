# This script sums all the images collected at a certain projection, and
# then cleans the sum

import numpy as np 
import matplotlib.pyplot as plt
import sys

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray.npy')

# Sum images collected at a certain angle
Sum_img = np.zeros([A.shape[2], A.shape[3], A.shape[4]])
Clean_sum_img = np.zeros([A.shape[2], A.shape[3], A.shape[4]])
Integrated_I = np.zeros([A.shape[2], 2])

for oo in range(A.shape[2]):
    for ii in range(A.shape[0]):
        for jj in range(A.shape[1]):
            int = np.sum(np.sum(A[ii,jj,oo,:,:]))
            if int > 0:
                Sum_img[oo,:,:] = Sum_img[oo,:,:] + A[ii,jj,oo,:,:]

    # For each quadrant composing the detector, take a region and calculate the average
    M1 = np.mean(Sum_img[oo, 0:30, 0:30])
    M2 = np.mean(Sum_img[oo, 270:300, 0:30])
    M3 = np.mean(Sum_img[oo, 0:30, 270:300])
    M4 = np.mean(Sum_img[oo, 270:300, 270:300])

    # For each region, subtract the average from the signal
    Clean_sum_img[oo, 0:150, 0:150] = Sum_img[oo, 0:150, 0:150] - M1
    Clean_sum_img[oo, 150:300, 0:150] = Sum_img[oo, 150:300, 0:150] - M2
    Clean_sum_img[oo, 0:150, 150:300] = Sum_img[oo, 0:150, 150:300] - M3
    Clean_sum_img[oo, 150:300, 150:300] = Sum_img[oo, 150:300, 150:300] - M4

    for i in range(Clean_sum_img.shape[1]):
        for j in range(Clean_sum_img.shape[2]):
            if (Clean_sum_img[oo,i,j] < 0 or Clean_sum_img[oo,i,j] > 1E+06):
                Clean_sum_img[oo,i,j] = 0

    Integrated_I[oo,0] = oo
    Integrated_I[oo,1] = np.sum(np.sum(Clean_sum_img[oo,:,:]))

fig = plt.figure()
plt.scatter(Integrated_I[:,0], Integrated_I[:,1])
plt.title('Integrated intensity after background cleaning. Summed images at a given projection', fontsize=18)
plt.xlabel('Projection number', fontsize=16)
plt.ylabel('Integrted intensity', fontsize=16)
plt.show()



