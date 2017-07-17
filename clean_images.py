# This script cleans the DFXRM images

import numpy as np
import matplotlib.pyplot as plt
import lib.EdfFile as EF
import sys
import scipy
import scipy.io

# Load the array with all the images

A = np.load('/u/data/alcer/DFXRM_rec/Rec_test/dataarray.npy')
A_omega = A.shape[2]
A_x = A.shape[3]
A_y = A.shape[4]

# Select projection number
Sum_img = np.zeros([A_omega, A_x, A_y])
Clean_sum_img = np.zeros([A_omega, A_x, A_y])

AAA = np.zeros([A.shape[0], A.shape[1], A.shape[3], A.shape[4]])
AAA[:,:,:,:] = A [:,:,3,:,:]

#scipy.io.savemat('All_img_proj_3.mat', {"foo":AAA})
np.save('All_img_proj_3.npy', AAA)

sys.exit()

for omega in range(A_omega):

    int = np.empty([A.shape[0], A.shape[1]])
    Img_array = np.empty([A.shape[0], A.shape[1], A_x, A_y])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            int[i,j] = sum(sum(A[i,j,omega,:,:]))
            if int[i,j] > 0:
                Img_array[i,j,:,:] = A[i,j,omega,:,:]

    # Sum images collected at a certain angle
    Array_ang_val = np.empty([A.shape[0], A.shape[1], A_x, A_y])
    int = np.empty([A.shape[0], A.shape[1]])
    for ii in range(A.shape[0]):
        for jj in range(A.shape[1]):
            int[ii,jj] = sum(sum(A[ii,jj,omega,:,:]))
            if int[ii,jj] > 0:
                Array_ang_val[ii, jj,:,:] = A[ii,jj,omega,:,:]
                Sum_img[omega,:,:] = Sum_img[omega,:,:] + A[ii,jj,omega,:,:]

    # For each quadrant composing the detector, take a region and calculate the average
    M1 = np.mean(Sum_img[omega,0:30, 0:30])
    M2 = np.mean(Sum_img[omega,270:300, 0:30])
    M3 = np.mean(Sum_img[omega,0:30, 270:300])
    M4 = np.mean(Sum_img[omega,270:300, 270:300])

    # For each region, subtract the average from the signal
    Clean_sum_img[omega, 0:150, 0:150] = Sum_img[omega, 0:150, 0:150] - M1
    Clean_sum_img[omega, 150:300, 0:150] = Sum_img[omega, 150:300, 0:150] - M2
    Clean_sum_img[omega, 0:150, 150:300] = Sum_img[omega, 0:150, 150:300] - M3
    Clean_sum_img[omega, 150:300, 150:300] = Sum_img[omega, 150:300, 150:300] - M4

    # Exclude negative values and hot pixels (value estimated using
    # Clean_hot_pixels.py)
    for i in range(Clean_sum_img.shape[1]):
        for j in range(Clean_sum_img.shape[2]):
            if (Clean_sum_img[omega, i,j] < 0 or Clean_sum_img[omega, i,j] > 4E+05):
                Clean_sum_img[omega, i,j] = 0

#scipy.io.savemat('Topotomo.mat',{"foo":Clean_sum_img})
#scipy.io.savemat('Topotomo_raw.mat', {"foo":Sum_img})

np.save('Topotomo_raw.npy', Sum_img)

sys.exit()

# Let's clean some memory
del A

# Calculate the integrated intensity and the mean for the cleaned images
Int_sum = np.zeros([A_omega,2])
Mean_sum = np.zeros([A_omega,2])
for oo in range(A_omega):
    Int_sum[oo,0] = oo
    Int_sum[oo,1] = sum(sum(Clean_sum_img[oo,:,:]))
    Mean_sum[oo,0] = oo
    Mean_sum[oo,1] = np.mean(Clean_sum_img[oo,:,:])

# Plot the distribution of the integrated intensity and of the mean as a
# function of the projection number
fig = plt.figure()
plt.scatter(Int_sum[:,0], Int_sum[:,1])
plt.title('Integrated intensity per projection after cleaning')
plt.xlabel('Projection number')
plt.ylabel('Integrated intensity')
plt.show()

fig = plt.figure()
plt.scatter(Mean_sum[:,0], Mean_sum[:,1])
plt.title('Average intensity per projection after cleaning')
plt.xlabel('Projection number')
plt.ylabel('Average intensity')
plt.show()

# Divide the summed images by the mean intensity
Scaled_int = np.zeros([A_omega,A_x,A_y])
Scaled_int_sum = np.zeros([A_omega,2])
for oo in range(0,5):#(A_omega):
    for pp in range(A_x):
        for qq in range(A_y):
            if Clean_sum_img[oo,pp,qq] > 0:
                Scaled_int[oo,pp,qq] = (Clean_sum_img[oo,pp,qq] / Mean_sum[oo,1]) * 100
    print np.mean(Scaled_int[oo,:,:])

# Plot an example for a single projection
fig=plt.figure()

ax1=fig.add_subplot(131)
plt.imshow(Sum_img[2,:,:])
ax1.title.set_text('Summed images -- raw')

ax2=fig.add_subplot(132)
plt.imshow(Clean_sum_img[2,:,:])
ax2.title.set_text('Sum - background')

ax2=fig.add_subplot(133)
plt.imshow(Scaled_int[2,:,:])
ax2.title.set_text('Sum - background')

plt.show()
