# For each projection, this script loads the relative images, sums
# them and clean the sum using a rolling median.style approach.

import numpy as np
import matplotlib.pyplot as plt
import fabio
import os
import scipy.io
import sys

# Directory where the IO data is stored
io_dir = '/u/data/alcer/DFXRM_rec/Rec_test_2/'
# Directory with the images
im_dir = '/u/data/andcj/hxrm/Al_april_2017/topotomo/monday/Al3/topotomoscan/'
# Figures to be used to clean the background (selected looking at the result
# from Plot_all_im_1omega.pys
first_im = 10
last_im = 112

# Load all datato get dimensions
A = np.load(os.path.join(io_dir + 'dataarray.npy'))
num_idx = A.shape[0]
Im_x = A.shape[3]
Im_y = A.shape[4]
num_proj = A.shape[2]
# Estremes of the ROI in the full frame
roi_lf = (512/2) - Im_x/2
roi_rg = (512/2) + Im_x/2
roi_dw = (512/2) - Im_y/2
roi_up = (512/2) + Im_y/2

del A

# List of the image files
im_paths = np.genfromtxt(os.path.join(io_dir + 'List_images.txt'), dtype = str)
# List of the files properties
im_prop = np.loadtxt(os.path.join(io_dir + 'Image_properties.txt'))

Fabio_array = np.zeros([num_proj, Im_x, Im_y])
# For each projection, sum the images and store the result
for j in range(num_proj):
    sum_om = np.zeros([roi_rg - roi_lf, roi_up - roi_dw])
    for i in range(im_prop.shape[0]):
        if im_prop[i,3] == j:
            img_name = os.path.join(im_dir + im_paths[i])
            I = fabio.open(img_name).data
            sum_om[:,:] += I[roi_lf:roi_rg, roi_dw:roi_up]
    Fabio_array[j,:,:] = sum_om[:,:]

Median_array = np.zeros([num_proj, Im_x, Im_y])
# For each projection, select the images to use for some median filtering
for j in range(num_proj):
    med_arr = np.zeros([num_idx*num_idx, 1])
    n_med = 0
    img_name = im_paths[i]
    for i in range(im_prop.shape[0]):
        if im_prop[i,3] == j:
            n_med = n_med + 1
            if n_med < num_idx*num_idx + 1:
                med_arr[n_med-1, 0] = i

    # Load selected images
    M1 = os.path.join(im_dir + im_paths[int(med_arr[first_im])])
    M2 = os.path.join(im_dir + im_paths[int(med_arr[last_im])])
    I1 = fabio.open(M1).data
    I2 = fabio.open(M1).data
    Median_array[j,:,:] = 0.5 * (I1[roi_lf:roi_rg, roi_dw:roi_up] + I2[roi_lf:roi_rg, roi_dw:roi_up])

print sum(sum(sum(Median_array)))

# To take into account the sample rotation, divide by the mean intensity
mean_int = np.zeros([num_proj,1])
for i in range(num_proj):
    mean_int[i] = np.mean(Fabio_array[i,:,:])

max_mean = int(max(mean_int))

Fabio_clean = np.zeros([num_proj, Im_x, Im_y])
for j in range(num_proj):
    for k in range(Fabio_array.shape[1]):
        for l in range(Fabio_array.shape[2]):
<<<<<<< HEAD
            Fabio_clean[j,k,l] = (Fabio_array[j,k,l] - 49 * Median_array[j,k,l])
=======
	    if (Fabio_array[j,k,l] - (num_idx * num_idx) * Median_array[j,k,l]) > 0:
                Fabio_clean[j,k,l] = (Fabio_array[j,k,l] - (num_idx * num_idx) * Median_array[j,k,l])

    fig = plt.figure()
    a1 = fig.add_subplot(1,3,1)
    plt.imshow(Fabio_array[3,:,:])

    a1 = fig.add_subplot(1,3,2)
    plt.imshow(Median_array[3,:,:])

    a1 = fig.add_subplot(1,3,3)
    plt.imshow(Fabio_clean[3,:,:])

    plt.show()
>>>>>>> 9aa571b3848fbe31310d636338d6ba54bb89417a

# Save data for matlab analysis
#scipy.io.savemat(os.path.join(io_dir + 'Fabio_clean.mat'),{"foo":Fabio_clean})
