# For each projection, this script loads the relative images, sums
# them and clean the sum using a rolling median.style approach. 

import numpy as np
import matplotlib.pyplot as plt
import fabio
import os
import scipy.io

# Directory where the IO data is stored
io_dir = '/u/data/alcer/DFXRM_rec/Rec_test'

# List of the image files
im_paths = np.genfromtxt(os.path.join(io_dir + '/List_images.txt'), dtype = str)
# List of the files properties
im_prop = np.loadtxt(os.path.join(io_dir + '/Image_properties.txt'))

Fabio_array = np.zeros([226, 300, 300])
# For each projection, sum the images and store the result
for j in range(226):
    sum_om = np.zeros([300,300])
    for i in range(im_prop.shape[0]):
        if im_prop[i,3] == j:
            img_name = im_paths[i]
            I = fabio.open(img_name).data
            sum_om[:,:] += I[106:406, 106:406]
    Fabio_array[j,:,:] = sum_om[:,:]

Median_array = np.zeros([226, 300, 300])
# For each projection, select the images to use for some median filtering
for j in range(226):
    # List all images relative to an omega, than take first and last of 
    # the first line (selected by looking at the data)
    med_arr = np.zeros([49,1])
    n_med = 0
    img_name = im_paths[i]
    for i in range(im_prop.shape[0]):
        if im_prop[i,3] == j:
            n_med = n_med + 1
            med_arr[n_med-1, 0] = i
    # Load first and seventh image
    M1 = im_paths[int(med_arr[0,0])]
    M2 = im_paths[int(med_arr[6,0])]
    I1 = fabio.open(M1).data
    I2 = fabio.open(M1).data
    Median_array[j,:,:] = 0.5 * (I1[106:406, 106:406] + I2[106:406, 106:406])

Fabio_clean = np.zeros([226, 300, 300])
for j in range(226):
    Fabio_clean[j,:,:] = Fabio_array[j,:,:] - 49*Median_array[j,:,:]

# Plot integrated intensity before and after cleaning
Int_bef_aft = np.zeros([226,3])
for i in range(226):
    Int_bef_aft[i,0] = i
    # Images before cleaning by the rolling median
    Int_bef_aft[i,1] = sum(sum(Fabio_array[i,:,:]))
    # Images after cleaning by the rolling median
    Int_bef_aft[i,2] = sum(sum(Median_array[i,:,:]))

fig = plt.figure()
#plt.scatter(Int_bef_aft[:,0], Int_bef_aft[:,1], color='k', label='Before cleaning')
plt.scatter(Int_bef_aft[3:226,0], Int_bef_aft[3:226,2], color='g', label='After cleaning')
plt.title('Integrated intensity after cleaning')
plt.show()

# Save data for matlab analysis
#scipy.io.savemat('Fabio_clean.mat',{"foo":Fabio_clean})
