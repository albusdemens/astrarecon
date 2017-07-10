# Script to clean hot pixels from the summed images recorded at different projections

import numpy as np
import matplotlib.pyplot as plt

# Load summed images, saved using clean_images.py
A = np.load('Clean_sum_img.npy')

Om = 2

B = np.zeros([A.shape[1], A.shape[2]])
M = np.mean(A[Om,:,:])

for i in range(A.shape[1]):
    for j in range(A.shape[2]):
        if A[Om,i,j] > 4e5:
            B[i,j] = M
        else:
            B[i,j] = A[Om,i,j]

plt.figure()
plt.imshow(B[:,:])
plt.show()
