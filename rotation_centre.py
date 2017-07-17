# Script to find the rotation centre in the summed topotomo images

import numpy as np
import matplotlib.pyplot as plt

# Load images that have been cleaned by the background
A = np.load('Clean_sum_img.npy')
A1 = A[1,:,:]
A2 = A[224,:,:]

B1 = np.zeros([A.shape[1], A.shape[2]])
B2 = np.zeros([A.shape[1], A.shape[2]])
# To make it easier to compare plots, show together half of images
B1[0:163,:] = A1[0:163,:]
B1[164:299,:] = A2[164:299,:]
B2[0:163,:] = A2[0:163,:]
B2[164:299,:] = A1[164:299,:]

# Calculate the integrated intensity for the two considered projections
Int1 = sum(map(sum, A1))
Int2 = sum(map(sum, A2))

print Int1, Int2

fig = plt.figure()
plt.subplot(221)
plt.title('Projection 1')
plt.imshow(A1)

plt.subplot(222)
plt.title('Projection 224')
plt.imshow(A2)

plt.subplot(223)
plt.title('Up: 1, down: 224')
plt.imshow(B1)

plt.subplot(224)
plt.title('Up: 224, down: 1')
plt.imshow(B2)

plt.show()
