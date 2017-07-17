import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

A_npy = np.load('/u/data/alcer/DFXRM_rec/Rec_test/Astra_input.npy')

fig = plt.figure()
plt.imshow(A_npy[:,3,:])
plt.title('mat')

plt.show()
