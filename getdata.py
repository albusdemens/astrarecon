# This script sums all the DFXRM images collected at the same angle
# Input: five-dimensional array written by getdata.py (mu, gamma, omega,
# pixel_x, pixel_y); output: three dimensional array (omega, pixel_x, pixel_y)

# python getdata.py /u/data/alcer/DFXRM_rec/Rec_test/ 1 0

import numpy as np
import sys
import matplotlib.pyplot as plt

'''
Inputs:
Dataset directory
Modality (1 if you need to determine threshold and rotation centre,
2 to determine rotation axis and 3 to prepare data for reconstruction)
Lower threshold (put 0 if you need to determine it)
Upper threshold (put 0 if you need to determine it)
'''

# Note: the standard frame size for the background subtraction is 20 pixels

class makematrix():
	def __init__(
	           self, datadir,
                        modality, thr_lo,
                        thr_up,
		sim=False):

            try:
                self.comm = MPI.COMM_WORLD
                self.rank = self.comm.Get_rank()
                self.size = self.comm.Get_size()
            except NameError:
                self.rank = 0
                self.size = 1

            mode = modality.split(',')
            thr_lo = thr_lo.split(',')
            thr_up = thr_up.split(',')

            # Load the npy file from getdata.py (on panda2)
            A = np.load(datadir + 'dataarray_final.npy')

            if not (int(mode[0]) == 1 or int(mode[0]) == 2 or int(mode[0]) == 3):
                print 'The only possible modes are 1 to 3'
                sys.exit()

            elif int(mode[0]) == 1:
                print 'Determine threhold from the following images'
                # Plot 4 images, to help the user select which threshold to use
                for aa in range(0, A.shape[2], int(A.shape[2]/10)):
                    Sum = np.zeros((A.shape[3], A.shape[4]))
                    for ii in range(A.shape[0]):
                        for jj in range(A.shape[1]):
                            Sum[:,:] += A[ii,jj,aa,:,:]
                    fig = plt.figure()
                    plt.imshow(np.rot90(Sum))
                    plt.show()

            elif int(mode[0]) == 2:
                print 'Determine the rotation axis'
                # Plot initial, final and middle image, together with their sum
                Sum_all = np.zeros([A.shape[3], A.shape[4]])
                count = 0
                fig = plt.subplots(2, 2, figsize=(8, 8))
                for aa in range(0, A.shape[2], int((A.shape[2]-1) /2)):
                        Sum = np.zeros([A.shape[3], A.shape[4]])
                        count = count + 1
                        for ii in range(A.shape[0]):
                                for jj in range(A.shape[1]):
                                        Sum[:,:] += A[ii,jj,aa,:,:]
                        if count == 1 or count == 3:
                                Sum_all += Sum
                        ax = plt.subplot(2, 2, count)
                        plt.imshow(np.rot90(Sum))
                        ax.set_title("Summed intensity at projection %i" % aa)

                ax1 = plt.subplot(2,2,4)
                plt.imshow(np.rot90(Sum_all))
                ax1.set_title("Combined firts and last image")
                plt.show()


            elif int(mode[0]) == 3:
                A_3d = np.zeros([A.shape[3], A.shape[2],  A.shape[4]])	# (x, omega, y)
                # Sum images recorded at the same omega
                for oo in range(A.shape[2]):
                    A_oo = np.zeros([A.shape[3], A.shape[4]])
                    A_oo_th = np.zeros([A.shape[3], A.shape[4]])
                    for ii in range(A.shape[0]):
                        for jj in range(A.shape[1]):
                            A_oo[:,:] += A[ii, jj, oo, :, :]

                    A_oo_th = np.rot90(A_oo)
                    #A_oo_th = A_oo
                    A_oo_th[A_oo < int(thr_lo[0])] = 0
                    if int(thr_up[0]) > 0:
			    A_oo_th[A_oo > int(thr_up[0])] = 0

                    for kk in range(A.shape[3]):
                        for ll in range(A.shape[4]):
                            A_3d[kk, oo, ll] = A_oo[kk,ll]

                np.save(datadir + '/summed_data_astra.npy', A_3d)

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print "Wrong number of input parameters. Data input should be:\n\
            Dataset directory\n\
            Modality (1 if you need to determine which threshold to use\n\
       	    2 to determine rotation axis, 3 to prepare data)\n\
            Lower threshold (put 0 if you need to determine it)\n\
            Upper threshold (put 0 if you need to determine it)\n\
	"
	else:
		mm = makematrix(
			sys.argv[1],
			sys.argv[2],
			sys.argv[3],
			sys.argv[4]
			)
