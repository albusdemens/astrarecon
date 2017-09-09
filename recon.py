#!/bin/python

# python recon.py /home/gpu/astra_input/recon4x4/

#!/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/astra/python')
import numpy as np
import astra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
Inputs:
Dataset directory
Estimated cente of the diffraction region
'''

class reconstr():
	def __init__(
        self, datadir,
        centre_est):

            centre_est = centre_est.split(',')
            mp = [centre_est[0], centre_est[1]]

            def makevectors(om):
                vectors = np.zeros((len(om), 12))
                mu = np.radians(10.2)
                factor = np.sin(mu) / np.tan(mu)

                for i, omi in enumerate(om):

                    # ray direction
                    vectors[i, 0] = np.cos(omi) * np.cos(mu)
                    vectors[i, 1] = np.sin(mu)
                    vectors[i, 2] = - np.sin(omi) * np.cos(mu)

                    # center of detector
                    vectors[i, 3] = 0
                    vectors[i, 4] = 0
                    vectors[i, 5] = 0

                    # vector from detector pixel (0,0) to (0,1)
                    vectors[i, 6] = np.sin(omi)
                    vectors[i, 7] = 0
                    vectors[i, 8] = np.cos(omi)

                    # vector from detector pixel (0,0) to (1,0)
                    vectors[i, 9] = np.cos(omi)
                    vectors[i, 10] = np.cos(mu)
                    vectors[i, 11] = - np.sin(mu) * np.sin(omi)

                return vectors


            def adjustcenter(dataarray, mp):
                new_array = dataarray[
                mp[0] - 100:mp[0] + 100,
                :,
                mp[1] - 100:mp[1] + 100]
                return new_array


            # Create volume geometry
            vol_geom = astra.create_vol_geom(300, 300, 300)

            # Omega angles, create vector array
            # angles = np.linspace(0, 2 * np.pi, 721, True)
            angles = np.load(datadir + 'omega.npy')
            vectors = makevectors(angles)

            # Import dataset as (u, angles, v). u and v are columns and rows.
            proj_data = np.load(datadir + 'summed_data_astra.npy')
            # proj_data = np.load('/u/data/andcj/astra-recon-data/recon90/dataarray.npy')
            #proj_data = adjustcenter(proj_data, [128, 125])

            # Create projection geometry from vector array
            proj_geom = astra.create_proj_geom('parallel3d_vec', proj_data.shape[0], proj_data.shape[2], vectors)
            # proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 180, 180, angles)


            # Create projection ID.
            proj_id = astra.data3d.create('-proj3d', proj_geom, proj_data)

            # Create reconstruction ID.
            rec_id = astra.data3d.create('-vol', vol_geom)
	    cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = proj_id
            # cfg['option'] = {}
            # cfg['option']['GPUindex'] = [0, 1, 2]

            # Create algorithm.
            alg_id = astra.algorithm.create(cfg)

            steps = 200
            print "Running algorithm, {} steps.".format(steps)
            # Run 150 steps.
            astra.algorithm.run(alg_id, steps)

            # Get the result
            rec = astra.data3d.get(rec_id)

            rec = (rec - np.min(rec)) / (-np.min(rec) + np.max(rec))

            # fig = pl.figure(3, figsize=pl.figaspect(1.0))
            # ax = p3.Axes3D(fig)

            # for ix in range(np.shape(rec)[0]):
            # 	print 'line {}'.format(ix)
            # 	for iy in range(np.shape(rec)[1]):
            # 		for iz in range(np.shape(rec)[2]):
            # 			if rec[ix, iy, iz] < 0.7 and rec[ix, iy, iz] > 0.2:
            # 				cax = ax.scatter3D(
            # 					ix, iy, iz, s=2, c=rec[ix, iy, iz])

            rs = np.shape(rec)
            b = 10

            cropped_rec = rec[b:rs[0] - b, b:rs[1] - b, b:rs[2] - b]

            fig = plt.figure(frameon=False)


            for i, image in enumerate(cropped_rec):
                fig.set_size_inches(1, 1)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                # ax.set_axis_off()
                fig.add_axes(ax)
                ax.set_axis_off()
                extent = ax.get_window_extent().transformed(
                    plt.gcf().dpi_scale_trans.inverted())

                ax.imshow(image, interpolation="none")
                fig.savefig('output/slice{:04d}.png'.format(i), dpi=np.shape(cropped_rec)[0])
		ax.clear()


        # pl.figure(1)
        # pl.imshow(rec[b:rs[0] - b, b:rs[2] - b, 56])
        # # pl.savefig('output/slice1.png')
        # pl.figure(2)
        # pl.imshow(rec[b:rs[0] - b, b:rs[2] - b, 58])
        # # pl.savefig('output/slice2.png')
        # pl.figure(3)
        # pl.imshow(rec[b:rs[0] - b, b:rs[2] - b, 60])
        # # pl.savefig('output/slice3.png')
        # pl.show()

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Wrong number of input parameters. Data input should be:\n\
            Dataset directory\n\
            Estimated cente of the diffraction region\n\
			"
	else:
		mm = reconstr(
			sys.argv[1],
            sys.argv[2]
			)
