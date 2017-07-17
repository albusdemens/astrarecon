#!/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/usr/local/astra/python')
import numpy as np
import astra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def makevectors(om):
	vectors = np.zeros((len(om), 12))
	theta = np.radians(90-10.38)	# Insert the correct diffraction angle
	#factor = np.sin(theta) / np.tan(theta)

	### Clockwise case
	for i, omi in enumerate(om):
		# ray direction
		vectors[i, 0] = 1.388E08*np.cos(omi)*np.sin(theta)
		vectors[i, 1] = 1.388E08*np.sin(theta)*np.sin(omi)
		vectors[i, 2] = - 1.388E08*np.cos(theta)

		# center of detector
		vectors[i, 3] = - np.sin(theta)*np.cos(omi)
		vectors[i, 4] =  - np.sin(theta)*np.sin(omi)
		vectors[i, 5] = np.cos(theta)

		# vector from detector pixel (0,0) to (0,1)
		vectors[i, 6] = - np.sin(omi)
		vectors[i, 7] = np.cos(omi)
		vectors[i, 8] = 0

		# vector from detector pixel (0,0) to (1,0)
		vectors[i, 9] = np.cos(theta)*np.cos(omi)
		vectors[i, 10] = np.cos(theta)*np.sin(omi)
		vectors[i, 11] = np.sin(theta)

	### Counterclockwise case
		# ray direction
		#vectors[i, 0] = np.sin(theta)
		#vectors[i, 1] = -np.cos(theta)*np.cos(omi)
		#vectors[i, 2] = np.cos(theta)*np.sin(omi)

		# center of detector
		#vectors[i, 3] = 0
		#vectors[i, 4] = 0
		#vectors[i, 5] = 0

		# vector from detector pixel (0,0) to (0,1)
		#vectors[i, 6] = 1
		#vectors[i, 7] = 0
		#vectors[i, 8] = 0

		# vector from detector pixel (0,0) to (1,0)
		#vectors[i, 9] = 0
		#vectors[i, 10] = np.cos(theta)*np.cos(omi) + np.sin(theta)*np.sin(omi)
		#vectors[i, 11] = np.cos(omi)*np.sin(theta) - np.cos(theta)*np.sin(omi)

	return vectors


#def adjustcenter(dataarray, mp):
#	new_array = dataarray[
#		mp[0] - 100:mp[0] + 100,
#		:,
#		mp[1] - 100:mp[1] + 100]
#	return new_array


# Create volume geometry
vol_geom = astra.create_vol_geom(200, 200, 200)

# Omega angles, create vector array
# angles = np.linspace(0, 2 * np.pi, 721, True)
angles = np.radians(np.load('/home/gpu/astra_data/April_2017_sundaynight/omega.npy'))
vectors = makevectors(angles)

# Create projection geometry from vector array
proj_geom = astra.create_proj_geom('parallel3d_vec', 300, 300, vectors)
# proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 180, 180, angles)

# Import dataset as (u, angles, v). u and v are columns and rows.
proj_data = np.load('/home/gpu/astra_data/April_2017_sundaynight/Astra_input.npy')
# proj_data = np.load('/u/data/andcj/astra-recon-data/recon90/dataarray.npy')
# proj_data = adjustcenter(proj_data, [128, 125])

# ASTRA is built to treat transmission, not diffraction data. Therefore, for
# each omega we calculate the complement of the summed image. We also flip the
# images, to simulate rotation around the vertical axis
proj_data_compl = np.zeros(proj_data.shape)
for oo in range(proj_data.shape[1]):
	max_proj = np.max(proj_data[:,oo,:])
	single_layer = np.zeros([proj_data.shape[0], proj_data.shape[2]])
	single_layer[:,:]= np.rot90(proj_data[:,oo,:])
	proj_data_compl[:,oo,:] = max_proj - single_layer[:,:]

np.save('roatated_3d.npy', proj_data_compl)

# Create projection ID.
proj_id = astra.data3d.create('-proj3d', proj_geom, proj_data_compl)

# Create reconstruction ID.
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
# cfg['option'] = {}
# cfg['option']['GPUindex'] = [0, 1, 2]

# Create algorithm.
alg_id = astra.algorithm.create(cfg)

steps = 150
print "Running algorithm, {} steps.".format(steps)
# Run 150 steps.
astra.algorithm.run(alg_id, steps)

# Get the result
rec = astra.data3d.get(rec_id)

print np.max(rec), np.min(rec), np.mean(rec)

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
