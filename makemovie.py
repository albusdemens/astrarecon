#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.rc('font', family='DejaVu Sans')
import matplotlib.pylab as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpisize = comm.Get_size()

# rank = 0
# mpisize = 1

print rank, mpisize

if rank == 0:
	start = time.time()

proj_data = np.load('/home/gpu/astra_input/recon4x4/dataarray.npy')
om_vals = np.load('/home/gpu/astra_input/recon4x4/omega.npy')

size = [np.shape(proj_data)[0], np.shape(proj_data)[2]]

print size

local_n = len(om_vals) / mpisize
istart = rank * local_n
istop = (rank + 1) * local_n
local_om = om_vals[istart:istop]
local_data = proj_data[:, istart:istop, :]

fig = plt.figure(frameon=False)
fig.set_size_inches(size[0], size[1])
ax = plt.Axes(fig, [0., 0., 1., 1.])
s = size[0]

print np.shape(proj_data)

for (i, om) in enumerate(local_om):
	print "Working on angle ", str(om), "."
	# ax.set_axis_off()
	fig.add_axes(ax)

	print i
	ta = local_data[:, i, :]

	ax.imshow(ta, interpolation="none")
	ax.autoscale(enable=False)
	# ax.text(50, 50, om, color='white')
	# ax.plot(
	# 	[s - 40 - 2 * 80, s - 40],
	# 	[s - 55, s - 55],
	# 	linewidth=3,
	# 	color='white')

	# ax.plot(
	# 	[30, 30],
	# 	[0, s],
	# 	color='blue')

	# ax.text(s - 180, s - 25, u'100 μm', color='white', fontsize=16)

	ax.set_axis_off()
	extent = ax.get_window_extent().transformed(
		plt.gcf().dpi_scale_trans.inverted())

	fig.savefig(
		'output/movie' +
		'/topo_im_' +
		str('%04d' % (i + rank * local_n)) +
		'.png',
		bbox_inches=extent,
		dpi=1)
	ax.clear()
	# pad_inches = 0)

	# if i == 1:
	# 	break
