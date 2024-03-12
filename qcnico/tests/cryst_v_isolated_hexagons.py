#!/usr/bin/env python

from os import path
import numpy as np
from qcnico import graph_tools as qcngt
from qcnico.coords_io import read_xyz
from qcnico import qcplots, plt_utils
import matplotlib.pyplot as plt


strucfile = "/Users/nico/Desktop/simulation_outputs/MAC_structures/pCNN/pCNN_MAC_102x102.xyz"
rCC = 1.8

pos = read_xyz(strucfile)
M = qcngt.adjacency_matrix_sparse(pos,rCC)


ring_data, cycles = qcngt.count_rings(pos,rCC,max_size=8,return_cycles=True, distinguish_hexagons=True)

cycles = list(cycles) # 'fix' the ordering of cycles
is_cryst = np.zeros(len(cycles), dtype=bool)


ring_sizes = np.array([len(c) for c in cycles])
cycle_coms = qcngt.cycle_centers(cycles, pos)

hexs = np.array([c for c in cycles if (len(c) == 6)])
Mhex = qcngt.hexagon_adjmat(hexs)

print(Mhex.sum(0))

# print(cycle_coms)
print(np.vstack(M.nonzero()))


# plt.show()


i6,c6 = qcngt.classify_hexagons(hexs)
print("Number of crystalline hexagons: ", len(c6))
print("Number of isolated hexagons: ", len(i6))

# Now, we want to find which index in the cycles array refer to isolated hexagons
i6 = np.sort(list(i6))

hex_inds = (ring_sizes == 6).nonzero()[0]
iso_inds = hex_inds[i6]

ring_sizes[iso_inds] *= -1 #isolated hexagons are assigned size -6

print(ring_data)

fig, ax = qcplots.plot_rings_MAC(pos,M,ring_sizes,cycle_coms,dotsize_atoms=10.0,dotsize_centers=50.0,show=False, return_plt_objs=True)

plt.show()



