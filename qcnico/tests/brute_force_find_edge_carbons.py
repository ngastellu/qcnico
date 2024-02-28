#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import histogram, setup_tex
from qcnico.qcplots import plot_atoms
from qcnico.coords_io import read_xyz
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.find_edge_carbons import concave_hull, brute_force_hull

rCC = 1.8

nn = 0 #structure index
pos_path = f'/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/t1-tdot25/t1/t1n{nn}.xyz'
pos = remove_dangling_carbons(read_xyz(pos_path),rCC)[:,:2]

X = pos[:,0]
Y = pos[:,1]

delta = 4
l = 400

setup_tex()
fig, ax = plt.subplots()

histogram(X,nbins=400,xlabel='Coord values',plt_objs=(fig,ax),show=False,plt_kwargs={'color':'red', 'alpha':0.5, 'label':'$x$'})
histogram(Y,nbins=400,xlabel='Coord values',plt_objs=(fig,ax),show=False,plt_kwargs={'color':'blue', 'alpha':0.5, 'label':'$y$'})

plt.legend()
plt.show()


L = (X < delta).nonzero()[0]
R = (X > l-delta).nonzero()[0]

B =(Y < delta).nonzero()[0]
T = (Y > l-delta).nonzero()[0]

colors = np.array(['k'] * pos.shape[0])
colors[L] = 'r'
colors[R] = 'b'
colors[B] = 'g'
colors[T] = 'm'

plot_atoms(pos,colour=colors, dotsize=0.1)

l_edge_bf, r_edge_bf = brute_force_hull(pos,edge_tol=6.66e-3)
edge_bf = np.hstack((l_edge_bf, r_edge_bf))


clrs = np.array(['k']*pos.shape[0])
clrs[edge_bf] = 'r'
plot_atoms(pos,colour=clrs, dotsize=0.1)

edge_ch = concave_hull(pos,2)
