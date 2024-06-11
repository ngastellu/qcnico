#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import get_cm
from qcnico.qcplots import plot_atoms
from qcnico.coords_io import read_xyz
from qcnico.jitted_cluster_utils import get_clusters


ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/subsample_tests/40x40/'

pos = read_xyz(ddir + 'bigMAC-1_relaxed_no-dangle.xyz')

with open(ddir + 'centres_hashmap-1.pkl', 'rb') as fo:
    centres_hashtable = pickle.load(fo)

centres = np.zeros((len(centres_hashtable), 2))
for r,k in centres_hashtable.items():
    centres[k] = r

Mhex = np.load(ddir + 'Mhex_global-1.npy')
nuclei = (Mhex.sum(0) == 6).nonzero()[0]

Mhex = Mhex.astype(np.int8)
nuclei_neighbs = Mhex[:,nuclei].nonzero()[0]
Mhex2 = Mhex @ Mhex
nuclei_next_neighbs = Mhex2[:,nuclei].nonzero()[0]
strict_6c = set(np.concatenate((nuclei,nuclei_neighbs,nuclei_next_neighbs)))
clusters = get_clusters(nuclei, Mhex, strict_6c)

cluster_clrs = get_cm(np.arange(len(clusters)),'rainbow',min_val=0.0,max_val=1.0)
cluster_centres = [np.array([centres[k] for k in C]) for C in clusters]


fig, ax  = plot_atoms(pos,dotsize=1.0,show=False)

for cc, clr in zip(cluster_centres, cluster_clrs):
    ax.scatter(*cc.T,c=clr,s=5.0,zorder=4)

plt.show()