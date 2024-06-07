#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.graph_tools import count_rings, classify_hexagons, cycle_centers
from qcnico.plt_utils import get_cm
from qcnico.qcplots import plot_atoms 

pos = np.load('/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/coords_no_dangle/coords-42.npy')
pos = pos[:,:2]

x, y = pos.T

mask = (x > 100) * (x < 200) * (y > 100) * (y < 200)
pos = pos[mask]

rCC = 1.8

_, rings = count_rings(pos,rCC,max_size=7,return_cycles=True)

hexs = [c for c in rings if len(c)==6]
_, _, clusters = classify_hexagons(hexs,return_cryst_clusters=True)

print(len(clusters))
cluster_sizes = [len(C) for C in clusters]
print(cluster_sizes)

cluster_clrs = get_cm(np.arange(len(clusters)),'rainbow',min_val=0.0,max_val=1.0)

clusters_hexs = [[hexs[n] for n in c] for c in clusters]
cluster_centres = [cycle_centers(c, pos) for c in clusters_hexs]

fig, ax  = plot_atoms(pos,dotsize=1.0,show=False)

for cc, clr in zip(cluster_centres, cluster_clrs):
    ax.scatter(*cc.T,c=clr,s=5.0,zorder=4)

plt.show()
