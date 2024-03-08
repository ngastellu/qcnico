#!/usr/bin/env python

import matplotlib.pyplot as plt
from qcnico.coords_io import read_xyz, read_xsf
from qcnico.qcplots import plot_atoms
from glob import glob

# lbls = range(8)
strucdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/tdot25/'
# filename_template = 'tdot25n'
filenames = glob(strucdir + '*xyz')
print(len(filenames))


for filename in filenames:
    # filename =  filename_template + f'{n}_relaxed.xsf'
    # pos, _ = read_xsf(strucdir + filename)
    pos = read_xyz(filename)
    fig, ax = plt.subplots()
    plot_atoms(pos, dotsize=0.5, plt_objs=(fig,ax),show=False)
    # ax.set_title(filename)
    ax.set_title(filename.split('/')[-1].split('n')[1].split('b')[0])
    plt.show()
