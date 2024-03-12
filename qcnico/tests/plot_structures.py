#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.coords_io import read_xyz, read_xsf
from qcnico.qcplots import plot_atoms
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from glob import glob

# lbls = range(8)
strucdir_unrelaxed = '/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/t1/'
strucdir_relaxed = '/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/t1/relaxed/'
labels = set(range(38)) - {4,16,18,21,27,32} #remove structures for which the diagonlisation was successful when unrelaxed
f_template = 't1n'
rCC = 1.8

for n in labels:
    f_unrelaxed = f_template + f'{n}.xyz'
    pos_unrelaxed = remove_dangling_carbons(read_xyz(strucdir_unrelaxed + f_unrelaxed),rCC)
    
    f_relaxed =  f_template + f'{n}_relaxed.xsf'
    pos_relaxed, _ = read_xsf(strucdir_relaxed + f_relaxed)
     
    print(f'{n} ---> {np.all(pos_unrelaxed == pos_relaxed)}')
    # print(f'{n} ---> {np.max(np.linalg.norm(pos_unrelaxed - pos_relaxed, axis=1 ))}')

    fig, ax = plt.subplots()
    plot_atoms(pos_unrelaxed, dotsize=0.5, plt_objs=(fig,ax),show=False)
    ax.set_title(f'{n} unrelaxed')
    # ax.set_title(filename.split('/')[-1].split('n')[1].split('b')[0])
    plt.show()

    fig, ax = plt.subplots()
    plot_atoms(pos_relaxed, dotsize=0.5, plt_objs=(fig,ax),show=False)
    ax.set_title(f'{n} relaxed')
    plt.show()