#!/usr/bin/env python

import numpy as np
from qcnico.plt_utils import multiple_histograms

dir_relaxed = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/Hao_ARPACK/hvals/'
dir_unrelaxed = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/Hao_ARPACK_unrelaxed/hvals/'

labels = set(range(38)) - {4,16,18,21,27,32} #remove structures for which the diagonlisation was successful when unrelaxed

k = 0
for n in labels:
    if k == 5:
        break
    hvals_relaxed = np.load(dir_relaxed + f'hvals-{n}.npy')
    hvals_unrelaxed = np.load(dir_unrelaxed + f'hvals-{n}.npy')
    multiple_histograms((hvals_unrelaxed,hvals_relaxed),('unrelaxed','relaxed'),nbins=30,show=True,title=f'$n = {n}$')
    k+=1


hvals_relaxed = np.hstack([np.load(dir_relaxed + f'hvals-{n}.npy') for n in labels])
hvals_unrelaxed = np.hstack([np.load(dir_unrelaxed + f'hvals-{n}.npy') for n in labels])
multiple_histograms((hvals_relaxed,hvals_unrelaxed),('relaxed','unrelaxed'),nbins=30,show=True,
                    xlabel = '$H_{ij}$ [eV]',
                    title='Distribution of tight-binding $H$ matrix elements for relaxed and unrelaxed $T=1$ MAC.\n ')
    

    
