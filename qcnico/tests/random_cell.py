#!/usr/bin/env python

import numpy as np
from qcnico.lattice import make_supercell
from qcnico.qcplots import plot_atoms



latt_vecs = np.eye(2) * 7

cell = np.random.rand(2,5) * 5


plot_atoms(cell.T,dotsize=10.0)

supercell = make_supercell(cell,latt_vecs,10,10)

plot_atoms(supercell.T,dotsize=2.0)
