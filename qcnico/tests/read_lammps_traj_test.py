#!/usr/bin/env python
import numpy as np
from qcnico.coords_io import read_lammps_traj


dump = '/Users/nico/Desktop/simulation_outputs/MO_dynamics/40x40/subsampled_trajfiles/100K_norotate_10000-100000-10.lammpstrj'

traj = read_lammps_traj(dump,0,10,2)
