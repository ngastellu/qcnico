#!/usr/bin/env python

from os import path
from time import perf_counter
import numpy as np
from qcnico.coords_io import get_lammps_frame, get_lammps_frame_bash


dump = '/Users/nico/Desktop/simulation_outputs/MO_dynamics/40x40/subsampled_trajfiles/300K_norotate_10000-100000-10.lammpstrj'
frames = np.arange(10000,50000,10000)

for n in frames:
    print(f"********** {n} **********")
    start = perf_counter()
    pos1 = get_lammps_frame(dump, n, step=10, frame0_index=10000)
    end = perf_counter()

    print(f"Python method took {end - start} seconds.")

    start = perf_counter()
    pos2 = get_lammps_frame_bash(dump, n, step=10, frame0=10000)
    end = perf_counter()

    print(f"Bash method took {end - start} seconds.")

    print(pos1 == pos2)
