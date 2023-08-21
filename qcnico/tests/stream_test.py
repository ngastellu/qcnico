#!/usr/bin/env python

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from qcnico.coords_io import stream_lammps_traj, get_lammps_frame


fdir = "/Users/nico/Desktop/simulation_outputs/MO_dynamics/40x40/subsampled_trajfiles/"

f100 = fdir + '100K_norotate_10000-100000-10.lammpstrj'

frames = np.array([10000, 20000, 30000])
times_stream = np.zeros_like(frames)
times_read = np.zeros_like(frames)

natoms_read = 10

selected_atoms = slice(0, natoms_read)
pos1 = np.zeros((frames.shape[0], natoms_read, 3))

frames100 = stream_lammps_traj(f100, frames[0], frames[-1], 10000, start_file=10000, step_file=10, atom_indices=selected_atoms)

for k,frame in enumerate(frames):
    print(f"----- {frame} -----")
    start = perf_counter()
    pos1[k] = next(frames100)
    end = perf_counter()

    times_stream[k] = end - start
    print(pos1[k,0,:])

np.save('slicepos.npy', pos1)

print("* * * * * * * * * * ")

#for k, frame in enumerate(frames):
#    print(f"----- {frame} -----")
#    start = perf_counter()
#    pos2[k] = get_lammps_frame(f100, frame, step=10, frame0_index=10000)
#    end = perf_counter()
#
#    times_read[k] = end - start
#    print(pos2[k,:10,:])
#
#
#print("* * * * * * * * * * ")
#plt.plot(times_stream,'r-',label='stream')
#plt.plot(times_read,'b-',label='read')
#plt.legend()
#plt.show()
#print(pos1[1,11:21,:])
#print(pos2[1,11:21,:])
#print(np.all(pos1 == pos2))
