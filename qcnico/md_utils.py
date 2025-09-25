#!/usr/bin/env python

import numpy as np
import subprocess as sbp


def parse_LAMMPS_log(logfile,nsteps,fields,nb_opening_lines=29):
    """Read data contained in the columns index by the `fields` array of LAMMPS log file.
    `nsteps` must include step 0
    Assumes the first `nb_opening_lines` do not contain relevant data."""

    out_data = np.zeros((nsteps, len(fields)))
    with open(logfile) as fo:
        for n in range(nb_opening_lines):
            fo.readline()
    
        for k in range(nsteps):
            l = fo.readline().strip()
            out_data[k,:] = np.array([float(s) for s in l.split()])[fields]
    
    return out_data


def subsample_structure(pos,xlim=(-np.inf,np.inf),ylim=(-np.inf, np.inf)):
    xmask = (pos[:,0] > xlim[0]) * (pos[:,0] < xlim[1])
    ymask = (pos[:,1] > ylim[0]) * (pos[:,1] < ylim[1])

    mask = xmask * ymask

    return pos[mask]


def count_natoms_lammpstrj(trajfile):
    return int(sbp.run(f"grep -m 1 -A 1 'ATOMS' {trajfile} | tail -1 ", shell=True, capture_output=True).stdout.decode())


    
def count_nsteps_lammpstrj(trajfile, Natoms=None, nb_context_lines=9):
    if Natoms is None:
        Natoms = count_natoms_lammpstrj(trajfile)

    return int(sbp.run(f"tail -{Natoms + nb_context_lines} {trajfile} | grep -A 1 'TIMESTEP' | tail -1", shell=True, capture_output=True ).stdout.decode())

def get_framerate_lammpstrj(trajfile, return_first_frame_index=False):
    cmd_out = sbp.run(f"grep -m 2 -A 1 'TIMESTEP' {trajfile}", shell=True, capture_output=True).stdout.decode().split('\n')
    print(cmd_out)
    iframe0 = int(cmd_out[1])
    iframe1 = int(cmd_out[4])
    framerate = iframe1 - iframe0

    if return_first_frame_index == True:
        return framerate, iframe0
    else:
        return framerate