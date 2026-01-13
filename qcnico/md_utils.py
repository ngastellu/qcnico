#!/usr/bin/env python

import numpy as np
import subprocess as sbp

def _get_nframes_LAMMPS_log(logfile):
    """Counts the number of frames captured in the LAMMPS logfile."""

    iframe0, iframe1 = [int(s.split()[0]) for s in sbp.run(f"grep -E '^[[:blank:]]+[0-9]+[[:blank:]]+[-]*[0-9]+' -m 2 {logfile}", shell=True, capture_output=True).stdout.decode().split('\n')[:-1]]

    step = iframe1 - iframe0

    iframe_last = int(sbp.run(f"tail -r {logfile} | grep -E '^[[:blank:]+[0-9]+[[:blank:]]+[-]*[0-9]+' -m 1", shell=True, capture_output=True).stdout.decode().split()[0])

    return 1 + (iframe_last - iframe0) // step

def parse_LAMMPS_log(logfile):
    """Reads data written to a logfile by a `thermo` command during a LAMMPS MD run. 
    Stores the output as dictionary whose keys are the column headers and the values are NumPy arrays containing
    the different time series recorded during the MD run."""
    cols = sbp.run(f"grep -E -m 1 '^[[:blank:]]+Step' {logfile}", shell=True, capture_output=True).stdout.decode().split()
    data = sbp.run(f"grep -E '^[[:blank:]]+[0-9]+[[:blank:]]+[-]?[0-9]+' {logfile}", shell=True, capture_output=True).stdout.decode().split('\n')[:-1]
    data = np.array([[float(l.split()[k]) for l in data] for k in range(len(cols))])
    return {c:arr for c, arr in zip(cols, data)}


def subsample_structure(pos,xlim=(-np.inf,np.inf),ylim=(-np.inf, np.inf)):
    xmask = (pos[:,0] > xlim[0]) * (pos[:,0] < xlim[1])
    ymask = (pos[:,1] > ylim[0]) * (pos[:,1] < ylim[1])

    mask = xmask * ymask

    return pos[mask]


def count_natoms_lammpstrj(trajfile):
    return int(sbp.run(f"grep -m 1 -A 1 'ATOMS' {trajfile} | tail -1 ", shell=True, capture_output=True).stdout.decode())


    
def count_nsteps_lammpstrj(trajfile, Natoms=None, nb_context_lines=9):
    """Gets index of last saved frame from LAMMPS dump file. This is usally equivalent to the number
      of frames of the MD run it describes, except in the event the run's first frame is not indexed as 1."""
    if Natoms is None:
        Natoms = count_natoms_lammpstrj(trajfile)

    return int(sbp.run(f"tail -{Natoms + nb_context_lines} {trajfile} | grep -A 1 'TIMESTEP' | tail -1", shell=True, capture_output=True ).stdout.decode())


def get_framerate_lammpstrj(trajfile, return_first_frame_index=False):
    cmd_out = sbp.run(f"grep -m 2 -A 1 'TIMESTEP' {trajfile}", shell=True, capture_output=True).stdout.decode().split('\n')
    iframe0 = int(cmd_out[1])
    iframe1 = int(cmd_out[4])
    framerate = iframe1 - iframe0

    if return_first_frame_index == True:
        return framerate, iframe0
    else:
        return framerate

def check_LAMMPS_success(logfile, return_last_line=False):
    normal_exit_message = 'Total wall time:'
    last_line = sbp.run(f"tail -1 {logfile}", shell=True, capture_output=True).stdout.decode()
    success = (last_line[:len(normal_exit_message)] == normal_exit_message)
    if return_last_line:
        return success, last_line
    else:
        return success

