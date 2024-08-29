#!/usr/bin/env python

import numpy as np


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
    
