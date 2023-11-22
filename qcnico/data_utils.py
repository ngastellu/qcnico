#!/usr/bin/env python

import numpy as np
from glob import glob

def avg_ydata(datadir_prefix,ydata_npy,xdata_npy=None):
    """Computes the average of the data produced by a SLURM job array (or any data which is contained
    in a homogeneously formatted fashion).

    Parameters
    ----------
    datadir_prefix: `str`
        The common name of all the directories produced by the job array (e.g. 'job-')
    ydata_npy: `str`
        Name of the NPY file containing the data produced by each job, of which we want the average.
    xdata_npy: `str`, optional
        If not `None`, this specifies the name of the NPY file containing the values of the 
        independent variable on which the data being averaged over depends. I.e. for any given
        datadir: y[i] = f(x[i]), where y = np.load(ydata_npy), x = np.load(xdata_npy), and f is the
        function being evaluated by the job array.
        ***** CAUTION: This is assumed to be the same for all jobs in the job array *****
    
    Returns
    -------
    y_avg: `np.ndarray`
        Average of the data contained in the '`datadir_prefix`+*/ydata_npy' files.
    x: `np.ndarray`
        Values of the dependent variable which yielded the values in `out`. This is only output if
        `x_data_npy` is not `None`.
    """
    datadirs = glob(datadir_prefix+'*')
    y_avg = np.load(datadirs[0]+'/'+ydata_npy)
    
    for d in datadirs[1:]:
        y_avg += np.load(d+'/'+ydata_npy)
    
    y_data /= len(datadirs)

    if xdata_npy is not None:
        x = np.load(xdata_npy)
        return x, y_data
    
    else:
        return y_data
    