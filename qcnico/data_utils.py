#!/usr/bin/env python

import numpy as np
from glob import glob
import os

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

    # Doing some formatting checks/fixes on the specified file names
    if ydata_npy[-4:] != '.npy':
        ydata_npy += '.npy'
    if xdata_npy[-4:] != '.npy':
        xdata_npy += '.npy'
    if datadir_prefix[-1] == '*':
        datadir_prefix = datadir_prefix[:-1]
    

    datadirs = glob(datadir_prefix+'*')
    nsuccess = 0
    k = 0
    while nsuccess == 0:
        d = datadirs[k]
        ynpy = d+'/'+ydata_npy
        if os.path.exists(ynpy):
            y_avg = np.load(ynpy)
            nsuccess = 1
            k+=1
        else:
            print(f'No data found in {d}/')
            k+=1
    
    for d in datadirs[k:]:
        ynpy = d+'/'+ydata_npy
        if os.path.exists(ynpy):
            y_avg += np.load(ynpy)
            nsuccess += 1
        else:
            print(f'No data found in {d}/')
    
    y_avg /= nsuccess

    print(f'**********************\n{nsuccess}/{len(datadirs)} successful runs.')

    if xdata_npy is not None:
        x = np.load(datadirs[k-1]+'/'+xdata_npy) # datadirs[k-1] contains ydata so we assume it also contains xdata
        return x, y_avg
    
    else:
        return y_avg
    

def read_dat_file(datfile):
    """Parses datfile produced by other scripts I wrote. The files are formatted as follows:
    line 1: [Name of x variable] | [Name of y variable]
    ....
    line n: x_n |  y_n
    """

    with open(datfile) as fo:
        lines = fo.readlines()

    x = np.array([float(l.split()[0]) for l in lines[1:]])
    y = np.array([float(l.split()[2]) for l in lines[1:]])

    return x, y

    
def get_successful_inds(datadir, filename_template, extension, pre_label='-'):
    """Looks in `datadir` for files of the form '{filename_template}{pre_label}n.{extension}',
    where `n` is an integer, `extension` is a file type (e.g. 'npy') and `pre_label` is usually
    '-' (by default). Returns the list of integers. 
    Useful to verify which runs in a job array ran successfully produced output files."""
    datfiles = glob(os.path.join(datadir, filename_template + pre_label + '*.' + extension))
    
    isucc = sorted([int(f.split(pre_label)[-1].split('.'+extension)[0]) for f in datfiles])
    return isucc #hehe

def save_npy(data, npyname, npydir=None):
    if npydir is None:
        npydir = '.'
    else:
        if not os.path.isdir(npydir):
            os.makedirs(npydir)
    np.save(os.path.join(npydir,npyname), data)
