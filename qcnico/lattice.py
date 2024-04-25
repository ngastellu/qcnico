#!/usr/bin/env python

import numpy as np

def cartesian_product(*arrays):
    """Make Cartesian (or direct) product of set arrays of equal length."""
    la = len(arrays)
    dtype = np.result_type(*arrays) #use type promotion to match types of all input arrays
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)): #ix_ magic
        arr[...,i] = a
    return arr.reshape(-1, la)


def make_supercell(unit_cell,lattice_vectors,m,n):
    """Generate a graphene supercell consisting of `m` x `n` unit cells (in lattice directions). 

    Parameters
    ----------

    `unit_cell`: `numpy.ndarray`, shape=(2,n)
        Array of COLUMN vectors of cartesian coordinates of unit cell
    `lattice_vectors`: `numpy.ndarray`, shape=(2,2)
        Array of COLUMN vectors containing the lattice vectors
    `m`: `int`
        Number of unit cells in the a1 direction.
    `n`: `int`
        Number of unit cells in the a2 direction.

    Output
    ------

    `supercell`: `np.ndarray`
        NumPy array containing the Cartesian coordinates of the atoms in the supercell.

    """

    mm = np.arange(m)
    nn = np.arange(n)

    #Positions of each unit cell center
    miller = cartesian_product(mm,nn).T
    bravais = (lattice_vectors @ miller).T

    #Re-center cell    
    cell_com = np.mean(unit_cell,axis=1)
    centered_cell = unit_cell - cell_com[:,None]
    print(centered_cell)

    #Generate supercell by decorating each Bravais position with centered unit cell
    print(bravais[:,:,None])
    supercell = np.concatenate(bravais[:,:,None] + centered_cell,axis=1)


    return supercell

