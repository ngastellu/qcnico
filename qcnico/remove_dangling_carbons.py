#!/usr/bin/env python

import numpy as np
from scipy import sparse
from qcnico.graph_tools import adjacency_matrix_sparse

def remove_dangling_carbons(coords,rcut):
    """Removes dangling carbons (i.e. carbon atoms with only one neighbour) from a given MAC structure.
    
    Parameters
    ----------
    coords: `numpy.ndarray`, shape=(N,3) or (N,2) dtype=float
        Array storing the cartesian coordinates of all atoms in the structure. 
    rcut: `float`
        Maximum distance between bonded atoms.

    Outputs
    -------
    new_coords: `ndarray`, shape=(P,3) or (P,2), dtype=float
        Array containg the cartesian coordinates of all atoms in the MAC molecule, apart from those 
        with only one neighbour."""
    
    N = coords.shape[0] #number of atoms
    dimension = coords.shape[1]

    M = adjacency_matrix_sparse(coords,rcut).tolil()

    removed = np.zeros(N,dtype=bool)

    dangle_free = False

    while not dangle_free:

        dangle_free = True

        for i, row in enumerate(M.rows):
            if removed[i]:
                continue
            if len(row) < 2:
                dangle_free = False
                removed[i] = True
                M[i,:] = 0
                M[:,i] = 0
        #for i, rbool in range(N):
        #    if rbool:
        #        continue
        #    c = M.getcol(i)
        #    if c.nonzero()[0].shape[0] < 2:
        #        remove[i] = True
                
    remove_indices = removed.nonzero()[0]
     
    P = N - remove_indices.shape[0]
    new_coords = np.zeros((P,dimension),dtype=float)
    j = 0

    for r, dangling_bool in zip(coords,removed):
        if dangling_bool:
            continue
        else:
            new_coords[j,:] = r
            j += 1
        
    return new_coords


def check_neighbours(coords,rcut):
    
    M = adjacency_matrix_sparse(coords,rcut)
    
    good_bonds = True
    problematic = []

    for k in range(M.shape[0]):
        num_neighbours = M.getcol(k).nnz
        if num_neighbours < 2 or num_neighbours > 3:
            good_bonds = False
            problematic.append((k,num_neighbours))
            print('Atom nb. {} has {} neighbours.'.format(k,num_neighbours))
    
    #if good_bonds:
    #    print('All atoms in the edited configuration have 2 or 3 nearest neighbours.')

    return good_bonds, problematic

if __name__ == '__main__':
    import sys
    from inputoutput_nico import get_coords, write_xyz
    
    inp_file = sys.argv[1]
    rCC = 1.75
    coords = get_coords(inp_file)
    print('Original configuration contains %d atoms.'%coords.shape[0])
    new_coords = remove_dangling_carbons(coords,rCC)
    print('Edited configuration now contains %d atoms.'%new_coords.shape[0])
    check = check_neighbours(new_coords,rCC)
    file_name = '.'.join(inp_file.split('.')[:-1]) #file name without extension
    write_xyz(new_coords,['C']*new_coords.shape[0],'dangling-C-free-%s.xyz'%file_name)
