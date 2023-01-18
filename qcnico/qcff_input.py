#!/usr/bin/env python

import numpy as np
from graph_tools import adjacency_matrix_sparse

def write_qcff_initconfig(coords, rcut):

    M = adjacency_matrix_sparse(coords,rcut)
    num_bonds = M.count_nonzero()/2
    num_atoms = coords.shape[0]
    print('Number of atoms: %d'%num_atoms)
    print('Number of bonds: %d'%num_bonds)
    
    index_field_size = len(str(coords.shape[0])) #for neat formatting
    dimension = coords.shape[1]

    if dimension not in [2,3]:
        print('ERROR: atomic positions must be expressed in 2D or 3D space.\nNo file written.')
        return

    with open('initconfig.inpt','w') as fo:

        fo.write('{:d}\n'.format(coords.shape[0]))

        for k,r in enumerate(coords):
            atom_index = k+1
            neighbours = M.getcol(k).nonzero()[0]
            neighbours += 1
            if dimension == 3: 
                x, y, z = r
            else: 
                x, y = r
                z = 0.0
            fo.write('\nA\t{0:{width}d}\t{1:2.8f}\t{2:2.8f}\t{3:2.8f}\t0'.format(atom_index,x,y,z,width=index_field_size))
            counter = 0
            for n in neighbours:
                counter += 1
                fo.write('\t{:{width}d}'.format(n,width=index_field_size))
            while counter < 4:
                counter += 1
                fo.write('\t{:{width}d}'.format(0,width=index_field_size))

if __name__ == '__main__':
    import sys
    from inputoutput_nico import get_coords, write_xyz  
    from remove_dangling_carbons import *

    atom_coords = get_coords(sys.argv[1])
    print(atom_coords)
    prefix = '.'.join(sys.argv[1].split('.')[:-1])
    #rCC = 1.8 #[angstroms] max C-C bond length
    rCC = 1.421 #[angstroms] C-C bond length in graphene
    print('Using rCC = ', rCC)
    
    remove_dangling_switch = input('Remove dangling carbons? [y/n]')

    if remove_dangling_switch == 'y':
        new_coords = remove_dangling_carbons(atom_coords,rCC)   
        check = check_neighbours(new_coords,rCC)
        write_xyz(new_coords, ['C']*new_coords.shape[0] ,'dangling-C-free_'+prefix+'.xyz')
        write_qcff_initconfig(new_coords,rCC)
    elif remove_dangling_switch == 'n':
        check = check_neighbours(atom_coords,rCC)
        write_qcff_initconfig(atom_coords,rCC)
    else:
        print('ERROR: Invalid user input.\n Enter \'y\' if you wish to\
        remove dangling Cs or from the input configuration, or \'n\' if not.')
