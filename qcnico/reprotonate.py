#!/usr/bin/env pythonw

import numpy as np

def add_H(atom,neigh1,neigh2):
    rCH = 1.086#[angstrom] sp2 C-H distance
    
    M = (neigh1 + neigh2)/2 #midpoint between both neighbours
    u = atom - M #direction of line between atom to protonate and the midpoint
    d = np.linalg.norm(u)
    t = rCH/d

    Hp = atom + t*u 
    Hm = atom - t*u

    Hs = np.array([Hp,Hm])

    return Hs[np.argmax(np.linalg.norm(Hs-M,axis=1))] #the H lies at the position on the line that maximises its distance w M

def add_H_inverted(atom,neigh1,neigh2):
    rCH = 1.086#[angstrom] sp2 C-H distance
    
    M = (neigh1 + neigh2)/2 #midpoint between both neighbours
    u = atom - M #direction of line between atom to protonate and the midpoint
    d = np.linalg.norm(u)
    t = rCH/d

    Hp = atom + t*u 
    Hm = atom - t*u

    Hs = np.array([Hp,Hm])

    return Hs[np.argmin(np.linalg.norm(Hs-M,axis=1))] #the H lies at the position on the line that minimises its distance w M


####### MAIN #######
if __name__ == '__main__':
    #from scipy import sparse
    from .graph_tools import adjacency_matrix_sparse
    from .coords_io import get_coords, write_xyz

    input_file = 'protonated_pCNN_MAC_4x4.xyz'

    atoms = get_coords(input_file)
    Natoms_old = atoms.shape[0]
    Ncarbons = 545

    M = adjacency_matrix_sparse(atoms[:Ncarbons],1.8)

    #list of C atoms to protonate
    to_protonate = np.array([9,
        37,
        68,
        184,
        220,
        295,
        337,
        392,
        457,
        516,
        531,
        542,
        541,
        540,
        539,
        537,
        511,
        493,
        450,
        210,
        209,
        172,
        118,
        85,
        60,
        5,
        6,
        512,
        4])

    to_protonate_inv = 512

    #spurious H atoms
    bad_Hs = np.array([557,567,576,575,573])
    bad_Hs.sort()

    print(bad_Hs)

    temp = np.delete(atoms,bad_Hs,0)

    Natoms_new = temp.shape[0] + to_protonate.shape[0] + 1
    new_atoms = np.zeros((Natoms_new, 3),dtype=float)

    new_atoms[:temp.shape[0],:] = temp[:,:]

    index = temp.shape[0]

    for i in to_protonate:
        carbon = atoms[i,:]
        neighbour_indices = M.getcol(i).nonzero()[0]
        assert len(neighbour_indices) == 2, 'Carbon nb. %d has %d neighbours!'%(i,len(neighbour_indices))
        n1, n2 = atoms[neighbour_indices]
        new_atoms[index,:] = add_H(carbon,n1,n2)
        index += 1

    neighbour_indices_inv = M.getcol(to_protonate_inv).nonzero()[0]
    n1inv,n2inv = atoms[neighbour_indices_inv]
    new_atoms[-1,:] = add_H_inverted(atoms[to_protonate_inv],n1inv,n2inv)

    Nhydrogens = Natoms_new - Ncarbons
    symbols = ['C']*Ncarbons + ['H']*Nhydrogens

    write_xyz(new_atoms,symbols,'newer_reprotonated_MAC_pCNN_4x4.xyz')
