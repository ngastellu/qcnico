#!/usr/bin/env python

import numpy as np

def rotate_pos(pos,theta = np.pi/2):
    """Rotates a MAC structure about the z-axis, through its center of mass."""
    
    pos = pos[:,:2]
    com = np.mean(pos,axis=0)
    pos -= com # place center of mass of structure at origin

    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    
    return (R @ (pos.T)).T + com[None,:]
        

if __name__ == '__main__':
    from qcnico.coords_io import read_xyz
    from qcnico.qcplots import plot_atoms_w_bonds
    from qcnico.graph_tools import adjacency_matrix_sparse

    rCC = 1.8

    posfile = '/Users/nico/Desktop/simulation_outputs/MAC_structures/kMC/sample-178l.xyz'
    pos = read_xyz(posfile)
    A = adjacency_matrix_sparse(pos, rCC)

    plot_atoms_w_bonds(pos,A,bond_lw=3)

    pos = rotate_pos(pos)
    plot_atoms_w_bonds(pos,A,bond_lw=3)