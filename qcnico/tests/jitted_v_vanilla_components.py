#!/usr/bin/env python

from percolate import jitted_components
from qcnico.graph_tools import components
import numpy as np
from scipy import sparse



N = 14

edges = [[0,1],
         [0,2],
         [0,3],
         [1,2],
         [4,5],
         [4,6],
         [4,7],
         [4,8],
         [4,9],
         [5,6],
         [5,7],
         [6,7],
         [6,8],
         [7,8],
         [8,9],
         [9,10],
         [11,12]
         ]

M = np.zeros((N,N),dtype='bool')

for e in edges:
    i,j = e
    print(f'({i,j})')
    M[i,j] = True
    M[j,i] = True

Msparse = sparse.csc_matrix(M)

seed = {0,4,11,13}
c1 = components(Msparse,seed_nodes=seed)
c2 = jitted_components(M,seed_nodes=seed)

print(f'Vanilla components found {len(c1)} clusters.')
print(f'Jitted components found {len(c2)} clusters.\n')

print('Vanilla clusters: ')
for c in c1:
    print(c)

print('\nJitted clusters: ')
for c in c2:
    print(c)