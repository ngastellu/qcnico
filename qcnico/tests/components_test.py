#!/usr/bin/env python

import numpy as np
from qcnico import graph_tools as gt


M = np.zeros((12,12),dtype=bool)

neighbour_inds = np.array( [ (0,1),
        (0,3),
        (1,2),
        (2,3),
        (3,4),
        (4,5),
        (6,7),
        (9,10),
        (10,11) ] ).T
neighbour_inds = tuple( tuple(n) for n in neighbour_inds )

M[neighbour_inds] = True

M += M.T

c = gt.components(M)
print(c)
