#!/usr/bin/env python


import pickle
import sys
from qcnico.lattice import cartesian_product
from qcnico.jitted_cluster_utils import get_clusters
import numpy as np


nn = sys.argv[1]
slice_inds = cartesian_product(np.arange(4),np.arange(4))

m,n = slice_inds[0,:]

print(f'Initialising hash maps (m,n) = ({m,n})',flush=True)
pos_global = {tuple(r):k for k,r in enumerate(np.load(f'hex_centers-{nn}_{m}_{n}.npy'))} # global hashtable mapping hexagin centers to integer indices

M = np.load(f'M_hex-{nn}_{m}_{n}.npy')
neighb_list = {k:tuple(M[k,:].nonzero()[0]) for k in M.shape[0]}

ncentres_tot = M.shape[0]

assert pos_global.shape[0] == ncentres_tot, f'Mismatch between number of centers ({pos_global.shape[0]}) and shape of hexagon adjacency matrix {M.shape}!'
print('Done! Commencing loop over other subsamples...',flush=True)


for mn in slice_inds[1:]:
    m,n = mn
    print(f'\n------ {(m,n)} ------',flush=True)
    pos = np.load(f'hex_centers-{nn}_{m}_{n}.npy')
    local_map = {k:-1 for k in range(pos.shape[0])} # hashtable that maps centre indices local to the NPY being processed to their global index (i.e. in `pos_global`)  

    print('Loop 1: ', end='')
    # first, update the global hashtable to properly index centers in subsample (m,n)
    for k, r in enumerate(pos):
        r = tuple(r)
        if r in pos_global:
            local_map[k] = pos_global[r]
        else:
            pos_global[r] = ncentres_tot
            local_map[k] = ncentres_tot
            ncentres_tot += 1
    print('Done!',flush=True)

    print('Loop 2: ', end='',flush=True)
    # next, update neighbour list using global hashmap
    M = np.load(f'M_hex-{nn}_{m}_{n}.npy')
    for k in range(pos.shape[0]):
        k_global = local_map[k]
        ineighbs_local = tuple(M.nonzero()[0])
        # handle case 
        if k_global in neighb_list:
            neighb_list[k_global] = neighb_list[k_global] + ineighbs_local
        else:
            neighb_list[k_global] = ineighbs_local
    print('Done!',flush=True)


print('Constructing global hexagon adjacency matrix...', flush=True)
Mglobal = np.zeros((ncentres_tot,ncentres_tot),dtype=bool)

isnucleus = np.zeros(ncentres_tot,dtype=bool)
isweird = np.zeros(ncentres_tot,dtype=bool)

for k in range(ncentres_tot):
    ineighbs = neighb_list[k]
    Mglobal[k,ineighbs] = True

    nb_neighbs = len(ineighbs)
    if nb_neighbs == 6:
         isnucleus[k] = True
    elif nb_neighbs > 6:
        isweird[k] = True
print('Done! Saving stuff.', flush=True)

np.save('Mhex_global-{nn}.npy',Mglobal)

with open('centres_hashmap-{nn}.pkl', 'wb') as fo:
    pickle.dump(pos_global,fo)

with open('neighbs_dict.pkl', 'wb') as fo:
    pickle.dump(neighb_list,fo)

nuclei = isnucleus.nonzero()[0]
print(f'*** Found {nuclei.shape[0]} crystalline nuclei ***')

if isweird.sum() > 0:
    weird = isweird.nonzero()[0]
    print(f'!!!! Found{weird.shape[0]} weird nuclei !!!!')


print('Searching for clusters...',flush=True)
Mglobal = Mglobal.astype(np.int8)
nuclei_neighbs = Mglobal[:,nuclei]
Mglobal2 = Mglobal @ Mglobal
nuclei_next_neighbs = Mglobal2[:,nuclei].nonzero()[0]
strict_6c = set(np.concatenate(nuclei,nuclei_neighbs,nuclei_next_neighbs))
crystalline_clusters = get_clusters(nuclei, Mglobal, strict_6c)
print('Done!',flush=True)

cluster_sizes = np.array([len(c) for c in crystalline_clusters])
np.save(f'cryst_cluster_sizes-{nn}.npy',cluster_sizes)