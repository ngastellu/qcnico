
from numba import njit, objmode, types
from numba.typed import List

@njit
def jitted_components(M, seed_nodes=None):
    '''The same function as `components` in `qcnico.graph_tools`, but modified to work with Numba: now only works with dense matrix arrays.'''

    N = M.shape[0]
    seen = set()
    clusters = []
    if seed_nodes is None:
        seed_nodes = set(range(N)) #if no seed nodes are given, initiate cluster search over the entire graph
    for i in seed_nodes:
        if i not in seen:
            # do a breadth-first search starting from each unseen node
            c = set()
            nextlevel = {i}
            while nextlevel:
                thislevel = nextlevel
                nextlevel = set()
                for j in thislevel:
                    if j not in c:
                        c.add(j)
                        nextlevel.update(M[j,:].nonzero()[0])
            seen.update(c)
            clusters.append(c)
    return clusters

@njit
def jitted_components_sparse(M, seed_nodes=None):
    '''The same function as `components` in `qcnico.graph_tools`, but modified to work with Numba: now only works with  SciPy sparse LIL or CSR arrays.'''
    N = M.shape[0]
    seen = set()
    clusters = []
    if seed_nodes is None:
        seed_nodes = set(range(N)) #if no seed nodes are given, initiate cluster search over the entire graph
    for i in seed_nodes:
        if i not in seen:
            # do a breadth-first search starting from each unseen node
            c = set()
            nextlevel = {i}
            while nextlevel:
                thislevel = nextlevel
                nextlevel = set()
                for j in thislevel:
                    if j not in c:
                        c.add(j)
                        nextlevel.update(M.getrow(j).nonzero()[1])
            seen.update(c)
            clusters.append(c)
    return clusters



@njit
def get_clusters(nuclei,Mhex,strict_6c):
    """Identifies the crystalline clusters in a MAC structure if the crystallinity criterion is strict (see graph_tools.classify_hexagons).
    
    * Strict criterion for hexagon crystallinity: a hexagon is at most thwo hexagons removed from a crystalline nucleus (see def. of `nuclei` 
    below).

    * Lax criterion for hexagon crystallinity: a hexagon is part of connected cluster containing at least one nucleus.

    Parameters
    ----------
    nuclei : `np.ndarray`, dtype  = `int`, shape = (Nhex,)
        Indices of hexagons who are surrounded by 6 other hexagons 
    Mhex: `np.ndarray`, dtype = `bool`, shape = (Nhex,Nhex)
        Adjacency matrix of hexagons; two hexagons are 'connected' if they share a vertex
    strict_6c : `np.ndarray`, dtype  = `int`, shape = (Nstrict,)
        Hexagons deemed crystalline, following the strict criterion.
    
    Returns
    -------
    strict_clusters: `set` of `tuple`s of `int`s
        Set of all crystalline clusters (i.e. tuples of integers indeing the hexagons).
    """
    lax_clusters = jitted_components(Mhex,seed_nodes=set(nuclei)) #crystalline clusters defined using loose criterion
    lax_6c = lax_clusters[0]
    for c in lax_clusters[1:]:
        lax_6c = lax_6c.union(c)
    ignore_6c = lax_6c - strict_6c #hexs that are crystalline by lax standards but noy by strict standards
    for i in list(ignore_6c):
        Mhex[i,:] = False
        Mhex[:,i] = False
    strict_clusters = jitted_components(Mhex, seed_nodes=set(nuclei))
    return strict_clusters
 

@njit
def get_clusters_sparse(nuclei,Mhex,strict_6c):
    """Identifies the crystalline clusters in a MAC structure if the crystallinity criterion is strict (see graph_tools.classify_hexagons).
    
    * Strict criterion for hexagon crystallinity: a hexagon is at most thwo hexagons removed from a crystalline nucleus (see def. of `nuclei` 
    below).

    * Lax criterion for hexagon crystallinity: a hexagon is part of connected cluster containing at least one nucleus.

    Parameters
    ----------
    nuclei : `np.ndarray`, dtype  = `int`, shape = (Nhex,)
        Indices of hexagons who are surrounded by 6 other hexagons 
    Mhex: `scipy.sparse._lil.lil_array`, dtype = `int`, shape = (Nhex,Nhex)
        Adjacency matrix of hexagons; two hexagons are 'connected' if they share a vertex.
        We're using LIL sparse matrix format because its supports efficient row-slicing and updates to the sparsity structure.
    strict_6c : `np.ndarray`, dtype  = `int`, shape = (Nstrict,)
        Hexagons deemed crystalline, following the strict criterion.
    
    Returns
    -------
    strict_clusters: `set` of `tuple`s of `int`s
        Set of all crystalline clusters (i.e. tuples of integers indeing the hexagons).
    """
    lax_clusters = jitted_components_sparse(Mhex,seed_nodes=set(nuclei)) #crystalline clusters defined using loose criterion
    lax_6c = lax_clusters[0]
    for c in lax_clusters[1:]:
        lax_6c = lax_6c.union(c)
    ignore_6c = lax_6c - strict_6c #hexs that are crystalline by lax standards but noy by strict standards
    for i in list(ignore_6c):
        Mhex[i,:] = 0 #only update rows since that's all that is used in `jitted_components_sparse`
    strict_clusters = jitted_components_sparse(Mhex, seed_nodes=set(nuclei))
    return strict_clusters
 
