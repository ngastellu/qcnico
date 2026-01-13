#!/usr/bin/env python

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from itertools import combinations
from functools import reduce
from collections import deque
from time import perf_counter


def adjacency_matrix_sparse(coords,rcut,supercell=None,return_pairs=False):
    """Creates the graph representation (in the form of a sparse adjacency matrix) of a set of atomic coordinates.
    
    Parameters
    ----------
    coords: `ndarray`, shape=(N,3) or (N,2), dtype=float
        Set of atomic coordinates that we wish to represent using a graph.
    rcut: `float`
        Maximum distance between bonded atoms. Any two atoms within `rcut` of each other will be
        considered bonded and will therefore an edge drawn between them in the graph representation
        of the system.
    supercell: `arraylike`, shape=(3,) or (2,), dtype=float
        Size of the box in all three (or two, for 2D systems) directions.
        If supercell is set, then PBC will be enforced.
    return_pairs: `bool`
        If set to `True`, this function will also return the set of all edges (i,j).

    Outputs
    -------
    Mcsc: `scipy.sparse.csc.csc_matrix`, shape=(N,N), dtype=bool
        Adjacency matrix of the graph (which is assumed to be unweighed and undirected) store in the
        sparse CSC format.
    pairs: `set`
        Set of all edges (i,j), where i and j are the indices of connected vertices in the graph and
        i < j. Only returned of the `return_pairs` is set to `True`.
    """

    N = coords.shape[0] #nb of atoms

     #if lattice vectors are specified, enforce PBC
    if supercell:        
        tree = cKDTree(coords,boxsize=supercell)
    else:
        tree = cKDTree(coords)

    neighbour_pairs = np.vstack(list(tree.query_pairs(rcut))).T

    Mcoo = sparse.coo_matrix((np.ones(neighbour_pairs.shape[1]),(neighbour_pairs[0,:],neighbour_pairs[1,:])),shape=(N,N),dtype=bool)
    Mcsc = Mcoo.tocsc()
    Mcsc += Mcsc.transpose()

    if return_pairs: return Mcsc, neighbour_pairs.T
    else: return Mcsc


def get_triangles(M, pairs):
    """Returns the set of all triangles in a graph.

    Parameters
    ----------
    M: `arraylike` shape=(N,N) dtype=bool
        Adjacency matrix of the graph.
    pairs: `ndarray`, shape=(m,2), dtype=int
        Set of all edges in the graph. Each edge is stored as (i,j), where i and j are the
        indices of the nodes connected by edge (i,j)

    Output
    ------
    triangles: `set`
        Set of all triangles (i,j,k), where i, j, and k are the indices of the nodes that form the
        triangle and i < j < k.
    """
    
    triangles = set()
    for i,j in pairs:
        third_vertex_indices = list((M.getcol(i).multiply(M.getcol(j))).nonzero()[0])
        for k in third_vertex_indices:
            triangle_indices = [i,j,k]
            triangle_indices.sort()
            triangles.add(tuple(triangle_indices))
    
    return triangles


def count_triangles(M):
    """Counts the number of triangles in a graph.

    Parameters
    ----------
    M: `scipy.sparse.csc_matrix` shape=(N,N) dtype=bool
        Adjacency matrix of the graph.
    
    Outputs
    -------
    nb_triangles: `int`
        Number of triangles in the graph.
    """
    
    M3 = M.dot(M.dot(M))
    nb_triangles = int(M3.diagonal().sum()/6.0)
    return nb_triangles


def depth_first_search(M,source_index,n,visited,paths):
    """Finds all of the paths of length n emanating from a given node.
    
    Parameters
    ----------
    M: `arraylike` shape=(N,N) dtype=bool
        Adjacency matrix of the graph.
    source_index: `int`
        Index of the source node.
    n: `int`
        Length of the paths we wish to find.
    visited: `ndarray`, shape=(N,), dtype=bool
        Array that keeps track of which nodes have already been visited by the algorithm.
    paths: `set`
        Set of paths starting from the source node. This set gets recursively as the search progresses.
        Upon completion of the algorithm, it will contain the desired set of all paths of length n starting
        from the source node.
    """

    #visited.append(source_index)
    visited[source_index] = 1

    if n == 0:
        #visited.sort()
        paths.append([source_index,visited.nonzero()[0]])
        #print(visited)
        #print(paths)
        #visited.remove(source_index)
        visited[source_index] = 0

    else: 
        for i in M.getcol(source_index).nonzero()[0]:
            #if i not in visited:
            if visited[i] == 0:
                depth_first_search(M,i,n-1,visited,paths)

        #visited.remove(source_index)
        visited[source_index] = 0


def get_n_cycles(M,index,n):
    """Returns all of the cycles of length n that include a given node v.
    
    Parameters
    ----------
    M: `arraylike` shape=(N,N) dtype=bool
        Adjacency matrix of the graph.
    index: `int`
        Index of the node whose cycles we want to find.
    n: `int`
        Length of the cycles we wish to find.

    Output
    ------
    cycles: `set`
        Set of all cycles of length `n` that include node `index`.
    """

    cycles = []

    visited = np.zeros(M.shape[0],dtype=bool)
    paths = []
    depth_first_search(M,index,n-1,visited,paths)

    for last_node, path in paths:
        if M[index,last_node]: 
            cycles.append(tuple(path))

    return cycles
    

def get_triplets(M):
    """Produces all triplets (i,j,k) for a given graph such that node j is linked to nodes i and k and j < i < k.
    
    Parameter
    ---------
    M: `arraylike` shape=(N,N) dtype=bool
        Adjacency matrix of the graph.
    
    Outputs
    -------
    T: `set`
        set of all ordered triplets (i,j,k) such that node j is linked to nodes i and k, j < i < k, 
        and (i,j,k) do not form a triangle.
    C: `set`
        set of all triplets (i,j,k) such that nodes i, j, and k form a triangle and j < i < k.
    """
    
    T = set()
    C = set()

    #print(M.shape[0])
    for j in range(M.shape[0]):
        neighbours = M.getcol(j).nonzero()[0]
        #print(neighbours)
        for i, k in combinations(neighbours, 2):
            if j > i: #combinations only outputs sorted tuples, so i < k
                continue
            triplet = (i,j,k)
            if M.getcol(i)[k]:
                C.add(triplet)
            else:
                T.add(triplet)
        
    return T, C

        
def chordless_cycles(M,path,max_size,source_neighbours,in_path,cycles):
    """
    Given a path in a graph, this function returns all of the set of all chordless cycles to which this path belongs.

    Parameters
    ----------
    M: `ndarray`, shape=(N,N), dtype=bool
        Adjacency matrix of the graph
    path: `tuple`
        Sequence of node indices describing a path in the graph
    cycles: `set`
        Set of cycles to append to.

    Output
    ------
    cycles: `set`
        Updated set of all cycles in th graph
    """

    u = path[-1] #last node in the path
    #print('\nNEW HEAD: ', u)
    #source_neighbours = M.getcol(source).nonzero()[0]
    neighbours = M.getcol(u).nonzero()[0]

    for v in neighbours:
        if in_path[v]:
            continue
        #neighbours_v = M.getcol(v).nonzero()[0]
        chordless = not np.any(M.getcol(v).toarray()[list(path)[1:-1]]) #make sure path remains chordless
        #print('neighbour: %d, chordless = %d, path: '%(v,chordless), path)
        if (v > path[1]) and chordless:
            path.append(v)
            in_path[v] = True
            #print('Appended.')
            if v in source_neighbours:
                #print('Cycle detected! Cycle: ', path)
                cycles.append(tuple(path))
                path.pop() #remove last node (i.e. v) from path
                in_path[v] = False
            elif len(path) >= max_size:
                path.pop() # remove from path
                in_path[v] = False
                break #exit the loop if the path is too large
            else:
                cycles = chordless_cycles(M,path,max_size,source_neighbours,in_path,cycles)
    
    path.pop() #remove last node (i.e. u) from path
    in_path[u] = False
    #print('Popping path: ', path)

    return cycles

def ray_crossings(M,cycle,u,coords):
    """Ray crossing method for determining whether a node in a graph ('test node') lies on the inside or the outside of a given polygon comprised of other nodes on the graph.
    
    Parameters
    ----------
    M: `ndarray`, shape=(N,N), dtype=bool
        Adjacency matrix of the graph
    cycle: `tuple`
        Set of vertex indices that characterise the polygon.
    u: `int`
        Index of the test node
    coords: `ndarray`, shape=(N,2), dtype=float
        Real-space cartesian coordinates of the atoms represented in the graph.

    Output
    -----
    crossings: `int`
        Number of intersections between a ray in the +x direction emanating from the test node
        and the edges of the polygon.
    """

    crossings = 0
    point = coords[u]
    x_pt, y_pt = coords[u,]

    for i, j in combinations(cycle,2):
        if not M[i,j]: #if (i,j) is not an edge, skip this pair of indices
            continue

        #print('edge: (%d,%d)'%(i,j))
        xi, yi = coords[i]
        xj, yj = coords[j]
        
        yshift_i = yi - y_pt
        yshift_j = yj - y_pt
        yflag_i = (yshift_i >= 0)
        yflag_j = (yshift_j >= 0)

        xshift_i = xi - x_pt
        xshift_j = xj - x_pt
        xflag_i = (xshift_i >= 0)
        xflag_j = (xshift_j >= 0)
        
        if yflag_i == yflag_j: # if both cycle points are on the same side of the ray; no intersection
            #print('Same y')
            continue

        elif (not xflag_i) and (not xflag_j): #both cycle nodes are to the left of the test point
            #print('Same x')
            continue

        if yshift_i == 0: #ray intersects with cycle node i
            #print('Same x as ', i)
            if yshift_j <= 0:
                crossings += 1 #assume that node i is infinitesimally above the ray
                #print('crossing!')
            else:
                continue

        elif yshift_j == 0: #ray intersects with cycle node j
            #print('Same x as ', j)
            if yshift_i <= 0:
                crossings += 1 #assume that node j is infinitesimally above the ray
                #print('crossing!')
            else:
                continue

        else:
            if xflag_i and xflag_j:
                crossings += 1
                #print('crossing! x')
            else: # x coords of the two cycle nodes straddle that of the test point
                a = (yj - yi)/(xj - xi)
                b = yj - a*xj
                intersection_pt = (y_pt - b)/a
                if intersection_pt >= x_pt:
                    crossings += 1
                    #print('crossing! inter')
    return crossings 

def check_bad_cycle(M,cycle,coords):
    """Checks whether or not a given chordless cycle contains smaller cycles. If it does, it is
    deemed bad.
    
    Parameters
    ----------
    M: `ndarray`, shape=(N,N), dtype=bool
        Adjacency matrix of the graph
    cycle: `tuple`
        Set of vertex indices that characterise the cycle.
    coords: `ndarray`, shape=(N,2), dtype=float
        Real-space cartesian coordinates of the atoms represented in the graph.

    Output
    -----
    bad_cycle: `bool`
        This boolean is `True` if the input cycle contains smaller cycles; `False` otherwise.
    """

    bad_cycle = False
    processed_nodes = np.zeros(M.shape[0],dtype=bool)
    processed_nodes[list(cycle)] = True
    #print('Initial processed nodes: ', processed_nodes)
    for v in cycle:
        #print('v: ', v)
        neighbours = M.getcol(v).nonzero()[0]
        for u in neighbours:
            #print('u: ', u)
            if processed_nodes[u]:
                continue
            else:
                processed_nodes[u] = True
                num_crossings = ray_crossings(M,cycle,u,coords)
                if num_crossings % 2 == 1:
                    bad_cycle = True
                    return bad_cycle
    return bad_cycle


def count_rings(coords,rcut,max_size=16, return_cycles=False, distinguish_hexagons=False, return_M=False):
    """Counts all of the atomic rings under a given length in a molecule.
    
    Parameters
    ----------
    coords: `ndarray`, shape=(N,2) or (N,3), dtype=float
        Real-space cartesian coordinates of the atoms in the molecule.
    rcut: `float`
        Maximum distance between bonded atoms. Any two atoms within `rcut` of each other will be
        considered bonded and will therefore an edge drawn between them in the graph representation
        of the system. 
    max_size: int
        Maximum size of the rings that we wish to count.

    Output
    ------
    ring_data: `ndarray`, shape=(max_size-2), dtype=int
        Array whose ith element corresponds to the number of chordless cycles of length (i+3) contained in the graph.
    """

    N = coords.shape[0]
    coords = coords[:,:2] # keep only x- and y-coords (assuming system is 2D)

    M = adjacency_matrix_sparse(coords,rcut)

    paths, cycles = get_triplets(M)
    #print(paths)
    #print(cycles)
    print('Number of triplets: ', len(paths))
    paths = deque(map(deque,paths))
    cycles = deque(cycles)


    while len(paths) > 0:
        p = paths.pop()
        print('\n\nTRIPLET: ', p)
        source = p[0]
        source_neighbours = set(M.getcol(source).nonzero()[0])
        in_path = np.zeros(N,dtype=bool)
        in_path[p] = True
        cycles = chordless_cycles(M,p,max_size,source_neighbours,in_path,cycles)
 
    print('Unsorted cycles: ', cycles)
    unique_cycles = set()
    for cycle in cycles:
        sorted_cycle = tuple(sorted(cycle))
        unique_cycles.add(sorted_cycle)

    #print('Unique cycles: ', unique_cycles)

    for cycle in unique_cycles.copy():
        print('\nCYCLE: ', cycle)
        if check_bad_cycle(M,cycle,coords) == True:
            print('REMOVED')
            unique_cycles.remove(cycle)

    print('Unique cycles post processing: ', unique_cycles)
    ring_data = np.zeros(max_size-2,dtype=int)
    for cycle in unique_cycles:
        if len(cycle) > max_size: 
            continue
        else:
            if len(cycle) == 6: print(cycle)
            ring_data[len(cycle)-3] += 1

    if distinguish_hexagons:
        hexs = np.array([c for c in unique_cycles if (len(c) == 6)]) 
        i6,c6 = classify_hexagons(hexs)

        new_ring_data = np.zeros(ring_data.shape[0]+1,dtype=int) # now need one more entry to distinguish hexagon types
        new_ring_data[:3] = ring_data[:3]
        new_ring_data[3] = len(i6)
        new_ring_data[4] = len(c6)
        new_ring_data[5:] = ring_data[4:]

        ring_data = new_ring_data
        
    out = (ring_data,unique_cycles,M)
    return_bools = [True, return_cycles, return_M]
    return tuple(out[i] for i, b in enumerate(return_bools) if b)


def components(M, seed_nodes=None):
    '''Returns the connected components of a graph characterised by adjacency matrix M.
    Originally shamelessly stolen from the NetworkX library.
    
    If seed_nodes is specified, the algorithm will only consider clusters containing the nodes indexed in
    seed_nodes (in other words the nodes in seed_nodes 'seed' the search for each cluster). 
    It will do so by having its outer loop run only over seed_nodes.'''

    N = M.shape[0]
    seen = set()
    clusters = []
    if seed_nodes is None:
        seed_nodes = range(N) #if no seed nodes are specified, search across all nodes in the graph
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
                        if isinstance(M, sparse.csc.csc_matrix):
                            nextlevel.update(M.getcol(j).nonzero()[0])
                        elif isinstance(M, sparse.csr.csr_matrix):
                            nextlevel.update(M.getrow(j).nonzero()[1])
                        else:
                            nextlevel.update(M[j,:].nonzero()[0])
            seen.update(c)
            clusters.append(c)
    return clusters


def pairwise_dists(pos):
    """Quick and dirty method for computing pairwise distance (in Euclidean space) between points in `pos` array."""
    return np.linalg.norm(pos[None,:] - pos[:,None], axis=2)

def hexagon_adjmat(hexagons):
    """Builds 'adjacency matrix' of hexagons: two hexagons are 'connected' if they share a vertex (i.e. C atom)."""

    hexagons = np.array(hexagons)
    Nhex = hexagons.shape[0]
    Mhex = np.zeros((Nhex,Nhex),dtype=bool)# adjacency matrix for hexagons
    pair_inds = np.vstack(np.triu_indices(Nhex,1)).T

    for ij in pair_inds:
        # print(f"\n***** Working on pair: {ij} *****")
        i, j = ij
        hexi = hexagons[i,:]
        hexj = hexagons[j,:]
        diff = hexi[:,None] - hexj # efficient pairwise difference of all vertex indices in both rings
        if np.any(diff == 0):
            Mhex[i,j] = True
            # common_bools = (diff == 0).nonzero()
            # common_inds_hexi = hexi[common_bools[0]]
            # common_inds_hexj = hexj[common_bools[1]]
            # print(common_inds_hexi)
            # print(common_inds_hexj)
    
    Mhex += Mhex.T
    return Mhex


def classify_hexagons(hexagons,strict_filter=True,return_cryst_clusters=False):
    """Separates hexagons into two classes: 
        * isolated 
        * crystallite
    Crystallites are regions of the MAC sample comprised of only hexagons whose surface area is 
    greater than or equal to one hexagon surrounded with six other hexagons (a 'snowflake motif'). If a hexagon is part of
    such a cluster of hexagons, but more than 2 hexagons removed from a snowflake motif, it is deemed isolated (i.e. not part of
    the crystallite).
    
   **** N.B. ****
   Returns two lists of INDICES which refer to elements of `hexagons`. For this function to work,
   ensure the hexagons is an ORDERED iterable (e.g. not a `set`)."""
    
    hexagons = np.array(hexagons) # cast hexagons as ndarray to make things easier down the road (None indexing, etc.)    
    nhex = hexagons.shape[0]
    print("Building hexagon adjacency matrix...")
    start = perf_counter()
    Mhex = hexagon_adjmat(hexagons)
    end = perf_counter()
    print(f"Done! [{end-start}s]")
    nb_neighbours = Mhex.sum(0) #Mhex is symmetric so summing over rows or cols is the same
    
    nuclei = (nb_neighbours == 6).nonzero()[0] # hexagons surrounded with six other hexagons
    weird_nuclei = (nb_neighbours > 6).nonzero()[0] # this is geometrically impossible so this list should be empty
    print(f"*** Found {nuclei.shape[0]} nuclei! ***")
    print(f"!!!! Found {weird_nuclei.shape[0]} weird nuclei !!!!")
    
    # Strict filtering crystalline hexs involves keeping only those who are the next-nearest-neighbours (at least) of
    # 'nuclei' hexagons.
    if strict_filter:
        nuclei_neighbs = Mhex[:,nuclei].nonzero()[0]

        Mhex = Mhex.astype(int) # gets next-nearest neighbour adjacency matrix
        Mhex2 = np.matmul(Mhex, Mhex)
        nuclei_next_neighbs = Mhex2[:,nuclei].nonzero()[0]

        crystalline_hexs = set(np.concatenate((nuclei,nuclei_neighbs,nuclei_next_neighbs)))

        if return_cryst_clusters:
            from .jitted_cluster_utils import get_clusters
            crystalline_clusters = get_clusters(nuclei,Mhex,crystalline_hexs)

    else:
        crystalline_clusters = components(Mhex, seed_nodes=nuclei)
        crystalline_hexs = reduce(set.union, crystalline_clusters) #set of all crystalline hexagons (one big set as opposed to list of sets)  
    
    all_hexs = set(range(nhex))
    isolated_hexs = all_hexs - crystalline_hexs

    if return_cryst_clusters:
        return isolated_hexs, crystalline_hexs, crystalline_clusters
    else:
        return isolated_hexs, crystalline_hexs

    
def cycle_centers(cycles, pos):
    """Returns the center of gravity of each cycle"""
    
    d = pos.shape[1] # nb of dimensions
    ncycles = len(cycles)
    centers = np.zeros((ncycles,d),dtype=float)

    for k, c in enumerate(cycles):
        centers[k,:] = np.mean(pos[list(c)],axis=0)

    return centers

    
def label_atoms(pos, cycles, ring_data, distinguish_hexagons=False):
    cycles_classified = [deque(maxlen=int(n+1)) for n in ring_data] #classifies rings based on their lengths, n+1 in case n=0
    N = pos.shape[0]
    n_ring_types = len(ring_data)

    # !!!! distinguish_hexagons=True option is BROKEN !!!!
    if distinguish_hexagons:
        hexs = np.array([c for c in cycles if len(c) == 6])
        i6, c6 = classify_hexagons(hexs)
        c6 = set([tuple(h) for h in hexs[list(c6)]])
        i6 = set([tuple(h) for h in hexs[list(i6)]])
        max_ring_size = n_ring_types + 1

        for c in cycles:
            l = len(c)
            if l < 6:
                cycles_classified[l-3].append(c)
            elif l == 6:
                if c in i6:
                    cycles_classified[3].append(c)
                else: 
                    cycles_classified[4].append(c)
            elif l > 6 and l <= max_ring_size:
                cycles_classified[l-2].append(c)
            else:
                continue
 
        # Cycle assignment strategy: assign each atom to the ring type to which it belongs the most
        # Tie breaking priority rules:
        # 1. c6
        # 2. i6
        # 3. smallest ring
        cycle_mem_counts = np.zeros((N,n_ring_types+1), dtype='int') # counts how many n-cycles each atom belongs to
        for k, lc in enumerate(cycles_classified):
            lc_arr = np.array([list(c) for c in lc]).flatten().astype(int) #list of all cycles w length k+3
            iatoms, counts = np.unique(lc_arr,return_counts=True)
            cycle_mem_counts[iatoms,k] = counts
        unassigned = (cycle_mem_counts.sum(1) == 0).nonzero()[0] #inds of atoms belonging to no cycles
        if unassigned.shape[0] > 0:
            cycle_mem_counts[unassigned,-1] = 1
        #permute inds to apply tie-break rules (np.argmax keeps 1st ind in case of a tie)
        permuted_inds = np.array([4,3,0,1,2] + list(range(5,n_ring_types+1)))
        cycle_mem_counts = cycle_mem_counts[:,permuted_inds]
        cycle_types = np.argmax(cycle_mem_counts,axis=1)
        print(cycle_types)
        
        # Change cycle_types s.t. cycle_types[k] = size of ring to which pos[k] belongs    
        cycle_types = permuted_inds[cycle_types] + 3
        print(cycle_types)
        cycle_types[cycle_types == 6] = -6 #isolated hexs are labelled by -6
        cycle_types[(cycle_types > 6)+(cycle_types == 0)] -= 1 # this assigns the correct labels to all rings AND labels unassigned atoms with -1
 
    else:
        max_ring_size = n_ring_types + 2
        for c in cycles:
            l = len(c)
            if l <= max_ring_size:
                cycles_classified[l-3].append(c)
            else: 
                continue

            # Cycle assignment: strategy same as for `distinguish_hexagons` case above
            # Tie breaking priority rules:
            # 1. hexagon
            # 2. smallest ring

        cycle_mem_counts = np.zeros((N,n_ring_types+1), dtype='int') # counts how many n-cycles each atom belongs to
        for k, lc in enumerate(cycles_classified):
            lc_arr = np.array([list(c) for c in lc]).flatten().astype(int)
            iatoms, counts = np.unique(lc_arr,return_counts=True)
            print(iatoms.dtype)
            cycle_mem_counts[iatoms,k] = counts
        unassigned = (cycle_mem_counts.sum(1) == 0).nonzero()[0] #inds of atoms belonging to no cycles
        if unassigned.shape[0] > 0:
            cycle_mem_counts[unassigned,-1] = 1

        #permute columns to apply priority rule (np.argmax keeps 1st ind in case of a tie)
        permuted_inds = np.array([3,0,1,2] + list(range(4,n_ring_types+1)))
        cycle_mem_counts = cycle_mem_counts[:,permuted_inds]
        cycle_types = np.argmax(cycle_mem_counts,axis=1)
        print(cycle_types)
        cycle_types = permuted_inds[cycle_types] #unpermute the indices to match the indexing of ring_data (e.g. 0 --> triangles, etc.)
        cycle_types += 3
        cycle_types[cycle_types>max_ring_size] = -1 # label unassigned atoms with -1
    
    return cycle_types


def label_6c_atoms(pos, rCC):
    _, cycles = count_rings(pos,rCC,max_size=7,return_cycles=True)
    hexs = np.array([c for c in cycles if len(c) == 6])
    i6, c6 = classify_hexagons(hexs)
    cryst_hexs = hexs[list(c6)]
    cryst_atoms = np.unique(cryst_hexs)
    labels = np.zeros(pos.shape[0],dtype='bool')
    labels[cryst_atoms] = True
    return labels





        


            
        
        
        

        






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import perf_counter
    from inputoutput_nico import read_xsf

    #indices = np.array([
    #    [1,2],
    #    [1,8],
    #    [1,9],
    #    [2,3],
    #    [2,4],
    #    [2,8],
    #    [2,9],
    #    [3,4],
    #    [3,5],
    #    [3,12],
    #    [4,6],
    #    [4,7],
    #    [5,6],
    #    [6,7],
    #    [7,8],
    #    [9,10],
    #    [10,11],
    #    [11,12]])
    #indices -= 1

    #coords = np.zeros((12,2),dtype=float)
    #a = 1 #edge length

    #coords[0,:] = np.array([0,0])
    #coords[1,:] = np.array([a,-a])
    #coords[2,:] = np.array([2*a, -3*a/2])
    #coords[3,:] = np.array([a,-2*a])
    #coords[4,:] = coords[3,:] + np.array([a,-a])
    #coords[5,:] = np.array([0,-3*a])
    #coords[6,:] = np.array([-a,-2*a])
    #coords[7,:] = np.array([-a,-a])
    #coords[8,:] = np.array([a,a])
    #coords[9,:] = np.array([2*a,3*a/2])
    #coords[10,:] = np.array([3*a,a])
    #coords[11,:] = np.array([3*a,-a])
    #plt.scatter(*coords.T)
    #plt.show()


    #matrix = sparse.coo_matrix((np.ones(indices.shape[0]),(indices[:,0],indices[:,1])),shape=(12,12))
    #matrix += matrix.transpose()
    #T, C = get_triplets(matrix)
    #print('T: ', T)
    #print('C: ', C)

    #M, pairs = adjacency_matrix_sparse(coords,3*a,return_pairs=True)
    #print(pairs)
    ##print(M.toarray())
    #print(count_rings(coords,2*a))

    start = perf_counter()
    #atomic_coords, _ = read_xsf('forces_14-12.xsf', read_forces=False)
    atomic_coords = read_xyz('../../../simulation_outputs/16x16_MAC_coords.xyz')
    print(len(atomic_coords))
    print(count_rings(atomic_coords[:,:2],1.8,max_size=8))
    end = perf_counter()
    print('Execution time: %f seconds.'%(end-start))
