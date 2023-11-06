#!/usr/bin/env python

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from itertools import combinations
from collections import deque


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
        set of all ordered triplets (i,j,k) such that node i is linked to nodes j and k, j < i < k, 
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
                path.pop() #remove last node from path
                in_path[v] = False
            elif len(path) >= max_size:
                path.pop()
                in_path[v] = False
                break #exit the loop if the path is too large
            else:
                cycles = chordless_cycles(M,path,max_size,source_neighbours,in_path,cycles)
    
    path.pop() #remove last node from path
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


def count_rings(coords,rcut,max_size=16):
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

    return ring_data

def components(M):
    '''Returns the connected components of a graph characterised by adjacency matrix M.
    Stolen from the NetworkX library.'''

    N = M.shape[0]
    seen = set()
    clusters = []
    for i in range(N):
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
