#!/usr/bin/env pythonw

import os
from collections import deque
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def min_y_pt(pointset):
    """Returns the point in `pointset` with the smallest y coordinate."""

    y_coords = pointset[:,1]
    return pointset[np.argsort(y_coords)[0]]


def remove_point(pointset, pt):
    """Returns NumPy array `pointset` without `pt`."""

    bool_array = np.all(pointset == pt, axis=1)
    if not np.any(bool_array): return pointset

    return pointset[~ bool_array]


def add_point(pointset, pt):
    """Returns numpy array `pointset` with `pt` added to it as its last element.
    If `pt` is already in `pointset`, `pointset` is returned without modification (no duplicates)."""

    if np.any(np.all(pointset == pt, axis=1)): return pointset

    return np.vstack((pointset,pt))


def clockwise_angle(origin,prev_pt,next_pt):
    """Computes the CLOCKWISE angle between line segments [`origin`,`prev_pt`] and 
    [`origin`,`next_pt`].
    
    Parameters
    ----------
    origin: `ndarray`, shape=(2,)
        Point common to both line segments whose clockwise angle we wish to compute angle.
        Both line segments are treated as vectors emanating from this point.
    prev_pt: `ndarray`, shape=(2,)
        Point on the hull that precedes `origin`
    next_pt: `ndarray`, shape=(2.)
        Candidate point to be added to the concave hull after `hull`

    Outputs
    -------
    angle: `float`
        Angle obtained from clockwise rotation between line segments [`origin`, `prev_pt`]
        and [`origin`,`next_pt`]
    """

    u = prev_pt - origin
    v = next_pt - origin

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    angle = np.arccos(np.dot(u,v)/(norm_u*norm_v))

    if np.cross(v,u) < 0: angle =  2*np.pi - angle

    return angle

def angle_sort(current_pt,prev_pt,c_pts):
    """Sorts the array `c_pts` in decreasing order of the clockwise angle formed between
    vectors [`current_pt`, `prev_pt`] and [`current_pt`,`c_pts[i]`].

    Parameters
    ----------
    current_pt: `ndarray`, shape=(2,)
        Last point on the concave hull. Common point between the last edge of the hull
        and all of the candidate edges which are being sorted based on the clockwise
        angle formed between them and the last edge (i.e. line segment [`current_pt`,`prev_pt`]).
    prev_pt: `ndarray`, shape=(2,)
        Before last point on the concave hull. The line segment [`current_pt`,`prev_pt`] defines the
        most recent edge of the concave hull.
    c_pts: `ndarray`, shape=(n,2)
        Array of the `n` nearest neighbours of `current_pt`. These are the candidate points for being
        the next point on the concave hull. Line segments [`current_pt`,`c_pts[i]`] are the candidate
        edges.

    Output
    ------
    c_pts_sorted: `ndarray`, shape=(n,2)
        Elements of `c_pts` sorted in decreasing order of the clockwise angle formed between line segments
        [`current_pt`,`prev_pt`] and [`current_pt`,`c_pts[i]`].
    """
    u = prev_pt - current_pt
    vs = c_pts - current_pt

    norm_u = np.linalg.norm(u)
    norm_vs = np.linalg.norm(vs,axis=1)

    angles = np.arccos((vs @ u)/(norm_vs*norm_u))

    cross_prods = np.cross(vs,u,axisa=1,axisb=0)
    bools_mask = (cross_prods < 0)
    angles[bools_mask] *= -1
    angles[bools_mask] += 2*np.pi
    sorted_inds = np.argsort(-angles) #all angles are positive, so this will reverse their order

    c_pts_sorted = c_pts[sorted_inds]
    return c_pts_sorted



def check_intersect(edgeA,edgeB):
    """Checks whether two line segments intersect. Accepts two 2-tuples (i.e. pairs) of
    points as input and returns `True` if the line segments defined by the input pairs of points
    intersect and `False` if they don't."""

    r1, r2 = edgeA
    r3, r4 = edgeB

    deltaA = r1 - r2
    deltaB = r3 - r4

    coefs = np.vstack((deltaA,-deltaB)).T
    constants = r4 - r2

    try:
        s,t = np.linalg.solve(coefs,constants)
    except np.linalg.LinAlgError as e: #this error only gets raised if the systems of eqs has no solution
        print(e)
        return False

    #print(s)
    #print(t)
    s_bool = (s >= 0) and (s <= 1)
    t_bool = (t >= 0) and (t <= 1)

    if s_bool and t_bool:
        return True
    else:
        return False


def ray_crossings_hull(pt,cycle):
    """Ray crossing method for determining whether a node in a graph ('test node') lies on the inside or the outside of a given polygon comprised of other nodes on the graph.
    Only works in 2D.
    
    Parameters
    ----------
    pt: `ndarray`, shape=(2,)
        Coords of the test node.
    cycle: `deque`
        Set of vertex positions that characterise the polygon.

    Output
    -----
    crossings: `int`
        Number of intersections between a ray in the +x direction emanating from the test node
        and the edges of the polygon.
    """

    crossings = 0
    x_pt, y_pt = pt

    for i in range(len(cycle)):
        xi, yi = cycle[i]
        xj, yj = cycle[i-1]
        
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


def concave_hull(points, k, return_k=False):
    """Computes the polygon envelope of a 2D point set using the k nearest-neighbours approach. 
       Obtained from: 
       http://repositorium.sdum.uminho.pt/bitstream/1822/6429/1/ConcaveHull_ACM_MYS.pdf
       
       Inputs
       ------
       points: `ndarray`, shape = (N,2)
            Set of points whose concave hull we wish to obtain.
            This set should not contain any duplicates.
       k: `int`
            Number of nearest neighbours to consider when building the hull.
            If k > 3, it will automatically be redefined as k = 3.
            If not all points are enclosed inside the final polygon, k is recursively
            incremented and the hull is re-computed with k = k+1 until the obtained hull
            contains the entire input point set.

       Outputs
       -------
       hull_points: `ndarray`
            Coords of pts on the concave hull of the input point set. These points (and the lines between them) 
            best describe the boundary of the point set.
       """
    
    if k < 3: k = 3 #number of nearest neighbour considered must >= 3

    if points.shape[1] != 2:
        print('concave_hull error: Point set must be 2D.\nReturning 0.')
        return 0

    N = points.shape[0]
    
    #if only 3 pts are in the set, then the hull is the set itself
    if N == 3: 
        if return_pos: return points
        else: return np.arange(3)

    dataset = np.array(list(set([tuple(p) for p in points]))) #removes duplicate pts

    #make sure that there are remain enough points in the dataset to find k nearest neighbours
    k = min(k,N-1) 

    first_pt = min_y_pt(dataset)

    hull = deque()
    hull.append(first_pt)

    current_pt = first_pt
    points = remove_point(dataset,current_pt)
    tree = cKDTree(dataset)

    step = 1
    unit_x = np.array([1,0])
    prev_pt = first_pt - unit_x

    while (np.any(current_pt != first_pt) or (step == 1)) and (dataset.shape[0] > 0):
        if step == 5: 
            dataset = add_point(dataset, first_pt)
            tree = cKDTree(dataset)


        #find k nearest neighbours and sort them by angle
        _, k_nn_indices = tree.query(current_pt,k=k)
        k_nn = dataset[k_nn_indices]
        c_pts = angle_sort(current_pt, prev_pt, k_nn)

        intersects = True
        i = 0

        while intersects and (i<c_pts.shape[0]):
            #if the candidate point is the first point of the hull, don't check if the
            #candidate edge intersects with the hull's first edge (bc of course it will)
            candidate = c_pts[i]
            if np.all(candidate == first_pt): last_pt = 1
            else: last_pt = 0

            j = 2
            intersects = False

            while (not intersects) and (j<len(hull) - last_pt):
                intersects = check_intersect((hull[-1],candidate),(hull[-j-1],hull[-j]))
                j += 1

            i += 1

        #if all candidate edges intersect the hull's pre-existing edges, restart the whole
        #procedure by considering k+1 nearest neighbours
        if intersects: 
            #print('All candidate edges intersect with previous edges.\
            #        Restarting with k = %d'%(k+1))
            #print(return_k,flush=True)
            return concave_hull(points,k+1,return_k)

        current_pt = candidate
        hull.append(current_pt)
        prev_pt = hull[-2]
        dataset = remove_point(dataset, current_pt)
        tree = cKDTree(dataset)
        step += 1

    #check that all points lie within the obtained polygon
    all_inside = True
    n = 0
    while all_inside and (n < dataset.shape[0]):
        all_inside = ray_crossings_hull(dataset[n],hull) % 2
        n += 1

    #if not all points lie in the hull, restart the whole procedure by considering
    #k+1 nearest neighbours
    if not all_inside:
        #print('Outliers remain. Restarting with k = %d'%(k+1))
        #print(return_k,flush=True)
        return concave_hull(points, k+1,return_k)
    
    #print('Final k = ', k)
    if return_k: return np.array(hull), k
    else: return np.array(hull)



if __name__ == "__main__":

    structures_dir = '/Users/nico/Desktop/McGill/Research/simulation_outputs/MAC_structures/'
    #gen_type = 'kMC/slurm-6727121_fixed' #type of generation method
    gen_type = 'pCNN'
    data_dir = os.path.join(structures_dir,gen_type)
    #pos_path = os.path.join(data_dir, 'sample-377.xsf')
    pos_path = os.path.join(data_dir, 'pCNN_MAC_102x102.xyz')

    pos = get_coords(pos_path)[:,:2]
    start = perf_counter()
    polygon = concave_hull(pos,3)
    end = perf_counter()
    print('Concave hull time = ',end-start)
    print(pos.shape)
    print(polygon.shape)

    fig, ax = plt.subplots()

    ax.scatter(*pos.T,c='k',s=10.0)
    ax.scatter(*polygon.T,c='r',s=10.0)
    ax.set_aspect('equal')
    plt.show()

    edge_indices = np.zeros(polygon.shape[0],dtype=int)

    for k, r in enumerate(polygon):
        edge_indices[k] = np.all(pos == r,axis=1).nonzero()[0]

    print(edge_indices)
    print(edge_indices.shape)
    print(np.unique(edge_indices).shape)

    fig, ax = plt.subplots()

    ax.scatter(*pos.T,c='k',s=10.0)
    ax.scatter(*pos[edge_indices].T,c='r',s=10.0)
    ax.set_aspect('equal')
    plt.show()
