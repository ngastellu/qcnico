
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
from scipy.sparse import find

def pair_correlation_hist(structure,L,rmin,rmax,Nbins,method='kdtree'):
    """Computes the radial pair correlation for a 2D system using histogram method. PBC (square box) are assumed
    for simplicity, so large structures should be used to avoid artefacts.
    
    PBC are used to avoid underestimating the pair correlation function due to edge atoms having less atoms than they
    'should'.
    """
    
    N = len(structure)
    V = L*L
    pair_density = N*(N-1)/(2*V)

    if method == 'pdist':
        # pdist needs to 2d arrays, so I'm building fake 2d arrays of just the x and y coords
        xs = np.zeros((N,2))
        ys = np.zeros((N,2))

        xs[:,0] = structure[:,0]
        ys[:,0] = structure[:,1]

        x_dists = pdist(xs)
        y_dists = pdist(ys)
        
        # Enforce PBC
        long_x_inds = (x_dists > L/2).nonzero()[0]
        long_y_inds = (y_dists > L/2).nonzero()[0]

        x_dists[long_x_inds] = L - x_dists[long_x_inds]
        y_dists[long_y_inds] = L - y_dists[long_y_inds]

        sep_vectors_pbc = np.vstack((x_dists,y_dists)).T #separation vectors w PBC 
        distances = np.linalg.norm(sep_vectors_pbc,axis=1)
    
    else:
        if method != 'kdtree':
            print(f'[pair_correlation_hist] Invalid `method` argument: {method}. Must be "pdist" or "kdtree".\
                  \nUsing default: method =  "kdtree", with rmax = {rmax}.')
        
        # Make sure structure fits into the periodic k-D tree
        xmin = np.min(structure[:,0])
        ymin = np.min(structure[:,1])
        if xmin < 0:
            structure[:,0] -= xmin
        if ymin < 0:
            structure[:,1] -= ymin
        
        Lx = np.max(structure[:,0]) - np.min(structure[:,0])
        Ly = np.max(structure[:,1]) - np.min(structure[:,1])
        eps = 0.5
        
        tree = KDTree(structure,boxsize=[Lx+eps,Ly+eps])
        Mdists = tree.sparse_distance_matrix(tree,max_distance=rmax)
        _, _, distances = find(Mdists)


    counts, bin_edges = np.histogram(distances, Nbins, range=(rmin, rmax))

    mid_points = (bin_edges[:-1] + bin_edges[1:])/2
    dr = bin_edges[1] - bin_edges[0]

    pair_func = counts.astype(float)/(2*np.pi*pair_density*mid_points*dr*2) # add extra factor of two to avoid double-counting dists

    return mid_points, pair_func