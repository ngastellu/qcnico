
import numpy as np
from scipy.spatial.distance import pdist

def pair_correlation_hist(structure,L,rmin,rmax,Nbins):
    """Computes the radial pair correlation for a 2D system using histogram method. PBC (square box) are assumed
    for simplicity, so large structures should be used to avoid artefacts.
    
    PBC are used to avoid underestimating the pair correlation function due to edge atoms having less atoms than they
    'should'.
    """
    
    N = len(structure)
    V = L*L
    pair_density = N*(N-1)/(2*V)

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

    counts, bin_edges = np.histogram(distances, Nbins, range=(rmin, rmax))

    mid_points = (bin_edges[:-1] + bin_edges[1:])/2
    dr = bin_edges[1] - bin_edges[0]

    pair_func = counts.astype(float)/(2*np.pi*pair_density*mid_points*dr)

    return mid_points, pair_func