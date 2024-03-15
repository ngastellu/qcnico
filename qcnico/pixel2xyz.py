import numpy as np

def discrete_to_xyz(image, maxrange, nbins): # convert discretized grid back to coordinate space
    # ok so we know that the reconstruction from full grid works, so all we actually have to do is to go
    # from discrete (image) space back to continuous (grid) space
    eps = 1e-8
    pos_bins = np.arange(eps, maxrange + eps + eps, 2 * maxrange / (nbins - 1)) # eps gives unassigned bins their own category
    bins = np.concatenate((-np.flip(pos_bins),pos_bins),0) # bins symmetric about zero
    delta = np.abs(bins[1]-bins[0])
    zero_element = nbins // 2 # the bin which corresponds to exactly zero
    grid = np.zeros(image.shape)
    # un-discretize the grid
    for i in range(image.shape[-2]):
        for j in range(image.shape[-1]):
            if image[0,i,j] != zero_element: # if there is a particle in this gridpoint
                for k in range(2):
                    if (zero_element - image[k,i,j]) > 0:
                        grid[k, i, j] = (image[k, i, j] - zero_element) * delta + delta / 2
                    else:
                        grid[k, i, j] = (image[k, i, j] - zero_element) * delta - delta / 2

    n_particles = np.sum(grid[0,:,:] != 0)
    # reconstruct coordinates
    re_coords = []
    # reconstruct coordinates from 2d grid to confirm the grid was properly made
    for i in range(grid.shape[-2]):
        for j in range(grid.shape[-1]):
            if (grid[0, i, j] != 0) and (grid[1,i,j] != 0):
                re_coords.append((i - grid[0, i, j], j - grid[1, i, j]))

    new_coords = np.zeros((n_particles, 2))
    for m in range(len(re_coords)):
        new_coords[m, :] = re_coords[m]  # the reconstructed coordinates

    return new_coords, grid

def pxl2xyz(image,pixel2angstroms):
    occupied_pixels = np.vstack(image.nonzero()).astype(float).T #get atomic coordinates in pixel space
    occupied_pixels *= pixel2angstroms #convert coordinates to angstroms

    return occupied_pixels
