import numpy as np

def distribute_inds(n, nprocs, rank):
    """determines indices of structures that will be relaxed by process nb. `rank`. tries to
    balance computational load evenly between all processes."""
    m = n // nprocs
    r = n % nprocs

    if r == 0:
        return np.arange(rank*m, (rank+1)*m, dtype='int')
    else: # if n is not divisible by n_procs, add 1 extra stt to the load of the last r proc
        if rank < nprocs - r:
            return np.arange(rank*m, (rank+1)*m, dtype='int')
        else:
            k = rank - (nprocs - r)
            return np.arange(rank*m + k, (rank+1)*m + k + 1, dtype='int') # the math checks out, i think


def gather_arrays(arr, comm, no_mix=False, root=0): 
    """Gathers multidimensional arrays of potentially different sizes along axis 0 from all ranks to root rank.
    If `no_mix` = True, then the arrays from each process are 'stacked' (as opposed to concatenated).
    The resulting array `recvarr` will have one dimension more than the arrays being gathered, where `recvarr[i,...]`
    corresponds to the array sent from process `i`.

    CAUTION: When using `mo_mix` =  True, all arrays being gathered must have the same shape, otherwise `recvarr` will 
    be an inhomogeneous array. At best, this will be inefficient and lead to unexpected behaviour. At worst, this will
    break."""

    sendcounts = comm.gather(arr.size, root=root)
    rank = comm.Get_rank()
    if rank == root:
        if no_mix:
            nprocs = comm.Get_size()
            # !!! All arrs must have the same shape for this to work !!!
            recvarr = np.empty((nprocs, *(arr.shape)), dtype=arr.dtype)
        else:
            n_elements = comm.gather(arr.shape[0], root=root) # varying nb of rows in arrays we want to gather
            n_elements_tot = sum(n_elements)
            if arr.ndim > 1:
                recvarr = np.empty((n_elements_tot, *(arr.shape[1:])), dtype=arr.dtype) # assume identical sizes for along all other axes
            else:
                recvarr = np.empty(n_elements_tot, dtype=arr.dtype) # assume identical sizes for along all other axes
        displs = np.zeros(len(sendcounts), dtype=np.int64)
        displs[1:] = np.cumsum(sendcounts[:-1])
        recvbuf = (recvarr.ravel(), (sendcounts, displs))
    else:
        recvarr = None
        recvbuf=None
    
    comm.Gatherv(sendbuf=arr.ravel(), recvbuf=recvbuf, root=0)
    return recvarr