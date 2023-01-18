import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg as sLA
from find_edge_carbons import concave_hull

# Set of functions that are useful to process QCFFPI data and extract 
# quantum chemical properties of MAC structures.


def AO_hamiltonian(M,energy_lvls,delta=-1):
    """Expresses the reduced Hamiltonian of MOs within `delta` hartrees of the HOMO/LUMO
    in the AO basis. If `delta` = -1, then the full Hamiltonian in the AO basis is returned;
    it is furthermore not split into an occupied and virtual Hamiltonian.
    *** ASSUMES ENERGIES ARE SORTED *** """

    N = M.shape[0]
    #_, M = read_MO_file(MO_file)
    #occ, virt = read_energies(orb_file)
    occ = energy_lvls[:int(N/2)]
    virt = energy_lvls[int(N/2):]

    for orbs in [occ,virt]:
        sorted_indices = np.argsort(orbs)

        if not np.all(sorted_indices == np.arange(N/2)):
            print('Energies unsorted in orb_file!')
        
    if delta > 0:
        E_homo = occ[-1]
        E_lumo = virt[0]

        relevant_occ_inds = (occ >= E_homo - delta).nonzero()[0]
        relevant_virt_inds = (virt <= E_lumo + delta).nonzero()[0] 

        print('Number of occupied MOs in reduced hamiltonian = ',relevant_occ_inds.shape)
        print('Number of virtual MOs in reduced hamiltonian = ',relevant_virt_inds.shape)
        
        occ_levels = occ[relevant_occ_inds]
        virt_levels = virt[relevant_virt_inds]

        D_occ = np.zeros((N,N))
        D_occ[relevant_occ_inds,relevant_occ_inds] = occ_levels
        print('D_occ:\n')
        print(D_occ)
        print('\n')

        D_virt = np.zeros((N,N))
        D_virt[relevant_virt_inds+(N//2),relevant_virt_inds+(N//2)] = virt_levels
        print('D_virt:\n')
        print(D_virt)
        print('\n')
        
        AO_hamiltonian_occ = M @ D_occ @ (M.T)
        AO_hamiltonian_virt = M @ D_virt @ (M.T)

        return AO_hamiltonian_occ, AO_hamiltonian_virt
    
    else: #delta = -1 ==> return full Hamiltonian in AO basis
        D = np.diag(energy_lvls)
        AO_hamiltonian = M @ D @ (M.T)
        return AO_hamiltonian


def greens_function_dense(e,Hao,gamL,gamR):

    Heff = Hao - (gamL + gamR)*(1j/2.0)
    E = np.eye(Heff.shape[0])*e

    G = scipy.linalg.inv(E - Heff)

    return G

def MCOs(Hao,gammaL,gammaR,sort=False,return_evals=False):
    
    Heff = Hao - (gammaL + gammaR) *(1j/2.0) #effective Hamiltonian for open system
    Heff_dagger = np.conj(Heff.T)

    if not sort:
        _, MCOs = scipy.linalg.eig(Heff)
        _, MCOs_bar = scipy.linalg.eig(Heff_dagger)

        return MCOs, MCOs_bar

    #sort eigenvectors in order of the real part of their corresponding eigenvalue
    else: 
        lambdas, MCOs = scipy.linalg.eig(Heff)
        lambdas2, MCOs_bar = scipy.linalg.eig(Heff_dagger)

        inds1 = np.argsort(np.real(lambdas))
        inds2 = np.argsort(np.real(lambdas2))

        MCOs = MCOs[:,inds1]
        lambdas = lambdas[inds1]

        MCOs_bar = MCOs_bar[:,inds2]
        lambdas2 = lambdas2[inds2]


        if return_evals:
            return lambdas, MCOs, lambdas2, MCOs_bar

        else:
            return MCOs, MCOs_bar

def MCOs_inv(Hao, gamL,gamR):

    Heff = Hao - (gamL+gamR)*(1j/2.0)

    zs, P = scipy.linalg.eig(Heff)

    # sort eigenvalues (and their corresponding eigenvectors using their real parts
    inds = np.argsort(np.real(zs))
    zs = zs[inds]
    P = P[:,inds]

    Pbar = scipy.linalg.inv(P).T
    zsbar = np.conj(zs)

    return zs, P, zsbar, Pbar


def inverse_participation_ratios(MO_matrix):

    return np.sum(np.abs(MO_matrix)**4, axis = 0)


def AO_gammas(pos, gamma, edge_tol=3.0, return_separate=True, graphene=False):
    """Returns the coupling matrices gamma_L and gamma_R represented in AO space (i.e. diagonal matrices).
    Edge atoms are detected using the concave hull algorithm.
    
    Parameters
    ----------
    pos: `numpy.ndarray`, shape=(N,2) or (N,3)
        Cartesian coordinates of the atoms in the conductor.
    M: `numpy.ndarray`, shape=(N,N)
        MO matrix of the conductor, represented in AO space. Elements of the MO matrix are given by:
        M_ij = <AO_i|MO_j>.
    gamma: `float`
        Coupling parameter (in eV). NOTE: We assume each edge atom is equally coupled to the leads.
    edge_tol: `float`
        Maximum allowed distance from left-/rightmost atom in the conductor for an edge atom to be considered
        of the structures left/right edge.
    return_separate: `bool` 
        If `True`, this function will return the coupling matrix to the left (gamma_L) and right (gamma_R) 
        leads separately. If it is set to `False`, gamma_L + gamma_R is returned.
    graphene: `bool`
        If `True`, the edge_atoms are identified by hand. Only use for graphene.
    
    Outputs
    ------
    gammaL, gammaR: `numpy.ndarray`, shape=(N,N)
        Matrices encoding the conductor's coupling to the left and right leads, respectively.
        If `return_separate` is set to `False`, gammaL+gammaR is returned."""

    if pos.shape[1] == 3:
        pos = pos[:,:2] #keep only x and y coords
    
    # for graphene; the edge is defined at the 
    if graphene: 
        sorted_xs = np.unique(pos[:,0])
        right_bools = (pos[:,0] == sorted_xs[-2]) + (pos[:,0] == sorted_xs[-1]) 
        right_inds = right_bools.nonzero()[0]

        left_bools = (pos[:,0] == sorted_xs[0]) + (pos[:,0] == sorted_xs[1]) 
        left_inds = left_bools.nonzero()[0]
    
    # if the edge is nontrivial, find it using the concave hull algorithm
    else: 
        edge_bois = concave_hull(pos,3)
        xmin = np.min(pos[:,0])
        xmax = np.max(pos[:,0])
        right_edge = edge_bois[edge_bois[:,0] > xmax - edge_tol]
        left_edge = edge_bois[edge_bois[:,0] < xmin + edge_tol]

        right_inds = np.zeros(right_edge.shape[0],dtype=int)
        left_inds = np.zeros(left_edge.shape[0],dtype=int)
        
        for k, r in enumerate(right_edge):
            right_inds[k] = np.all(pos == r, axis=1).nonzero()[0]

        for k, r in enumerate(left_edge):
            left_inds[k] = np.all(pos == r, axis=1).nonzero()[0]
    

    N = pos.shape[0]
    gammaR = np.zeros((N,N),dtype=float)
    gammaL = np.zeros((N,N),dtype=float)

    gammaR[right_inds,right_inds] = gamma
    gammaL[left_inds,left_inds] = gamma

    if return_separate:
        return gammaL, gammaR
    else:
        return gammaL + gammaR


def MO_gammas(M,gamL,gamR,return_diag=False,return_separate=True):
    """Transforms the coupling (broadening) matrices from their AO representation to their MCO
    representations."""


    Mdagger = M.conj().T
    gamL_MO = np.linalg.multi_dot((Mdagger,gamL,M))
    gamR_MO = np.linalg.multi_dot((Mdagger,gamR,M))

    if return_diag:
        couplingsR = np.diag(gamR_MO)
        couplingsL = np.diag(gamL_MO)

        if return_separate:
            return couplingsL, couplingsR
        else:
            return couplingsL + couplingsR 
    else:
        if return_separate:
            return gamL_MO, gamR_MO
        else:
            return gamL_MO + gamR_MO


def MCO_gammas(P,Pbar,gamL,gamR,return_diag=False,return_separate=True):
    """Transforms the coupling (broadening) matrices from their AO representation to their MCO
    representations."""

    Pbar_dagger = np.conj(Pbar.T)

    gamL_MCO = np.linalg.multi_dot((Pbar_dagger,gamL,P))
    gamR_MCO = np.linalg.multi_dot((Pbar_dagger,gamR,P))

    if return_diag:
        couplingsR = np.diag(gamR_MCO)
        couplingsL = np.diag(gamL_MCO)

        if return_separate:
            return couplingsL, couplingsR
        else:
            return couplingsL + couplingsR 
    else:
        if return_separate:
            return gamL_MCO, gamR_MCO
        else:
            return gamL_MCO + gamR_MCO


def interference_matrix_MCO_evals(e, Hao, gamL, gamR, diag_gf=False, dense_gf=True):

    #Hao = AO_hamiltonian(M, energy_lvls)
    N = Hao.shape[0]

    if diag_gf:

        if dense_gf:
            G = greens_function_dense(e, Hao, gamL, gamR)
        else:
            edgeL = np.diag(gamL).nonzero()[0]
            edgeR = np.diag(gamR).nonzero()[0]
            gamma = np.max(gamL)
            
            G = greens_function(e, Hao, gamma, edgeL, edgeR)
        
        evals, P = scipy.linalg.eig(G)
        
        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals))
        evals = evals[inds]
        P = P[:,inds]

        evals_bar, Pbar = scipy.linalg.eig(np.conj(G.T))

        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals_bar))
        evals_bar = evals_bar[inds]
        Pbar = Pbar[:,inds]

        gammaL = np.linalg.multi_dot((np.conj(P.T), gamL, P))
        gammaR = np.linalg.multi_dot((np.conj(Pbar.T), gamR, Pbar))

        A = gammaL * evals.reshape(N,1)
        B = gammaR * evals_bar.reshape(N,1)
            

    else:
        zs, P, zbars, Pbar = MCOs(Hao, gamL, gamR, sort=True, return_evals=True)
        #zs, P, zbars, Pbar = MCOs_inv(Hao, gamL, gamR)

        eZ = e - zs.reshape(1,N)
        eZbar = e - zbars.reshape(1,N)

        gammaL = np.linalg.multi_dot((np.conj(P.T), gamL, P))
        gammaR = np.linalg.multi_dot((np.conj(Pbar.T), gamR, Pbar))

        A = gammaL / eZ
        B = gammaR / eZbar

    return A * (B.T)

def interference_matrix_MCO_matrix_product(e,Hao,gamL,gamR,diag_gf=False, dense_gf=True):
    
    #Hao = AO_hamiltonian(M, energy_lvls)

    if dense_gf:
        G = greens_function_dense(e, Hao, gamL, gamR)

    else:
        edgeL = np.diag(gamL).nonzero()[0]
        edgeR = np.diag(gamR).nonzero()[0]
        gamma = np.max(gamL)
        
        G = greens_function(e, Hao, gamma, edgeL, edgeR)

    if diag_gf:
        evals, P = scipy.linalg.eig(G)
        
        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals))
        evals = evals[inds]
        P = P[:,inds]

        evals_bar, Pbar = scipy.linalg.eig(np.conj(G.T))

        #sort eigenvalues in order of their real parts
        inds = np.argsort(np.real(1/evals_bar))
        evals_bar = evals_bar[inds]
        Pbar = Pbar[:,inds]

    else:
        P, Pbar = MCOs(Hao, gamL, gamR, sort=True)

    A = np.linalg.multi_dot((np.conj(P.T), gamL, G, P))
    B = np.linalg.multi_dot((np.conj(Pbar.T), gamR, np.conj(G.T), Pbar))

    return A * (B.T)

def interference_matrix_MO(e,M,energy_lvls,gamL,gamR):

    d = np.diag(e - energy_lvls)
    
    Sigma = (gamL+gamR)*(-1j/2)

    G = scipy.linalg.inv((d - Sigma)) #Green's function

    Gdagger = np.conj(G.T) #Hermitian adjoint of G

    A = gamL @ G
    B = gamR @ Gdagger

    return A * (B.T)

def MO_com(pos, MO_matrix, n=None):

    if n:
        psi = np.abs(MO_matrix[:,n]**2)
    else:
        psi = np.abs(MO_matrix**2)
    return psi @ pos


def MO_rgyr(pos,MO_matrix,n,center_of_mass=None):

    psi = np.abs(MO_matrix[:,n])**2

    if np.all(center_of_mass) == None:
        com = psi @ pos

    else: #if center of mass has already been computed, do not recompute
        com = center_of_mass

    R_squared = (pos*pos).sum(axis=-1) #fast way to compute square length of all position vectors
    R_squared_avg = R_squared @ psi

    #return np.sqrt(R_squared_avg - (com @ com))
    return np.sqrt(R_squared_avg - (com*com).sum(1))


def MCO_com(pos, P, Pbar, n):

    Pbar_dagger = np.conj(Pbar).T
    psi = P[:,n]*Pbar_dagger[n,:]
    center_of_mass = psi @ pos

    return center_of_mass
    

def MCO_rgyr(pos,P,Pbar,n,center_of_mass=None):

    Pbar_dagger = np.conj(Pbar).T
    psi = P[:,n]*Pbar_dagger[n,:]

    if np.all(center_of_mass) == None:
        center_of_mass = psi @ pos

    else: #if center of mass has already been computed, do not recompute
        com = center_of_mass

    R_squared = (pos*pos).sum(axis=-1) #fast way to compute square length of all position vectors
    R_squared_avg = R_squared @ psi

    return np.sqrt(R_squared_avg - (com @ com))


def all_rgyrs(pos,MO_matrix,centers_of_mass=None):

    psis = np.abs(MO_matrix)**2

    if np.all(centers_of_mass) == None:
        coms = (psis.T) @ pos

    else: #if centers of mass have already been computed, do not recompute
        coms = centers_of_mass

    R_squared = (pos*pos).sum(-1)
    R_squared_avg = R_squared @ psis

    coms_squared = (coms*coms).sum(-1)

    return np.sqrt(R_squared_avg - coms_squared)


def all_rgyrs_MCO(pos,P,Pbar,centers_of_mass=None):
    """Potentially finished."""

    N = pos.shape[0]

    xs = pos[:,0]
    ys = pos[:,1]
    
    Pbar_dagger = np.conj(Pbar.T)

    if np.any(centers_of_mass) == None:
        xs_MCOs = np.diag(Pbar_dagger @ (P * xs.reshape(N,1)))
        ys_MCOs = np.diag(Pbar_dagger @ (P * ys.reshape(N,1)))

        coms = np.vstack((xs_MCOs, ys_MCOs)).T

    else: #if centers of mass have already been computed, do not recompute
        coms = centers_of_mass

    R_squared = (pos*pos).sum(-1) #square all positions and sum along last axis (i.e. each vector's coords)
    R_squared_avg = np.diag(Pbar_dagger @ (R_squared.reshape(N,1) * P))

    coms_squared = (coms*coms).sum(-1)

    return np.sqrt(R_squared_avg - coms_squared)
