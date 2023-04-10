import numpy as np
from .graph_tools import adjacency_matrix_sparse


# Set of functions useful to extract data from QCFFPI calculations of MAC structures.

def get_Natoms(infile):
    """Returns the number of atoms contained in a MAC structure whose MO coefficients
       are stored in `infile`."""
    with open(infile) as fo:
        line1 = fo.readline()
        L = len(line1.split())
        Natoms = L - 5
        if L < 502:
            return Natoms
        else:
            init_line = False
            L = 0
            while not init_line:
                Natoms += L
                l = fo.readline()
                L = len(l.split())
                init_line = (L == 502)

    return Natoms


def read_MO_file(infile, Natoms=None, MO_inds=None):
    """Reads MO coefs output file from QCFFPI and returns a list of atomic positions and a AO -> MO
     transformation matrix with elements M_ij = <AO_i|MO_j>. If `MO_inds` is specified, then only
     the columns indexed by `MO_inds` (corresponding to specific MOs) are returned, rather thanthe
     full MO matrix."""
    
    if Natoms == None:
        Natoms = get_Natoms(infile)
    with open(infile) as fo:
        lines = fo.readlines()

    positions = np.zeros((Natoms,3),dtype=float)
    MO_matrix = np.zeros((Natoms,Natoms),dtype=np.float64)

    if Natoms <= 497:
        nlines_per_atom = 1
    else:
        nlines_per_atom = int(1 + np.ceil((Natoms-497)/500))

    for k, line in enumerate(lines):
        #print(k)
        atom_index = k // nlines_per_atom
        if atom_index == Natoms: break
        split_line = line.split()
        if k % nlines_per_atom == 0:
            counter = 0
            positions[atom_index,:] = list(map(float,split_line[2:5]))
            MO_matrix[atom_index,:497] = list(map(float,split_line[5:]))
            counter += 497
        else:
            n = len(split_line)
            MO_matrix[atom_index,counter:counter+n] = list(map(float,split_line))
            counter += n

    if MO_inds:
        return positions, MO_matrix[:,MO_inds]
    else:
        return positions, MO_matrix


def read_energies(orb_file,Natoms=-1,convert2eV=True):
    """Reads energies from QCCFPI output file `orb_file` and returns them in an array.
    
    *** ASSUMES ENERGIES ARE SORTED *** 

    If Natoms is specified, only `Natoms` lines are read from `orbfile`. Otherwise, precisely
    the first half of the lines are read (regular QCFFPI runs return the energies for the 
    initial config and for the config after the 1st MD timestep.
    """

    with open(orb_file) as fo:
        lines = fo.readlines()
    if Natoms == -1:
        nlines_to_read = int(len(lines)/2)

    else:
        nlines_to_read = Natoms

    if convert2eV:
        Ha2eV = 27.2114
        return np.array(list(map(float,[l.split()[1] for l in lines[:nlines_to_read]]))) * Ha2eV

    else:
        return np.array(list(map(float,[l.split()[1] for l in lines[:nlines_to_read]])))


def read_Hao(Hao_file, Natoms, convert2eV=True):
    """Reads AO Hamiltonian from Hao.dat file output by QCFFPI."""
    
    Hao = np.zeros((Natoms,Natoms),dtype=float)

    if Natoms <= 500:
        nlines_per_row = 1
    elif Natoms % 500 == 0:
        nlines_per_row = Natoms // 500
    else:
        nlines_per_row = 1 + (Natoms // 500)

    nlines_to_read = nlines_per_row * Natoms

    with open(Hao_file) as fo:
        lines = fo.readlines()[:nlines_to_read]
 
    for k, line in enumerate(lines):
        row_index = k // nlines_per_row
        split_line = line.lstrip().rstrip().split()

        if k % nlines_per_row == 0:
            counter = 0
            Hao[row_index,:500] = list(map(float, split_line))
            counter += 500
        else:
            n = len(split_line)
            Hao[row_index,counter:counter+n] = list(map(float, split_line))
            counter += n

    if convert2eV:
        Ha2eV = 27.2114
        Hao *= Ha2eV

    return Hao


def write_qcff_initconfig(coords, rcut):
    """Writes the `initconfig.inpt` (contains the position/type of the nuclei and describes which 
    atoms share a covalent bond) necessary to  run a QCFFPI calculation. This function does not
    return any value; it merely creates (or overwrites a file).

    Parameters
    ---------
    coords: `np.ndarray`, shape=(N,3) or (N,2)
        Array whose rows contain the Cartesian coordinate of the nuclei.
        If only two spatial dimensions are specified, z = 0 is assumed.

    rcut: `float`
        Max neighbour-neighboour distance; atoms i and j are bonded iff |ri - rj| < rcut.
    """

    M = adjacency_matrix_sparse(coords,rcut)
    num_bonds = M.count_nonzero()/2
    num_atoms = coords.shape[0]
    print('Number of atoms: %d'%num_atoms)
    print('Number of bonds: %d'%num_bonds)
    
    index_field_size = len(str(coords.shape[0])) #for neat formatting
    dimension = coords.shape[1]

    if dimension not in [2,3]:
        print('ERROR: atomic positions must be expressed in 2D or 3D space.\nNo file written.')
        return

    with open('initconfig.inpt','w') as fo:

        fo.write('{:d}\n'.format(coords.shape[0]))

        for k,r in enumerate(coords):
            atom_index = k+1
            neighbours = M.getcol(k).nonzero()[0]
            neighbours += 1
            if dimension == 3: 
                x, y, z = r
            else: 
                x, y = r
                z = 0.0
            fo.write('\nA\t{0:{width}d}\t{1:2.8f}\t{2:2.8f}\t{3:2.8f}\t0'.format(atom_index,x,y,z,width=index_field_size))
            counter = 0
            for n in neighbours:
                counter += 1
                fo.write('\t{:{width}d}'.format(n,width=index_field_size))
            while counter < 4:
                counter += 1
                fo.write('\t{:{width}d}'.format(0,width=index_field_size))
