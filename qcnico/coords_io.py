#!/usr/bin/env python

import numpy as np
from itertools import islice
from os import path

def get_coords(infile, dump=True):

    if dump:
        coords, *_ = read_dump(infile)

    else:
    
        suffix = infile.split('.')[-1]

        if suffix == 'npy':
            coords = np.load(infile)
        
        elif suffix == 'xyz':
            coords = read_xyz(infile)
        
        elif suffix == 'xsf':
            coords, *_ = read_xsf(infile)
        
        else:
            print('Invalid file type: %s\nValid file types: npy, xyz, xsf\nReturning 0'%suffix)
            coords = 0
    
    return coords

def xsf2xyz_MAC(filepath):
    prefix = '.'.join(filepath.split('.')[:-1])
    atoms, *_ = read_xsf(filepath)
    symbols = ['C']*len(atoms)
    outfile = prefix + '.xyz'
    write_xyz(atoms,symbols,outfile)


def read_xyz(filepath):
    """Returns the coordinates of all atoms stored in a .xyz file. It assumes all atoms are of the same
    element and thus does not keep track of the chemical symbols in the input file.

    Parameter
    ----------
    filepath: `str`
        Path to the .xyz file whose coordinates we wish to obtain.

    Output
    ------
    coords: `ndarray`, shape=(N,3)
        Array of coordinates stored in the input file.
    """
    with open(filepath) as fo:
        natoms = int(fo.readline().rstrip().lstrip().lstrip('#'))
        fo.readline() #skip 2nd line
        lines = fo.readlines()
    coords = np.array([list(map(float,line.lstrip().rstrip().split()[1:])) for line in lines])

    return coords


def write_xyz(coords,symbols,filename,append=False):
    """Writes the coordinates stored in NumPy array to a .xyz file.

    Parameter
    ----------
    coords: `numpy.ndarray`, shape=(N,M) with M > 1, dtype=float
        Cartesian coordinates that we wish to write to a file
    symbols: `numpy.ndarray`, shape=(N,), dtype=str
        List of strings that label the entities whose coordinates are stored in `coords`.
        For atoms, these are usually chemical symbols (e.g. 'C' for carbon).
    filepath: `str`
        Path to the .xyz file whose coordinates we wish to obtain.

    Output
    ------
    coords: `ndarray`, shape=(N,3)
        Array of coordinates stored in the input file.
    """
    symbol_field_size = len(sorted(symbols,key=len)[-1]) #get maximum size of symbols
    if coords.shape[1] == 2:
        new_coords = np.zeros((coords.shape[0],3),dtype=float)
        new_coords[:,:2] = coords
        coords = new_coords
    if append: iostyle = 'a'
    else: iostyle = 'w'
    with open(filename,iostyle) as fo:
        fo.write(' {:d}\n\n'.format(coords.shape[0]))
        for s, r in zip(symbols,coords):
            x,y,z = r
            fo.write('{0:{width}}\t{1:2.8f}\t{2:2.8f}\t{3:2.8f}\n'.format(s,x,y,z,width=symbol_field_size))

def write_xsf(atoms, supercell, symbols=None, force_array=None, filename="carbon.xsf"):
    f=open(filename, "w")

    if symbols == None:
        symbols = ['C']*atoms.shape[0]

    if supercell==None:
        f.write("ATOMS\n")
        for s, atom in zip(symbols,atoms):
            f.write("%s %f %f %f\n" % (s, atom[1][0], atom[1][1], atom[1][2]))

    else:
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        f.write("%f %f %f\n" % (supercell[0], 0.0, 0.0))
        f.write("%f %f %f\n" % (0.0, supercell[1], 0.0))
        f.write("%f %f %f\n" % (0.0, 0.0, 20.0))
        f.write("PRIMCOORD\n")
        f.write("%d 1\n" % (len(atoms)))
        
        if np.all(force_array == None) or np.any(force_array.shape!=atoms.shape):
            if np.any(force_array != None) and force_array.shape != atoms.shape:
                print('ERROR: write_xsf: atoms and forces arrays need to have the same shape.\nWriting only atoms.')
            for s, atom in zip(symbols, atoms):
                f.write("%s %f %f %f\n" % (s, atom[0], atom[1], atom[2]))
        else:
            for atom,force in zip(atoms,force_array):
                f.write('%s %f %f %f %f %f %f\n'%(s, *atom,*force))

    f.close()

def read_xsf(filename,read_forces=True):
    f=open(filename)
    for i in range(2): f.readline()

    supercell = []
    supercell.append( float( f.readline().strip().split()[0] ) )
    supercell.append( float( f.readline().strip().split()[1] ) )

    for i in range(2): f.readline()
    
    na = int( f.readline().strip().split()[0] )
    atoms = np.zeros((na,3),dtype=float)
    forces = np.zeros((na,3),dtype=float)
    forces_in_file = False
    for k in range(na):
        split_line = f.readline().strip().split()
        x,y,z = split_line[1:4]
        atoms[k,:] = np.array([x,y,z])
        if len(split_line) > 4:
            forces_in_file = True
            fx,fy,fz = split_line[4:]
            forces[k] = np.array([fx,fy,fz])
    f.close()
    if forces_in_file and read_forces:
        return atoms, forces, supercell
    else:
        return atoms, supercell

def write_LAMMPS_data(atoms, supercell, filename="carbon.data",minimum_coords=None):
    if np.all(minimum_coords == None):
        minimum_coords = np.zeros(3,dtype=float)
    f=open(filename,"w")
    f.write("carbon\n\n")
    f.write("%d atoms\n\n" % (len(atoms)))
    f.write("1 atom types\n\n")
    f.write("%f %f xlo xhi\n" % (minimum_coords[0], supercell[0]))
    f.write("%f %f ylo yhi\n" % (minimum_coords[1], supercell[1]))
    f.write("%f 20.0 zlo zhi\n\n" % (minimum_coords[2]))

    f.write("Masses\n\n")
    f.write("1 12.0\n\n")

    f.write("Atoms\n\n")
    for i in range(len(atoms)):
        f.write("%d 1 %f %f %f\n" % (i+1,
                                     atoms[i][0],
                                     atoms[i][1],
                                     atoms[i][2] )   #+(np.random.rand()-0.5)*0.5)
        )  # atom_ID atom_type x y z
    f.close()

def read_dump(dump):

    f=open(dump)

    line = f.readline()

    step = int(f.readline().rstrip().split()[0])

    f.readline()

    natoms = int( f.readline().strip().split()[0] )
    pos = np.zeros((natoms,3),dtype=np.float64)
    symbols = [None] * natoms
    #print("number of atoms: %d" % (natoms))
    f.readline()

    xmin, xmax = f.readline().strip().split()
    ymin, ymax = f.readline().strip().split()
    zmin, zmax = f.readline().strip().split()

    f.readline()

    for i in range(natoms):
        ll = f.readline().rstrip().split()
        symbols[i] = ll[0]
        pos[i,:] = list(map(float, ll[1:4]))
    f.close()

    return pos, symbols, step
    
def LAMMPS2XSF(dump):
    from dump2xsf import dump2xsf
    dump2xsf(dump)

def get_lammps_frame(dump, nframe, return_symbols=False):
    from itertools import islice
    '''Fetches the coordinates corresponding to frame number `nframe` of the LAMMPS MD simulation
    contained in the dump file `dump`.
    If `return_symbols`, then the chemical identity of the atoms will also be returned in a
    separate array.'''

    nb_non_coord_lines = 9

    print('ye')

    with open(dump) as fo:
       for n in range(3): fo.readline()
       Natoms = int(fo.readline().lstrip())
       nlines_per_frame = Natoms + nb_non_coord_lines
       fo.seek(0)
       lines_gen = islice(fo,nframe*nlines_per_frame, (nframe+1)*nlines_per_frame)
       relevant_lines = [l.rstrip().split() for l in list(lines_gen)]
    pos = np.array([ list( map(float, l[1:4]) ) for l in relevant_lines[nb_non_coord_lines:]])
    if return_symbols:
        symbols = [l[0] for l in relevant_lines]
        return pos, symbols
    else:
        print('yo')
        return pos
    

def write_subsampled_trajfile(dump, start, end, step):
    """Keep only a subset of frames from a trajectory file and write them to a new trajectory file."""
    nb_non_coord_lines = 9
    prefix = path.basename(dump).split('.')[0]
    dumpdir = path.dirname(dump)
    outfile = path.join(dumpdir, prefix + f'_frames_{start}-{end}-{step}.lammpstrj')
    with open(dump) as fo:
        for n in range(3): fo.readline()
        Natoms = int(fo.readline().lstrip().rstrip())
        nlines_per_frame = Natoms + nb_non_coord_lines
        fo.seek(0)
        n = start
        fp = open(outfile,'w')
        for i in range(n*nlines_per_frame): fo.readline() #get to 1st desired frame
        while n <= end:
            print("n = ", n,flush=True)
            first=True
            second = False
            ct = 0
            for i in range(nlines_per_frame):
                ct += 1
                l = fo.readline()
                fp.write(l)
                if second:
                    print('~~ L2: ', l)
                    second = False
                if first:  
                    print('** L1: ', l)
                    first = False
                    second = True
            n += step
            print("NLINES = ", ct)
            for i in range((step-1) * nlines_per_frame): fo.readline()
        
