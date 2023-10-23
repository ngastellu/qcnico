#!/usr/bin/env python

import numpy as np
from itertools import islice
from os import path
import subprocess as sbp

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


def read_xyz(filepath,return_symbols=False):
    """Returns the coordinates (and symbols, optionally) of all atoms stored in a .xyz file.

    Parameter
    ----------
    filepath: `str`
        Path to the .xyz file whose coordinates we wish to obtain.

    Output
    ------
    coords: `ndarray`, shape=(N,3), dtype=`float`
        Array of coordinates stored in the input file.
    symbols: `ndarray`, shape=(N,), dtype=`str`
        Symbols corresponding to the atoms whose coordinates are stored in the input file.
    """
    with open(filepath) as fo:
        natoms = int(fo.readline().rstrip().lstrip().lstrip('#'))
        fo.readline() #skip 2nd line
        lines = fo.readlines()

    if return_symbols:
        N = len(lines)
        symbols = [None] * N
        coords = np.zeros((N,3),dtype=float)

        for k, line in enumerate(lines):
            split_line = line.rstrip().lstrip().split()
            symbols[k] = split_line[0]
            coords[k,:] = list(map(float, split_line[1:4]))

        return coords, symbols

    else:
        coords = np.array([list(map(float,line.lstrip().rstrip().split()[1:4])) for line in lines])

        return coords


def write_xyz(coords,symbols,filename,append=False):
    """Writes the coordinates stored in NumPy array to a .xyz file.

    Parameters
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

def get_lammps_frame(dump, nframe, return_symbols=False, step=1, frame0_index=0):
    from itertools import islice
    '''Fetches the coordinates corresponding to frame number `nframe` of the LAMMPS MD simulation
    contained in the dump file `dump`.
    If `return_symbols`, then the chemical identity of the atoms will also be returned in a
    separate array.'''

    nb_non_coord_lines = 9
    nframe = int((nframe-frame0_index)/step)

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

def stream_lammps_traj(dump, start, end, step=1, nb_non_coord_lines=9,stream_cols=slice(1,4), start_file=0, step_file=1, atom_indices=None):
    """Parse a LAMMPS MD trajectory contained in dump file `dump` using islice and a yield statement to save time.
    `stream_cols` argument corresponds to the indices of `line.split()` which are desired.
    Args `start_file` and `step_file` are used to account for the fact that the file being read may not necessarily 
    start at frame 0, or have a unit step between sampled frames; this way arguments `start`, `end`, and `frame` correspond 
    to the  ACTUAL frame indices/step desired WITHOUT needing to account for the way the file is put together."""
    with open(dump) as fo:
        for n in range(3): fo.readline()
        Natoms = int(fo.readline().lstrip())

        nlines_per_frame = Natoms + nb_non_coord_lines
        nframes = (end - start + 1) // step

        if stream_cols.step is None:
            ncoords = stream_cols.stop - stream_cols.start
        else:
            ncoords = (stream_cols.stop - stream_cols.start) // stream_cols.step

        traj_data = np.zeros((Natoms,ncoords))

        # Renormalise `start` and `step` to account for the fact that the file may have its own non-trivial start and step
        start = (start - start_file) // step_file
        step = step // step_file

        fo.seek(0)
        cnt = 0
        lines_gen = islice(fo,start*nlines_per_frame, (start+1)*nlines_per_frame)
        relevant_lines = [l.rstrip().split() for l in list(lines_gen)]
        traj_data = np.array([list( map(float, l[stream_cols]) ) for l in relevant_lines[nb_non_coord_lines:]])
        if atom_indices is None:
            yield traj_data
        else:
            yield traj_data[atom_indices]

        cnt += 1
                
        while cnt <= nframes:
            lines_gen = islice(fo, (step-1)*nlines_per_frame, (step)*nlines_per_frame)
            relevant_lines = [l.rstrip().split() for l in list(lines_gen)]
            traj_data = np.array([list( map(float, l[stream_cols]) ) for l in relevant_lines[nb_non_coord_lines:]])
            if atom_indices is None:
                yield traj_data
            else:
                yield traj_data[atom_indices]
            cnt += 1
       

def get_lammps_frame_bash(dump,nframe,Natoms,step=1,frame0=0):
    nb_non_coord_lines = 9
    nlines_per_frame = nb_non_coord_lines + Natoms
    nframe = int((nframe - frame0) / step) + 1
    with open(dump) as fo:
        p1 = sbp.Popen(f'head -n {nlines_per_frame * nframe}', stdin=fo, stdout=sbp.PIPE, shell=True)
        poslines = sbp.Popen(f'tail -n {Natoms}',stdin=p1.stdout,stdout=sbp.PIPE,shell=True).communicate()[0].decode()
        pos = [list(map(float,l.rstrip().lstrip().split()[1:4])) for l in poslines.split('\n')[:-1]]
        pos = np.array(pos)
    return pos


def write_subsampled_trajfile(dump, start, end, step, outfile=None):
    """Keep only a subset of frames from a trajectory file and write them to a new trajectory file."""
    nb_non_coord_lines = 9
    prefix = path.basename(dump).split('.')[0]
    dumpdir = path.dirname(dump)
    if outfile is None:
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
        
