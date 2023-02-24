from os import path
from .qcffpi_io import read_MO_file, read_energies



class Molecule:

    '''This class packages QCFFPI data into an object for easier handling and processing.
    This is mainly written for convenience and assumes adhesions to the following conventions:

        * MO files are stored in a directory called \'MO_coefs\' and are named as follows: 
            \'MOs_x.dat\', where \'x\' is a label.
        
        * Energy files are stored in a directory called \'orbital_energies\' and are named as follows:
            \'orb_x.dat\', where x is the **same** label as the one used in the name of the MO file.

    This class offers some flexibility; the user can reset the names of the directories that store the
    MO and energy files. Furthermore the prefixes \'MOs_\' and '\'orb_\' can also be redefined.
    However, the label \'x\' should remain the same for the MO file and the energy file.

    The constructor for this class accepts two string arguments: 

        * `label`: The label \'x\' used to identify the MO/energy files.

        * `project_dir`: The directory containing the \'MO_coefs\' and \'orbital_energies\' directories
        (or their renamed analogs).
    '''

    def __init__(self, label, project_dir):

        self.label = label
        self.projdir = path.expanduser(project_dir)
        self.N = -1

    def fetch_pos_MOs(self, prefix='MOs_', MOdir='MO_coefs'):
        lbl = self.label
        MOfile = path.join(self.projdir,MOdir,f'{prefix}{lbl}.dat')
        self.pos, self.M = read_MO_file(MOfile)
        self.N = self.pos.shape[0]

    def fetch_energies(self, prefix='orb',orbdir='orbital_energies',convert_to_eV=True):
        lbl = self.label
        orbfile = path.join(self.projdir,orbdir,f'{prefix}{lbl}.dat')
        self.energies = read_energies(orbfile,Natoms=self.N)
        if convert_to_eV:
            Ha2eV = 27.2114
            self.energies *= Ha2eV

    def fetch_QCFFPI_data(self, rMO_kwargs, orb_kwargs):
        fetch_pos_MOs(self, **rMO_kwargs)
        fetch_energies(self, **orb_kwargs)
