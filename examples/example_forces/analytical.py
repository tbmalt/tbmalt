import time
from os.path import join
import urllib, tempfile, tarfile
import torch
from tbmalt.io.skf import Skf
from ase.build import molecule
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

# Define global constants
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=20, sci_mode=False, linewidth=200, profile="full")

#Function to obtain skf file
def skf_file(output_path: str):
    """Path to auorg-1-1 HDF5 database.

    This function downloads the auorg-1-1 Slater-Koster parameter set & converts
    it to HDF5 database stored at the path provided.

    Arguments:
         output_path: location to where the auorg-1-1 HDF5 database file should
            be stored.

    Warnings:
        This will fail without an internet connection.

    """
    # Link to the auorg-1-1 parameter set
    #link = 'https://dftb.org/fileadmin/DFTB/public/slako/auorg/auorg-1-1.tar.xz'
    link = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'

    # Elements of interest
    elements = ['H', 'C', 'N', 'O', 'S', 'Au']

    with tempfile.TemporaryDirectory() as tmpdir:

        # Download and extract the auorg parameter set to the temporary directory
        urllib.request.urlretrieve(link, path := join(tmpdir, 'auorg-1-1.tar.xz'))
        with tarfile.open(path) as tar:
            tar.extractall(tmpdir)

        # Select the relevant skf files and place them into an HDF5 database
        skf_files = [join(tmpdir, 'auorg-1-1', f'{i}-{j}.skf')
                     for i in elements for j in elements]

        for skf_file in skf_files:
            Skf.read(skf_file).write(output_path)

print('-------------------------')
print('Analytical')

# File with the sk data
#path = './auorg.hdf5'
skf_file('./test_auorg.hdf5')
path = './test_auorg.hdf5'

species = [1, 6, 8, 16, 79]
#species = [1]
#species = [6, 8]


shell_dict = {1: [0], 6: [0,1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
#shell_dict = {6: [0,1], 8: [0, 1]}
#shell_dict = {1: [0]}
# Set up geometry
#H2O = Geometry(torch.tensor([8, 1, 1]), 
#               torch.tensor([[0.0, 0.0, 0.0],
#                             [0.0, 0.8, -0.5],
#                             [0.0, -0.8, -0.5]], requires_grad=False),
#               units='angstrom'
#               )
#CO2 = Geometry(torch.tensor([6, 8, 8]), 
#               torch.tensor([[0.0, 0.0, 0.0],
#                             [0.0, 0.0, 1.16],
#                             [0.0, 0.0, -1.16]], requires_grad=False),
#               units='angstrom'
#               )
C2H2Au2S3 = Geometry(torch.tensor([1, 6, 16, 79, 16, 79, 16, 6, 1]),
                     torch.tensor([
                         [+0.00, +0.00, +0.00],
                         [-0.03, +0.83, +0.86],
                         [-0.65, +1.30, +1.60],
                         [+0.14, +1.80, +2.15],
                         [-0.55, +0.42, +2.36],
                         [+0.03, +2.41, +3.46],
                         [+1.12, +1.66, +3.23],
                         [+1.10, +0.97, +0.86],
                         [+0.19, +0.93, +4.08]], requires_grad=False),
                     units='angstrom'
                     )

#geos = Geometry.from_ase_atoms([molecule('H2O'), molecule('CO2')])
#print(geos._positions)
#print(geos.atomic_numbers)

H2 = Geometry(torch.tensor([1, 1]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.5]], requires_grad=False),
               units='angstrom'
               )
#geos = H2O + CO2
#geos = Geometry.from_ase_atoms(molecule('CO2'))
#geos = CO2
#print(geos)

geos = C2H2Au2S3
#geos = H2

print("Atomic numbers:", geos.atomic_numbers)
print("Positions:", geos._positions)

orbital_info = OrbitalInfo(geos.atomic_numbers, shell_dict, shell_resolved=False)

# Set up the feeds
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')
overlap_feed = SkFeed.from_database(path, species, 'overlap')
occupation_feed = SkfOccupationFeed.from_database(path, species)
hubbard_feed = HubbardFeed.from_database(path, species)
repulsive_feed = RepulsiveSplineFeed.from_database(path, species)

# Set up the calculator
#dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)
dftb_calculator = Dftb1(hamiltonian_feed, overlap_feed, occupation_feed, r_feed=repulsive_feed)

# Run a SCF calculation
start_time = time.time()
energy = dftb_calculator(geos, orbital_info)
end_time = time.time()
print('Energy:', energy)
print('Time:', end_time - start_time)

# Get total energy
total_energy = dftb_calculator.total_energy
print('Total energy:', total_energy)

#Get repulsive energy
repulsive_energy = dftb_calculator.repulsive_energy
print('Repulsive energy:', repulsive_energy)

#Get forces
start_time = time.time()
forces = dftb_calculator.forces
end_time = time.time()
print('Forces:', forces)
print('Time:', end_time - start_time)
