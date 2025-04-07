import time
from os.path import join
import urllib, tempfile, tarfile
import torch
from tbmalt.io.skf import Skf
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

# Define global constants
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=15, sci_mode=False, linewidth=200, profile="full")

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
print('autograd')

# File with the sk data
#path = './auorg.hdf5'
skf_file('./test_auorg.hdf5')
path = './test_auorg.hdf5'
print('finished skf_file')
#species = [1, 6, 8]
species = [1, 6, 8, 16, 79]
#species = [1]

#shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}
shell_dict = {1: [0], 6: [0,1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
#shell_dict = {1: [0]}

# Set up geometry
H2O = Geometry(torch.tensor([8, 1, 1]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.8, -0.5],
                             [0.0, -0.8, -0.5]], requires_grad=True),
               units='angstrom'
               )
CO2 = Geometry(torch.tensor([6, 8, 8]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.16],
                             [0.0, 0.0, -1.16]], requires_grad=True),
               units='angstrom'
               )
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
                         [+0.19, +0.93, +4.08]], requires_grad=True),
                     units='angstrom'
                     )

#H2O.positions.requires_grad_(True)
#print('Positions:', H2O.positions)

H2 = Geometry(torch.tensor([1, 1]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.5]], requires_grad=True),
               units='angstrom'
               )
#H2O = C2H2Au2S3
#H20 = CO2

#geo = H2
#geo = C2H2Au2S3
#geo = H2O
geo = H2O + H2 + CO2 + C2H2Au2S3
#geo = H2
#geo = C2H2Au2S3

orbital_info = OrbitalInfo(geo.atomic_numbers, shell_dict, shell_resolved=False)

# Set up the feeds
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')
overlap_feed = SkFeed.from_database(path, species, 'overlap')
occupation_feed = SkfOccupationFeed.from_database(path, species)
hubbard_feed = HubbardFeed.from_database(path, species)
repulsive_feed = RepulsiveSplineFeed.from_database(path, species)

# Set up the calculator
dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed, filling_scheme=None)
#dftb_calculator = Dftb1(hamiltonian_feed, overlap_feed, occupation_feed, r_feed=repulsive_feed)

# Run a SCF calculation
#start_time = time.time()

#start_time_direct = time.time()

#start_time_direct_energy = time.time()
#energy_direct = dftb_calculator(geo, orbital_info, grad_mode='direct')
#end_time_direct_energy = time.time()
#print('energy direct time', end_time_direct_energy - start_time_direct_energy)
#start_time_direct_force = time.time()
#forces_direct = - torch.autograd.grad(energy_direct, geo.positions, grad_outputs=torch.ones_like(energy_direct))[0]
#end_time_direct_force = time.time()
#print('force direct time', end_time_direct_force - start_time_direct_force)
#print('energy direct:', energy_direct)
#print('forces direct:', forces_direct)

start_time_imp_energy = time.time()
energy_imp = dftb_calculator(geo, orbital_info, grad_mode='implicit')
end_time_imp_energy = time.time()
print('energy imp time', end_time_imp_energy - start_time_imp_energy)
start_time_imp_force = time.time()
forces_imp = - torch.autograd.grad(energy_imp, geo.positions, grad_outputs=torch.ones_like(energy_imp))[0]
end_time_imp_force = time.time()
print('force imp time', end_time_imp_force - start_time_imp_force)
print('energy imp:', energy_imp)
print('forces imp:', forces_imp)

#start_time_laststep_energy = time.time()
#energy_laststep = dftb_calculator(geo, orbital_info, grad_mode='last_step')
#end_time_laststep_energy = time.time()
#print('energy last step time', end_time_laststep_energy - start_time_laststep_energy)
#start_time_laststep_force = time.time()
#forces_laststep = - torch.autograd.grad(energy_laststep, geo.positions, grad_outputs=torch.ones_like(energy_laststep))[0]
#end_time_laststep_force = time.time()
#print('force last step time', end_time_laststep_force - start_time_laststep_force)
#print('energy last step:', energy_laststep)
#print('forces last step:', forces_laststep)

#
##energy = dftb_calculator(geo, orbital_info, grad_mode='implicit')
#end_time = time.time()
#print('force imp diff:', forces_direct - forces_imp)
#print('force last step diff:', forces_direct - forces_laststep)
#print('Time:', end_time - start_time)
#
## Get total energy
##total_energy = dftb_calculator.total_energy
##print('Total energy:', total_energy)
#
##Get repulsive energy
##repulsive_energy = dftb_calculator.repulsive_energy
##print('Repulsive energy:', repulsive_energy)
#
## Calculate the gradient
##start_time = time.time()
##forces = - torch.autograd.grad(total_energy, geo.positions, grad_outputs=torch.ones_like(total_energy))[0]
##
##end_time = time.time()
##
##print('Forces:', forces)
##print('Time:', end_time - start_time)
