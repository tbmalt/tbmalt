import torch
from ase.build import molecule
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

# Define global constants
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=15, sci_mode=False, linewidth=200, profile="full")

print('-------------------------')
print('Analytical')

# File with the sk data
path = './auorg.hdf5'

species = [1, 6, 8]
#species = [1]

shell_dict = {1: [0], 6: [0,1], 8: [0, 1]}
#shell_dict = {1: [0]}
# Set up geometry
H2O = Geometry(torch.tensor([8, 1, 1]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.8, -0.5],
                             [0.0, -0.8, -0.5]], requires_grad=False),
               units='angstrom'
               )
#CO2 = Geometry(torch.Tensor([6, 8, 8]), 
#               torch.tensor([[0.0, 0.0, 0.0],
#                             [0.0, 0.0, 1.16],
#                             [0.0, 0.0, -1.16]], requires_grad=False)
#               )

#geos = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
#print(geos._positions)
#print(geos.atomic_numbers)

H2 = Geometry(torch.tensor([1, 1]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.5]], requires_grad=False),
               units='angstrom'
               )
geos = H2O

orbital_info = OrbitalInfo(geos.atomic_numbers, shell_dict, shell_resolved=False)

# Set up the feeds
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')
overlap_feed = SkFeed.from_database(path, species, 'overlap')
occupation_feed = SkfOccupationFeed.from_database(path, species)
hubbard_feed = HubbardFeed.from_database(path, species)
repulsive_feed = RepulsiveSplineFeed.from_database(path, species)

# Set up the calculator
#dftb_calculator = Dftb1(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)
dftb_calculator = Dftb1(hamiltonian_feed, overlap_feed, occupation_feed, r_feed=repulsive_feed)

# Run a SCF calculation
energy = dftb_calculator(geos, orbital_info)
print('Energy:', energy)

# Get total energy
total_energy = dftb_calculator.total_energy
print('Total energy:', total_energy)

#Get repulsive energy
repulsive_energy = dftb_calculator.repulsive_energy
print('Repulsive energy:', repulsive_energy)

#Get forces
forces = dftb_calculator.forces
print('Forces:', forces)

