import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

# Define global constants
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

# File with the sk data
path = './auorg.hdf5'

species = [1, 8]

shell_dict = {1: [0], 8: [0, 1]}

# Set up geometry
H2O = Geometry(torch.tensor([8, 1, 1]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.8, -0.5],
                             [0.0, -0.8, -0.5]], requires_grad=True),
               units='angstrom'
               )
#H2O.positions.requires_grad_(True)
print('Positions:', H2O.positions)

orbital_info = OrbitalInfo(H2O.atomic_numbers, shell_dict, shell_resolved=False)

# Set up the feeds
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')
overlap_feed = SkFeed.from_database(path, species, 'overlap')
occupation_feed = SkfOccupationFeed.from_database(path, species)
hubbard_feed = HubbardFeed.from_database(path, species)
repulsive_feed = RepulsiveSplineFeed.from_database(path, species)

# Set up the calculator
dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)

# Run a SCF calculation
energy = dftb_calculator(H2O, orbital_info)
print('Energy:', energy)

# Get total energy
total_energy = dftb_calculator.total_energy
print('Total energy:', total_energy)

#Get repulsive energy
repulsive_energy = dftb_calculator.repulsive_energy
print('Repulsive energy:', repulsive_energy)

# Calculate the gradient
#gradient = torch.autograd.grad(total_energy, H2O.positions, grad_outputs=torch.ones_like(total_energy))[0]
(gradient,) = torch.autograd.grad(total_energy, H2O.positions)
print(H2O.positions)
forces = -gradient

print('Forces:', forces)
