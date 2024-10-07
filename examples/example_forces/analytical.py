import torch
from ase.build import molecule
from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

# Define global constants
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

# File with the sk data
path = './auorg.hdf5'

species = [1, 6, 8]

shell_dict = {1: [0], 6: [0,1], 8: [0, 1]}
# Set up geometry
H2O = Geometry(torch.tensor([8, 1, 1]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.8, -0.5],
                             [0.0, -0.8, 0.5]], requires_grad=False)
               )
CO2 = Geometry(torch.Tensor([6, 8, 8]), 
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.16],
                             [0.0, 0.0, -1.16]], requires_grad=False)
               )

geos = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
print(geos._positions)
print(geos.atomic_numbers)


orbital_info = OrbitalInfo(geos.atomic_numbers, shell_dict, shell_resolved=False)

# Set up the feeds
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')
overlap_feed = SkFeed.from_database(path, species, 'overlap')
occupation_feed = SkfOccupationFeed.from_database(path, species)
hubbard_feed = HubbardFeed.from_database(path, species)
repulsive_feed = RepulsiveSplineFeed.from_database(path, species)

# Set up the calculator
dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)

# Run a SCF calculation
dftb_calculator(geos, orbital_info)

forces = dftb_calculator.forces
print(forces)
