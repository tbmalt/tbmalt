import torch
from ase.build import molecule
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed

torch.set_default_dtype(torch.float64)

path = './auorg.hdf5'

species = [1, 6, 8]
shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

geos = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
orbital_info = OrbitalInfo(geos.atomic_numbers, shell_dict, shell_resolved=False)

# Set up the feeds
hamiltonian_feed = SkFeed.from_database(path, species, 'hamiltonian')
overlap_feed = SkFeed.from_database(path, species, 'overlap')
occupation_feed = SkfOccupationFeed.from_database(path, species)
hubbard_feed = HubbardFeed.from_database(path, species)
repulsive_feed = RepulsiveSplineFeed.from_database(path, species)

#dftb_calculator = Dftb1(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)
print(geos._positions)
x = hamiltonian_feed.matrix(geos, orbital_info)

print('-------------------------')

geos._positions[0, 4, 1] += 100
#geos._positions[1, 4, 1] += 100
print(geos._positions)
y = hamiltonian_feed.matrix(geos, orbital_info)

print(x-y)

