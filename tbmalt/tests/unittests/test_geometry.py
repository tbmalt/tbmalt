import os
import pytest
import h5py
import numpy as np
from scipy.spatial import distance_matrix
import torch
from torch.autograd import gradcheck
from ase.build import molecule
from tbmalt.tests.test_utils import *
from tbmalt.structures.geometry import Geometry
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units


######################
# Geometry Test Data #
######################
@fix_seed
def positions_data(device, batch=False, requires_grad=False):
    rg = requires_grad
    if not batch:
        return torch.rand(5, 3, device=device, requires_grad=rg)
    else:
        return [torch.rand((5, 3), device=device, requires_grad=rg),
                torch.rand((3, 3), device=device, requires_grad=rg),
                torch.rand((2, 3), device=device, requires_grad=rg)]


@fix_seed
def atomic_numbers_data(device, batch=False):
    if not batch:
        return torch.tensor([6, 1, 1, 1, 1], device=device)
    else:
        return [torch.tensor([6, 1, 1, 1, 1], device=device),
                torch.tensor([1, 1, 8], device=device),
                torch.tensor([1, 1], device=device)]


##################
# Geometry Basic #
##################
def geometry_basic_helper(device, positions, atomic_numbers):
    """Function to reduce code duplication when testing basic functionality."""
    # Pack the reference data, if multiple systems provided
    batch = isinstance(atomic_numbers, list)
    if batch:
        atomic_numbers_ref = pack(atomic_numbers)
        positions_ref = pack(positions)
        positions_angstrom = [i / length_units['angstrom'] for i in positions]
    else:
        atomic_numbers_ref = atomic_numbers
        positions_ref = positions
        positions_angstrom = positions / length_units['angstrom']

    # Check 1: Ensure the geometry entity is correct constructed
    geom_1 = Geometry(atomic_numbers, positions)
    check_1 = (torch.allclose(geom_1.atomic_numbers, atomic_numbers_ref)
               and torch.allclose(geom_1.positions, positions_ref))
    assert check_1, 'Geometry was not instantiated correctly'

    # Check 2: Check unit conversion proceeds as anticipated.
    geom_2 = Geometry(atomic_numbers, positions_angstrom, units='angstrom')
    check_2 = torch.allclose(geom_1.positions, geom_2.positions)
    assert check_2, 'Geometry failed to correctly convert length units'

    # Check 3: Check that __repr__ does not crash when called. No assert is
    # needed here as a failure will result in an exception being raised.
    _t = repr(geom_1)

    # Test with a larger number of systems to ensure the string gets truncated.
    # This is only applicable to batched Geometry instances.
    if batch:
        geom_3 = Geometry([atomic_numbers[0] for _ in range(10)],
                          [positions[0] for _ in range(10)])
        _t2 = repr(geom_3)
        check_3 = '...' in _t2
        assert check_3, 'String representation was not correctly truncated'

    # Check 4: Test the device on which the Geometry's tensor are located
    # can be changed via the `.to()` method. Note that this check will only
    # be performed if a cuda device is present.
    if torch.cuda.device_count():
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        geom_1.to(new_device)
        check_4 = (geom_1.atomic_numbers.device == new_device
                   and geom_1.positions.device == new_device)

        assert check_4, '".to" method failed to set the correct device'


def _calculate_vector_position(positions):
    """Calculate vector of positions between atoms."""
    if positions.dim() == 2:
        vector = torch.zeros(positions.shape[0], positions.shape[0], 3)
        for iat in range(positions.shape[0]):
            for jat in range(positions.shape[0]):
                vector[iat, jat] = positions[iat] - positions[jat]

    else:
        vector = torch.zeros(positions.shape[0], positions.shape[1],
                             positions.shape[1], 3)
        for ib in range(positions.shape[0]):
            for iat in range(positions.shape[1]):
                for jat in range(positions.shape[1]):
                    vector[ib, iat, jat] = positions[ib, iat] - positions[ib, jat]
    return vector



def test_geometry_single(device):
    """Test the basic single system functionality of the Geometry class"""
    geometry_basic_helper(device, positions_data(device),
                          atomic_numbers_data(device))


def test_geometry_batch(device):
    """Test the basic batch system functionality of the Geometry class"""
    geometry_basic_helper(device, positions_data(device, True),
                          atomic_numbers_data(device, True))


#####################
# Geometry Atoms IO #
#####################
def test_geometry_single_from_ase_atoms(device):
    """Check single system instances can be instantiated from ase.Atoms objects."""

    # Create an ase.Atoms object
    atoms = molecule('CH4')

    # Check 1: Ensure that the from_ase_atoms method correctly constructs
    # a geometry instance. This includes the unit conversion operation.
    geom_1 = Geometry.from_ase_atoms(atoms, device=device)
    check_1 = np.allclose(geom_1.positions.sft(), atoms.positions * length_units['angstrom'])

    assert check_1, 'from_ase_atoms did not correctly parse the positions'

    # Check 2: Check the tensors were placed on the correct device
    check_2 = (geom_1.positions.device == device
               and geom_1.atomic_numbers.device == device)

    assert check_2, 'from_ase_atoms did not place tensors on the correct device'

    check_3_1 = (geom_1.global_numbers == torch.tensor([1, 6])).all()
    check_3_2 = (geom_1.global_number_pairs == torch.tensor(
        [[1, 1], [6, 1], [1, 6], [6, 6]])).all()

    assert check_3_1, 'global atomic numbers'
    assert check_3_2, 'global atomic number pairs'

    # test to element with Tensor and list input
    check_4_1 = geom_1.to_element(atomic_numbers_data(device)) == \
                 ['C', 'H', 'H', 'H', 'H']
    check_4_2 = geom_1.to_element(geom_1.atomic_numbers) == ['C', 'H', 'H', 'H', 'H']

    assert check_4_1, 'specie name is correct'
    assert check_4_2, 'specie name is correct'


def test_geometry_batch_from_ase_atoms(device):
    """Check batch instances can be instantiated from ase.Atoms objects."""

    # Create an ase.Atoms object
    atoms = [molecule('CH4'), molecule('H2O')]
    ref_pos = pack([torch.tensor(i.positions) for i in atoms]).sft()
    ref_pos = ref_pos * length_units['angstrom']

    # Check 1: Ensure that the from_ase_atoms method correctly constructs
    # a geometry instance. This includes the unit conversion operation.
    geom_1 = Geometry.from_ase_atoms(atoms, device=device)
    check_1 = np.allclose(geom_1.positions.sft(), ref_pos),

    assert check_1, 'from_ase_atoms did not correctly parse the positions'

    # Check 2: Check the tensors were placed on the correct device
    check_2 = (geom_1.positions.device == device
               and geom_1.atomic_numbers.device == device)

    assert check_2, 'from_ase_atoms did not place tensors on the correct device'

    check_3_1 = (geom_1.global_numbers == torch.tensor([1, 6, 8])).all()
    check_3_2 = (geom_1.global_number_pairs == torch.tensor(
        [[1, 1], [6, 1], [8, 1], [1, 6], [6, 6], [8, 6], [1, 8], [6, 8], [8, 8]])).all()

    assert check_3_1, 'global atomic numbers'
    assert check_3_2, 'global atomic number pairs'

    # test to element with Tensor and list input
    check_4_1 = geom_1.to_element(atomic_numbers_data(device, batch=True)) == \
                 [['C', 'H', 'H', 'H', 'H'], ['H', 'H', 'O'], ['H', 'H']]
    check_4_2 = geom_1.to_element(geom_1.atomic_numbers) == \
        [['C', 'H', 'H', 'H', 'H'], ['O','H', 'H']]

    assert check_4_1, 'specie name is correct'
    assert check_4_2, 'specie name is correct'


####################
# Geometry HDF5 IO #
####################
def geometry_hdf5_helper(path, atomic_numbers, positions):
    """Function to reduce code duplication when testing the HDF5 functionality."""
    # Ensure any test hdf5 database is erased before running
    if os.path.exists(path):
        os.remove(path)

    # Pack the reference data, if multiple systems provided
    batch = isinstance(atomic_numbers, list)
    atomic_numbers_ref = pack(atomic_numbers) if batch else atomic_numbers
    positions_ref = pack(positions) if batch else positions

    # Construct a geometry instance
    geom_1 = Geometry(atomic_numbers, positions)

    # Infer target device
    device = geom_1.positions.device

    # Open the database
    with h5py.File(path, 'w') as db:
        # Check 1: Write to the database and check that the written data
        # matches the reference data.
        geom_1.to_hdf5(db)
        check_1 = (np.allclose(db['atomic_numbers'][()], atomic_numbers_ref.sft())
                   and np.allclose(db['positions'][()], positions_ref.sft()))
        assert check_1, 'Geometry not saved the database correctly'

        # Check 2: Ensure geometries are correctly constructed from hdf5 data
        geom_2 = Geometry.from_hdf5(db, device=device)
        check_2 = (torch.allclose(geom_2.positions, geom_1.positions)
                   and torch.allclose(geom_2.atomic_numbers, geom_1.atomic_numbers))
        assert check_2, 'Geometry could not be loaded from hdf5 data'

        # Check 3: Make sure that the tensors were placed on the correct device
        check_3 = (geom_2.positions.device == device
                   and geom_2.atomic_numbers.device == device)
        assert check_3, 'Tensors not placed on the correct device'

    # If this is a batch test then repeat test 2 but pass in a list of HDF5
    # groups rather than one batch HDF5 group.
    if batch:
        os.remove(path)
        with h5py.File(path, 'w') as db:
            for n, (an, pos) in enumerate(zip(atomic_numbers, positions)):
                Geometry(an, pos).to_hdf5(db.create_group(f'geom_{n + 1}'))
            geom_3 = Geometry.from_hdf5([db[f'geom_{i}'] for i in range(1, 4)])
            check_4 = torch.allclose(geom_3.positions.to(device), geom_1.positions)
            assert check_4, 'Instance could not be loaded from hdf5 data (batch)'

    # Remove the test database
    os.remove(path)


def test_dist_posvec_single(device):
    """Test single distance and position vector."""
    atomic_numbers = atomic_numbers_data(device)
    positions = positions_data(device)
    geo = Geometry(atomic_numbers, positions)
    distances = torch.sqrt(((geo.positions.unsqueeze(-3) -
                             geo.positions.unsqueeze(-2)) ** 2).sum(-1))
    assert torch.max(abs(geo.distances - distances)) < 1E-14

    # calculate position vector
    vec = _calculate_vector_position(geo.positions)
    assert torch.max(abs(geo.positions_vec - vec)) < 1E-14


def test_dist_posvec_batch(device):
    """Test batch distance and its mask, position vectors."""
    atomic_numbers = atomic_numbers_data(device, batch=True)
    positions = positions_data(device, batch=True)
    geo = Geometry(atomic_numbers, positions)
    distances = pack([torch.sqrt(((ipos[:inat].repeat(inat, 1) -
                       ipos[:inat].repeat_interleave(inat, 0))
                      ** 2).sum(1)).reshape(inat, inat)
          for ipos, inat in zip(geo.positions, geo.n_atoms)])
    assert torch.max(abs(geo.distances - distances)) < 1E-14

    # calculate position vector
    vec = _calculate_vector_position(geo.positions)
    assert torch.max(abs(geo.positions_vec - vec)) < 1E-14


def test_geometry_single_hdf5(device):
    """Ensure a Geometry instance can be witten to & read from an HDF5 database."""
    # Generate input data and run the tests
    geometry_hdf5_helper('.tbmalt_test_s.hdf5',
                         atomic_numbers_data(device),
                         positions_data(device))


def test_geometry_batch_hdf5(device):
    """Ensure Geometry instances can be witten to & read from an HDF5 database."""
    geometry_hdf5_helper('.tbmalt_test_b.hdf5',
                         atomic_numbers_data(device, True),
                         positions_data(device, True))


#####################
# Geometry.distance #
#####################
def geometry_distance_helper(geom):
    """Function to reduce code duplication when distance."""
    # Infer target device
    device = geom.positions.device
    # Calculate the distance matrix and its reference
    dmat = geom.distances
    if geom.atomic_numbers.dim() == 1:
        dmat_ref = distance_matrix(geom.positions.sft(), geom.positions.sft())
    else:
        dmat_ref = np.stack([distance_matrix(i, i) for i in geom.positions.sft()])
        dmat_ref[~geom.mask_dist] = 0.0

    # Ensure distances are within tolerance thresholds.
    check_1 = np.allclose(dmat.sft(), dmat_ref)
    assert check_1, 'Distances are not within tolerance thresholds'

    # Confirm that results are on the correct device
    check_2 = dmat.device == device
    assert check_2, 'Distances were not returned on the correct device'


def test_geometry_distance_single(device):
    """Geometry single system distance test."""
    # Construct a geometry object
    geom = Geometry(atomic_numbers_data(device),
                    positions_data(device))
    geometry_distance_helper(geom)


def test_geometry_distance_batch(device):
    """Geometry batch system distance test."""
    # Construct a geometry object
    geom = Geometry(atomic_numbers_data(device, True),
                    positions_data(device, True))
    geometry_distance_helper(geom)
