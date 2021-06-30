import os
import pytest
import h5py
import numpy as np
from scipy.spatial import distance_matrix
import torch
from ase.build import molecule
from tests.test_utils import fix_seed
from tbmalt.structures.geometry import Geometry, unique_atom_pairs
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units
from tbmalt.data import chemical_symbols


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
    from time import time
    # Pack the reference data, if multiple systems provided
    batch = isinstance(atomic_numbers, list) or atomic_numbers.ndim == 2
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

    # Check 4: Verify that the `.chemical_symbols` returns the correct value
    check_4 = all([chemical_symbols[int(j)] == i if isinstance(i, str)
                   else [chemical_symbols[int(k)] for k in j if k != 0] == i
                   for i, j in zip(geom_1.chemical_symbols, atomic_numbers)])
    assert check_4, 'The ".chemical_symbols" property is incorrect'

    # Check 5: ensure that the dtype and device properties are correct.
    check_5a = geom_1.device == device
    check_5b = geom_1.dtype == positions_ref.dtype
    assert check_5a, 'Geometry.device is incorrect'
    assert check_5b, 'Geometry.dtype is incorrect'

    # Check 6: Test the device on which the Geometry's tensor are located
    # can be changed via the `.to()` method. Note that this check will only
    # be performed if a cuda device is present.
    if torch.cuda.device_count():
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        geom_1_copy = geom_1.to(new_device)
        check_6a = (geom_1_copy.atomic_numbers.device == new_device
                    and geom_1_copy.positions.device == new_device)

        # Check that the .device property was correctly set
        check_6b = new_device == geom_1_copy.device
        check_6 = check_6a and check_6b

        assert check_6, '".to" method failed to set the correct device'

    # Check 7: Ensure slicing proceeds as anticipated. This should raise an
    # error for single systems and slice a batch of systems.
    if batch:
        attrs = ['positions', 'atomic_numbers', '_mask_dist', 'n_atoms',
                 '_n_batch', 'device', 'dtype']
        for slc in [slice(None, 2), slice(-2, None), slice(None, None, 2)]:
            # Create sliced and reference geometry objects
            geom_slc = geom_1[slc]
            geom_ref = Geometry(atomic_numbers_ref[slc], positions_ref[slc])


            # Loop over and ensure the attributes are the same
            for attr in attrs:
                check_7 = geom_slc.__getattribute__(attr) == geom_ref.__getattribute__(attr)
                # Some checks will be multidimensional
                if isinstance(check_7, torch.Tensor):
                    check_7 = check_7.all()
                assert check_7, f'Slicing created malformed "{attr}" attribute'

    else:
        with pytest.raises(IndexError, match=r'Geometry slicing is only *'):
            _ = geom_1[0]  # <- should fail if geom_1 is a single system

    # Check 8: Error should be raised if the number of systems in the positions
    # & atomic_numbers arguments disagree with one another
    if batch:
        with pytest.raises(AssertionError, match=r'`atomic_numbers` & `pos*'):
            _ = Geometry(atomic_numbers[slice(None, None, 2)], positions)


def test_geometry_single(device):
    """Test the basic single system functionality of the Geometry class"""
    geometry_basic_helper(device, positions_data(device),
                          atomic_numbers_data(device))


def test_geometry_batch(device):
    """Test the basic batch system functionality of the Geometry class"""
    # Run test with a list of tensors & using pre-packed tensor
    geometry_basic_helper(device, positions_data(device, True),
                          atomic_numbers_data(device, True))
    geometry_basic_helper(device, pack(positions_data(device, True)),
                          pack(atomic_numbers_data(device, True)))


def test_geometry_addition(device):
    """Ensure that geometry objects an be added together."""
    kw = {'device': device}
    an_1 = torch.tensor([1], **kw)
    an_2 = torch.tensor([2, 3], **kw)
    an_3 = torch.tensor([4, 5, 6], **kw)
    an_4 = torch.tensor([7, 8, 9, 10], **kw)

    pos_1 = torch.tensor([[1, 1, 1.]], **kw)
    pos_2 = torch.tensor([[2, 2, 2.], [3, 3, 3]], **kw)
    pos_3 = torch.tensor([[4, 4, 4.], [5, 5, 5], [6, 6, 6]], **kw)
    pos_4 = torch.tensor([[7, 7, 7.], [8, 8, 8], [9, 9, 9], [10, 10, 10]], **kw)

    geom_b1 = Geometry([an_1, an_2], [pos_1, pos_2])
    geom_b2 = Geometry([an_3, an_4], [pos_3, pos_4])

    geom_2 = Geometry(an_2, pos_2)
    geom_3 = Geometry(an_3, pos_3)
    geom_4 = Geometry(an_4, pos_4)

    # Check 1: two single system geometry objects
    check_1 = geom_3 + geom_4 == geom_b2
    assert check_1, 'Single system Geometry addition failed'

    # Check 2: two batched geometry objects
    check_2 = geom_b1 + geom_b2 == Geometry([an_1, an_2, an_3, an_4],
                                            [pos_1, pos_2, pos_3, pos_4])
    assert check_2, 'Batched Geometry addition failed'

    # Check 3: one single and one batched geometry object
    check_3 = geom_2 + geom_b2 == Geometry([an_2, an_3, an_4],
                                           [pos_2, pos_3, pos_4])
    assert check_3, 'Mixed batch/single Geometry addition failed'


#####################
# Geometry Atoms IO #
#####################
def test_geometry_from_ase_atoms_single(device):
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


def test_geometry_from_ase_atoms_batch(device):
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


def test_geometry_hdf5_single(device):
    """Ensure a Geometry instance can be witten to & read from an HDF5 database."""
    # Generate input data and run the tests
    geometry_hdf5_helper('.tbmalt_test_s.hdf5',
                         atomic_numbers_data(device),
                         positions_data(device))


def test_geometry_hdf5_batch(device):
    """Ensure Geometry instances can be witten to & read from an HDF5 database."""
    geometry_hdf5_helper('.tbmalt_test_b.hdf5',
                         atomic_numbers_data(device, True),
                         positions_data(device, True))


#####################
# Geometry.distance #
#####################
def geometry_distance_helper(geom):
    """Function to reduce code duplication when checking .distances."""
    # Infer target device
    device = geom.positions.device
    # Calculate the distance matrix and its reference
    dmat = geom.distances
    if geom.atomic_numbers.dim() == 1:
        dmat_ref = distance_matrix(geom.positions.sft(), geom.positions.sft())
    else:
        pos = [i[:j.count_nonzero()].sft() for i, j in
               zip(geom.positions, geom.atomic_numbers)]
        dmat_ref = pack([torch.tensor(distance_matrix(i, i))
                         for i in pos]).sft()

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


#############################
# Geometry.distance_vectors #
#############################
def geometry_distance_vectors_helper(atomic_numbers, positions):
    """Function to reduce code duplication when checking .distance_vectors."""
    geom = Geometry(atomic_numbers, positions)

    # Check 1: Calculate distance vector tolerance
    if isinstance(positions, torch.Tensor):
        ref_d_vec = positions.unsqueeze(1) - positions
    else:
        ref_d_vec = pack([i.unsqueeze(1) - i for i in positions])
    d_vec = geom.distance_vectors
    check_1 = torch.allclose(d_vec, ref_d_vec)
    assert check_1, 'Distance vectors are outside of tolerance thresholds'

    # Check 2: Device persistence check
    check_2 = d_vec.device == geom.positions.device
    assert check_2, 'Distance vectors were not returned on the correct device'


def test_geometry_distance_vectors_single(device):
    """Geometry single system distance vector test."""
    geometry_distance_vectors_helper(atomic_numbers_data(device),
                                     positions_data(device))


def test_geometry_distance_vectors_batch(device):
    """Geometry single system distance vector test."""
    geometry_distance_vectors_helper(atomic_numbers_data(device, batch=True),
                                     positions_data(device, batch=True))


##############################
# geometry.unique_atom_pairs #
##############################
def test_unique_atom_pairs(device):
    """Tests the 'unique_atom_pairs' helper function."""
    geom = Geometry(atomic_numbers_data(device, True),
                    positions_data(device, True))
    ref = torch.tensor(
        [[1, 1], [6, 1], [8, 1], [1, 6], [6, 6],
         [8, 6], [1, 8], [6, 8], [8, 8]], device=device)
    check = (unique_atom_pairs(geom) == ref).all()
    assert check, "unique_atom_pairs returned an unexpected result"
