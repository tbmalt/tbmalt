"""Periodic-related module unit-tests.

Here the functionality of the cell module and periodic module
is tested."""

import os
from os.path import join
import pytest
import torch
from ase.build import molecule
from ase.lattice import cubic, tetragonal, orthorhombic, triclinic, monoclinic, hexagonal
import h5py
import numpy as np
from tbmalt import Geometry
from tbmalt.common.batch import pack
from tests.test_utils import fix_seed
from tbmalt.data.units import length_units

torch.set_default_dtype(torch.float64)


######################
# Geometry Test Data #
######################
def cells_data(device, batch=False, requires_grad=False):
    rg = requires_grad
    if not batch:
        return torch.tensor([[4, 0, 0.], [0, 4, 0], [0, 0, 4]],
                            device=device, requires_grad=rg)
    else:
        return [torch.tensor([[4, 0, 0.], [0, 4, 0], [0, 0, 4]],
                             device=device, requires_grad=rg),
                torch.tensor([[5, 0, 0.], [0, 5, 0], [0, 0, 5]],
                             device=device, requires_grad=rg),
                torch.tensor([[6, 0, 0.], [0, 6, 0], [0, 0, 6]],
                             device=device, requires_grad=rg)]


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
def periodic_geometry_basic_helper(device, positions, atomic_numbers, cells):
    """Function to reduce code duplication when testing basic functionality."""
    from time import time
    # Pack the reference data, if multiple systems provided
    batch = isinstance(atomic_numbers, list) or atomic_numbers.ndim == 2
    if batch:
        atomic_numbers_ref = pack(atomic_numbers)
        positions_ref = pack(positions)
        positions_angstrom = [i / length_units['angstrom'] for i in positions]
        cells_ref = pack(cells)
        cells_angstrom = [i / length_units['angstrom'] for i in cells]
    else:
        atomic_numbers_ref = atomic_numbers
        positions_ref = positions
        positions_angstrom = positions / length_units['angstrom']
        cells_ref = cells
        cells_angstrom = cells / length_units['angstrom']

    # Check 1: Ensure the geometry entity is correct constructed
    geom_1 = Geometry(atomic_numbers, positions, cells)
    check_1 = (torch.allclose(geom_1.atomic_numbers, atomic_numbers_ref)
               and torch.allclose(geom_1.positions, positions_ref)
               and torch.allclose(geom_1.cells, cells_ref))
    assert check_1, 'Periodic geometry was not instantiated correctly'

    # Check 2: Check unit conversion
    geom_2 = Geometry(atomic_numbers, positions_angstrom,
                      cells_angstrom, units='a')
    check_2 = (torch.allclose(geom_2.cells, geom_1.cells)
               and torch.allclose(geom_2.positions, geom_1.positions))
    assert check_2, 'Cell failed to correctly convert length units'

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

    # Check 4: ensure that the dtype and device properties are correct.
    check_4a = geom_1.device == device
    check_4b = (geom_1.dtype == positions_ref.dtype
                and geom_1.dtype == cells_ref.dtype)
    assert check_4a, 'Geometry.device is incorrect'
    assert check_4b, 'Geometry.dtype is incorrect'

    # Check 5: Test the device on which the Geometry's tensor are located
    # can be changed via the `.to()` method. Note that this check will only
    # be performed if a cuda device is present.
    if torch.cuda.device_count():
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        geom_1_copy = geom_1.to(new_device)
        check_5a = (geom_1_copy.atomic_numbers.device == new_device
                    and geom_1_copy.positions.device == new_device
                    and geom_1_copy.cells.device == new_device)

        # Check that the .device property was correctly set
        check_5b = new_device == geom_1_copy.device
        check_5 = check_5a and check_5b

        assert check_5, '".to" method failed to set the correct device'

    # Check 6: Ensure slicing proceeds as anticipated. This should raise an
    # error for single systems and slice a batch of systems.
    if batch:
        attrs = ['positions', 'atomic_numbers', 'cells', '_mask_dist',
                 'n_atoms', '_n_batch', 'device', 'dtype']
        for slc in [slice(None, 2), slice(-2, None), slice(None, None, 2)]:
            # Create sliced and reference geometry objects
            geom_slc = geom_1[slc]
            geom_ref = Geometry(atomic_numbers_ref[slc], positions_ref[slc],
                                cells_ref[slc])

            # Loop over and ensure the attributes are the same
            for attr in attrs:
                check_6 = geom_slc.__getattribute__(attr) == geom_ref.__getattribute__(attr)
                # Some checks will be multidimensional
                if isinstance(check_6, torch.Tensor):
                    check_6 = check_6.all()
                assert check_6, f'Slicing created malformed "{attr}" attribute'

    else:
        with pytest.raises(IndexError, match=r'Geometry slicing is only *'):
            _ = geom_1[0]  # <- should fail if geom_1 is a single system

    # Check 7: Error should be raised if the number of systems in the positions
    # & atomic_numbers & cells arguments disagree with one another
    if batch:
        with pytest.raises(AssertionError, match=r'`atomic_numbers` & `pos*'):
            _ = Geometry(atomic_numbers, positions, cells[slice(None, None, 2)])


def test_periodic_geometry_single(device):
    """General test of Geometry and Cell modules for single periodic system."""
    periodic_geometry_basic_helper(device, positions_data(device),
                                   atomic_numbers_data(device),
                                   cells_data(device))

    # Check 2: Check fraction position conversion
    frac = True
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    positions_frac = torch.tensor(
        [[0., 0., 0.], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6],
         [0.8, 0.8, 0.8]], device=device)
    cells = torch.tensor([[3., 3., 0], [0., 3., 3.], [3., 0., 3.]],
                         device=device)
    positions_frac_ref = torch.matmul(
            positions_frac, cells * length_units['angstrom'])

    geom_1 = Geometry(numbers, positions_frac, cells, frac, units='a')
    geom_2 = Geometry(numbers, positions_frac,
                      cells * length_units['angstrom'], frac)

    check_2 = (torch.allclose(geom_1.positions, positions_frac_ref)
               and torch.allclose(geom_2.positions, positions_frac_ref))
    assert check_2, 'Cell failed to correctly convert fraction position'


def test_periodic_geometry_batch(device):
    """General test of Geometry and Cell modules for batch periodic systems."""
    periodic_geometry_basic_helper(device, positions_data(device, True),
                                   atomic_numbers_data(device, True),
                                   cells_data(device, True))
    periodic_geometry_basic_helper(device, pack(positions_data(device, True)),
                                   pack(atomic_numbers_data(device, True)),
                                   pack(cells_data(device, True)))

    # Check 2: Check fraction position conversion
    frac = True
    cells = [torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]],
                          device=device),
             torch.tensor([[3., 0., 0.], [0., 4., 0.], [0., 0., 5.]],
                          device=device),
             torch.tensor([[3., 4., 0.], [0., 5., 6.], [4., 0., 5.]],
                          device=device)]
    positions_frac = [torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]],
                                   device=device),
                      torch.tensor([[0., 0., 0.], [0.5, 0.5, 0.5]],
                                   device=device),
                      torch.tensor([[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]],
                                   device=device)]
    numbers = [torch.tensor([1, 1], device=device),
               torch.tensor([1, 1], device=device),
               torch.tensor([1, 1], device=device)]
    positions_frac_ref = pack(positions_frac) * length_units['angstrom']
    positions_frac_ref = torch.matmul(pack(positions_frac), pack(cells)
                                      * length_units['angstrom'])

    geom_1 = Geometry(numbers, positions_frac, cells, frac, units='a')
    geom_2 = Geometry(pack(numbers), pack(positions_frac),
                      pack(cells) * length_units['angstrom'], frac)

    check_2 = (torch.allclose(geom_1.positions, positions_frac_ref)
               and torch.allclose(geom_2.positions, positions_frac_ref))
    assert check_2, 'Cell failed to correctly convert fraction position'


def test_periodic_geometry_addition(device):
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

    cell_1 = torch.tensor([[1, 0, 0.], [0, 1, 0], [0, 0, 1]], **kw)
    cell_2 = torch.tensor([[2, 0, 0.], [0, 2, 0], [0, 0, 2]], **kw)
    cell_3 = torch.tensor([[3, 0, 0.], [0, 3, 0], [0, 0, 3]], **kw)
    cell_4 = torch.tensor([[4, 0, 0.], [0, 4, 0], [0, 0, 4]], **kw)

    geom_b1 = Geometry([an_1, an_2], [pos_1, pos_2], [cell_1, cell_2])
    geom_b2 = Geometry([an_3, an_4], [pos_3, pos_4], [cell_3, cell_4])

    geom_2 = Geometry(an_2, pos_2, cell_2)
    geom_3 = Geometry(an_3, pos_3, cell_3)
    geom_4 = Geometry(an_4, pos_4, cell_4)
    geom_5 = Geometry(an_4, pos_4)

    # Check 1: two single system geometry objects
    check_1 = geom_3 + geom_4 == geom_b2
    assert check_1, 'Single system Geometry addition failed'

    # Check 2: two batched geometry objects
    check_2 = geom_b1 + geom_b2 == Geometry([an_1, an_2, an_3, an_4],
                                            [pos_1, pos_2, pos_3, pos_4],
                                            [cell_1, cell_2, cell_3, cell_4])
    assert check_2, 'Batched Geometry addition failed'

    # Check 3: one single and one batched geometry object
    check_3 = geom_2 + geom_b2 == Geometry([an_2, an_3, an_4],
                                           [pos_2, pos_3, pos_4],
                                           [cell_2, cell_3, cell_4])
    assert check_3, 'Mixed batch/single Geometry addition failed'

    # Check 4: one PBC object and one non-PBC object
    with pytest.raises(TypeError, match=r'Addition can not take*'):
        _ = geom_2 + geom_5


def test_periodic_geometry_property(device):
    """Test property generation of periodic geometry."""
    # Build 1D/2D/3D pbc geometry
    cells_3d = torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]],
                            device=device)
    cells_2d = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 0.]],
                            device=device)
    cells_1d = torch.tensor([[4., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                            device=device)
    cells_npbc = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                              device=device)
    numbers = torch.tensor([1, 1], device=device)
    positions = torch.rand((2, 3), device=device)
    geom_3d = Geometry(numbers, positions, cells_3d, frac=True)
    geom_2d = Geometry(numbers, positions, cells_2d)
    geom_1d = Geometry(numbers, positions, cells_1d, frac=True)
    geom_npbc = Geometry(numbers, positions, cells_npbc)

    # Check 1: check pbc information
    check_1a = (geom_3d.pbc == torch.tensor([True, True, True], device=device)).all()
    check_1b = (geom_2d.pbc == torch.tensor([True, True, False], device=device)).all()
    check_1c = (geom_1d.pbc == torch.tensor([True, False, False], device=device)).all()
    check_1d = (geom_npbc.pbc == torch.tensor([False, False, False], device=device)).all()
    check_1 = check_1a and check_1b and check_1c and check_1d
    assert check_1, 'Cell failed to return pbc information'

    # Check 2: Non-pbc system should not be fractional
    with pytest.raises(ValueError, match=r'Cluster should not be*'):
        _ = Geometry(numbers, positions, cells_npbc, frac=True)


#####################
# Geometry Atoms IO #
#####################
def test_periodic_geometry_from_ase_atoms_single(device):
    """Check single system instances can be instantiated from ase.Atoms objects."""

    # Create an ase.Atoms object
    atoms = molecule('CH4', pbc=True, cell=[3.0, 3.0, 3.0])

    # Check 1: Ensure that the from_ase_atoms method correctly constructs
    # a geometry instance. This includes the unit conversion operation.
    geom_1 = Geometry.from_ase_atoms(atoms, device=device)
    check_1 = (np.allclose(geom_1.positions.cpu().numpy(),
                           atoms.positions * length_units['angstrom'])
               and np.allclose(geom_1.cells.cpu().numpy(),
                               atoms.cell[:] * length_units['angstrom']))

    assert check_1, 'from_ase_atoms did not correctly parse the positions'

    # Check 2: Check the tensors were placed on the correct device
    check_2 = (geom_1.positions.device == device
               and geom_1.atomic_numbers.device == device
               and geom_1.cells.device == device)

    assert check_2, 'from_ase_atoms did not place tensors on the correct device'


def test_periodic_geometry_from_ase_atoms_batch(device):
    """Check batch instances can be instantiated from ase.Atoms objects."""

    # Create an ase.Atoms object
    atoms = [molecule('CH4', pbc=True, cell=[3.0, 3.0, 3.0]),
             molecule('H2O', pbc=True, cell=[3.0, 4.0, 5.0])]
    ref_pos = pack([torch.tensor(i.positions) for i in atoms])
    ref_pos = ref_pos * length_units['angstrom']
    ref_cell = pack([torch.tensor(i.cell[:]) for i in atoms])
    ref_cell = ref_cell * length_units['angstrom']

    # Check 1: Ensure that the from_ase_atoms method correctly constructs
    # a geometry instance. This includes the unit conversion operation.
    geom_1 = Geometry.from_ase_atoms(atoms, device=device)
    check_1 = (np.allclose(geom_1.positions.cpu().numpy(), ref_pos)
               and np.allclose(geom_1.cells.cpu().numpy(), ref_cell))

    assert check_1, 'from_ase_atoms did not correctly parse the positions'

    # Check 2: Check the tensors were placed on the correct device
    check_2 = (geom_1.positions.device == device
               and geom_1.atomic_numbers.device == device
               and geom_1.cells.device == device)

    assert check_2, 'from_ase_atoms did not place tensors on the correct device'

    # Check 3: Ensure that the mixing batch of PBC and non-PBC can be detected
    atoms_2 = [molecule('CH4'),
               molecule('H2O', pbc=True, cell=[3.0, 4.0, 5.0])]
    with pytest.raises(NotImplementedError, match=r'Mixing of PBC *'):
        _ = Geometry.from_ase_atoms(atoms_2, device=device)


def test_cell_from_ase_lattice(device):
    """Test importing lattice information from ase.Lattice objects."""

    # Creat an ase.Lattice object
    cell_1 = cubic.SimpleCubic('H', latticeconstant=5.0)
    cell_2 = tetragonal.SimpleTetragonal(
        'H', latticeconstant={'a': 4.0, 'c': 6.0})
    cell_3 = orthorhombic.SimpleOrthorhombic(
        'H', latticeconstant={'a': 4.0, 'b': 5.0, 'c': 6.0})
    cell_4 = triclinic.Triclinic(
        'H', latticeconstant={'a': 4.0, 'b': 5.0, 'c': 6.0,
                              'alpha': 30, 'beta': 40, 'gamma': 50})
    cell_5 = monoclinic.SimpleMonoclinic(
        'H', latticeconstant={'a': 4.0, 'b': 5.0, 'c': 6.0, 'alpha': 30})
    cell_6 = hexagonal.Hexagonal(
        'H', latticeconstant={'a': 4.0, 'c': 6.0})

    # Refernce cell information
    a = length_units['angstrom']
    ref_1 = torch.tensor([cell_1.cell[:]], device=device) * a
    ref_2 = torch.tensor([cell_2.cell[:]], device=device) * a
    ref_3 = torch.tensor([cell_3.cell[:]], device=device) * a
    ref_4 = torch.tensor([cell_4.cell[:]], device=device) * a
    ref_5 = torch.tensor([cell_5.cell[:]], device=device) * a
    ref_6 = torch.tensor([cell_6.cell[:]], device=device) * a

    # Build a geometry object
    numbers = torch.tensor([1, 1], device=device)
    positions = torch.tensor([[0., 0., 0.], [2., 2., 2.]], device=device)
    geom_1 = Geometry(numbers, positions, cell_1, units='a')
    geom_2 = Geometry(numbers, positions, cell_2, units='a')
    geom_3 = Geometry(numbers, positions, cell_3, units='a')
    geom_4 = Geometry(numbers, positions, cell_4, units='a')
    geom_5 = Geometry(numbers, positions, cell_5, units='a')
    geom_6 = Geometry(numbers, positions, cell_6, units='a')

    # Check1: Ensure that cells of pbc systems can be created via customized
    # objects from ase.Lattice.
    check_1a = torch.allclose(geom_1.cells, ref_1)
    check_1b = torch.allclose(geom_2.cells, ref_2)
    check_1c = torch.allclose(geom_3.cells, ref_3)
    check_1d = torch.allclose(geom_4.cells, ref_4)
    check_1e = torch.allclose(geom_5.cells, ref_5)
    check_1f = torch.allclose(geom_6.cells, ref_6)

    check_1 = (check_1a and check_1b and check_1c and check_1d
               and check_1e and check_1f)
    assert check_1, 'Cells are not created correctly from ase.Lattice'

    # check2: Ensure that the dtype and device properties are correct.
    check_2a = (geom_1.cells.device == device and
                geom_2.cells.device == device and
                geom_3.cells.device == device and
                geom_4.cells.device == device and
                geom_5.cells.device == device and
                geom_6.cells.device == device)
    check_2b = (geom_1.cells.dtype == ref_1.dtype and
                geom_2.cells.dtype == ref_2.dtype and
                geom_3.cells.dtype == ref_3.dtype and
                geom_4.cells.dtype == ref_4.dtype and
                geom_5.cells.dtype == ref_5.dtype and
                geom_6.cells.dtype == ref_6.dtype)

    assert check_2a, 'Geometry.device is incorrect'
    assert check_2b, 'Geometry.dtype is incorrect'


####################
# Geometry HDF5 IO #
####################
def periodic_geometry_hdf5_helper(path, atomic_numbers, positions, cells):
    """Function to reduce code duplication when testing the HDF5 functionality."""

    # Pack the reference data, if multiple systems provided
    batch = isinstance(atomic_numbers, list)
    atomic_numbers_ref = pack(atomic_numbers) if batch else atomic_numbers
    positions_ref = pack(positions) if batch else positions
    cells_ref = pack(cells) if batch else cells

    # Construct a geometry instance
    geom_1 = Geometry(atomic_numbers, positions, cells)

    # Infer target device
    device = geom_1.positions.device

    # Open the database
    with h5py.File(path, 'w') as db:
        # Check 1: Write to the database and check that the written data
        # matches the reference data.
        geom_1.to_hdf5(db)
        check_1 = (np.allclose(db['atomic_numbers'][()], atomic_numbers_ref.cpu())
                   and np.allclose(db['positions'][()], positions_ref.cpu())
                   and np.allclose(db['cells'][()], cells_ref.cpu()))
        assert check_1, 'Geometry not saved the database correctly'

        # Check 2: Ensure geometries are correctly constructed from hdf5 data
        geom_2 = Geometry.from_hdf5(db, device=device)
        check_2 = (torch.allclose(geom_2.positions, geom_1.positions)
                   and torch.allclose(geom_2.atomic_numbers, geom_1.atomic_numbers)
                   and torch.allclose(geom_2.cells, geom_1.cells))
        assert check_2, 'Geometry could not be loaded from hdf5 data'

        # Check 3: Make sure that the tensors were placed on the correct device
        check_3 = (geom_2.positions.device == device
                   and geom_2.atomic_numbers.device == device
                   and geom_2.cells.device == device)
        assert check_3, 'Tensors not placed on the correct device'

    # If this is a batch test then repeat test 2 but pass in a list of HDF5
    # groups rather than one batch HDF5 group.
    if batch:
        os.remove(path)
        with h5py.File(path, 'w') as db:
            for n, (an, pos, cell) in enumerate(zip(atomic_numbers, positions, cells)):
                Geometry(an, pos, cell).to_hdf5(db.create_group(f'geom_{n + 1}'))
            geom_3 = Geometry.from_hdf5([db[f'geom_{i}'] for i in range(1, 4)])
            check_4 = (torch.allclose(geom_3.positions.to(device), geom_1.positions)
                       and torch.allclose(geom_3.cells.to(device), geom_1.cells))
            assert check_4, 'Instance could not be loaded from hdf5 data (batch)'


def test_periodic_geometry_hdf5_single(device, tmpdir):
    """Ensure a Geometry instance can be witten to & read from an HDF5 database."""
    # Generate input data and run the tests
    periodic_geometry_hdf5_helper(join(tmpdir, 'tbmalt_test_s.hdf5'),
                                  atomic_numbers_data(device),
                                  positions_data(device),
                                  cells_data(device))


def test_periodic_geometry_hdf5_batch(device, tmpdir):
    """Ensure Geometry instances can be witten to & read from an HDF5 database."""
    periodic_geometry_hdf5_helper(join(tmpdir, 'tbmalt_test_b.hdf5'),
                                  atomic_numbers_data(device, True),
                                  positions_data(device, True),
                                  cells_data(device, True))


####################
# Cell translation #
####################
def _get_cell_trans(latVec, cutoff, negExt=1, posExt=1, unit='bohr'):
    """Reproduce code originally from DFTB+ to test TBMaLT.
    This code is for single geometry and not vectorized, to act
    as a reference for cell translation code in TBMaLT."""
    device = latVec.device

    if unit == 'angstrom':
        latVec = latVec * length_units[unit]
    recVec = torch.inverse(latVec)

    # get ranges of periodic boundary condition from negative to positive
    ranges = torch.zeros((2, 3), dtype=torch.int, device=device)
    for ii in range(3):
        iTmp = torch.floor(cutoff * torch.sqrt(sum(recVec[:, ii] ** 2)))
        ranges[0, ii] = -negExt - iTmp
        ranges[1, ii] = posExt + iTmp

    # Length of the first, second and third column in ranges
    leng1, leng2, leng3 = ranges[1, :] - ranges[0, :] + 1
    ncell = leng1 * leng2 * leng3  # -> Number of lattice cells

    # Cell translation vectors in relative coordinates
    cellvec = torch.zeros(ncell, 3, device=device)
    col3 = torch.linspace(ranges[0, 2], ranges[1, 2], leng3, device=device)
    col2 = torch.linspace(ranges[0, 1], ranges[1, 1], leng2, device=device)
    col1 = torch.linspace(ranges[0, 0], ranges[1, 0], leng1, device=device)
    cellvec[:, 2] = col3.repeat(int(ncell / leng3))
    col2 = col2.repeat(leng3, 1)
    col2 = torch.cat([(col2[:, ii]) for ii in range(leng2)])
    cellvec[:, 1] = col2.repeat(int(ncell / (leng2 * leng3)))
    col1 = col1.repeat(leng3 * leng2, 1)
    cellvec[:, 0] = torch.cat([(col1[:, ii]) for ii in range(leng1)])

    # Cell translation vectors in absolute units
    rcellvec = torch.stack([torch.matmul(
        torch.transpose(latVec, 0, 1), cellvec[ii]) for ii in range(ncell)])

    return cellvec, rcellvec


def test_cell_translation_single(device):
    """General test of Periodic module (single)."""
    # Generate data
    numbers = atomic_numbers_data(device)
    positions = positions_data(device)
    cutoff = torch.tensor([9.98], device=device)

    # Cubic
    latvec1 = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
                           device=device)
    geom1 = Geometry(numbers, positions, latvec1, units='a')
    cellvec_ref1, rcellvec_ref1 = _get_cell_trans(geom1.cells, cutoff + 1)

    # Trigonal
    latvec2 = torch.tensor([[3., 3., 0.], [0., 3., 3.], [3., 0., 3.]],
                           device=device)
    geom2 = Geometry(numbers, positions, latvec2, units='a')
    cellvec_ref2, rcellvec_ref2 = _get_cell_trans(geom2.cells, cutoff + 1)

    # Triclinc
    latvec3 = torch.tensor([[-2., 2., 2.], [3., -3., 3.], [4., 4., -4.]],
                           device=device)
    geom3 = Geometry(numbers, positions, latvec3, units='a')
    cellvec_ref3, rcellvec_ref3 = _get_cell_trans(geom3.cells, cutoff + 1)

    # Check: Check cell translation
    periodic1 = geom1.periodic
    periodic2 = geom2.periodic
    periodic3 = geom3.periodic
    check1 = (torch.allclose(periodic1.cellvec, cellvec_ref1)
              and torch.allclose(periodic1.rcellvec, rcellvec_ref1))
    check2 = (torch.allclose(periodic2.cellvec, cellvec_ref2)
              and torch.allclose(periodic2.rcellvec, rcellvec_ref2))
    check3 = (torch.allclose(periodic3.cellvec, cellvec_ref3)
              and torch.allclose(periodic3.rcellvec, rcellvec_ref3))
    check = check1 and check2 and check3

    assert check, 'Periodic failed to correctly implement cell translation (single)'


def test_cell_translation_batch(device):
    """General test of Periodic module (batch)."""
    # Generate data
    numbers = atomic_numbers_data(device, batch=True)
    positions = positions_data(device, batch=True)
    latvec = [torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]], device=device),
              torch.tensor([[-2., 2., 2.], [3., -3., 3.], [4., 4., -4.]], device=device),
              torch.tensor([[6., 0., 0.], [0., 5., 0.], [0., 0., 4.]], device=device)]
    geom = Geometry(numbers, positions, latvec, units='a')
    cutoff = torch.tensor([9.98], device=device)

    # Reference data
    cellvec_ref = pack([_get_cell_trans(icell, cutoff + 1)[0]
                        for icell in geom.cells], value=1e3)

    # Check: Check cell translation
    periodic = geom.periodic
    check = torch.allclose(periodic.cellvec, cellvec_ref)

    assert check, 'Periodic failed to correctly implement cell translation (batch)'
