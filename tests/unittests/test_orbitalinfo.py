"""OrbitalInfo module unit-tests.

Here the functionality of the orbs module is tested. However, it should be
noted that no gradient stability checks are performed as such tests are not
applicable to the `OrbitalInfo` class.
"""
import os
import pytest
import numpy as np
import h5py
import torch
from torch import allclose as close
from tbmalt.structures.orbitalinfo import OrbitalInfo
from tbmalt.common.batch import pack

# Developers Notes
# This test module could do with some cleaning up:
#   - Better doc-strings
#   - More helper functions to abstract repeated code
#   - Reference data should either be moved to a set of external files
#     or abstracted to a data function.

#########################
# General Functionality #
#########################
def test_orbs_operations(device):
    """Ensure general operators work as intended.

    These are operations like slicing, addition and equality. However, some
    other general methods are also tested here.
    """
    bd = {1: [0], 6: [0, 1]}


    H2 = OrbitalInfo(torch.tensor([1, 1], device=device), bd)
    H2_shell_resolved = OrbitalInfo(torch.tensor([1, 1], device=device), bd, True)
    batched_H2 = OrbitalInfo([torch.tensor([1, 1], device=device)], bd)

    CH4 = OrbitalInfo(torch.tensor([6, 1, 1, 1, 1], device=device), bd)

    H2_CH4 = OrbitalInfo([torch.tensor([1, 1], device=device),
                          torch.tensor([6, 1, 1, 1, 1], device=device)], bd)

    # Check that error is raise when combining systems with difering shell resolutions

    # Equality
    check_1a = H2 == H2 and CH4 == CH4  # Should evaluate to True
    check_1b = H2 != CH4 and CH4 != H2_CH4 # Should evaluate to False
    check_1c = H2 != H2_shell_resolved  # Special False case 1
    check_1d = H2 != batched_H2  # Special False case 2
    check_1 = all([check_1a, check_1b, check_1c, check_1d])

    assert check_1, 'OrbitalInfo equality evaluates incorrect'


    # Addition
    check_2 = (H2 + CH4) == H2_CH4 == (batched_H2 + CH4)
    assert check_2, 'OrbitalInfo addition did not perform as expected'
    with pytest.raises(AttributeError):
        H2 + H2_shell_resolved

    # Slicing
    # Padding should be automatically removed when sliced
    check_3 = H2_CH4[0] == H2 and H2_CH4[1] == CH4 and H2_CH4[0:1] == batched_H2
    assert check_3, 'Batch slicing operations did not perform as expected'

    # n_orbs_on_species
    # Single value acquisition
    check_4a = H2.n_orbs_on_species(1) == 1 and CH4.n_orbs_on_species(6) == 4
    # Multi-value acquisition
    check_4b = torch.all(H2_CH4.n_orbs_on_species(torch.tensor([1, 6]))
                == torch.tensor([1, 4], device=device))
    check_4 = check_4a and check_4b
    assert check_4, "Batch.n_orbs_on_species failed to yield the correct results"

    # n_shells_on_species
    # Single value acquisition
    check_5a = H2.n_shells_on_species(1) == 1 and CH4.n_shells_on_species(6) == 2
    # Multi-value acquisition
    check_5b = torch.all(H2_CH4.n_shells_on_species(torch.tensor([1, 6]))
                == torch.tensor([1, 2], device=device))
    check_5 = check_5a and check_5b
    assert check_5, "Batch.n_orbs_on_species failed to yield the correct results"

def test_orbs_basic_single(device):
    """General operational test of basic OrbitalInfo functionality (single)."""
    # Generate test orbs entity
    atomic_numbers = torch.tensor([1, 1, 6, 6, 79, 79], device=device)
    shell_dict = {1: [0], 6: [0, 1], 79: [0, 1, 2]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    # Ensure exceptions are raised if shell_dict keys are not lists of ints
    with pytest.raises(ValueError):
        _ = OrbitalInfo(atomic_numbers, {1: [torch.tensor(0)], 6: [0, 1]})

    with pytest.raises(ValueError):
        _ = OrbitalInfo(atomic_numbers, {1: torch.tensor([0]), 6: [0, 1]})

    # Reference data
    orbs_per_atom = torch.tensor([1, 1, 4, 4, 9, 9], device=device)
    orbs_per_shell = torch.tensor([1, 1, 1, 3, 1, 3, 1, 3, 5, 1, 3, 5], device=device)
    on_atoms = torch.tensor([0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5], device=device)
    on_shells = torch.tensor([0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7,
                              8, 8, 8, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 11], device=device)
    shell_ls = torch.tensor([0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2], device=device)

    # Counter properties
    check_1a = orbs.n_atoms == 6
    check_1b = orbs.n_shells == 12
    check_1c = orbs.n_orbitals == 28

    assert check_1a, 'n_atoms is incorrect'
    assert check_1b, 'n_subshells is incorrect'
    assert check_1c, 'n_orbitals is incorrect'

    # Shape properties
    check_2a = orbs.atomic_matrix_shape == torch.Size([6, 6])
    check_2b = orbs.shell_matrix_shape == torch.Size([12, 12])
    check_2c = orbs.orbital_matrix_shape == torch.Size([28, 28])

    assert check_2a, 'atomic_matrix_shape is incorrect'
    assert check_2b, 'subshell_matrix_shape is incorrect'
    assert check_2c, 'orbital_matrix_shape is incorrect'

    check_2da = orbs.matrix_shape('atomic') == torch.Size([6, 6])
    check_2db = orbs.matrix_shape('shell') == torch.Size([12, 12])
    check_2dc = orbs.matrix_shape('full') == torch.Size([28, 28])
    check_2d = check_2da and check_2db and check_2dc

    assert check_2d, 'matrix_shape function returned an unexpected shape'

    # Orbital counter properties (location specific)
    check_3a = close(orbs.orbs_per_atom, orbs_per_atom)
    check_3b = close(orbs.orbs_per_shell, orbs_per_shell)

    assert check_3a, 'orbs_per_atom is incorrect'
    assert check_3b, 'orbs_per_shell is incorrect'

    # Orbital location properties
    check_4a = close(orbs.on_atoms, on_atoms)
    check_4b = close(orbs.on_shells, on_shells)

    assert check_4a, 'on_atoms is incorrect'
    assert check_4b, 'on_atoms is incorrect'

    # Resolution convenience functions
    orbs.shell_resolved = False
    check_5a = close(orbs.orbs_per_res, orbs_per_atom)
    check_9a = close(orbs.on_res, on_atoms)
    orbs.shell_resolved = True
    check_5b = close(orbs.orbs_per_res, orbs_per_shell)
    check_9b = close(orbs.on_res, on_shells)
    check_5 = check_5a and check_5b
    check_6 = check_9a and check_9b

    assert check_5, 'orbs_per_res is incorrect'
    assert check_6, 'on_res is incorrect'

    # Check shell_ls
    check_7 = close(orbs.shell_ls, shell_ls)

    assert check_7, 'shell_ls is incorrect'

    # Check results are all on the correct device.
    attrs = ['n_atoms', 'n_shells', 'n_orbitals', 'orbs_per_atom',
             'orbs_per_shell', 'on_atoms', 'on_shells', 'shell_ls']

    for attr in attrs:
        check_8 = orbs.__getattribute__(attr).device == device
        assert check_8, f'Attribute {attr} returned on the wrong device.'

    # Ensure "to" method functions as expected
    if torch.cuda.device_count():  # Can only test if there is a gpu present
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        orbs_copy = orbs.to(new_device)
        check_9a = orbs_copy.atomic_numbers.device == new_device

        # Check that the .device property was correctly set
        check_9b = new_device == orbs_copy.device
        check_6 = check_9a and check_9b

        assert check_6, '".to" method failed to set the correct device'


def test_orbs_basic_batch(device):
    """General operational test of basic OrbitalInfo functionality (batch)."""
    # Generate test orbs entity
    dev = {'device': device}
    atomic_numbers = [torch.tensor([1, 1, 6, 6, 79, 79], **dev),
                      torch.tensor([1, 6], **dev)]
    shell_dict = {1: [0], 6: [0, 1], 79: [0, 1, 2]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    # Reference data
    orbs_per_atom = torch.tensor([[1, 1, 4, 4, 9, 9],
                                  [1, 4, 0, 0, 0, 0]], **dev)
    orbs_per_shell = torch.tensor([[1, 1, 1, 3, 1, 3, 1, 3, 5, 1, 3, 5],
                                   [1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]], **dev)
    on_atoms = torch.tensor([
        [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [0, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], **dev)
    on_shells = torch.tensor([
        [0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7,
         8, 8, 8, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 11],
        [0, 1, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], **dev)
    shell_ls = torch.tensor([[0, 0, 0,  1, 0, 1, 0, 1, 2, 0, 1, 2],
                             [0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], **dev)

    # Counter properties
    check_1a = close(orbs.n_atoms, torch.tensor([6, 2], **dev))
    check_1b = close(orbs.n_shells, torch.tensor([12, 3], **dev))
    check_1c = close(orbs.n_orbitals, torch.tensor([28, 5], **dev))

    assert check_1a, 'n_atoms is incorrect'
    assert check_1b, 'n_subshells is incorrect'
    assert check_1c, 'n_orbitals is incorrect'

    # Shape properties
    check_2a = orbs.atomic_matrix_shape == torch.Size([2, 6, 6])
    check_2b = orbs.shell_matrix_shape == torch.Size([2, 12, 12])
    check_2c = orbs.orbital_matrix_shape == torch.Size([2, 28, 28])

    assert check_2a, 'atomic_matrix_shape is incorrect'
    assert check_2b, 'subshell_matrix_shape is incorrect'
    assert check_2c, 'orbital_matrix_shape is incorrect'

    # Orbital counter properties (location specific)
    check_3a = close(orbs.orbs_per_atom, orbs_per_atom)
    check_3b = close(orbs.orbs_per_shell, orbs_per_shell)

    assert check_3a, 'orbs_per_atom is incorrect'
    assert check_3b, 'orbs_per_shell is incorrect'

    # Orbital location properties
    check_4a = close(orbs.on_atoms, on_atoms)
    check_4b = close(orbs.on_shells, on_shells)

    assert check_4a, 'on_atoms is incorrect'
    assert check_4b, 'on_atoms is incorrect'

    # Resolution convenience functions
    orbs.shell_resolved = False
    check_5a = close(orbs.orbs_per_res, orbs_per_atom)
    check_6a = close(orbs.on_res, on_atoms)
    orbs.shell_resolved = True
    check_5b = close(orbs.orbs_per_res, orbs_per_shell)
    check_6b = close(orbs.on_res, on_shells)
    check_5 = check_5a and check_5b
    check_6 = check_6a and check_6b

    assert check_5, 'orbs_per_res is incorrect'
    assert check_6, 'on_res is incorrect'

    # Check shell_ls
    check_7 = close(orbs.shell_ls, shell_ls)

    assert check_7, 'shell_ls is incorrect'

    # Make sure pre-packed tensors can be passed in
    orbs_2 = OrbitalInfo(pack(atomic_numbers), shell_dict)
    # If orbs_2's "shell_ls" attribute is the same as orbs, then it will
    # perform identically.
    check_8 = torch.allclose(orbs_2.shell_ls, orbs.shell_ls)

    assert check_8, 'OrbitalInfo instantiate incorrectly with pre-packed arguments'


#################
# OrbitalInfo HDF5 IO #
##################
def orbs_hdf5_helper(orbs):
    """Function to reduce code duplication when testing the HDF5 functionality."""
    path = 'orbs_test_db.hdf5'
    device = orbs.device

    # Ensure any test hdf5 database is erased before running
    if os.path.exists(path):
        os.remove(path)

    # Open the database
    with h5py.File(path, 'w') as db:
        # Check 1: Write to the database and check that the written data
        # matches the reference data.
        orbs.to_hdf5(db)
        check_1a = np.allclose(db['atomic_numbers'][()], orbs.atomic_numbers.cpu().numpy())
        check_1b = [np.allclose(np.array(orbs.shell_dict[int(k)]), v[()])
                    for k, v in db['shell_dict'].items()]
        check_1c = db['shell_resolved'][()] == orbs.shell_resolved
        check_1 = check_1a and check_1b and check_1c

        assert check_1, 'OrbitalInfo not saved the database correctly'

        # Check 2: Ensure geometries are correctly constructed from hdf5 data
        orbs_2 = OrbitalInfo.from_hdf5(db, device=device)
        check_2 = (torch.allclose(orbs_2.atomic_numbers, orbs.atomic_numbers)
                   and torch.allclose(orbs_2.shell_ls, orbs.shell_ls)
                   and orbs.shell_resolved == orbs_2.shell_resolved)

        assert check_2, 'OrbitalInfo could not be loaded from hdf5 data'

        # Check 3: Make sure that the tensors were placed on the correct device
        check_3 = orbs_2.atomic_numbers.device == device
        assert check_3, 'OrbitalInfo not placed on the correct device'

    # Remove the test database
    os.remove(path)


def test_orbs_hdf5_single(device):
    atomic_numbers = torch.tensor([1, 1, 6, 6, 79, 79], device=device)
    shell_dict = {1: [0], 6: [0, 1], 79: [0, 1, 2]}
    orbs_hdf5_helper(OrbitalInfo(atomic_numbers, shell_dict, shell_resolved=True))


def test_orbs_hdf5_batch(device):
    atomic_numbers = torch.tensor([[1, 1, 6, 6, 79, 79],
                                   [1, 6, 0, 0, 0, 0]], device=device)
    shell_dict = {1: [0], 6: [0, 1], 79: [0, 1, 2]}
    orbs_hdf5_helper(OrbitalInfo(atomic_numbers, shell_dict, shell_resolved=True))


########################
# OrbitalInfo Matrix Methods #
########################
def test_orbs_azimuthal_matrix_single(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([1, 1, 1, 1, 6], device=device)
    shell_dict = {1: [0], 6: [0, 1]}

    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = torch.stack([
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ]),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1]
            ])], -1).to(device)

    block_ref = torch.stack([
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1]
            ]),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1]
            ])], -1).to(device)

    # Check un-masked and un-sorted full and block azimuthal matrices
    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.azimuthal_matrix(option, False, False, False, False)
        check_1 = close(mat, ref)
        assert check_1, f'{option} azimuthal matrix is incorrect, no sort/mask'

    # Sorting
    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.azimuthal_matrix(option, True, False, False, False)
        check_2 = close(mat, ref.sort(-1)[0])
        assert check_2, f'{option} sorted azimuthal matrix is incorrect'

    # Diagonal masking (only applicable to full)
    mat = orbs.azimuthal_matrix('full', False, False, True, False)
    full_ref_masked = full_ref.clone()
    full_ref_masked.diagonal()[:] = -1
    check_3 = close(mat, full_ref_masked)
    assert check_3, f'Diagonal masking check failed'

    # Lower triangle masking
    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.azimuthal_matrix(option, False, False, False, True)
        ref = ref.clone()
        ref[[*torch.tril_indices(*ref.shape[:-1], -1)]] = -1
        check_4 = close(mat, ref)
        assert check_4, f'{option}: lower triangular masking is incorrect'

    # On-site masking
    full_mat = orbs.azimuthal_matrix('full', False, True, False, False)
    block_mat = orbs.azimuthal_matrix('shell', False, True, False, False)
    # Build reference matrices for full and block
    full_ref_masked = full_ref.clone()
    mask = torch.block_diag(*[torch.full((i, i), True) for i in [1, 1, 1, 1, 4]])
    mask.diagonal()[:] = False
    full_ref_masked[mask] = -1

    block_ref_masked = block_ref.clone()
    idx_mat = orbs.index_matrix('shell')
    block_ref_masked[idx_mat[..., 0] == idx_mat[..., 1]] = -1

    check_5a = close(full_mat, full_ref_masked)
    check_5b = close(block_mat, block_ref_masked)
    assert check_5a, 'full azimuthal matrix onsite masking failed'
    assert check_5b, 'block azimuthal matrix onsite masking failed'

    # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(ValueError):
        orbs.azimuthal_matrix('invalid_option')
    with pytest.raises(ValueError):
        orbs.azimuthal_matrix('atomic')


def test_orbs_azimuthal_matrix_batch(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([[1, 1, 1, 1, 6],
                                   [1, 6, 0, 0, 0]], device=device)
    shell_dict = {1: [0], 6: [0, 1]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1]
                ])], -1)], value=-1).to(device)

    block_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]
                ])], -1)], value=-1).to(device)

    # Check un-masked and un-sorted full and block azimuthal matrices
    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.azimuthal_matrix(option, False, False, False, False)
        check_1 = close(mat, ref)
        assert check_1, f'{option} azimuthal matrix is incorrect, no sort/mask'

    # Sorting
    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.azimuthal_matrix(option, True, False, False, False)
        check_2 = close(mat, ref.sort(-1)[0])
        assert check_2, f'{option} sorted azimuthal matrix is incorrect'

    # Diagonal masking (only applicable to full)
    mat = orbs.azimuthal_matrix('full', False, False, True, False)
    full_ref_masked = full_ref.clone()
    full_ref_masked.diagonal(dim1=1, dim2=2)[:] = -1
    check_3 = close(mat, full_ref_masked)
    assert check_3, f'Diagonal masking check failed'

    # Lower triangle masking
    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.azimuthal_matrix(option, False, False, False, True)
        # Create masked reference matrix
        ref = ref.clone()
        a, b = [*torch.tril_indices(*ref.shape[1:-1], -1)]
        ref[:, a, b] = -1

        check_4 = close(mat, ref)
        assert check_4, f'{option}: lower triangular masking is incorrect'

    # On-site masking
    full_mat = orbs.azimuthal_matrix('full', False, True, False, False)
    block_mat = orbs.azimuthal_matrix('shell', False, True, False, False)
    # Build reference matrices for full and block
    full_ref_masked = full_ref.clone()
    mask = pack([
        torch.block_diag(*[torch.full((i, i), True) for i in [1, 1, 1, 1, 4]]),
        torch.block_diag(*[torch.full((i, i), True) for i in [1, 4]])
    ], value=True)

    mask.diagonal(dim1=1, dim2=2)[:] = False
    full_ref_masked[mask] = -1

    block_ref_masked = block_ref.clone()
    idx_mat = orbs.index_matrix('shell')
    block_ref_masked[idx_mat[..., 0] == idx_mat[..., 1]] = -1

    check_5a = close(full_mat, full_ref_masked)
    check_5b = close(block_mat, block_ref_masked)
    assert check_5a, 'full azimuthal matrix onsite masking failed'
    assert check_5b, 'block azimuthal matrix onsite masking failed'


def test_orbs_shell_number_matrix_single(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([1, 1, 1, 1, 6], device=device)
    shell_dict = {1: [0], 6: [0, 1]}

    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = torch.stack([
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ]),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1]
            ])], -1).to(device)

    block_ref = torch.stack([
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1]
            ]),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1]
            ])], -1).to(device)

    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.shell_number_matrix(option)
        check_1 = close(mat, ref)
        check_2 = mat.device == device

        assert check_1, f'{option} shell number matrix is incorrect'
        assert check_2, f'{option} shell number matrix is on the wrong device'

        # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(ValueError):
        orbs.shell_number_matrix('invalid_option')
    with pytest.raises(ValueError):
        orbs.shell_number_matrix('atomic')


def test_orbs_shell_number_matrix_batch(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([[1, 1, 1, 1, 6],
                                   [1, 6, 0, 0, 0]], device=device)
    shell_dict = {1: [0], 6: [0, 1]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1]
                ])], -1)], value=-1).to(device)

    block_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]
                ])], -1)], value=-1).to(device)

    for option, ref in zip(['full', 'shell'], [full_ref, block_ref]):
        mat = orbs.shell_number_matrix(option)
        check_1 = close(mat, ref)
        check_2 = mat.device == device

        assert check_1, f'{option} shell number matrix is incorrect'
        assert check_2, f'{option} shell number matrix is on the wrong device'

        # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(ValueError):
        orbs.shell_number_matrix('invalid_option')
    with pytest.raises(ValueError):
        orbs.shell_number_matrix('atomic')
        

def test_orbs_atomic_number_matrix_single(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([1, 1, 1, 1, 6], device=device)
    shell_dict = {1: [0], 6: [0, 1]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = torch.stack([
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [6, 6, 6, 6, 6, 6, 6, 6],
                [6, 6, 6, 6, 6, 6, 6, 6],
                [6, 6, 6, 6, 6, 6, 6, 6],
                [6, 6, 6, 6, 6, 6, 6, 6]
            ]),
        torch.tensor(
            [
                [1, 1, 1, 1, 6, 6, 6, 6],
                [1, 1, 1, 1, 6, 6, 6, 6],
                [1, 1, 1, 1, 6, 6, 6, 6],
                [1, 1, 1, 1, 6, 6, 6, 6],
                [1, 1, 1, 1, 6, 6, 6, 6],
                [1, 1, 1, 1, 6, 6, 6, 6],
                [1, 1, 1, 1, 6, 6, 6, 6],
                [1, 1, 1, 1, 6, 6, 6, 6]
            ])], -1).to(device)

    block_ref = torch.stack([
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [6, 6, 6, 6, 6, 6],
                [6, 6, 6, 6, 6, 6]
            ]),
        torch.tensor(
            [
                [1, 1, 1, 1, 6, 6],
                [1, 1, 1, 1, 6, 6],
                [1, 1, 1, 1, 6, 6],
                [1, 1, 1, 1, 6, 6],
                [1, 1, 1, 1, 6, 6],
                [1, 1, 1, 1, 6, 6]
            ])], -1).to(device)

    atomic_ref = torch.stack([
        torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [6, 6, 6, 6, 6]
            ]),
        torch.tensor(
            [
                [1, 1, 1, 1, 6],
                [1, 1, 1, 1, 6],
                [1, 1, 1, 1, 6],
                [1, 1, 1, 1, 6],
                [1, 1, 1, 1, 6]
            ])], -1).to(device)

    for option, ref in zip(['full', 'shell', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = orbs.atomic_number_matrix(option)
        check_1 = close(mat, ref)
        check_2 = mat.device == device

        assert check_1, f'{option} atomic number matrix is incorrect'
        assert check_2, f'{option} atomic number matrix is on the wrong device'

    # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(ValueError):
        orbs.atomic_number_matrix('invalid_option')


def test_orbs_atomic_number_matrix_batch(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([[1, 1, 1, 1, 6],
                                   [1, 6, 0, 0, 0]], device=device)
    shell_dict = {1: [0], 6: [0, 1]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [6, 6, 6, 6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6, 6, 6, 6]
                ]),
            torch.tensor(
                [
                    [1, 1, 1, 1, 6, 6, 6, 6],
                    [1, 1, 1, 1, 6, 6, 6, 6],
                    [1, 1, 1, 1, 6, 6, 6, 6],
                    [1, 1, 1, 1, 6, 6, 6, 6],
                    [1, 1, 1, 1, 6, 6, 6, 6],
                    [1, 1, 1, 1, 6, 6, 6, 6],
                    [1, 1, 1, 1, 6, 6, 6, 6],
                    [1, 1, 1, 1, 6, 6, 6, 6]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6]
                ]),
            torch.tensor(
                [
                    [1, 6, 6, 6, 6],
                    [1, 6, 6, 6, 6],
                    [1, 6, 6, 6, 6],
                    [1, 6, 6, 6, 6],
                    [1, 6, 6, 6, 6]
                ])], -1)], value=0).to(device)

    block_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [6, 6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6, 6]
                ]),
            torch.tensor(
                [
                    [1, 1, 1, 1, 6, 6],
                    [1, 1, 1, 1, 6, 6],
                    [1, 1, 1, 1, 6, 6],
                    [1, 1, 1, 1, 6, 6],
                    [1, 1, 1, 1, 6, 6],
                    [1, 1, 1, 1, 6, 6]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [1, 1, 1],
                    [6, 6, 6],
                    [6, 6, 6]
                ]),
            torch.tensor(
                [
                    [1, 6, 6],
                    [1, 6, 6],
                    [1, 6, 6]
                ])], -1)], value=0).to(device)

    atomic_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [6, 6, 6, 6, 6]
                ]),
            torch.tensor(
                [
                    [1, 1, 1, 1, 6],
                    [1, 1, 1, 1, 6],
                    [1, 1, 1, 1, 6],
                    [1, 1, 1, 1, 6],
                    [1, 1, 1, 1, 6]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [1, 1],
                    [6, 6]
                ]),
            torch.tensor(
                [
                    [1, 6],
                    [1, 6]
                ])], -1)], value=0).to(device)

    for option, ref in zip(['full', 'shell', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = orbs.atomic_number_matrix(option)
        check_1 = close(mat, ref)
        check_2 = mat.device == device

        assert check_1, f'{option} atomic number matrix is incorrect'
        assert check_2, f'{option} atomic number matrix is on the wrong device'

def test_orbs_index_matrix_single(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([1, 1, 1, 1, 6], device=device)
    shell_dict = {1: [0], 6: [0, 1]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = torch.stack([
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4]
            ]),
        torch.tensor(
            [
                [0, 1, 2, 3, 4, 4, 4, 4],
                [0, 1, 2, 3, 4, 4, 4, 4],
                [0, 1, 2, 3, 4, 4, 4, 4],
                [0, 1, 2, 3, 4, 4, 4, 4],
                [0, 1, 2, 3, 4, 4, 4, 4],
                [0, 1, 2, 3, 4, 4, 4, 4],
                [0, 1, 2, 3, 4, 4, 4, 4],
                [0, 1, 2, 3, 4, 4, 4, 4]
            ])], -1).to(device)

    block_ref = torch.stack([
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4]
            ]),
        torch.tensor(
            [
                [0, 1, 2, 3, 4, 4],
                [0, 1, 2, 3, 4, 4],
                [0, 1, 2, 3, 4, 4],
                [0, 1, 2, 3, 4, 4],
                [0, 1, 2, 3, 4, 4],
                [0, 1, 2, 3, 4, 4]
            ])], -1).to(device)

    atomic_ref = torch.stack([
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4]
            ]),
        torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4]
            ])], -1).to(device)

    for option, ref in zip(['full', 'shell', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = orbs.index_matrix(option)
        check_1 = close(mat, ref)
        check_2 = mat.device == device

        assert check_1, f'{option} index matrix is incorrect'
        assert check_2, f'{option} index matrix is on the wrong device'

    # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(ValueError):
        orbs.index_matrix('invalid_option')


def test_orbs_index_matrix_batch(device):
    # Generate test orbs entity
    atomic_numbers = torch.tensor([[1, 1, 1, 1, 6],
                                   [1, 6, 0, 0, 0]], device=device)
    shell_dict = {1: [0], 6: [0, 1]}
    orbs = OrbitalInfo(atomic_numbers, shell_dict)

    full_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4, 4, 4]
                ]),
            torch.tensor(
                [
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 4, 4]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1]
                ])], -1)], value=-1).to(device)

    block_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4]
                ]),
            torch.tensor(
                [
                    [0, 1, 2, 3, 4, 4],
                    [0, 1, 2, 3, 4, 4],
                    [0, 1, 2, 3, 4, 4],
                    [0, 1, 2, 3, 4, 4],
                    [0, 1, 2, 3, 4, 4],
                    [0, 1, 2, 3, 4, 4]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1]
                ]),
            torch.tensor(
                [
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1]
                ])], -1)], value=-1).to(device)

    atomic_ref = pack([
        torch.stack([
            torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4]
                ]),
            torch.tensor(
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4]
                ])], -1),
        torch.stack([
            torch.tensor(
                [
                    [0, 0],
                    [1, 1]
                ]),
            torch.tensor(
                [
                    [0, 1],
                    [0, 1]
                ])], -1)], value=-1).to(device)

    for option, ref in zip(['full', 'shell', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = orbs.index_matrix(option)
        check_1 = close(mat, ref)
        check_2 = mat.device == device

        assert check_1, f'{option} index matrix is incorrect'
        assert check_2, f'{option} index matrix is on the wrong device'
