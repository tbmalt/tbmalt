"""Basis module unit-tests.

Here the functionality of the basis module is tested. However, it should be
noted that no gradient stability or device persistence checks are performed;
as such tests are not applicable to
"""

import pytest
import torch
from tbmalt.structures.basis import Basis
from tbmalt.common.batch import pack
from torch import allclose as close


def test_basis_basic_single(device):
    """General operational test of basic Basis functionality (single)."""
    # Generate test basis entity
    atomic_numbers = torch.tensor([1, 1, 6, 6, 79, 79])
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

    # Ensure an exception is raised if max_ls keys are not integers
    with pytest.raises(ValueError):
        basis = Basis(atomic_numbers, {torch.tensor(1): 0, torch.tensor(6): 1,
                                       torch.tensor(79): 2})

    # Reference data
    orbs_per_atom = torch.tensor([1, 1, 4, 4, 9, 9])
    orbs_per_shell = torch.tensor([1, 1, 1, 3, 1, 3, 1, 3, 5, 1, 3, 5])
    on_atoms = torch.tensor([0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                             4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    on_shells = torch.tensor([0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7,
                              8, 8, 8, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 11])
    # Counter properties
    check_1 = basis.n_atoms == 6
    check_2 = basis.n_subshells == 12
    check_3 = basis.n_orbitals == 28

    assert check_1, 'n_atoms is incorrect'
    assert check_2, 'n_subshells is incorrect'
    assert check_3, 'n_orbitals is incorrect'

    # Shape properties
    check_4 = basis.atomic_matrix_shape == torch.Size([6, 6])
    check_5 = basis.subshell_matrix_shape == torch.Size([12, 12])
    check_6 = basis.orbital_matrix_shape == torch.Size([28, 28])

    assert check_4, 'atomic_matrix_shape is incorrect'
    assert check_5, 'subshell_matrix_shape is incorrect'
    assert check_6, 'orbital_matrix_shape is incorrect'

    # Orbital counter properties (location specific)
    check_7 = close(basis.orbs_per_atom, orbs_per_atom)
    check_8 = close(basis.orbs_per_shell, orbs_per_shell)

    assert check_7, 'orbs_per_atom is incorrect'
    assert check_8, 'orbs_per_shell is incorrect'

    # Orbital location properties
    check_9 = close(basis.on_atoms, on_atoms)
    check_10 = close(basis.on_shells, on_shells)

    assert check_9, 'on_atoms is incorrect'
    assert check_10, 'on_atoms is incorrect'

    # Resolution convenience functions
    basis.shell_resolved = False
    check_11a = close(basis.orbs_per_res, orbs_per_atom)
    check_12a = close(basis.on_res, on_atoms)
    basis.shell_resolved = True
    check_11b = close(basis.orbs_per_res, orbs_per_shell)
    check_12b = close(basis.on_res, on_shells)
    check_11 = check_11a and check_11b
    check_12 = check_12a and check_12b

    assert check_11, 'orbs_per_res is incorrect'
    assert check_12, 'on_res is incorrect'


def test_basis_basic_batch(device):
    """General operational test of basic Basis functionality (batch)."""
    # Generate test basis entity
    atomic_numbers = [torch.tensor([1, 1, 6, 6, 79, 79]),
                      torch.tensor([1, 6])]
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

    # Reference data
    orbs_per_atom = torch.tensor([[1, 1, 4, 4, 9, 9],
                                  [1, 4, 0, 0, 0, 0]])
    orbs_per_shell = torch.tensor([[1, 1, 1, 3, 1, 3, 1, 3, 5, 1, 3, 5],
                                   [1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    on_atoms = torch.tensor([
        [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [0, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    on_shells = torch.tensor([
        [0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7,
         8, 8, 8, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 11],
        [0, 1, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    # Counter properties
    check_1 = close(basis.n_atoms, torch.tensor([6, 2]))
    check_2 = close(basis.n_subshells, torch.tensor([12, 3]))
    check_3 = close(basis.n_orbitals, torch.tensor([28, 5]))

    assert check_1, 'n_atoms is incorrect'
    assert check_2, 'n_subshells is incorrect'
    assert check_3, 'n_orbitals is incorrect'

    # Shape properties
    check_4 = basis.atomic_matrix_shape == torch.Size([2, 6, 6])
    check_5 = basis.subshell_matrix_shape == torch.Size([2, 12, 12])
    check_6 = basis.orbital_matrix_shape == torch.Size([2, 28, 28])

    assert check_4, 'atomic_matrix_shape is incorrect'
    assert check_5, 'subshell_matrix_shape is incorrect'
    assert check_6, 'orbital_matrix_shape is incorrect'

    # Orbital counter properties (location specific)
    check_7 = close(basis.orbs_per_atom, orbs_per_atom)
    check_8 = close(basis.orbs_per_shell, orbs_per_shell)

    assert check_7, 'orbs_per_atom is incorrect'
    assert check_8, 'orbs_per_shell is incorrect'

    # Orbital location properties
    check_9 = close(basis.on_atoms, on_atoms)
    check_10 = close(basis.on_shells, on_shells)

    assert check_9, 'on_atoms is incorrect'
    assert check_10, 'on_atoms is incorrect'

    # Resolution convenience functions
    basis.shell_resolved = False
    check_11a = close(basis.orbs_per_res, orbs_per_atom)
    check_12a = close(basis.on_res, on_atoms)
    basis.shell_resolved = True
    check_11b = close(basis.orbs_per_res, orbs_per_shell)
    check_12b = close(basis.on_res, on_shells)
    check_11 = check_11a and check_11b
    check_12 = check_12a and check_12b

    assert check_11, 'orbs_per_res is incorrect'
    assert check_12, 'on_res is incorrect'

    # Make sure pre-packed tensors can be passed in
    basis_2 = Basis(pack(atomic_numbers), max_ls)
    # If basis_2 has the "_masks" attribute & its "max_l_on_atom" attribute is
    # the same as basis, then it will perform identically.
    check_13 = (hasattr(basis_2, '_masks')
                and torch.allclose(basis_2.max_l_on_atom, basis.max_l_on_atom))

    assert check_13, 'Basis instantiate incorrectly with pre-packed arguments'


def test_basis_azimuthal_matrix_single(device):
    # Generate test basis entity
    atomic_numbers = torch.tensor([1, 1, 1, 1, 6])
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

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
            ])], -1)

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
            ])], -1)

    # Check un-masked and un-sorted full and block azimuthal matrices
    for option, ref in zip(['full', 'block'], [full_ref, block_ref]):
        mat = basis.azimuthal_matrix(option, False, False, False, False)
        check_1 = close(mat, ref)
        assert check_1, f'{option} azimuthal matrix is incorrect, no sort/mask'

    # Sorting
    for option, ref in zip(['full', 'block'], [full_ref, block_ref]):
        mat = basis.azimuthal_matrix(option, True, False, False, False)
        check_2 = close(mat, ref.sort(-1)[0])
        assert check_2, f'{option} sorted azimuthal matrix is incorrect'

    # Diagonal masking (only applicable to full)
    mat = basis.azimuthal_matrix('full', False, False, False, True)
    full_ref_masked = full_ref.clone()
    full_ref_masked.diagonal()[:] = -1
    check_3 = close(mat, full_ref_masked)
    assert check_3, f'Diagonal masking check failed'

    # Lower triangle masking
    for option, ref in zip(['full', 'block'], [full_ref, block_ref]):
        mat = basis.azimuthal_matrix(option, False, False, True, False)
        ref = ref.clone()
        ref[[*torch.tril_indices(*ref.shape[:-1], -1)]] = -1
        check_4 = close(mat, ref)
        assert check_4, f'{option}: lower triangular masking is incorrect'

    # On-site masking
    full_mat = basis.azimuthal_matrix('full', False, True, False, False)
    block_mat = basis.azimuthal_matrix('block', False, True, False, False)
    # Build reference matrices for full and block
    full_ref_masked = full_ref.clone()
    mask = torch.block_diag(*[torch.full((i, i), True) for i in [1, 1, 1, 1, 4]])
    mask.diagonal()[:] = False
    full_ref_masked[mask] = -1

    block_ref_masked = block_ref.clone()
    block_ref_masked.diagonal()[:] = -1

    check_5a = close(full_mat, full_ref_masked)
    check_5b = check_4 = close(block_mat, block_ref_masked)
    assert check_5a, 'full azimuthal matrix onsite masking failed'
    assert check_5a, 'block azimuthal matrix onsite masking failed'

    # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(KeyError):
        basis.azimuthal_matrix('invalid_option')



def test_basis_azimuthal_matrix_batch(device):
    # Generate test basis entity
    atomic_numbers = [torch.tensor([1, 1, 1, 1, 6]),
                      torch.tensor([1, 6])]
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

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
                ])], -1)], value=-1)

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
                ])], -1)], value=-1)

    # Check un-masked and un-sorted full and block azimuthal matrices
    for option, ref in zip(['full', 'block'], [full_ref, block_ref]):
        mat = basis.azimuthal_matrix(option, False, False, False, False)
        check_1 = close(mat, ref)
        assert check_1, f'{option} azimuthal matrix is incorrect, no sort/mask'

    # Sorting
    for option, ref in zip(['full', 'block'], [full_ref, block_ref]):
        mat = basis.azimuthal_matrix(option, True, False, False, False)
        check_2 = close(mat, ref.sort(-1)[0])
        assert check_2, f'{option} sorted azimuthal matrix is incorrect'

    # Diagonal masking (only applicable to full)
    mat = basis.azimuthal_matrix('full', False, False, False, True)
    full_ref_masked = full_ref.clone()
    full_ref_masked.diagonal(dim1=1, dim2=2)[:] = -1
    check_3 = close(mat, full_ref_masked)
    assert check_3, f'Diagonal masking check failed'

    # Lower triangle masking
    for option, ref in zip(['full', 'block'], [full_ref, block_ref]):
        mat = basis.azimuthal_matrix(option, False, False, True, False)
        # Create masked reference matrix
        ref = ref.clone()
        a, b = [*torch.tril_indices(*ref.shape[1:-1], -1)]
        ref[:, a, b] = -1

        check_4 = close(mat, ref)
        assert check_4, f'{option}: lower triangular masking is incorrect'

    # On-site masking
    full_mat = basis.azimuthal_matrix('full', False, True, False, False)
    block_mat = basis.azimuthal_matrix('block', False, True, False, False)
    # Build reference matrices for full and block
    full_ref_masked = full_ref.clone()
    mask = pack([
        torch.block_diag(*[torch.full((i, i), True) for i in [1, 1, 1, 1, 4]]),
        torch.block_diag(*[torch.full((i, i), True) for i in [1, 4]])
    ], value=True)

    mask.diagonal(dim1=1, dim2=2)[:] = False
    full_ref_masked[mask] = -1

    block_ref_masked = block_ref.clone()
    block_ref_masked.diagonal(dim1=1, dim2=2)[:] = -1

    check_5a = close(full_mat, full_ref_masked)
    check_5b = check_4 = close(block_mat, block_ref_masked)
    assert check_5a, 'full azimuthal matrix onsite masking failed'
    assert check_5a, 'block azimuthal matrix onsite masking failed'


def test_basis_atomic_number_matrix_single(device):
    # Generate test basis entity
    atomic_numbers = torch.tensor([1, 1, 1, 1, 6])
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

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
            ])], -1)

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
            ])], -1)

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
            ])], -1)

    for option, ref in zip(['full', 'block', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = basis.atomic_number_matrix(option)
        check_1 = close(mat, ref)
        assert check_1, f'{option} atomic number matrix is incorrect'

    # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(KeyError):
        basis.atomic_number_matrix('invalid_option')


def test_basis_atomic_number_matrix_batch(device):
    # Generate test basis entity
    atomic_numbers = [torch.tensor([1, 1, 1, 1, 6]),
                      torch.tensor([1, 6])]
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

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
                ])], -1)], value=0)

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
                ])], -1)], value=0)

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
                ])], -1)], value=0)

    for option, ref in zip(['full', 'block', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = basis.atomic_number_matrix(option)
        check_1 = close(mat, ref)
        assert check_1, f'{option} atomic number matrix is incorrect'

def test_basis_index_matrix_single(device):
    # Generate test basis entity
    atomic_numbers = torch.tensor([1, 1, 1, 1, 6])
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

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
            ])], -1)

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
            ])], -1)

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
            ])], -1)

    for option, ref in zip(['full', 'block', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = basis.index_matrix(option)
        check_1 = close(mat, ref)
        assert check_1, f'{option} index matrix is incorrect'

    # Ensure an exception is raised if an invalid option was passed
    with pytest.raises(KeyError):
        basis.index_matrix('invalid_option')


def test_basis_index_matrix_batch(device):
    # Generate test basis entity
    atomic_numbers = [torch.tensor([1, 1, 1, 1, 6]),
                      torch.tensor([1, 6])]
    max_ls = {1: 0, 6: 1, 79: 2}
    basis = Basis(atomic_numbers, max_ls)

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
                ])], -1)], value=-1)

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
                ])], -1)], value=-1)

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
                ])], -1)], value=-1)

    for option, ref in zip(['full', 'block', 'atomic'],
                           [full_ref, block_ref, atomic_ref]):
        mat = basis.index_matrix(option)
        check_1 = close(mat, ref)
        assert check_1, f'{option} index matrix is incorrect'
