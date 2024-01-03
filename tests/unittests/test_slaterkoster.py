# -*- coding: utf-8 -*-
"""Unit tests associated with `tbmalt.physics.dftb.slaterkoster`."""
from typing import Tuple, List, Callable, Optional, Union
from re import findall

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.autograd import gradcheck

from tbmalt.physics.dftb.slaterkoster import (
    _rot_yz_s, _rot_xy_s, _rot_yz_p, _rot_xy_p, _rot_yz_d,
    _rot_xy_d, _rot_yz_f, _rot_xy_f,
    sub_block_ref, sub_block_rot
)
from tbmalt import Geometry

####################
# Helper Functions #
####################
l_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
max_ls = {1: 0, 6: 1, 79: 2, 57: 3}
shell_dict = {1: [0], 6: [0, 1], 79: [0, 1, 2], 57: [0, 1, 2, 3]}


def from_file(path: str, **kwargs) -> Tensor:
    """Reads data from a numpy text file into a `Tensor`.

    This function extracts and returns n-dimensional tensors from a specified
    numpy text file.

    Arguments:
        path: Path to the target numpy text file.
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            the device on which the tensor is placed to be controlled.

    Returns:
        tensor: Data from the specified file.
    """
    shape = np.array(findall('[0-9]+', open(path).readline()), dtype=int)
    return torch.tensor(np.loadtxt(path).reshape(shape), **kwargs)


def build_geom(*atomic_numbers: int, **kwargs) -> Geometry:
    """Constructs a `Geometry` object for a given number of atoms.

    Takes an arbitrary number of atoms and returns a `Geometry` object. Note
    that the positions are randomly generated.

    Arguments:
        *atomic_numbers: An arbitrary number atomic numbers.
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            the device on which the tensor is placed to be controlled.

    Returns:
        geometry: `Geometry` object with the desired atoms at random positions.
    """

    return Geometry(torch.tensor(*[atomic_numbers],),
                    torch.rand(len(atomic_numbers), 3, **kwargs))


def molecules(device: torch.device, n: Optional[int] = None
              ) -> Union[
                         Tuple[List[Tensor], List[Tensor]],
                         Tuple[Tensor, Tensor]
                        ]:
    """Atomic numbers & positions of a small selection of molecules.

    This data is used in testing the ``hs_matrix`` function. This contains
    three molecules: i) H2, simple diatomic molecule; ii) CH4, a slightly more
    complex molecules; and iii) CH4+Au2La a contrived system that permits
    testing of all supported orbital interactions.

    Arguments:
        device: The device onto which the position tensors are to be placed.
        n: If only a single system is desired ``n`` can be used to specify the
            index of the desired system. By default, ``n`` is None; meaning
            that all systems are returned.

    Returns:
        atomic_numbers: Atomic numbers of the systems.
        positions: Positions of said systems.
    """
    atomic_numbers = [
        torch.tensor([1, 1]),
        torch.tensor([6,  1,  1,  1,  1]),
        torch.tensor([57, 79, 57, 6, 1, 1, 1, 1])
    ]

    positions = [
        torch.tensor([[0.00,   0.00,  0.00],
                      [0.00,   0.00,  1.40]], device=device),
        torch.tensor([[0.00,   0.00,  0.00],
                      [1.19,   1.19,  1.19],
                      [-1.19, -1.19,  1.19],
                      [1.19,  -1.19, -1.19],
                      [-1.19,  1.19, -1.19]], device=device),
        torch.tensor([[3.78,   3.78,  3.78],
                      [-2.83, -2.83, -2.83],
                      [-3.78,  3.78,  3.78],
                      [0.00,   0.00,  0.00],
                      [1.19,   1.19,  1.19],
                      [-1.19, -1.19,  1.19],
                      [1.19,  -1.19, -1.19],
                      [-1.19,  1.19, -1.19]], device=device)
    ]
    if n is None:
        return atomic_numbers, positions
    else:
        return atomic_numbers[n], positions[n]


def sk_rotation_data(batch: bool = False, **kwargs
                     ) -> Tuple[Callable, Tensor, Tensor]:
    """Slater-Koster rotation matrix data.

    This function returns data required to run and validate the Slater-Koster
    rotation sub-blocks.

    Arguments:
        batch: Controls whether single data-points or a batch thereof should
            be yielded at each iteration. [DEFAULT=False]
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Yields:
        func: Rotation function to be tested.
        unit_vectors: Unit vectors (input data for the functions)
        reference: Reference results.
    """
    skt_functions = [_rot_yz_s, _rot_xy_s, _rot_yz_p, _rot_xy_p, _rot_yz_d,
                     _rot_xy_d, _rot_yz_f, _rot_xy_f]

    # Read in unit vectors & the reference rotation matrices. Then create a mas
    # identifying vectors where |y| > |z|.
    path = 'tests/unittests/data/slaterkoster'
    u_vecs = from_file(f'{path}/unit_vectors.dat', **kwargs)
    r_mats = {ll: from_file(f'{path}/rot_{ll}.dat', **kwargs) for ll in range(4)}
    yz_mask = u_vecs[:, 1].abs() > u_vecs[:, 2].abs()

    for f in skt_functions:
        # Extract azimuthal number and rotation plane from the function's name.
        mask = yz_mask if 'yz' in f.__name__ else ~yz_mask
        s = ... if batch else 0
        yield f, u_vecs[mask][s], r_mats[l_dict[f.__name__[-1]]][mask][s]


def sub_block_ref_data(batch: bool = False, **kwargs
                       ) -> Tuple[Tensor, Tensor, Tensor]:
    """Data for testing the `sub_block_ref` function.

    Yields synthetic data for testing `sub_block_ref`. Synthetic data is used
    as `sub_block_ref` is effectively a fancy diagonal embedding algorithm.

    Arguments:
        batch: Controls whether single data-points or a batch thereof should
            be yielded at each iteration. [DEFAULT=False]
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Yields:
        l_pair: Azimuthal pair (input data).
        integrals: Associated Slater-Koster integrals (input data).
        reference: Reference results.
    """
    dtype = torch.rand(1).dtype

    def to_batch(data):
        """Convert single system into a batch of systems"""
        size = torch.Size((3, *([1] * data[-1].ndim)))
        fac = torch.arange(3, **kwargs).view(size)
        return [torch.stack([i] * 3) * fac for i in data]

    requires_grad = kwargs.pop('requires_grad', False)

    # Azimuthal quantum number pairs. The device on which l_pairs is placed is
    # irrelevant, as it is only ever used for indexing.
    l_pairs = torch.tensor([[i, j] for i in range(4) for j in range(4) if i <= j])

    # Reference block matrices
    ff_m = torch.diag_embed(torch.arange(-3, 4, dtype=dtype, **kwargs).abs() + 1.)
    slices = list(reversed([slice(i, 7 - i) for i in range(4)]))
    results = [torch.atleast_1d(ff_m[r, c].squeeze()).clone()
               for l1, r in enumerate(slices) for c in slices[l1:]]

    # Generate some dummy off-site integral values; ordered as follows:
    #   [ss_i, sp_i, sd_i, sf_i, pp_i, pd_i, pf_i, dd_i, df_i, ff_i]
    integrals = [i for l, n in [(2, 4), (3, 3), (4, 2), (5, 1)] for i in
                 torch.arange(1, l, **kwargs, dtype=dtype).tile(n, 1)]

    if batch:  # Create additional systems if a batch has been requested
        results = to_batch(results)
        integrals = to_batch(integrals)

    # Enable gradient tracking for integrals if requested, this must be done
    # after "batchification" to prevent graph flow issues.
    if requires_grad:
        for i in integrals:
            i.requires_grad = True

    for l_pair, integral, result in zip(l_pairs, integrals, results):
        yield l_pair, integral, result


def sub_block_rot_data(batch: bool = False, **kwargs
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Data for testing the `sub_block_rot` function.

    Yields reference data required to test the `sub_block_rot` function.

    Arguments:
        batch: Controls whether single data-points or a batch thereof should
            be yielded at each iteration. [DEFAULT=False]
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Yields:
        l_pair: Azimuthal pair (input data).
        unit_vectors: Unit vectors (input data).
        integrals:  Slater-Koster integrals (input data).
        reference: Reference results.
    """
    path = 'tests/unittests/data/slaterkoster'
    l_pairs = [(l1, l2) for l1 in range(4) for l2 in range(4)]
    u_vecs = from_file(f'{path}/unit_vectors.dat', **kwargs)

    kwargs.pop('requires_grad', None)
    for l_pair in l_pairs:
        integrals = from_file(
            f'{path}/integrals_{min(l_pair)}_{max(l_pair)}.dat', **kwargs)
        ref = from_file(f'{path}/block_rot_{l_pair[0]}_{l_pair[1]}.dat', **kwargs)
        s = ... if batch else 0
        yield torch.tensor(l_pair), u_vecs[s], integrals[s], ref[s]


###########################################
# Slater-Koster Rotation Matrix Functions #
###########################################
def _sk_rotation_matrices_test_helper(batch, **kwargs):
    """Tests the Slater-Koster rotation matrix functions."""
    for func, u_vec, ref in sk_rotation_data(batch, **kwargs):
        pred = func(u_vec)
        check_1 = torch.allclose(pred, ref)
        check_2 = pred.device == kwargs['device']

        name = f'{func.__name__} {"[batch]" if batch else "[single]"}'

        assert check_1, f'Result of {name} exceed tolerance limits'
        assert check_2, f'{name} failed device persistence check'


def test_sk_rotation_matrices_single(device):
    """Runs single system tests of the SK rotation matrix functions."""
    _sk_rotation_matrices_test_helper(False, device=device)


def test_sk_rotation_matrices_batch(device):
    """Runs batch system tests of the SK rotation matrix functions."""
    _sk_rotation_matrices_test_helper(True, device=device)


@pytest.mark.grad
def test_sk_rotation_matrices_grad(device):
    """Checks the gradient stability of the SK rotation matrix functions."""
    # Note this test will always take a long time as the final
    for func, u_vec, _ in sk_rotation_data(
            True, device=device, requires_grad=True):
        name = f'{func.__name__}'
        grad_s = gradcheck(func, (u_vec[0],), raise_exception=False)
        grad_b = gradcheck(func, (u_vec[[0, -1]],), raise_exception=False)
        assert grad_s, f'{name} failed single system gradient stability test'
        assert grad_b, f'{name} failed batch system gradient stability test'


#############################################
# tbmalt.physics.slaterkoster.sub_block_ref #
#############################################
def _sub_block_ref_helper(batch=False, **kwargs):
    """Used when testing the `sub_block_ref` function.

    The `sub_block_ref` function is responsible for constructing the unrotated
    diatomic sub-blocks.
    """
    for l_pair, integrals, res in sub_block_ref_data(batch, **kwargs):
        # Ensure results are within acceptable tolerance when in both
        # azimuthal minor & azimuthal major modes.
        pred_1 = sub_block_ref(l_pair, integrals)
        pred_2 = sub_block_ref(l_pair.flip(-1), integrals)

        res = torch.atleast_2d(res)
        check_1a = torch.allclose(pred_1, res)
        check_1b = torch.allclose(pred_2.transpose(-1, -2), res)
        check_1 = check_1a and check_1b
        check_2 = pred_1.device == kwargs['device']

        form = '[batch]' if batch else '[single]'
        name = str(l_pair.tolist()) + form
        assert check_1, f'Integral matrix tolerances exceeded; {name}'
        assert check_2, f'Device persistence check failed; {name}'


def test_sub_block_ref_single(device):
    """Runs single system tests on the `sub_block_ref` function."""
    _sub_block_ref_helper(False, device=device)


def test_sub_block_ref_batch(device):
    """Runs batch system tests on the `sub_block_ref` function."""
    _sub_block_ref_helper(True, device=device)


@pytest.mark.grad
def test_sub_block_ref_grad(device):
    """Runs gradient stability tests on the `sub_block_ref` function."""
    for l_pair, integrals, res in sub_block_ref_data(
            True, device=device, requires_grad=True):
        grad_s = gradcheck(sub_block_ref, (l_pair, integrals[0],),
                           raise_exception=False)
        grad_b = gradcheck(sub_block_ref, (l_pair, integrals[::3],),
                           raise_exception=False)
        name = str(l_pair.tolist())
        assert grad_s, f'Single system gradient stability test failed on {name}'
        assert grad_b, f'Batch system gradient stability test failed on {name}'


#############################################
# tbmalt.physics.slaterkoster.sub_block_rot #
#############################################
def _sub_block_rot_helper(batch=False, **kwargs):
    """Used in testing the `sub_block_rot` function.

    The `sub_block_rot` function is responsible for constructing the rotated
    Slater-Koster diatomic sub-block from a set of SK-integrals & unit vectors.

    Notes:
         `sub_block_rot` utilises the`sub_block_ref` & rotation matrix
         functions. Thus, this test will likely fail if either of the
         following tests fail:
            - test_sk_rotation_matrices_single
            - test_sub_block_ref_single
    """

    for l_pair, u_vecs, ints, ref in sub_block_rot_data(batch, **kwargs):
        pred = sub_block_rot(l_pair, u_vecs, ints)
        check_1 = torch.allclose(ref, pred)
        check_2 = pred.device == kwargs['device']

        form = '[batch]' if batch else '[single]'
        name = str(l_pair.tolist()) + form

        assert check_1, f'Result of {name} exceed tolerance limits'
        assert check_2, f'{name} failed device persistence check'


def test_sub_block_rot_single(device):
    """Performs a single system test on the `sub_block_rot` function.

    If this test fails, read the doc-string in `_sub_block_rot_helper`."""
    _sub_block_rot_helper(False, device=device)


def test_sub_block_rot_batch(device):
    """Performs a batch system test on the `sub_block_rot` function.

    If this test fails, read the doc-string in `_sub_block_rot_helper`."""
    _sub_block_rot_helper(True, device=device)


@pytest.mark.grad
def test_sub_block_rot_grad(device):
    """Checks the gradient stability of the `sub_block_rot` function.

    If this test fails, read the doc-string in `_sub_block_rot_helper`."""
    for l_pair, u_vecs, ints, ref in sub_block_rot_data(
            True, device=device, requires_grad=True):
        grad_s = gradcheck(sub_block_rot, (l_pair, u_vecs[0], ints[0]),
                           raise_exception=False)
        grad_b = gradcheck(sub_block_rot, (l_pair, u_vecs[::3], ints[::3]),
                           raise_exception=False)

        name = str(l_pair.tolist())
        assert grad_s, f'Single system gradient stability test failed on {name}'
        assert grad_b, f'Batch system gradient stability test failed on {name}'
