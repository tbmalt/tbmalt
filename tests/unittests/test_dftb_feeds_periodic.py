from os.path import join, dirname
import pytest
import urllib
import torch
import tarfile
from typing import List
import numpy as np
from tbmalt.io.skf import Skf
from tbmalt.physics.dftb.feeds import ScipySkFeed, SkfOccupationFeed, SkFeed
from tbmalt import Geometry, Basis, Periodic
from tbmalt.common.batch import pack
from functools import reduce

from tests.test_utils import skf_file

torch.set_default_dtype(torch.float64)


def systems(device) -> List[Geometry]:
    """Returns a selection of `Geometry` entities for testing.

    Currently returned systems are CH4, H2O, C2H6 and C2H2Au2S3. The last of
    which is designed to ensure most possible interaction types are checked.

    Arguments:
        device: device onto which the `Geometry` objects should be placed.
            [DEFAULT=None]

    Returns:
        geometries: A list of `Geometry` objects.
    """
    CH4 = Geometry(torch.tensor([6, 1, 1, 1, 1], device=device),
                   torch.tensor([
                       [3.0, 3.0, 3.0],
                       [3.6, 3.6, 3.6],
                       [2.4, 3.6, 3.6],
                       [3.6, 2.4, 3.6],
                       [3.6, 3.6, 2.4]],
                       device=device),
                   torch.tensor(
                       [[4.0, 4.0, 0.0],
                        [0.0, 5.0, 0.0],
                        [0.0, 0.0, 6.0]],
                       device=device),
                   units='angstrom')

    H2O = Geometry(torch.tensor([1, 8, 1], device=device),
                   torch.tensor([
                       [0.965, 0.075, 0.088],
                       [1.954, 0.047, 0.056],
                       [2.244, 0.660, 0.778]],
                       device=device),
                   torch.tensor(
                       [[4.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0],
                        [0.0, 0.0, 6.0]],
                       device=device),
                   units='angstrom')

    C2H6 = Geometry(torch.tensor([6, 6, 1, 1, 1, 1, 1, 1], device=device),
                    torch.tensor([
                       [0.949, 0.084, 0.020],
                       [2.469, 0.084, 0.020],
                       [0.573, 1.098, 0.268],
                       [0.573, -0.638, 0.775],
                       [0.573, -0.209, -0.982],
                       [2.845, 0.376, 1.023],
                       [2.845, 0.805, -0.735],
                       [2.845, -0.931, -0.227]],
                       device=device),
                    torch.tensor(
                       [[5.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0],
                        [0.0, 4.0, 4.0]],
                       device=device),
                    units='angstrom')

    C2H2Au2S3 = Geometry(torch.tensor([1, 6, 16, 79, 16, 79, 16, 6, 1], device=device),
                         torch.tensor([
                             [+0.00, +0.00, +0.00],
                             [-0.03, +0.83, +0.86],
                             [-0.65, +1.30, +1.60],
                             [+0.14, +1.80, +2.15],
                             [-0.55, +0.42, +2.36],
                             [+0.03, +2.41, +3.46],
                             [+1.12, +1.66, +3.23],
                             [+1.10, +0.97, +0.86],
                             [+0.19, +0.93, +4.08]],
                             device=device),
                         torch.tensor(
                             [[5.0, 0.0, 0.0],
                              [0.0, 5.0, 0.0],
                              [0.0, 0.0, 5.0]],
                             device=device),
                         units='angstrom')

    return [CH4, H2O, C2H6, C2H2Au2S3]


def hamiltonians(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['CH4', 'H2O', 'C2H6', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_pbc_H.csv'), delimiter=','),
            device=device))
    return matrices


def overlaps(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['CH4', 'H2O', 'C2H6', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_pbc_S.csv'), delimiter=','),
            device=device))
    return matrices


#########################################
# tbmalt.physics.dftb.feeds.ScipySkFeed #
#########################################

# Single
def test_scipyskfeed_pbc_single(skf_file: str, device):
    """ScipySkFeed matrix single system operability tolerance test"""
    b_def = {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)
    cutoff = torch.tensor([18.38])  # Au-Au pair has a large cutoff

    for sys, H_ref, S_ref in zip(
            systems(device), hamiltonians(device), overlaps(device)):
        periodic = Periodic(sys, sys.cells, cutoff)
        H = H_feed.matrix(sys, Basis(sys.atomic_numbers, b_def), periodic)
        S = S_feed.matrix(sys, Basis(sys.atomic_numbers, b_def), periodic)

        check_1 = torch.allclose(H, H_ref, atol=1E-3)
        check_2 = torch.allclose(S, S_ref, atol=1E-3)
        check_3 = H.device == device
        assert check_1, f'ScipySkFeed H matrix outside of tolerance ({sys})'
        assert check_2, f'ScipySkFeed S matrix outside of tolerance ({sys})'
        assert check_3, 'ScipySkFeed.matrix returned on incorrect device'


# Batch
def test_scipyskfeed_pbc_batch(skf_file: str, device):
    """ScipySkFeed matrix batch operability tolerance test"""
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)
    cutoff = torch.tensor([18.38])

    sys = reduce(lambda i, j: i+j, systems(device))
    basis = Basis(sys.atomic_numbers,
                  {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]})
    periodic = Periodic(sys, sys.cells, cutoff)

    H = H_feed.matrix(sys, basis, periodic)
    S = S_feed.matrix(sys, basis, periodic)

    check_1 = torch.allclose(H, pack(hamiltonians(device)), atol=1E-3)
    check_2 = torch.allclose(S, pack(overlaps(device)), atol=1E-3)
    check_3 = H.device == device

    # Check that batches of size one do not cause problems
    check_4 = (H_feed.matrix(sys[0:1], basis[0:1], Periodic(
        sys[0:1], sys[0:1].cells, cutoff)).ndim == 3)

    assert check_1, 'ScipySkFeed H matrix outside of tolerance (batch)'
    assert check_2, 'ScipySkFeed S matrix outside of tolerance (batch)'
    assert check_3, 'ScipySkFeed.matrix returned on incorrect device'
    assert check_4, 'Failure to operate on batches of size "one"'


#########################################
# tbmalt.physics.dftb.feeds.SkFeed #
#########################################

# Single
def test_skffeed_pbc_single(skf_file: str, device):
    """SkFeed matrix single system operability tolerance test"""
    b_def = {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    H_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)
    cutoff = torch.tensor([18.38])

    for sys, H_ref, S_ref in zip(
            systems(device), hamiltonians(device), overlaps(device)):
        periodic = Periodic(sys, sys.cells, cutoff)
        H = H_feed.matrix(sys, Basis(sys.atomic_numbers, b_def), periodic)
        S = S_feed.matrix(sys, Basis(sys.atomic_numbers, b_def), periodic)

        check_1 = torch.allclose(H, H_ref, atol=1E-9)
        check_2 = torch.allclose(S, S_ref, atol=1E-9)
        check_3 = H.device == device
        assert check_1, f'SkFeed H matrix outside of tolerance ({sys})'
        assert check_2, f'SkFeed S matrix outside of tolerance ({sys})'
        assert check_3, 'SkFeed.matrix returned on incorrect device'


# Batch
def test_skffeed_pbc_batch(skf_file: str, device):
    """SkFeed matrix batch operability tolerance test"""
    H_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)
    cutoff = torch.tensor([18.38])

    sys = reduce(lambda i, j: i+j, systems(device))
    basis = Basis(sys.atomic_numbers,
                  {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]})
    periodic = Periodic(sys, sys.cells, cutoff)

    H = H_feed.matrix(sys, basis, periodic)
    S = S_feed.matrix(sys, basis, periodic)

    check_1 = torch.allclose(H, pack(hamiltonians(device)), atol=1E-9)
    check_2 = torch.allclose(S, pack(overlaps(device)), atol=1E-9)
    check_3 = H.device == device

    # Check that batches of size one do not cause problems
    check_4 = (H_feed.matrix(sys[0:1], basis[0:1], Periodic(
        sys[0:1], sys[0:1].cells, cutoff)).ndim == 3)

    assert check_1, 'SkFeed H matrix outside of tolerance (batch)'
    assert check_2, 'SkFeed S matrix outside of tolerance (batch)'
    assert check_3, 'SkFeed.matrix returned on incorrect device'
    assert check_4, 'Failure to operate on batches of size "one"'
