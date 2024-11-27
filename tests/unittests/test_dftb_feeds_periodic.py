from os.path import join, dirname
import pytest
import urllib
import torch
import tarfile
from typing import List
import numpy as np
from tbmalt.io.skf import Skf
from tbmalt.physics.dftb.feeds import ScipySkFeed, SkfOccupationFeed, SkFeed
from tbmalt import Geometry, OrbitalInfo
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units
from functools import reduce

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

    # Cutoff in bohr
    cutoff = torch.tensor([9.98], device=device)
    cutoff2 = torch.tensor([18.38], device=device)  # Au-Au cutoff

    H2 = Geometry(
        torch.tensor([1, 1], device=device),
        torch.tensor([
            [+0.000000000000000E+00, +0.000000000000000E+00, 0.696520874048385252],
            [+0.000000000000000E+00, +0.000000000000000E+00, -0.696520874048385252]],
            device=device),
        torch.tensor([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5]],
            device=device), cutoff=cutoff)

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
                   units='angstrom',
                   cutoff = cutoff / length_units['angstrom'])

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
                   units='angstrom',
                   cutoff = cutoff / length_units['angstrom'])

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
                    units='angstrom',
                    cutoff = cutoff / length_units['angstrom'])

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
                         units='angstrom',
                         cutoff = cutoff2 / length_units['angstrom'])

    return [H2, CH4, H2O, C2H6, C2H2Au2S3]


def hamiltonians(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'H2O', 'C2H6', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_pbc_H.csv'), delimiter=','),
            device=device))
    return matrices


def overlaps(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'H2O', 'C2H6', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_pbc_S.csv'), delimiter=','),
            device=device))
    return matrices


#########################################
# tbmalt.physics.dftb.feeds.ScipySkFeed #
#########################################

# Single
def test_scipyskfeed_pbc_single(device, skf_file: str):
    """ScipySkFeed matrix single system operability tolerance test"""
    if device.type == "cuda":
        pytest.skip("Scipy splines do not support CUDA.")

    b_def = {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)

    for sys, H_ref, S_ref in zip(
            systems(device), hamiltonians(device), overlaps(device)):
        H = H_feed.matrix(sys, OrbitalInfo(sys.atomic_numbers, b_def))
        S = S_feed.matrix(sys, OrbitalInfo(sys.atomic_numbers, b_def))

        check_1 = torch.allclose(H, H_ref, atol=1E-2)
        check_2 = torch.allclose(S, S_ref, atol=1E-1)
        check_3 = H.device == device
        assert check_1, f'ScipySkFeed H matrix outside of tolerance ({sys})'
        assert check_2, f'ScipySkFeed S matrix outside of tolerance ({sys})'
        assert check_3, 'ScipySkFeed.matrix returned on incorrect device'


# Batch
def test_scipyskfeed_pbc_batch(device, skf_file: str):
    """ScipySkFeed matrix batch operability tolerance test"""

    if device.type == "cuda":
        pytest.skip("Scipy splines do not support CUDA.")

    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)


    sys = reduce(lambda i, j: i+j, systems(device))
    orbs = OrbitalInfo(sys.atomic_numbers,
                        {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]})

    H = H_feed.matrix(sys, orbs)
    S = S_feed.matrix(sys, orbs)

    check_1 = torch.allclose(H, pack(hamiltonians(device)), atol=1E-2)
    check_2 = torch.allclose(S, pack(overlaps(device)), atol=1E-1)
    check_3 = H.device == device

    # Check that batches of size one do not cause problems
    check_4 = (H_feed.matrix(sys[0:1], orbs[0:1]).ndim == 3)

    assert check_1, 'ScipySkFeed H matrix outside of tolerance (batch)'
    assert check_2, 'ScipySkFeed S matrix outside of tolerance (batch)'
    assert check_3, 'ScipySkFeed.matrix returned on incorrect device'
    assert check_4, 'Failure to operate on batches of size "one"'

#########################################
# tbmalt.physics.dftb.feeds.SkFeed #
#########################################


# Single
def test_skffeed_pbc_single(device, skf_file: str):
    """SkFeed matrix single system operability tolerance test"""

    b_def = {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    H_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)

    for sys, H_ref, S_ref in zip(
            systems(device), hamiltonians(device), overlaps(device)):
        H = H_feed.matrix(sys, OrbitalInfo(sys.atomic_numbers, b_def))
        S = S_feed.matrix(sys, OrbitalInfo(sys.atomic_numbers, b_def))

        check_1 = torch.allclose(H, H_ref, rtol=1E-8, atol=1E-8)
        check_2 = torch.allclose(S, S_ref, rtol=1E-8, atol=1E-8)
        check_3 = H.device == device

        assert check_1, f'SkFeed H matrix outside of tolerance ({sys})'
        assert check_2, f'SkFeed S matrix outside of tolerance ({sys})'
        assert check_3, 'SkFeed.matrix returned on incorrect device'


# Batch
def test_skffeed_pbc_batch(device, skf_file: str):
    """SkFeed matrix batch operability tolerance test"""
    H_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'hamiltonian', device=device)
    S_feed = SkFeed.from_database(
        skf_file, [1, 6, 8, 16, 79], 'overlap', device=device)

    sys = reduce(lambda i, j: i+j, systems(device))
    orbs = OrbitalInfo(sys.atomic_numbers,
                        {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]})

    H = H_feed.matrix(sys, orbs)
    S = S_feed.matrix(sys, orbs)

    check_1 = torch.allclose(H, pack(hamiltonians(device)), rtol=1E-8, atol=1E-8)
    check_2 = torch.allclose(S, pack(overlaps(device)), rtol=1E-8, atol=1E-8)
    check_3 = H.device == device

    # Check that batches of size one do not cause problems
    check_4 = (H_feed.matrix(sys[0:1], orbs[0:1]).ndim == 3)

    assert check_1, 'SkFeed H matrix outside of tolerance (batch)'
    assert check_2, 'SkFeed S matrix outside of tolerance (batch)'
    assert check_3, 'SkFeed.matrix returned on incorrect device'
    assert check_4, 'Failure to operate on batches of size "one"'
