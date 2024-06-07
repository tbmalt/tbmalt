import os
from os.path import join, dirname
import pytest
import torch
from typing import List
import numpy as np
from ase.build import molecule

from tbmalt.physics.dftb.feeds import ScipySkFeed, SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt import Geometry, OrbitalInfo
from tbmalt.common.batch import pack
from functools import reduce

from tests.test_utils import skf_file, skf_file_vcr

torch.set_default_dtype(torch.float64)


def molecules(device) -> List[Geometry]:
    """Returns a selection of `Geometry` entities for testing.

    Currently, returned systems are H2, CH4, and C2H2Au2S3. The last of which
    is designed to ensure most possible interaction types are checked.

    Arguments:
        device: device onto which the `Geometry` objects should be placed.
            [DEFAULT=None]

    Returns:
        geometries: A list of `Geometry` objects.
    """
    H2 = Geometry(torch.tensor([1, 1], device=device),
                  torch.tensor([
                      [+0.00, +0.00, +0.37],
                      [+0.00, +0.00, -0.37]],
                      device=device),
                  units='angstrom')

    CH4 = Geometry(torch.tensor([6, 1, 1, 1, 1], device=device),
                   torch.tensor([
                       [+0.00, +0.00, +0.00],
                       [+0.63, +0.63, +0.63],
                       [-0.63, -0.63, +0.63],
                       [+0.63, -0.63, -0.63],
                       [-0.63, +0.63, -0.63]],
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
                         units='angstrom')

    return [H2, CH4, C2H2Au2S3]


def hamiltonians(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_H.csv'), delimiter=','),
            device=device))
    return matrices


def overlaps(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_S.csv'), delimiter=','),
            device=device))
    return matrices


#########################################
# tbmalt.physics.dftb.feeds.ScipySkFeed #
#########################################
# Single
def test_scipyskfeed_single(skf_file: str, device):
    """ScipySkFeed matrix single system operability tolerance test"""
    b_def = {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'overlap', device=device)

    for mol, H_ref, S_ref in zip(
            molecules(device), hamiltonians(device), overlaps(device)):
        H = H_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))
        S = S_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))

        check_1 = torch.allclose(H, H_ref, atol=1E-7)
        check_2 = torch.allclose(S, S_ref, atol=1E-7)
        check_3 = H.device == device

        assert check_1, f'ScipySkFeed H matrix outside of tolerance ({mol})'
        assert check_2, f'ScipySkFeed S matrix outside of tolerance ({mol})'
        assert check_3, 'ScipySkFeed.matrix returned on incorrect device'


# Batch
def test_scipyskfeed_batch(skf_file: str, device):
    """ScipySkFeed matrix batch operability tolerance test"""
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'overlap', device=device)

    mols = reduce(lambda i, j: i+j, molecules(device))
    orbs = OrbitalInfo(mols.atomic_numbers,
                        {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]})

    H = H_feed.matrix(mols, orbs)
    S = S_feed.matrix(mols, orbs)

    check_1 = torch.allclose(H, pack(hamiltonians(device)), atol=1E-7)
    check_2 = torch.allclose(S, pack(overlaps(device)), atol=1E-7)
    check_3 = H.device == device

    # Check that batches of size one do not cause problems
    check_4 = (H_feed.matrix(mols[0:1], orbs[0:1]).ndim == 3)

    assert check_1, 'ScipySkFeed H matrix outside of tolerance (batch)'
    assert check_2, 'ScipySkFeed S matrix outside of tolerance (batch)'
    assert check_3, 'ScipySkFeed.matrix returned on incorrect device'
    assert check_4, 'Failure to operate on batches of size "one"'

# Note that gradient tests are not performed on the ScipySkFeed as it is not
# backpropagatable due to its use of Scipy splines for interpolation.

#########################################
# tbmalt.physics.dftb.feeds.SkFeed #
#########################################
# Hamiltonian and overlap data for CH4 and H2O
H_ref_ch4 = torch.tensor(
    [[-5.048917654803000E-01, 0.000000000000000E+00, 0.000000000000000E+00,
      0.000000000000000E+00, -3.235506247119279E-01, -3.235506247119279E-01,
      -3.235506247119279E-01, -3.235506247119279E-01],
     [0.000000000000000E+00, -1.943551799182000E-01, 0.000000000000000E+00,
      0.000000000000000E+00, -1.644218947194303E-01, 1.644218947194303E-01,
      1.644218947194303E-01, -1.644218947194303E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, -1.943551799182000E-01,
      0.000000000000000E+00, -1.644218947194303E-01, -1.644218947194303E-01,
      1.644218947194303E-01, 1.644218947194303E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
      -1.943551799182000E-01, -1.644218947194303E-01, 1.644218947194303E-01,
      -1.644218947194303E-01, 1.644218947194303E-01],
     [-3.235506247119279E-01, -1.644218947194303E-01, -1.644218947194303E-01,
      -1.644218947194303E-01, -2.386005440483000E-01, -6.680097953235961E-02,
      -6.680097953235961E-02, -6.680097953235961E-02],
     [-3.235506247119279E-01, 1.644218947194303E-01, -1.644218947194303E-01,
      1.644218947194303E-01, -6.680097953235961E-02, -2.386005440483000E-01,
      -6.680097953235961E-02, -6.680097953235961E-02],
     [-3.235506247119279E-01, 1.644218947194303E-01, 1.644218947194303E-01,
      -1.644218947194303E-01, -6.680097953235961E-02, -6.680097953235961E-02,
      -2.386005440483000E-01, -6.680097953235961E-02],
     [-3.235506247119279E-01, -1.644218947194303E-01, 1.644218947194303E-01,
      1.644218947194303E-01, -6.680097953235961E-02, -6.680097953235961E-02,
      -6.680097953235961E-02, -2.386005440483000E-01]])

S_ref_ch4 = torch.tensor(
    [[1.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
      0.000000000000000E+00, 4.084679411526629E-01, 4.084679411526629E-01,
      4.084679411526629E-01, 4.084679411526629E-01],
     [0.000000000000000E+00, 1.000000000000000E+00, 0.000000000000000E+00,
      0.000000000000000E+00, 2.598016488988212E-01, -2.598016488988212E-01,
      -2.598016488988212E-01, 2.598016488988212E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, 1.000000000000000E+00,
      0.000000000000000E+00, 2.598016488988212E-01, 2.598016488988212E-01,
      -2.598016488988212E-01, -2.598016488988212E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
      1.000000000000000E+00, 2.598016488988212E-01, -2.598016488988212E-01,
      2.598016488988212E-01, -2.598016488988212E-01],
     [4.084679411526629E-01, 2.598016488988212E-01, 2.598016488988212E-01,
      2.598016488988212E-01, 1.000000000000000E+00, 8.616232232870227E-02,
      8.616232232870227E-02, 8.616232232870227E-02],
     [4.084679411526629E-01, -2.598016488988212E-01, 2.598016488988212E-01,
      -2.598016488988212E-01, 8.616232232870227E-02, 1.000000000000000E+00,
      8.616232232870227E-02, 8.616232232870227E-02],
     [4.084679411526629E-01, -2.598016488988212E-01, -2.598016488988212E-01,
      2.598016488988212E-01, 8.616232232870227E-02, 8.616232232870227E-02,
      1.000000000000000E+00, 8.616232232870227E-02],
     [4.084679411526629E-01, 2.598016488988212E-01, -2.598016488988212E-01,
      -2.598016488988212E-01, 8.616232232870227E-02, 8.616232232870227E-02,
      8.616232232870227E-02, 1.000000000000000E+00]])

H_ref_h2o = torch.tensor(
    [[-8.788325840766993E-01, 0.000000000000000E+00, 0.000000000000000E+00,
      0.000000000000000E+00, -4.882459882956572E-01, -4.882459882956572E-01],
     [0.000000000000000E+00, -3.321317735287993E-01, 0.000000000000000E+00,
      0.000000000000000E+00, -2.760418336133891E-01, 2.760418336133891E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, -3.321317735287993E-01,
      0.000000000000000E+00, 2.156680014519258E-01, 2.156680014519258E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
      -3.321317735287993E-01, 0.000000000000000E+00, 0.000000000000000E+00],
     [-4.882459882956572E-01, -2.760418336133891E-01, 2.156680014519258E-01,
      0.000000000000000E+00, -2.386005440482994E-01, -1.056313505954941E-01],
     [-4.882459882956572E-01, 2.760418336133891E-01, 2.156680014519258E-01,
      0.000000000000000E+00, -1.056313505954941E-01, -2.386005440482994E-01]])

S_ref_h2o = torch.tensor(
    [[1.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
      0.000000000000000E+00, 4.090587540889245E-01, 4.090587540889245E-01],
     [0.000000000000000E+00, 1.000000000000000E+00, 0.000000000000000E+00,
      0.000000000000000E+00, 3.165866606110211E-01, -3.165866606110211E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, 1.000000000000000E+00,
      0.000000000000000E+00, -2.473451631825645E-01, -2.473451631825645E-01],
     [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
      1.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00],
     [4.090587540889245E-01, 3.165866606110211E-01, -2.473451631825645E-01,
      0.000000000000000E+00, 1.000000000000000E+00, 1.554584894565836E-01],
     [4.090587540889245E-01, -3.165866606110211E-01, -2.473451631825645E-01,
      0.000000000000000E+00, 1.554584894565836E-01, 1.000000000000000E+00]])


def test_skfeed_single(device):
    """SkFeed matrix single system operability tolerance test"""
    species = [1, 6, 7, 8]
    path_to_file = 'example_dftb_vcr.h5'
    b_def = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

    skf_file_vcr(path_to_file)

    # Load the Hamiltonian, overlap feed model
    h_feed = SkFeed.from_database(path_to_file, species, 'hamiltonian',
                                  interpolation='bicubic')
    s_feed = SkFeed.from_database(path_to_file, species, 'overlap',
                                  interpolation='bicubic')

    # Define (wave-function) compression radii, keep s and p the same
    vcrs = [torch.tensor([2.7, 2.5, 2.5, 2.5, 2.5], device=device),
            torch.tensor([2.3, 2.5, 2.5], device=device)]
    H_ref = [H_ref_ch4, H_ref_h2o]
    S_ref = [S_ref_ch4, S_ref_h2o]

    for ii, mol in enumerate([molecule('CH4'), molecule('H2O')]):
        mol = Geometry.from_ase_atoms(mol)
        h_feed.vcr = vcrs[ii]
        s_feed.vcr = vcrs[ii]

        H = h_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))
        S = s_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))

        check_1 = torch.allclose(H, H_ref[ii], atol=2E-5)
        check_2 = torch.allclose(S, S_ref[ii], atol=1E-4)
        check_3 = H.device == device

        assert check_1, f'SkFeed H matrix outside of tolerance ({mol})'
        assert check_2, f'SkFeed S matrix outside of tolerance ({mol})'
        assert check_3, 'SkFeed.matrix returned on incorrect device'

    os.system(f'rm {path_to_file}')


# Batch
def test_skfeed_batch(device):
    """SkFeed matrix batch system operability tolerance test"""
    species = [1, 6, 7, 8]
    path_to_file = 'example_dftb_vcr.h5'
    b_def = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

    skf_file_vcr(path_to_file)

    # Load the Hamiltonian, overlap feed model
    h_feed = SkFeed.from_database(path_to_file, species, 'hamiltonian',
                                  interpolation='bicubic')
    s_feed = SkFeed.from_database(path_to_file, species, 'overlap',
                                  interpolation='bicubic')

    # Define (wave-function) compression radii, keep s and p the same
    vcrs = torch.tensor([[2.7, 2.5, 2.5, 2.5, 2.5], [2.3, 2.5, 2.5, 0, 0]],
                        device=device)
    H_ref = pack([H_ref_ch4, H_ref_h2o])
    S_ref = pack([S_ref_ch4, S_ref_h2o])

    mol = Geometry.from_ase_atoms([molecule('CH4'), molecule('H2O')])
    h_feed.vcr = vcrs
    s_feed.vcr = vcrs

    H = h_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))
    S = s_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))

    check_1 = torch.allclose(H, H_ref, atol=2E-5)
    check_2 = torch.allclose(S, S_ref, atol=1E-4)
    check_3 = H.device == device

    assert check_1, f'SkFeed H matrix outside of tolerance ({mol})'
    assert check_2, f'SkFeed S matrix outside of tolerance ({mol})'
    assert check_3, 'SkFeed.matrix returned on incorrect device'

    os.system(f'rm {path_to_file}')


###############################################
# tbmalt.physics.dftb.feeds.SkfOccupationFeed #
###############################################
# General
def test_skfoccupationfeed_general(device, skf_file):

    # Check 0: ensure that the feed can be constructed from a HDF5 skf database
    # without encountering an error.
    o_feed = SkfOccupationFeed.from_database(skf_file, [1, 6], device=device)

    # Check 1: ensure the feed is constructed on the correct device.
    check_1 = o_feed.device == device == list(o_feed.occupancies.values())[0].device
    assert check_1, 'SkfOccupationFeed has been placed on the incorrect device'

    # Check 2: verify that the '.to' method moves the feed and its contents to
    # the specified device as intended. Note that this test cannot be performed
    # if there is no other device present on the system to which a move can be
    # made.

    if torch.cuda.device_count():
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        o_feed_copy = o_feed.to(new_device)

        # In this instance the `.device` property can be trusted as it is
        # calculated ad-hoc from the only pytorch attribute.
        check_2 = o_feed_copy.device == new_device != device

        assert check_2, '".to" method failed to set the correct device'


# Single
def test_skfoccupationfeed_single(device, skf_file):
    o_feed = SkfOccupationFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    # Check 1: verify that results are returned on the correct device.
    check_1 = device == o_feed(
        OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)).device

    assert check_1, 'Results were placed on the wrong device'

    # Check 2: ensure results are within tolerance
    check_2a = torch.allclose(
        o_feed(OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)),
        torch.tensor([1., 1], device=device))

    check_2b = torch.allclose(
        o_feed(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1], device=device), shell_dict)),
        torch.tensor([2., 2/3, 2/3, 2/3, 1, 1, 1, 1], device=device))

    check_2c = torch.allclose(
        o_feed(OrbitalInfo(torch.tensor([1, 1, 8], device=device), shell_dict)),
        torch.tensor([1., 1, 2, 4/3, 4/3, 4/3], device=device))

    check_2 = check_2a and check_2b and check_2c

    assert check_2, 'Predicted occupation values errors exceed allowed tolerance'


# Batch
def test_skfoccupationfeed_batch(device, skf_file):
    o_feed = SkfOccupationFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    orbs = OrbitalInfo(torch.tensor([
        [1, 1, 0, 0, 0],
        [6, 1, 1, 1, 1],
        [1, 1, 8, 0, 0],
    ], device=device), shell_dict)

    reference = torch.tensor([
        [1, 1,   0,   0,   0,   0,   0, 0],
        [2, 2/3, 2/3, 2/3, 1,   1,   1, 1],
        [1, 1,   2,   4/3, 4/3, 4/3, 0, 0]
    ], device=device)

    predicted = o_feed(orbs)

    check_1 = predicted.device == device
    assert check_1, 'Results were placed on the wrong device'

    check_2 = torch.allclose(predicted, reference)
    assert check_2, 'Predicted occupation value errors exceed allowed tolerance'


###############################################
# tbmalt.physics.dftb.feeds.HubbardFeed #
###############################################
# General
def test_hubbardfeed_general(device, skf_file):

    # Check 0: ensure that the feed can be constructed from a HDF5 skf database
    # without encountering an error.
    u_feed = HubbardFeed.from_database(skf_file, [1, 6], device=device)

    # Check 1: ensure the feed is constructed on the correct device.
    check_1 = u_feed.device == device == list(u_feed.hubbard_u.values())[0].device
    assert check_1, 'HubbardFeed has been placed on the incorrect device'

    # Check 2: verify that the '.to' method moves the feed and its contents to
    # the specified device as intended. Note that this test cannot be performed
    # if there is no other device present on the system to which a move can be
    # made.

    if torch.cuda.device_count():
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        u_feed_copy = u_feed.to(new_device)

        # In this instance the `.device` property can be trusted as it is
        # calculated ad-hoc from the only pytorch attribute.
        check_2 = u_feed_copy.device == new_device != device

        assert check_2, '".to" method failed to set the correct device'


# Single
def test_hubbardfeed_single(device, skf_file):
    u_feed = HubbardFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    # Check 1: verify that results are returned on the correct device.
    check_1 = device == u_feed(
        OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)).device

    assert check_1, 'Results were placed on the wrong device'

    # Check 2: ensure results are within tolerance
    check_2a = torch.allclose(
        u_feed(OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)),
        torch.tensor([0.4196174261, 0.4196174261], device=device))

    check_2b = torch.allclose(
        u_feed(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1], device=device), shell_dict)),
        torch.tensor([0.3646664974, 0.4196174261, 0.4196174261,
                      0.4196174261, 0.4196174261], device=device))

    check_2c = torch.allclose(
        u_feed(OrbitalInfo(torch.tensor([1, 1, 8], device=device), shell_dict)),
        torch.tensor([0.4196174261, 0.4196174261, 0.4954041702], device=device))

    check_2 = check_2a and check_2b and check_2c

    assert check_2, 'Predicted hubbard value errors exceed allowed tolerance'


# Batch
def test_hubbardfeed_batch(device, skf_file):
    u_feed = HubbardFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    orbs = OrbitalInfo(torch.tensor([
        [1, 1, 0, 0, 0],
        [6, 1, 1, 1, 1],
        [1, 1, 8, 0, 0],
    ], device=device), shell_dict)

    reference = torch.tensor([
        [0.4196174261, 0.4196174261, 0,            0,            0],
        [0.3646664974, 0.4196174261, 0.4196174261, 0.4196174261, 0.4196174261],
        [0.4196174261, 0.4196174261, 0.4954041702, 0,            0]
    ], device=device)

    predicted = u_feed(orbs)

    check_1 = predicted.device == device
    assert check_1, 'Results were placed on the wrong device'

    check_2 = torch.allclose(predicted, reference)
    assert check_2, 'Predicted hubbard value errors exceed allowed tolerance'