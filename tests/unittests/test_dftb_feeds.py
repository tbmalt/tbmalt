from os.path import join, dirname
import pytest
import urllib
import torch
import tarfile
from typing import List
import numpy as np
from tbmalt.io.skf import Skf
from tbmalt.physics.dftb.feeds import ScipySkFeed
from tbmalt import Geometry, Basis
from tbmalt.common.batch import pack
from functools import reduce


@pytest.fixture
def skf_file(tmpdir):
    """Path to auorg-1-1 HDF5 database.

    This fixture downloads the auorg-1-1 Slater-Koster parameter set, converts
    it to HDF5, and returns the path to the resulting database.

    Returns:
         path: location of auorg-1-1 HDF5 database file.

    Warnings:
        This will fail i) without an internet connection, ii) if the auorg-1-1
        parameter sets moves, or iii) it is used outside of a PyTest session.

    """
    # Link to the auorg-1-1 parameter set
    link = 'https://dftb.org/fileadmin/DFTB/public/slako/auorg/auorg-1-1.tar.xz'

    # Elements of interest
    elements = ['H', 'C', 'Au', 'S']

    # Download and extract the auorg parameter set to the temporary directory
    urllib.request.urlretrieve(link, path := join(tmpdir, 'auorg-1-1.tar.xz'))
    with tarfile.open(path) as tar:
        tar.extractall(tmpdir)

    # Select the relevant skf files and place them into an HDF5 database
    skf_files = [join(tmpdir, 'auorg-1-1', f'{i}-{j}.skf')
                 for i in elements for j in elements]

    for skf_file in skf_files:
        Skf.read(skf_file).write(path := join(tmpdir, 'auorg.hdf5'))

    return path


# @pytest.fixture
def molecules(device) -> List[Geometry]:
    """Returns a selection of `Geometry` entities for testing.

    Currently returned systems are H2, CH4, and C2H2Au2S3. The last of which
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
    # join(dirname(tbmalt.__file__), '..', ''
    # path = 'tests/unittests/data/skfeed'
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_H.csv'), delimiter=','),
            device=device))
    return matrices


def overlaps(device):
    matrices = []
    # path = 'tests/unittests/data/skfeed'
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_S.csv'), delimiter=','),
            device=device))
    return matrices


#########################################
# tbmalt.physics.dftb.feeds.ScipySkFeed #
#########################################
# General

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
        H = H_feed.matrix(mol, Basis(mol.atomic_numbers, b_def))
        S = S_feed.matrix(mol, Basis(mol.atomic_numbers, b_def))

        check_1 = torch.allclose(H, H_ref, atol=1E-7)
        check_2 = torch.allclose(S, S_ref, atol=1E-7)
        check_3 = H.device == device

        assert check_1, f'ScipySkFeed H matrix outside of tolerance ({mol})'
        assert check_2, f'ScipySkFeed S matrix outside of tolerance ({mol})'
        assert check_3, 'ScipySkFeed.matrix returned on incorrect device'


def test_scipyskfeed_batch(skf_file:str, device):
    """ScipySkFeed matrix batch operability tolerance test"""
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'hamiltonian',device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'overlap', device=device)

    mols = reduce(lambda i, j: i+j, molecules(device))
    basis = Basis(mols.atomic_numbers,
                  {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]})

    H = H_feed.matrix(mols, basis)
    S = S_feed.matrix(mols, basis)

    check_1 = torch.allclose(H, pack(hamiltonians(device)), atol=1E-7)
    check_2 = torch.allclose(S, pack(overlaps(device)), atol=1E-7)
    check_3 = H.device == device

    # Check that batches of size one do not cause problems
    check_4 = (H_feed.matrix(mols[0:1], basis[0:1]).ndim == 3)

    assert check_1, 'ScipySkFeed H matrix outside of tolerance (batch)'
    assert check_2, 'ScipySkFeed S matrix outside of tolerance (batch)'
    assert check_3, 'ScipySkFeed.matrix returned on incorrect device'
    assert check_4, 'Failure to operate on batches of size "one"'

# Note that gradient tests are not performed on the ScipySkFeed as it is not
# backpropagatable due to its use of Scipy splines for interpolation.
