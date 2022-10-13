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

# periodic test

#########################################
# tbmalt.physics.dftb.feeds.ScipySkFeed #
#########################################


def test_scipyskfeed_periodic_ch4_single(device, skf_file):
    """ScipySkFeed matrix single system operability tolerance test"""
    shell_dict = {1: [0], 6: [0, 1]}
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6], 'overlap', device=device)
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    positions = torch.tensor([[3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
                              [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    cells = torch.tensor([[4., 4., 0.], [0., 5., 0.], [0., 0., 6.]],
                         device=device)
    geo = Geometry(atomic_numbers, positions, cells, units='a')
    cutoff = torch.tensor([9.98])
    periodic = Periodic(geo, geo.cells, cutoff=cutoff)
    H = H_feed.matrix(geo, Basis(geo.atomic_numbers, shell_dict), periodic)
    S = S_feed.matrix(geo, Basis(geo.atomic_numbers, shell_dict), periodic)
    check_1 = torch.allclose(H, H_CH4_ref, atol=1E-4)
    check_2 = torch.allclose(S, S_CH4_ref, atol=1E-4)
    assert check_1, 'ScipySkFeed H matrix outside of tolerance.'
    assert check_2, 'ScipySkFeed S matrix outside of tolerance.'


def test_skffeed_periodic_ch4_single(device, skf_file):
    """SkFeed matrix single system operability tolerance test"""
    shell_dict = {1: [0], 6: [0, 1]}
    H_feed = SkFeed.from_database(
        skf_file, [1, 6], 'hamiltonian', device=device)
    S_feed = SkFeed.from_database(
        skf_file, [1, 6], 'overlap', device=device)
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    positions = torch.tensor([[3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
                              [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    cells = torch.tensor([[4., 4., 0.], [0., 5., 0.], [0., 0., 6.]],
                         device=device)
    geo = Geometry(atomic_numbers, positions, cells, units='a')
    cutoff = torch.tensor([9.98])
    periodic = Periodic(geo, geo.cells, cutoff=cutoff)
    H = H_feed.matrix(geo, Basis(geo.atomic_numbers, shell_dict), periodic)
    S = S_feed.matrix(geo, Basis(geo.atomic_numbers, shell_dict), periodic)
    check_1 = torch.allclose(H, H_CH4_ref, atol=1E-9)
    check_2 = torch.allclose(S, S_CH4_ref, atol=1E-9)
    assert check_1, 'SkFeed H matrix outside of tolerance.'
    assert check_2, 'SkFeed S matrix outside of tolerance.'


H_CH4_ref = torch.tensor(
       [[-5.051168718302566640e-01,  1.465366998749940068e-19,
          0.000000000000000000e+00, -4.633270221481022855e-19,
         -3.543467491831184812e-01, -3.552731186921234130e-01,
         -3.552731186921234130e-01, -3.543467491831184257e-01],
        [-1.465366998749940068e-19, -1.942936797945294114e-01,
          0.000000000000000000e+00, -2.446615944544236953e-04,
         -1.711060541438707006e-01, -1.705300217006210928e-01,
          1.705300217006213981e-01, -1.711060541438705895e-01],
        [ 0.000000000000000000e+00,  0.000000000000000000e+00,
         -1.944182900442237016e-01,  0.000000000000000000e+00,
         -1.710388914972194863e-01, -1.713041647154286973e-01,
         -1.713041647154286973e-01,  1.710388914972195973e-01],
        [ 4.633270221481022855e-19, -2.446615944544236953e-04,
          0.000000000000000000e+00, -1.934379186795915917e-01,
         -1.701979019501085877e-01,  1.686404485847345924e-01,
         -1.686404485847342871e-01, -1.701979019501085044e-01],
        [-3.543467491831184812e-01, -1.711060541438707006e-01,
         -1.710388914972194863e-01, -1.701979019501085877e-01,
         -2.388853960937410981e-01, -1.865983248258180904e-01,
         -1.819001134203488135e-01, -1.813412148975178939e-01],
        [-3.552731186921234130e-01, -1.705300217006210928e-01,
         -1.713041647154286973e-01,  1.686404485847345924e-01,
         -1.865983248258180904e-01, -2.388853960937410981e-01,
         -9.343856924695721766e-02, -8.763494734368823535e-02],
        [-3.552731186921234130e-01,  1.705300217006213981e-01,
         -1.713041647154286973e-01, -1.686404485847342871e-01,
         -1.819001134203488135e-01, -9.343856924695721766e-02,
         -2.388853960937410981e-01, -8.520135062508535362e-02],
        [-3.543467491831184257e-01, -1.711060541438705895e-01,
          1.710388914972195973e-01, -1.701979019501085044e-01,
         -1.813412148975178939e-01, -8.763494734368823535e-02,
         -8.520135062508535362e-02, -2.388853960937410981e-01]])

S_CH4_ref = torch.tensor(
       [[ 1.000031443559250999e+00, -1.406074692442139044e-19,
          0.000000000000000000e+00,  1.711006553453686926e-19,
          4.600293233645404989e-01,  4.606513101978418900e-01,
          4.606513101978417790e-01,  4.600293233645402768e-01],
        [ 1.406074692442139044e-19,  9.999069391184520761e-01,
          0.000000000000000000e+00,  1.536226648837873097e-04,
          2.695749912031358275e-01,  2.692069272263438173e-01,
         -2.692069272263442059e-01,  2.695749912031356055e-01],
        [ 0.000000000000000000e+00,  0.000000000000000000e+00,
          1.000038338137340999e+00,  0.000000000000000000e+00,
          2.696612624178841844e-01,  2.698221245781374789e-01,
          2.698221245781374789e-01, -2.696612624178841844e-01],
        [-1.711006553453686926e-19,  1.536226648837873097e-04,
          0.000000000000000000e+00,  9.994355633863640787e-01,
          2.690818161696568178e-01, -2.680693457888900233e-01,
          2.680693457888895237e-01,  2.690818161696567068e-01],
        [ 4.600293233645404989e-01,  2.695749912031358275e-01,
          2.696612624178841844e-01,  2.690818161696568178e-01,
          1.000085370448456024e+00,  3.440968651637770770e-01,
          3.403802412755939089e-01,  3.397225432074815199e-01],
        [ 4.606513101978418900e-01,  2.692069272263438173e-01,
          2.698221245781374789e-01, -2.680693457888900233e-01,
          3.440968651637770770e-01,  1.000085370448456024e+00,
          1.422093873117618867e-01,  1.361860077928115920e-01],
        [ 4.606513101978417790e-01, -2.692069272263442059e-01,
          2.698221245781374789e-01,  2.680693457888895237e-01,
          3.403802412755939089e-01,  1.422093873117618867e-01,
          1.000085370448456024e+00,  1.345064648732615109e-01],
        [ 4.600293233645402768e-01,  2.695749912031356055e-01,
         -2.696612624178841844e-01,  2.690818161696567068e-01,
          3.397225432074815199e-01,  1.361860077928115920e-01,
          1.345064648732615109e-01,  1.000085370448456024e+00]])
