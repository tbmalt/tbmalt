# -*- coding: utf-8 -*-
r"""This contains a collection of general utilities used then running PyTests.

This module should be imported as:
    from tbmalt.tests.test_utils import *

This ensures that the default dtype and autograd anomaly detection settings
are all inherited.
"""

from os.path import join
import urllib, tempfile, tarfile

import numpy as np
import torch
import functools
import pytest

from tbmalt.io.skf import Skf, VCRSkf


def fix_seed(func):
    """Sets torch's & numpy's random number generator seed.

    Fixing the random number generator's seed maintains consistency between
    tests by ensuring that the same test data is used every time. If this is
    not done it can make debugging problems very difficult.

    Arguments:
        func (function):
            The function that is being wrapped.

    Returns:
        wrapped (function):
            The wrapped function.
    """
    # Use functools.wraps to maintain the original function's docstring
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # Set both numpy's and pytorch's seed to zero
        np.random.seed(0)
        torch.manual_seed(0)

        # Call the function and return its result
        return func(*args, **kwargs)

    # Return the wapped function
    return wrapper


def clean_zero_padding(m, sizes):
    """Removes perturbations induced in the zero padding values by gradcheck.

    When performing gradient stability tests via PyTorch's gradcheck function
    small perturbations are induced in the input data. However, problems are
    encountered when these perturbations occur in the padding values. These
    values should always be zero, and so the test is not truly representative.
    Furthermore, this can even prevent certain tests from running. Thus, this
    function serves to remove such perturbations in a gradient safe manner.

    Note that this is intended to operate on 3D matrices where. Specifically a
    batch of square matrices.

    Arguments:
        m (torch.Tensor):
            The tensor whose padding is to be cleaned.
        sizes (torch.Tensor):
            The true sizes of the tensors.

    Returns:
        cleaned (torch.Tensor):
            Cleaned tensor.

    Notes:
        This is only intended for 2D matrices packed into a 3D tensor.
    """

    # Identify the device
    device = m.device

    # First identify the maximum tensor size
    max_size = torch.max(sizes)

    # Build a mask that is True anywhere that the tensor should be zero, i.e.
    # True for regions of the tensor that should be zero padded.
    mask_1d = (
            (torch.arange(max_size, device=device) - sizes.unsqueeze(1)) >= 0
    ).repeat(max_size, 1, 1)

    # This, rather round about, approach to generating and applying the masks
    # must be used as some PyTorch operations like masked_scatter do not seem
    # to function correctly
    mask_full = torch.zeros(*m.shape, device=device).bool()
    mask_full[mask_1d.permute(1, 2, 0)] = True
    mask_full[mask_1d.transpose(0, 1)] = True

    # Create and apply the subtraction mask
    temp = torch.zeros_like(m, device=device)
    temp[mask_full] = m[mask_full]
    cleaned = m - temp

    return cleaned


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
    elements = ['H', 'C', 'O', 'Au', 'S']

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


def skf_file_vcr(output_path: str):
    """Path to Slater-Koster files.

    This function downloads the Slater-Koster parameter set & converts it to
     HDF5 database stored at the path provided. Slater-Koster parameter set of
     each atom has different compression radii.

    Arguments:
         output_path: location to where the database file should be stored.

    Warnings:
        This will fail without an internet connection.

    """
    # Link to the VCR Slater-Koster parameter set
    link = 'https://zenodo.org/record/8109578/files/compr_wav.tar.gz?download=1'

    # Elements of interest and compression radii grids
    elements = ['H', 'C', 'N', 'O']
    compr = ['01.00', '01.50', '02.00', '02.50', '03.00', '03.50', '04.00',
             '04.50', '05.00', '06.00', '08.00', '10.00']

    with tempfile.TemporaryDirectory() as tmpdir:

        # Download and extract the auorg parameter set to the temporary directory
        urllib.request.urlretrieve(link, path := join(tmpdir, 'compr_wav.tar.gz'))

        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(tmpdir)

        # Read all Slater-Koster parameter sets
        [join(tmpdir, 'compr_wav', f'{i}-{j}.skf.{ic}.{jc}')
         for i in elements for j in elements for ic in compr for jc in compr]
        VCRSkf.from_dir(join(tmpdir, 'compr_wav'), output_path)
