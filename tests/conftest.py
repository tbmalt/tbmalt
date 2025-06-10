import pytest
import torch
from os.path import join
import urllib, tarfile
from tbmalt.io.skf import Skf, VCRSkf
from tbmalt.tools.downloaders import download_dftb_parameter_set
# Default must be set to float64 otherwise gradcheck will not function
torch.set_default_dtype(torch.float64)


def pytest_addoption(parser):
    """Set up command line options."""
    parser.addoption(  # Enable device selection
        "--device", action="store", default="cpu",
        help="specify test device (cpu/cuda/etc/...)"
    )

    parser.addoption(  # Should more comprehensive gradient test be performed?
        "--detect-anomaly", action='store_true',
        help='this flag enables more comprehensive, but time consuming, '
             'gradient tests.'
    )


@pytest.fixture(scope='session')
def device(request) -> torch.device:
    """Defines the device on which each test should be run.

    Returns:
        device: The device on which the test will be run.

    """
    # Device checks require CPU to be specified *without* a device number and
    # cuda to be specified *with* one.
    device_name = request.config.getoption("--device")
    if device_name == 'cuda':
        return torch.device('cuda:0')
    else:
        return torch.device(device_name)


@pytest.fixture(scope='session')
def skf_file(tmpdir_factory):
    """Path to auorg-1-1 HDF5 database.

    This fixture downloads the auorg-1-1 Slater-Koster parameter set, converts
    it to HDF5, and returns the path to the resulting database.

    Returns:
         path: location of auorg-1-1 HDF5 database file.

    Warnings:
        This will fail i) without an internet connection, ii) if the auorg-1-1
        parameter sets moves, or iii) it is used outside a PyTest session.

    Notes:
        This fixture just wraps the DFTB parameter set downloader method
        `tbmalt.tools.downloaders.download_dftb_parameter_set` so that a
        parameter set can be downloaded when running the unit tests.
        The parameter set will be deleted after the tests have run.
    """
    tempdir = tmpdir_factory.mktemp('tmp')
    parameter_set_file_path = join(tempdir, "auorg.h5")
    download_dftb_parameter_set(
        "https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz",
        parameter_set_file_path)

    return parameter_set_file_path


@pytest.fixture(scope='session')
def skf_file_siband(tmpdir_factory):
    """Path to siband-1-1 HDF5 database.

    This fixture downloads the siband-1-1  Slater-Koster parameter set, converts
    it to HDF5, and returns the path to the resulting database.

    Returns:
         path: location of siband-1-1 HDF5 database file.

    Warnings:
        This will fail i) without an internet connection, ii) if the siband-1-1
        parameter sets moves, or iii) it is used outside a PyTest session.

    Notes:
        This fixture just wraps the DFTB parameter set downloader method
        `tbmalt.tools.downloaders.download_dftb_parameter_set` so that a
        parameter set can be downloaded when running the unit tests.
        The parameter set will be deleted after the tests have run.
    """
    tempdir = tmpdir_factory.mktemp('tmp')
    parameter_set_file_path = join(tempdir, "siband.h5")
    download_dftb_parameter_set(
        "https://github.com/dftbparams/siband/releases/download/v1.1.0/siband-1-1.tar.xz",
        parameter_set_file_path)

    return parameter_set_file_path


@pytest.fixture(scope='session')
def skf_file_pbc(tmpdir_factory):
    """Path to pbc-0-3 HDF5 database.

    This fixture downloads the pbc-0-3  Slater-Koster parameter set, converts
    it to HDF5, and returns the path to the resulting database.

    Returns:
         path: location of pbc-0-3 HDF5 database file.

    Warnings:
        This will fail i) without an internet connection, ii) if the pbc-0-3
        parameter sets moves, or iii) it is used outside a PyTest session.

    Notes:
        This fixture just wraps the DFTB parameter set downloader method
        `tbmalt.tools.downloaders.download_dftb_parameter_set` so that a
        parameter set can be downloaded when running the unit tests.
        The parameter set will be deleted after the tests have run.
    """
    tempdir = tmpdir_factory.mktemp('tmp')
    parameter_set_file_path = join(tempdir, "pbc.h5")
    download_dftb_parameter_set(
        "https://github.com/dftbparams/pbc/releases/download/v0.3.0/pbc-0-3.tar.xz",
        parameter_set_file_path)

    return parameter_set_file_path


@pytest.fixture(scope='session')
def skf_file_vcr(tmpdir_factory):
    """Path to Slater-Koster files.

    This function downloads the Slater-Koster parameter set & converts it to
     HDF5 database stored at the path provided. Slater-Koster parameter set of
     each atom has different compression radii.

    Arguments:
         path: location of auorg-1-1 HDF5 database file.

    Warnings:
        This will fail without an internet connection.

    """
    tempdir = tmpdir_factory.mktemp('tmp')

    # Link to the VCR Slater-Koster parameter set
    link = 'https://zenodo.org/record/8109578/files/compr_wav.tar.gz?download=1'

    # Elements of interest and compression radii grids
    elements = ['H', 'C', 'N', 'O']
    compr = ['01.00', '01.50', '02.00', '02.50', '03.00', '03.50', '04.00',
             '04.50', '05.00', '06.00', '08.00', '10.00']

    # Download and extract the auorg parameter set to the temporary directory
    urllib.request.urlretrieve(link, location := join(tempdir, 'compr_wav.tar.gz'))

    with tarfile.open(location, 'r:gz') as tar:
        tar.extractall(tempdir)

    # Read all Slater-Koster parameter sets
    [join(tempdir, 'compr_wav', f'{i}-{j}.skf.{ic}.{jc}')
     for i in elements for j in elements for ic in compr for jc in compr]
    path = join(tempdir, "compr_wav.hdf5")
    VCRSkf.from_dir(join(tempdir, 'compr_wav'), path)

    return path


def pytest_configure(config):
    """Pytest configuration hook."""
    # Check if the "--detect-anomaly" flag was passed, if so then turn on
    # autograd anomaly detection.
    if config.getoption('--detect-anomaly'):
        torch.autograd.set_detect_anomaly(True)


def __sft(self):
    """Alias for calling .cpu().numpy()"""
    return self.cpu().numpy()


torch.Tensor.sft = __sft
