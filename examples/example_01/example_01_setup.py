from os.path import join
import urllib, tempfile, tarfile

import torch

from tbmalt.io.skf import Skf
torch.set_default_dtype(torch.float64)


def skf_file(output_path: str):
    """Path to auorg-1-1 HDF5 database.

    This function downloads the auorg-1-1 Slater-Koster parameter set & converts
    it to HDF5 database stored at the path provided.

    Arguments:
         output_path: location to where the auorg-1-1 HDF5 database file should
            be stored.

    Warnings:
        This will fail without an internet connection.

    """
    # Link to the auorg-1-1 parameter set
    link = 'https://dftb.org/fileadmin/DFTB/public/slako/auorg/auorg-1-1.tar.xz'

    # Elements of interest
    elements = ['H', 'C', 'N', 'O', 'S', 'Au']

    with tempfile.TemporaryDirectory() as tmpdir:

        # Download and extract the auorg parameter set to the temporary directory
        urllib.request.urlretrieve(link, path := join(tmpdir, 'auorg-1-1.tar.xz'))
        with tarfile.open(path) as tar:
            tar.extractall(tmpdir)

        # Select the relevant skf files and place them into an HDF5 database
        skf_files = [join(tmpdir, 'auorg-1-1', f'{i}-{j}.skf')
                     for i in elements for j in elements]

        for skf_file in skf_files:
            Skf.read(skf_file).write(output_path)


# STEP 1: Inputs
parameter_db_path = "example_dftb_parameters.h5"

# STEP 2: Execution
skf_file(parameter_db_path)


