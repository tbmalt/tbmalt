from os.path import join
import urllib, tempfile, tarfile

import torch

from tbmalt.io.skf import VCRSkf
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
    link = 'https://seafile.zfn.uni-bremen.de/f/82656301e2bb4d4a8d77/?dl=1'

    # Elements of interest
    elements = ['H', 'C', 'N', 'O']
    compr = ['01.00', '01.50', '02.00', '02.50', '03.00', '03.50', '04.00',
             '04.50', '05.00', '06.00', '08.00', '10.00']

    with tempfile.TemporaryDirectory() as tmpdir:

        # Download and extract the auorg parameter set to the temporary directory
        urllib.request.urlretrieve(link, path := join(tmpdir, 'compr_wav.tar.gz'))

        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(tmpdir)

        # Select the relevant skf files and place them into an HDF5 database
        skf_files = [join(tmpdir, 'compr_wav', f'{i}-{j}.skf.{ic}.{jc}')
                     for i in elements for j in elements
                     for ic in compr for jc in compr]

        # for skf_file in skf_files:
        VCRSkf.from_dir(join(tmpdir, 'compr_wav'), output_path)


# STEP 1: Inputs
parameter_db_path = "example_dftb_vcr.h5"

# STEP 2: Execution
skf_file(parameter_db_path)


