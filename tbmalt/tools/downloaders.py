from os.path import join

from tbmalt.io.skf import Skf
import torch
import os
import re
import tarfile
import tempfile

from urllib.request import urlopen

torch.set_default_dtype(torch.float64)


def download_dftb_parameter_set(
        url: str, output_path: str):
    """Download a Slater-Koster parameter set.

    This function is intended to help users quickly download and make use of
    the various density functional tight binding parameters sets offered
    by the DFTB organisation (https://github.com/dftbparams).

    When provided with a link to one of the offered parameter sets, this
    function will download, and extract the archived. Locate the Slater-
    Koster files, and then parse them into a TBMaLT compatible HDF5 formatted
    file.

    To get the URL address for a parameter set users should i) navigate to
    https://github.com/dftbparams, ii) select the repository corresponding to
    the desired parameter set, iii) navigate to the "Releases" tab, iv) select
    a release, and v) copy the URL for the "<PARAMETER_SET_NAME>.tar.xy" file
    listed under the "Assets" section. This should look something like this:
    https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz

    Arguments:
        url: The URL from which the archive should be downloaded.
        output_path: The location to which the HDF5 structured parameter set
            should be written.

    Examples:
        >>> from tbmalt.tools.downloaders import download_dftb_parameter_set
        >>> from tbmalt.io.skf import Skf
        >>> import matplotlib.pyplot as plt
        >>> parameter_url = "https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz"
        >>> file_path = "auorg.h5"
        >>> download_dftb_parameter_set(parameter_url, file_path)
        >>> hh_skf = Skf.read(file_path, (1, 1))
        >>> plt.plot(hh_skf.grid, hh_skf.overlap[(0, 0)][0])
        >>> plt.show()
    """
    # Raise an exception early if performing the operation would
    # overwrite an existing file.
    if os.path.exists(output_path):
        raise FileExistsError(
            f"A file already exists at this path: {output_path}")

    regex = re.compile(r"[A-Z][a-z]?-[A-Z][a-z]?\.skf")

    # A temporary directory is used to ensure that intermediate files get
    # deleted once they are no longer needed.
    with tempfile.TemporaryDirectory() as tmpdir:

        # The archive containing the desired parameter set is downloaded from
        # the specified address and extracted into the temporary directory.
        local_archive_path = os.path.join(tmpdir, "archive.tar.xz")

        with urlopen(url) as response:
            with open(local_archive_path, "wb") as f:
                f.write(response.read())

        with tarfile.open(local_archive_path, mode="r:xz") as tar:
            tar.extractall(tmpdir)

        # Locate all skf files matching the expected naming convention.
        skf_files = [join(root, file) for root, _, files in os.walk(tmpdir)
                     for file in files if regex.fullmatch(file)]

        # Each matching SKF file is parsed and written into the HDF5 formatted
        # file specified by the output path.
        for skf_file in skf_files:
            Skf.read(skf_file).write(output_path)

