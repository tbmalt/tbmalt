from os.path import exists, join
import urllib, tempfile, zipfile
import shutil

from tbmalt.tools.downloaders import download_dftb_parameter_set


# Location at which the DFTB parameter set database is located
parameter_db_path = 'example_dftb_parameters.h5'

# Ensure that the DFTB parameter set database actually exists first.
if not exists(parameter_db_path):
    download_dftb_parameter_set(
        "https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz",
        parameter_db_path)

# Link to the training data
link = 'https://zenodo.org/records/15592694/files/tbmalt_data.zip?download=1'

# Download the data
with tempfile.TemporaryDirectory() as tmpdir:
    output_file = join(tmpdir, 'data.zip')
    req = urllib.request.Request(link)
    with urllib.request.urlopen(req) as response, open(output_file, 'wb') as out_file:
        out_file.write(response.read())

    # Extract the data
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(tmpdir)

    shutil.copyfile(join(tmpdir, 'data', 'dataset.h5'), 'dataset.h5')
