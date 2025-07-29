from os.path import exists
from typing import Any, List
import random
import torch
import h5py

from tbmalt import OrbitalInfo
from tbmalt.ml.calculator import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.io.dataset import DataSetIM


Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# ============== #
# STEP 1: Inputs #
# ============== #

# 1.1: System settings
# --------------------

# Provide a list of molecules upon which TBMaLT is to be run
size = [1000]
sources = ['run1/train', 'run2/train', 'run3/train']
targets = ['dipole']

# Provide information about the orbitals on each atom; this is keyed by atomic
# numbers and valued by azimuthal quantum numbers like so:
#   {Z₁: [ℓᵢ, ℓⱼ, ..., ℓₙ], Z₂: [ℓᵢ, ℓⱼ, ..., ℓₙ], ...}
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

# 1.2: Model settings
# -------------------
# Location at which the DFTB parameter set database is located
parameter_db_path = 'example_dftb_parameters.h5'

# Should fitting be performed here?
fit_model = True

# Number of fitting cycles, number of batch size each cycle
number_of_epochs = 10
n_batch = 300

# Location of a file storing the properties that will be fit to.
target_path = './dataset.h5'


# ============= #
# STEP 2: Setup #
# ============= #

# load data set
def load_target_data(path: str, sources: List[str], targets: List[str]) -> Any:
    """Load fitting target data.

    Arguments:
        path: path to a database in which the fitting data can be found.
        sources: a list of paths specifying the groups from which data
            should be loaded; one for each system.
        targets: paths relative to `source` specifying the HDF5 datasets
                to load.

    Returns:
        targets: returns an <OBJECT> storing the data to which the model is to
            be fitted.
    """
    # Data could be loaded from a json file or an hdf5 file; use your own
    # discretion here. A dictionary might be the best object in which to store
    # the target data.
    with h5py.File(path, 'r') as f:
        _sources = []
        for sou in sources:
            _sources.extend([sou + '/' + i for i in (f[sou].keys())])
        return DataSetIM.load_data(path, _sources, targets)


def init_model():
    raise NotImplementedError()


# 2.1: Target system specific objects
# -----------------------------------
if fit_model:
    dataloder = load_target_data(target_path, sources, targets)
else:
    raise NotImplementedError()


# 2.2: Loading of the DFTB parameters into their associated feed objects
# ----------------------------------------------------------------------
# Construct the `Geometry` and `OrbitalInfo` objects. The former is analogous to the
# ase.Atoms object while the latter provides information about what orbitals
# are present and which atoms they belong two. `OrbitalInfo` is perhaps a poor choice
# of name and `OrbitalInfo` would be more appropriate.
# geometry = Geometry(atomic_numbers, positions, units='a')
# orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)

# Construct the Hamiltonian and overlap matrix feeds; but ensure that the DFTB
# parameter set database actually exists first.
if not exists(parameter_db_path):
    raise FileNotFoundError(
        f'The DFTB parameter set database "{parameter_db_path}" could '
        f'not be found, please ensure "example_01_setup.py" has been run.')

# Identify which species are present
species = torch.tensor([1, 6, 7, 8])
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# Load the Hamiltonian feed model
h_feed = SkFeed.from_database(
    parameter_db_path, species, 'hamiltonian', interpolation=CubicSpline)

# Load the overlap feed model
s_feed = SkFeed.from_database(
    parameter_db_path, species, 'overlap', interpolation=CubicSpline)

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)

# 2.3: Construct the SCC-DFTB calculator object
# ---------------------------------------------
# As this is a minimal working example, no optional settings are provided to the
# calculator object.
dftb_calculator = Dftb2(
    h_feed, s_feed, o_feed, u_feed, filling_scheme=None,
    filling_temp=None, grad_mode="direct")

# Construct machine learning object
lr = 0.003
criterion = getattr(torch.nn, 'MSELoss')(reduction='mean')
h_var, s_var = [], []
for key in h_feed._off_sites.keys():

    # Collect spline parameters and add to optimizer
    h_feed._off_sites[key].coefficients.requires_grad_(True)
    s_feed._off_sites[key].coefficients.requires_grad_(True)
    h_var.append({'params': h_feed._off_sites[key].coefficients, 'lr': lr})
    s_var.append({'params': s_feed._off_sites[key].coefficients, 'lr': lr})

optimizer = torch.optim.Adam(h_var + s_var, lr=lr)


# ================= #
# STEP 3: Execution #
# ================= #
def calculate_losses(calculator: Calculator, data: Any) -> Tensor:
    """An example function computing the loss of the model.

    Args:
        calculator: calculator object via which target properties can be
            calculated.
        targets: target data to which the model should be fitted.

    Returns:
        loss: the computed loss.

    """
    loss = 0.0

    for key in targets:
        key = 'q_final_atomic' if key == 'charge' else key
        loss += criterion(calculator.__getattribute__(key), data.data[key])

    return loss


def update_model(calculator: Calculator):
    """Update the model feed objects.

    Arguments:
        calculator: calculator object containing the feeds that are to be
            updated.
    """
    raise NotImplementedError()


if fit_model:
    random_idx = random.sample(torch.arange(len(dataloder)).tolist(), len(dataloder))
    indice = torch.split(torch.tensor(random_idx), n_batch)

    for epoch in range(number_of_epochs):

        data = dataloder[indice[epoch % len(indice)]]
        orbs = OrbitalInfo(data.geometry.atomic_numbers, shell_dict, shell_resolved=False)

        # Perform the forwards operation
        dftb_calculator(data.geometry, orbs)

        # Calculate the loss
        loss = calculate_losses(dftb_calculator, data)
        print(loss)

        optimizer.zero_grad()

        # Invoke the autograd engine
        loss.backward()

        # Update the model
        optimizer.step()
else:
    # Run the DFTB calculation
    raise NotImplementedError()
