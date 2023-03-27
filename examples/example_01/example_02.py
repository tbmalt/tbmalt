"""Training on small molecules from data set."""
import pickle
from os.path import exists
from typing import Any, List

import torch
import h5py

from tbmalt import Geometry, Basis
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.io.dataset import DataSetIM

from ase.build import molecule

Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# ============== #
# STEP 1: Inputs #
# ============== #

# 1.1: System settings
# --------------------

# Provide a list of moecules upon which TBMaLT is to be run
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
test_model = True

# Number of fitting cycles, number of batch size each cycle
number_of_epochs = 120
n_batch = [1000, 1000, 1000]  # Batch size of three fitting run
lr = 0.003
onsite_lr = 3e-4
tolerance = 1e-6
criterion = getattr(torch.nn, 'MSELoss')(reduction='mean')
shell_resolved = False
# Location of a file storing the properties that will be fit to.
target_path = './dataset.h5'


# ============= #
# STEP 2: Setup #
# ============= #

# load data set
def load_target_data(path: str, properties: List, groups) -> Any:
    """Load fitting target data.

    Arguments:
        molecules: Molecules for which fitting targets should be returned.
        path: path to a database in which the fitting data can be found.

    Returns:
        targets: returns an <OBJECT> storing the data to which the model is to
            be fitted.
    """
    # Data could be loaded from a json file or an hdf5 file; use your own
    # discretion here. A dictionary might be the best object in which to store
    # the target data.
    return DataSetIM.load_reference(groups, properties)


def init_model():
    raise NotImplementedError()


# 2.1: Target system specific objects
# -----------------------------------
if fit_model or test_model:
    with h5py.File(target_path, 'r') as f:
        dataloder_fit = [load_target_data(target_path, targets, g) for g in
                         [f['run1']['train'], f['run2']['train'], f['run3']['train']]]
        dataloder_test = [load_target_data(target_path, targets, g) for g in
                          [f['run1']['test'], f['run2']['test'], f['run3']['test']]]


# 2.2: Loading of the DFTB parameters into their associated feed objects
# ----------------------------------------------------------------------
# Construct the `Geometry` and `Basis` objects. The former is analogous to the
# ase.Atoms object while the latter provides information about what orbitals
# are present and which atoms they belong two. `Basis` is perhaps a poor choice
# of name and `OrbitalInfo` would be more appropriate.
# geometry = Geometry(atomic_numbers, positions, units='a')
# basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

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
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=CubicSpline)

# Load the overlap feed model
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap',
                              interpolation=CubicSpline)

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)

# 2.3: Construct the SCC-DFTB calculator object
# ---------------------------------------------
# As this is a minimal working example, no optional settings are provided to the
# calculator object.
dftb_calculator_init = Dftb2(h_feed, s_feed, o_feed, u_feed)


# Construct machine learning object
def build_optim(dftb_calculator):
    h_var, s_var = [], []

    for key in dftb_calculator.h_feed.off_sites.keys():

        # Collect spline parameters and add to optimizer
        dftb_calculator.h_feed.off_sites[key].abcd.requires_grad_(True)
        dftb_calculator.s_feed.off_sites[key].abcd.requires_grad_(True)
        h_var.append({'params': dftb_calculator.h_feed.off_sites[key].abcd, 'lr': lr})
        s_var.append({'params': dftb_calculator.s_feed.off_sites[key].abcd, 'lr': lr})

    ml_onsite, onsite_dict = [], {}
    for key, val in dftb_calculator.h_feed.on_sites.items():
        for l in shell_dict[key]:
            onsite_dict.update({(key, l): val[int(l ** 2)].requires_grad_(True)})
            ml_onsite.append({'params': onsite_dict[(key, l)], 'lr': onsite_lr})

    optimizer = getattr(torch.optim, 'Adam')(h_var + s_var + ml_onsite, lr=lr)
    return optimizer, onsite_dict


with open('dftb_calculator_init.pkl', 'wb') as w:
    pickle.dump(dftb_calculator_init, w)

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
        loss += criterion(calculator.__getattribute__(key), data[key])

    return loss


def update_model(calculator: Calculator):
    """Update the model feed objects.

    Arguments:
        calculator: calculator object containing the feeds that are to be
            updated.
    """
    raise NotImplementedError()


def single_fit(dftb_calculator, dataloder, size):
    indice = torch.split(torch.tensor(dataloder.random_idx), size)
    optimizer, onsite_dict = build_optim(dftb_calculator)
    loss_old = 0

    for epoch in range(number_of_epochs):

        data = dataloder[indice[epoch % len(indice)]]

        geometry = Geometry(data['atomic_numbers'], data['positions'], units='a')
        basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

        if not shell_resolved:
            dftb_calculator.h_feed.on_sites = {
                iatm: torch.cat([onsite_dict[(iatm, l)].repeat(2 * l + 1).T
                                 for l in shell_dict[iatm]], -1)
                for iatm in geometry.unique_atomic_numbers().tolist()}

        # Perform the forwards operation
        dftb_calculator(geometry, basis)

        # Calculate the loss
        loss = calculate_losses(dftb_calculator, data)
        print(epoch, loss)

        optimizer.zero_grad()

        # Invoke the autograd engine
        loss.backward()

        # Update the model
        optimizer.step()

        if torch.abs(loss_old - loss.detach()).lt(tolerance):
            break
        loss_old = loss.detach().clone()

        # Reset the calculator
        # dftb_calculator.reset()

    return dftb_calculator


def single_test(dftb_calculator, dftb_calculator_init, dataloder):

    geometry = Geometry(dataloder.dataset['atomic_numbers'],
                        dataloder.dataset['positions'], units='a')
    basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

    # Perform DFTB calculations
    dftb_calculator_init(geometry, basis)
    dftb_calculator(geometry, basis)


# STEP 3.1: Execution fitting
if fit_model:
    for ii, dataloder in enumerate(dataloder_fit):
        assert len(n_batch) == len(dataloder_fit), 'size and Nr. fit run inconsistent'

        with open('dftb_calculator_init.pkl', 'rb') as r:
            dftb_calculator_init = pickle.load(r)

        dftb_calculator = single_fit(dftb_calculator_init, dataloder, n_batch[ii])

        with open(f'dftb_calculator_{ii}.pkl', 'wb') as w:
            pickle.dump(dftb_calculator, w)


# STEP 3.2: Execution testing
if test_model:
    for ii, dataloder in enumerate(dataloder_test):
        assert len(n_batch) == len(dataloder_test), 'size and Nr. fit run inconsistent'

        with open('dftb_calculator_init.pkl', 'rb') as r:
            dftb_calculator_init = pickle.load(r)

        with open(f'dftb_calculator_{ii}.pkl', 'rb') as r:
            dftb_calculator = pickle.load(r)

        single_test(dftb_calculator, dftb_calculator_init, dataloder)

# spl: [0.1331, 0.1340, 0.1356], [0.0175, 0.0170, 0.0172]
