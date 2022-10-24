from os.path import exists
from typing import List, Dict, Any
import torch
from tbmalt import Geometry, Basis
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import ScipySkFeed, SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import CubicSpline

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
# molecule_names = ['H2', 'CH4', 'C2H4', 'H2O']
molecule_names = ['CH4', 'H2O']
targets = {'q_final_atomic': torch.tensor(
    [[4.251914, 0.937022, 0.937022, 0.937022, 0.937022],
     [6.526248, 0.736876, 0.736876, 0, 0]])}

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

# Number of fitting cycles
number_of_epochs = 10

# Location of a file storing the properties that will be fit to.
target_path = 'target_data.json'

# ============= #
# STEP 2: Setup #
# ============= #

# 2.1: Target system specific objects
# -----------------------------------

# Construct the `Geometry` and `Basis` objects. The former is analogous to the
# ase.Atoms object while the latter provides information about what orbitals
# are present and which atoms they belong two. `Basis` is perhaps a poor choice
# of name and `OrbitalInfo` would be more appropriate.
geometry = Geometry.from_ase_atoms(list(map(molecule, molecule_names)))
basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

# geometry[True, False, False, False]
# 2.2: Loading of the DFTB parameters into their associated feed objects
# ----------------------------------------------------------------------

# Construct the Hamiltonian and overlap matrix feeds; but ensure that the DFTB
# parameter set database actually exists first.
if not exists(parameter_db_path):
    raise FileNotFoundError(
        f'The DFTB parameter set database "{parameter_db_path}" could '
        f'not be found, please ensure "example_01_setup.py" has been run.')

# Identify which species are present
species = torch.unique(geometry.atomic_numbers)
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# Load the Hamiltonian feed model
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=CubicSpline, requires_grad=True)

# Load the overlap feed model
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap',
                              interpolation=CubicSpline, requires_grad=True)

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)

# 2.3: Construct the SCC-DFTB calculator object
# ---------------------------------------------
# As this is a minimal working example, no optional settings are provided to the
# calculator object.
dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed)
dftb_calculator(geometry, basis)

# dftb_calculator._geometry = geometry
# dftb_calculator._basis = basis
# dftb_calculator.forward_2()
# #
# dftb_calculator_o(geometry, basis)

# Construct machine learning object
lr = 0.002
criterion = getattr(torch.nn, 'MSELoss')(reduction='mean')
optimizer = getattr(torch.optim, 'Adam')(h_feed.variables + s_feed.variables, lr=lr)


def load_target_data(molecules: Geometry, path: str
                     ) -> Any:
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
    raise NotImplementedError()


def init_model():
    raise NotImplementedError()


if fit_model:
    # targets = load_target_data(molecule_names, target_path)
    pass


# ================= #
# STEP 3: Execution #
# ================= #
def calculate_losses(calculator: Calculator, targets: Any) -> Tensor:
    """An example function computing the loss of the model.

    Args:
        calculator: calculator object via which target properties can be
            calculated.
        targets: target data to which the model should be fitted.

    Returns:
        loss: the computed loss.

    """
    return criterion(calculator.__getattribute__('q_final_atomic'),
                     targets['q_final_atomic'])


def update_model(calculator: Calculator):
    """Update the model feed objects.

    Arguments:
        calculator: calculator object containing the feeds that are to be
            updated.
    """
    raise NotImplementedError()


if fit_model:
    for epoch in range(1, number_of_epochs + 1):
        # Perform the forwards operation
        dftb_calculator(geometry, basis)

        # Calculate the loss
        loss = calculate_losses(dftb_calculator, targets)

        optimizer.zero_grad()

        # Invoke the autograd engine
        loss.backward()

        # Update the model
        # update_model(dftb_calculator, loss)
        optimizer.step()

        # Reset the calculator
        # dftb_calculator.reset()
else:
    # Run the DFTB calculation
    dftb_calculator(geometry, basis)

_ = ...

# Tasks @FanGuozheng:
#   i  ) Create a database that stores some properties to which the feed models
#     can be fitted then add code to the `load_target_data` function to load it.
#   ii ) Fill out the `calculate_losses` function so that it can return a final
#     loss for the model.
#   iii) Add the code required to update the feeds, this will require:
#       1) adding code to the in `update_model`.
#       2) swapping out the `ScipySkFeed` objects for `SkFeed` objects.
#       3) adding a `requires_grad` argument to the `SkFeed` object's
#          `from_database` method. Otherwise feeds loaded from HDF5 databases
#          will never be trainable.
#   iv ) Replace the Dftb1 calculator with the Dftb2 calculator once @mcsloy has
#     refactored the Dftb2 code and resolved the issues associated with it. This
#     step can be ignored for now.

# Tasks @mcsloy:
#   i  ) Refactor Dftb2 code to simplify it, resolve current issues, and ensure
#     the SCC cycle is performed outside of the gradient.
