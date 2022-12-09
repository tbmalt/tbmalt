import os
import pickle
from os.path import exists
from typing import Any, List

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

from tbmalt import Geometry, Basis
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.io.dataset import Dataloader

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
size = [1000]
targets = ['charge']

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
number_of_epochs = 6
n_batch = 1000
lr = 0.001
criterion = getattr(torch.nn, 'MSELoss')(reduction='mean')

# Location of a file storing the properties that will be fit to.
# TODO add download links
target_path = './dataset.h5'


# ============= #
# STEP 2: Setup #
# ============= #

# load data set
def load_target_data(path: str, properties: List) -> Any:
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
    with h5py.File(path, 'r') as f:
        groups = f['run1']['train']
        return Dataloader.load_reference(groups, properties)


# 2.1: Target system specific objects
# -----------------------------------
if fit_model:
    dataloder = load_target_data(target_path, targets)
else:
    raise NotImplementedError()


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
    os.system('python ../example_01/example_01_setup.py')

# Identify which species are present
species = torch.tensor([1, 6, 7, 8])
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
with open("dftb_calculator.pkl", "wb") as f:
    pickle.dump(Dftb2(h_feed, s_feed, o_feed, u_feed), f)


# 2.4: Construct and initialize ML parameters
def init_model(dftb_calculator):
    h_var = [val.abcd for key, val in dftb_calculator.h_feed.off_sites.items()]
    s_var = [val.abcd for key, val in dftb_calculator.s_feed.off_sites.items()]
    optimizer = getattr(torch.optim, 'Adam')(h_var + s_var, lr=lr)

    return h_var, s_var, optimizer


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
        key_dftb = 'q_final_atomic' if key == 'charge' else key
        loss += criterion(calculator.__getattribute__(key_dftb), data[key])

    return loss


def update_model(calculator: Calculator):
    """Update the model feed objects.

    Arguments:
        calculator: calculator object containing the feeds that are to be
            updated.
    """
    raise NotImplementedError()


if fit_model:
    indice = torch.split(torch.tensor(dataloder.random_idx), n_batch)

    loss1_list, loss2_list, loss4_list = [], [], []
    grad_h1, grad_h2, grad_h4 = {}, {}, {}
    grad_s1, grad_s2, grad_h4 = {}, {}, {}

    # Initialize DFTB and ML parameters
    with open("dftb_calculator.pkl", "rb") as f:
        dftb_calculator = pickle.load(f)
        h_var, s_var, optimizer = init_model(dftb_calculator)

    for epoch in range(number_of_epochs):
        this_grad_h1, this_grad_s1 = {}, {}
        data = dataloder[indice[epoch % len(indice)]]
        geometry = Geometry(data['atomic_numbers'], data['positions'], units='a')
        basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

        # Perform the forwards operation
        dftb_calculator(geometry, basis, scc_grad=True)

        # Calculate the loss
        loss1 = calculate_losses(dftb_calculator, data)
        loss1_list.append(loss1.detach())
        print('loss1', loss1)
        # for key, var in dftb_calculator.h_feed.off_sites.items():
        #     var.abcd.register_hook(lambda grad: grad_h1.append(grad))
        # for key, var in dftb_calculator.s_feed.off_sites.items():
        #     var.abcd.register_hook(lambda grad: grad_s1.append(grad))

        optimizer.zero_grad()

        # Invoke the autograd engine
        loss1.backward(retain_graph=True)
        for key, var in dftb_calculator.h_feed.off_sites.items():
            if var.abcd.grad is not None:
                this_grad_h1.update({key: var.abcd.grad[var.abcd.grad.ne(0)]})
        for key, var in dftb_calculator.s_feed.off_sites.items():
            if var.abcd.grad is not None:
                this_grad_s1.update({key: var.abcd.grad[var.abcd.grad.ne(0)]})

        # Update the model
        # update_model(dftb_calculator, loss)
        optimizer.step()
        grad_h1.update({epoch: this_grad_h1})
        grad_s1.update({epoch: this_grad_s1})

    # Initialize DFTB and ML parameters
    with open("dftb_calculator.pkl", "rb") as f:
        dftb_calculator = pickle.load(f)
        h_var, s_var, optimizer = init_model(dftb_calculator)

    for epoch in range(number_of_epochs):
        this_grad_h2, this_grad_s2 = {}, {}
        data = dataloder[indice[epoch % len(indice)]]

        geometry = Geometry(data['atomic_numbers'], data['positions'], units='a')
        basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

        # Perform the forwards operation
        dftb_calculator(geometry, basis, scc_grad=False)

        # Calculate the loss
        loss2 = calculate_losses(dftb_calculator, data)
        loss2_list.append(loss2.detach())
        print('loss', loss2)
        # for key, var in dftb_calculator.h_feed.off_sites.items():
        #     var.abcd.register_hook(lambda grad: grad_h2.append(grad))
        # for key, var in dftb_calculator.s_feed.off_sites.items():
        #     var.abcd.register_hook(lambda grad: grad_s2.append(grad))

        optimizer.zero_grad()

        # Invoke the autograd engine
        loss2.backward(retain_graph=True)
        for key, var in dftb_calculator.h_feed.off_sites.items():
            if var.abcd.grad is not None:
                this_grad_h2.update({key: var.abcd.grad[var.abcd.grad.ne(0)]})
        for key, var in dftb_calculator.s_feed.off_sites.items():
            if var.abcd.grad is not None:
                this_grad_s2.update({key: var.abcd.grad[var.abcd.grad.ne(0)]})

        optimizer.step()
        grad_h2.update({epoch: this_grad_h2})
        grad_s2.update({epoch: this_grad_s2})

    plt.plot(np.arange(number_of_epochs), loss1_list)
    plt.plot(np.arange(number_of_epochs), loss2_list)
    plt.show()

    for step in grad_h1.keys():
        for key in grad_h1[step].keys():
            ii = grad_h1[step][key]
            jj = grad_h2[step][key]
            plt.plot(ii.flatten(), ii.flatten(), 'k')
            if torch.abs(ii - jj).gt(0.0005).any():
                plt.plot(ii.flatten(), jj.flatten(), 'x',
                         label='step: ' + str(step) + ', key: ' + str(key))
            else:
                plt.plot(ii.flatten(), jj.flatten(), 'c.')
    plt.xlabel('full SCC grad')
    plt.ylabel('SCC without grad')
    plt.title(f'Hamiltonian gradients of {epoch} steps')
    plt.legend()
    plt.show()

    for step in grad_s1.keys():
        for key in grad_s1[step].keys():
            ii = grad_s1[step][key]
            jj = grad_s2[step][key]
            plt.plot(ii.flatten(), ii.flatten(), 'k')
            if torch.abs(ii - jj).gt(0.0001).any():
                plt.plot(ii.flatten(), jj.flatten(), 'x',
                         label='step: ' + str(step) + ', key: ' + str(key))
            else:
                plt.plot(ii.flatten(), jj.flatten(), 'c.')
    plt.xlabel('full SCC grad')
    plt.ylabel('SCC without grad')
    plt.title(f'overlap gradients of {epoch} steps')
    plt.legend()
    plt.show()


else:
    # Run the DFTB calculation
    raise NotImplementedError()
