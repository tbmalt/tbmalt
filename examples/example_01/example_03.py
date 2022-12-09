import pickle
from os.path import exists
from typing import Any, List

import numpy as np
import torch
import h5py
from h5py import Group
from sklearn.ensemble import RandomForestRegressor

from tbmalt import Geometry, Basis
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import BicubInterp, PolyInterpU
from tbmalt.io.dataset import Dataloader
from tbmalt.ml.acsf import Acsf
from tbmalt.common.batch import pack

Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

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
parameter_db_path = 'example_dftb_vcr.h5'
parameter_db_path_std = 'example_dftb_parameters.h5'

# Should fitting be performed here?
fit_model = True
pred_model = True

# Number of fitting cycles, number of batch size each cycle
number_of_epochs = 5
n_fit_batch = 1000

# Location of a file storing the properties that will be fit to.
target_path = './dataset.h5'


# ============= #
# STEP 2: Setup #
# ============= #

# load data set
def load_target_data(properties: List, groups: Group) -> Any:
    """Load fitting target data.

    Arguments:
        properties: Target properties should be returned.
        groups: Names of groups.

    Returns:
        targets: returns an <OBJECT> storing the data to which the model is to
            be fitted.
    """
    # Data could be loaded from a json file or an hdf5 file; use your own
    # discretion here. A dictionary might be the best object in which to store
    # the target data.
    return Dataloader.load_reference(groups, properties)


# 2.1: Target system specific objects
# -----------------------------------
if fit_model:
    with h5py.File(target_path, 'r') as f:
        dataloder_fit = [load_target_data(targets, g) for g in
                         [f['run1']['train'], f['run2']['train'], f['run3']['train']]]
        dataloder_test = [load_target_data(targets, g) for g in
                          [f['run1']['test'], f['run2']['test'], f['run3']['test']]]
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
    raise FileNotFoundError(
        f'The DFTB parameter set database "{parameter_db_path}" could '
        f'not be found, please ensure "example_01_setup.py" has been run.')

# Identify which species are present
species = torch.tensor([1, 6, 7, 8])
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# Load the Hamiltonian feed model
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=BicubInterp)
h_feed_std = SkFeed.from_database(
    parameter_db_path_std, species, 'hamiltonian')

# Load the overlap feed model
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap',
                              interpolation=BicubInterp)
s_feed_std = SkFeed.from_database(
    parameter_db_path_std, species, 'overlap')

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)
o_feed_std= SkfOccupationFeed.from_database(parameter_db_path_std, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)
u_feed_std = HubbardFeed.from_database(parameter_db_path_std, species)

# 2.3: Construct the SCC-DFTB calculator object
# ---------------------------------------------
# As this is a minimal working example, no optional settings are provided to the
# calculator object.
with open('dftb_calculator_vcr_init.pkl', 'wb') as w:
    dftb_calculator_init = Dftb2(h_feed, s_feed, o_feed, u_feed)
    pickle.dump(dftb_calculator_init, w)
with open('dftb_calculator_vcr_init_std.pkl', 'wb') as w:
    dftb_calculator_init_std = Dftb2(
        h_feed_std, s_feed_std, o_feed_std, u_feed_std)
    pickle.dump(dftb_calculator_init_std, w)

# 2.4: Construct machine learning object
lr = 0.02
onsite_lr = 1e-4
criterion = getattr(torch.nn, 'MSELoss')(reduction='mean')
g1_params = 6.0
g2_params = torch.tensor([0.5, 1.0])
g4_params = torch.tensor([[0.02, 1.0, -1.0]])
element_resolve = True
n_estimators = 100
global_r = True
tolerance = 1e-5


def build_optim(dftb_calculator, dataloder, global_r):
    """Build optimizer for VCR training."""
    comp_r = torch.ones(dataloder.dataset['atomic_numbers'].shape) * 3.5

    if not global_r:
        comp_r[dataloder.dataset['atomic_numbers'] == 1] = 3.0
        comp_r[dataloder.dataset['atomic_numbers'] == 6] = 2.7
        comp_r[dataloder.dataset['atomic_numbers'] == 7] = 2.2
        comp_r[dataloder.dataset['atomic_numbers'] == 8] = 2.3
        comp_r.requires_grad_(True)

        ml_onsite, ml_onsite_dict = [], {}
        numbers = dataloder.dataset['atomic_numbers']

        for key, val in h_feed.on_sites.items():
            n_atoms = (numbers == key).sum()
            ml_onsite_dict.update({key: val.repeat(n_atoms, 1)})

        dftb_calculator.h_feed.on_sites = ml_onsite_dict
        for key, val in dftb_calculator.h_feed.on_sites.items():
            dftb_calculator.h_feed.on_sites[key].requires_grad_(True)
            ml_onsite.append(
                {'params': dftb_calculator.h_feed.on_sites[key], 'lr': onsite_lr})

        optimizer = getattr(torch.optim, 'Adam')(
            [{'params': comp_r, 'lr': lr}] + ml_onsite, lr=lr)

        return comp_r, ml_onsite_dict, optimizer
    else:
        # For global compression radii, optimize each atom specie parameters
        unq_atm = torch.unique(dataloder.dataset['atomic_numbers'])
        unq_atm = unq_atm[unq_atm.ne(0)]
        comp_r0 = torch.tensor([3.0, 2.7, 2.2, 2.3])
        comp_r0.requires_grad_(True)

        ml_onsite = []
        for key, val in h_feed.on_sites.items():
            h_feed.on_sites[key].requires_grad_(True)
            ml_onsite.append({'params': val, 'lr': onsite_lr})

        optimizer = getattr(torch.optim, 'Adam')(
            [{'params': comp_r0, 'lr': lr}] + ml_onsite, lr=lr)
        return comp_r0, optimizer


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


def single_fit(dftb_calculator, dataloder, n_batch, global_r):
    indice = torch.split(torch.tensor(dataloder.random_idx), n_batch)

    if not global_r:
        comp_r, ml_onsite_dict, optimizer = build_optim(dftb_calculator, dataloder, global_r)
        dftb_calculator.h_feed.is_local_onsite = True
        dftb_calculator.h_feed.on_sites = ml_onsite_dict
    else:
        comp_r, optimizer = build_optim(dftb_calculator, dataloder, global_r)

    loss_old = 0
    for epoch in range(number_of_epochs):

        data = dataloder[indice[epoch % len(indice)]]

        geometry = Geometry(data['atomic_numbers'], data['positions'], units='a')
        basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

        if not global_r:
            this_cr = comp_r[indice[epoch % len(indice)]]
        else:
            this_cr = torch.ones(geometry.atomic_numbers.shape)
            for ii, iatm in enumerate(geometry.unique_atomic_numbers()):
                this_cr[iatm == geometry.atomic_numbers] = comp_r[ii]

        # Perform the forwards operation
        dftb_calculator.h_feed.vcr = this_cr
        dftb_calculator.s_feed.vcr = this_cr
        dftb_calculator(geometry, basis)

        # Calculate the loss
        loss = calculate_losses(dftb_calculator, data)
        optimizer.zero_grad()
        print(epoch, loss)
        print('dftb_calculator.h_feed.on_sites', dftb_calculator.h_feed.on_sites)
        # print('dftb_calculator.h_feed.vcr', dftb_calculator.h_feed.vcr)

        # Invoke the autograd engine
        loss.backward()
        optimizer.step()

        if torch.abs(loss_old - loss.detach()).lt(tolerance):
            break
        loss_old = loss.detach().clone()

        this_cr = this_cr.detach().clone()
        min_mask = this_cr[this_cr != 0].lt(2.0)
        max_mask = this_cr[this_cr != 0].gt(9.0)

        # To make sure compression radii inside grid points
        if min_mask.any():
            with torch.no_grad():
                comp_r.clamp_(min=2.0)
        if max_mask.any():
            with torch.no_grad():
                comp_r.clamp_(max=9.0)

    if global_r:
        dftb_calculator.h_feed.vcr = comp_r

    return dftb_calculator


def build_feature(geometry, basis):
    acsf = Acsf(geometry, basis, basis.shell_dict, g1_params, g2_params, g4_params,
                element_resolve=element_resolve)
    acsf()

    return acsf.g


def scikit_learn_model(x_train, y_train, x_test,
                       geometry_test: Geometry, atom_like: bool = False):
    """ML process with random forest method.

    Arguments:
        x_train:
        y_train:
        x_test:
        geometry_test:
        atom_like:
    """
    reg = RandomForestRegressor(n_estimators=n_estimators)
    reg.fit(x_train, y_train)
    y_pred = torch.from_numpy(reg.predict(x_test))

    if not atom_like:
        size_sys = geometry_test.n_atoms
        return pack(torch.split(y_pred, tuple(size_sys)))
    else:
        return y_pred


def single_test(dftb_calculator: Dftb2,
                dftb_calculator_std: Dftb2,
                dataloder_fit,
                dataloder_test,
                global_r):

    geometry_fit = Geometry(dataloder_fit.dataset['atomic_numbers'],
                            dataloder_fit.dataset['positions'], units='a')
    basis_fit = Basis(geometry_fit.atomic_numbers, shell_dict, shell_resolved=False)

    geometry_test = Geometry(dataloder_test.dataset['atomic_numbers'],
                             dataloder_test.dataset['positions'], units='a')
    basis_test = Basis(geometry_test.atomic_numbers, shell_dict, shell_resolved=False)

    if not global_r:
        # flatten and remove padding atomic numbers
        numbers_fit = geometry_fit.atomic_numbers[geometry_fit.atomic_numbers.ne(0)]
        numbers_test = geometry_test.atomic_numbers[geometry_test.atomic_numbers.ne(0)]

        # Collect ML input and reference
        x_fit = build_feature(geometry_fit, basis_fit)
        y_fit = dftb_calculator.h_feed.vcr.detach()
        y_fit = y_fit[geometry_fit.atomic_numbers.ne(0)]
        x_test = build_feature(geometry_test, basis_test)

        # predict compression radii
        y_pred = scikit_learn_model(x_fit, y_fit, x_test, geometry_test)

        # predict on-site terms
        y_pred_on_dict = {}
        for key, val in dftb_calculator.h_feed.on_sites.items():
            y_pred_on = scikit_learn_model(
                x_fit[numbers_fit == key], val.detach(),
                x_test[numbers_test == key], geometry_test, True)

            y_pred_on_dict.update({key: y_pred_on if y_pred_on.dim() == 2
                else y_pred_on.unsqueeze(-1)})

        # Update compression radii and onsite
        dftb_calculator.h_feed.on_sites = y_pred_on_dict
        dftb_calculator.h_feed.vcr = y_pred
        dftb_calculator.s_feed.vcr = y_pred
    else:
        comp_r = dftb_calculator.h_feed.vcr
        this_cr = torch.ones(geometry_test.atomic_numbers.shape)
        for ii, iatm in enumerate(geometry_test.unique_atomic_numbers()):
            this_cr[iatm == geometry_test.atomic_numbers] = comp_r[ii]
        print('comp_r', dftb_calculator.h_feed.on_sites)
        # Perform the forwards operation
        dftb_calculator.h_feed.vcr = this_cr
        dftb_calculator.s_feed.vcr = this_cr

    # Perform DFTB calculations
    dftb_calculator_std(geometry_test, basis_test)
    dftb_calculator(geometry_test, basis_test)
    print(torch.sum(torch.abs(dftb_calculator_std.dipole - dataloder_test.dataset['dipole']))
          / dftb_calculator.geometry._n_batch)
    print(torch.sum(torch.abs(dftb_calculator.dipole - dataloder_test.dataset['dipole']))
          / dftb_calculator.geometry._n_batch)
    import matplotlib.pyplot as plt
    plt.plot(dataloder_test.dataset['dipole'], dataloder_test.dataset['dipole'], 'k')
    plt.plot(dataloder_test.dataset['dipole'], dftb_calculator_std.dipole, 'r.')
    plt.plot(dataloder_test.dataset['dipole'], dftb_calculator.dipole.detach(), 'b.')
    plt.show()


# STEP 3.1: Execution training
if fit_model:

    for ii, data_fit in enumerate(dataloder_fit):

        with open('dftb_calculator_vcr_init.pkl', 'rb') as r:
            dftb_calculator_init = pickle.load(r)

        dftb_calculator = single_fit(
            dftb_calculator_init, data_fit, n_fit_batch, global_r=global_r)

        if global_r:
            with open(f'dftb_calculator_vcr_{ii}_global.pkl', 'wb') as w:
                pickle.dump(dftb_calculator, w)
        else:
            with open(f'dftb_calculator_vcr_{ii}_local.pkl', 'wb') as w:
                pickle.dump(dftb_calculator, w)

else:
    # Run the DFTB calculation
    raise NotImplementedError()


# STEP 3.2: Execution testing
if pred_model:

    # Standard DFTB model
    with open('dftb_calculator_vcr_init_std.pkl', 'rb') as r:
        dftb_calculator_std = pickle.load(r)

    # Load DFTB model with optmized parameters
    for ii, (d_fit, d_test) in enumerate(zip(dataloder_fit, dataloder_test)):

        if global_r:
            with open(f'dftb_calculator_vcr_{ii}_global.pkl', 'rb') as w:
                dftb_calculator = pickle.load(w)

            single_test(dftb_calculator, dftb_calculator_std, d_fit, d_test, global_r)
        else:
            with open(f'dftb_calculator_vcr_{ii}_local.pkl', 'rb') as w:
                dftb_calculator = pickle.load(w)

            single_test(dftb_calculator, dftb_calculator_std, d_fit, d_test, global_r)

