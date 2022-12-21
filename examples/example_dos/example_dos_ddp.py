import os
from os.path import exists
from typing import Any, List

import torch
import numpy as np
import h5py

from tbmalt import Geometry, Basis
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.io.dataset import Dataloader
from tbmalt.physics.dftb.properties import dos
import tbmalt.common.maths as tb_math
from tbmalt.common.batch import pack
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# Number of threads should be set accrodingly due to the information of cluster
torch.set_num_threads(1)

# Set the printoption
torch.set_printoptions(6, profile='full')

# ================ #
# STEP 1: Settings #
# ================ #

# 1.1: System settings
# --------------------

# Provide a list of systems upon which TBMaLT is to be run
training_size = 30
testing_size = 20
targets = ['dos']

# Provide information about the orbitals on each atom; this is keyed by atomic
# numbers and valued by azimuthal quantum numbers like so:
#   {Z₁: [ℓᵢ, ℓⱼ, ..., ℓₙ], Z₂: [ℓᵢ, ℓⱼ, ..., ℓₙ], ...}
shell_dict = {14: [0, 1, 2]}

# Identify which species are present
species = torch.tensor([14])
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# 1.2: Model settings
# -------------------
# Location at which the DFTB parameter set database is located
parameter_db_path = './siband.hdf5'

# Should fitting be performed here?
fit_model = True

# Should test the trained mode?
test = True

# Number of fitting cycles, number of batch size each cycle
number_of_epochs = 500
n_batch = 5

# learning rate
lr = 0.000005

# type of loss function
loss_function = 'Hellinger'

# Location of a file storing the properties that will be fit to.
target_path = './dataset_dos.h5'

# Choose which training and testing dataset to be loaded
# To run training and testing using the rattled type silicon systems, target_run
#    can be set as 'run1', 'run2' and 'run3'.
target_run = 'run1'

# Energy window for dos sampling
points = torch.linspace(-3.3, 1.6, 491)

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


# ======================== #
# STEP 2: Data preparation #
# ======================== #


# 2.1 load data set
def load_target_data(path: str, group1: str, group2: str, properties: List,
                     size: int) -> Any:
    """Load fitting target data.

    Arguments:
        path: path to a database in which the fitting data can be found.
        group 1: string to select the training run.
        group 2: string to select training or testing data.
        properties: list to select properties.
        size: size to load data.

    Returns:
        targets: returns an <OBJECT> storing the data to which the model is to
            be fitted.
    """
    # Data could be loaded from a json file or an hdf5 file; use your own
    # discretion here. A dictionary might be the best object in which to store
    # the target data.
    with h5py.File(path, 'r') as f:
        groups = f[group1][group2]
        return Dataloader.load_reference(groups, properties, size, pbc=True,
                                         rand=False)


# 2.2 prepare training and testing data
class SiliconDataset(Dataset):
    """Class to build training and testing dataset."""

    def __init__(self, numbers, positions, latvecs, homo_lumos, eigenvalues):
        self.numbers = numbers
        self.positions = positions
        self.latvecs = latvecs
        self.homo_lumos = homo_lumos
        self.eigenvalues = eigenvalues

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        number = self.numbers[idx]
        position = self.positions[idx]
        latvec = self.latvecs[idx]
        homo_lumo = self.homo_lumos[idx]
        eigenvalue = self.eigenvalues[idx]
        system = {"number": number, "position": position, "latvec": latvec,
                  "homo_lumo": homo_lumo, "eigenvalue": eigenvalue, "idx": idx}

        return system


def prepare_data(run):
    """Prepare training and testing data."""
    try:
        os.mkdir('./result')
    except FileExistsError:
        pass

    # Build a random index to select systems for training and testing
    dataloder_train = load_target_data(target_path, run, 'train', ['homo_lumos', 'eigenvalues'], training_size)
    dataloder_test = load_target_data(target_path, run, 'test', ['homo_lumos', 'eigenvalues'], testing_size)
    indice = torch.arange(training_size + testing_size).tolist()

    data_train = dataloder_train[indice[: training_size]]
    data_test = dataloder_test[indice[: testing_size]]

    numbers_train, positions_train, cells_train = (data_train['atomic_numbers'],
                                                   data_train['positions'],
                                                   data_train['cells'])

    numbers_test, positions_test, cells_test = (data_test['atomic_numbers'],
                                                data_test['positions'],
                                                data_test['cells'])
    # Build datasets
    dataset_train = SiliconDataset(numbers_train, positions_train, cells_train,
                                   data_train['homo_lumos'],
                                   data_train['eigenvalues'])
    dataset_test = SiliconDataset(numbers_test, positions_test, cells_test,
                                  data_test['homo_lumos'],
                                  data_test['eigenvalues'])

    # Calculate reference dos and implement sampling
    ref_ev = data_train['eigenvalues']
    fermi_train = data_train['homo_lumos'].mean(dim=-1)
    energies_train = fermi_train.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(
            training_size, 0)
    dos_ref = dos((ref_ev), energies_train, 0.09)
    data_train_dos = torch.cat((energies_train.unsqueeze(-1),
                                dos_ref.unsqueeze(-1)), -1)

    return dataset_train, dataset_test, data_train_dos


# split the data according to batch size and number of processers
def data_split(rank, world_size, dataset, batch_size, pin_memory=False,
               num_workers=0):
    """Prepare the training data for distributed environment."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


# ================= #
# STEP 3: Execution #
# ================= #
def loss_fn(results, ref_dos, ibatch):
    """Calculate loss during training."""
    loss = 0.
    # Get type of loss function.
    if loss_function == 'MSELoss':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif loss_function == 'Hellinger':
        criterion = tb_math.HellingerLoss()

    # Calculate the loss
    ref = ref_dos[..., 1][ibatch] if ref_dos.ndim == 3 else ref_dos[..., 1]
    loss = loss + criterion(results, ref)

    return loss


def dftb_results(numbers, positions, cells, h_feed_n, s_feed_n, **kwargs):
    """Perform forward DFTB calculatoins."""
    # Build objects for DFTB calculations
    geometry = Geometry(numbers, positions, cells, units='a',
                        cutoff=torch.tensor([18.0]))
    basis = Basis(geometry.atomic_numbers, shell_dict, shell_resolved=False)

    mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2,
              'generations': 3, 'tolerance': 1e-10}
    kwargs = {}
    kwargs['mix_params'] = mix_params

    # Build objects for DFTB calculaitons
    dftb_calculator = Dftb2(h_feed_n, s_feed_n, o_feed, u_feed,
                            supress_SCF_error=True, **kwargs)
    dftb_calculator(geometry, basis)

    return dftb_calculator


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


class DFTB_DDP(torch.nn.Module):
    """Implement DFTB calculation within the framework of nn.Module."""

    def __init__(self):
        super(DFTB_DDP, self).__init__()
        h_var = [val.abcd for key, val in h_feed.off_sites.items()]
        s_var = [val.abcd for key, val in s_feed.off_sites.items()]
        variable = h_var + s_var

        self.parameters = torch.nn.ParameterList([torch.nn.Parameter(ivar)
                                                  for ivar in variable])

    def forward(self, data):
        h_feed_p = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=CubicSpline, requires_grad=True)
        s_feed_p = SkFeed.from_database(parameter_db_path, species, 'overlap',
                              interpolation=CubicSpline, requires_grad=True)
        var = list(self.parameters())
        h_var_u = var[:len(var)//2]
        s_var_u = var[len(var)//2:]
        ii = 0
        for key, val in h_feed_p.off_sites.items():
            val.abcd = h_var_u[ii]
            ii = ii + 1
        ii = 0
        for key, val in s_feed_p.off_sites.items():
            val.abcd = s_var_u[ii]
            ii = ii + 1

        scc = dftb_results(data['number'], data['position'],
                           data['latvec'], h_feed_p, s_feed_p)
        fermi_dftb = getattr(scc, 'homo_lumo').mean(dim=-1)
        energies_dftb = fermi_dftb.unsqueeze(-1) + points.unsqueeze(
            0).repeat_interleave(n_batch, 0)
        dos_dftb = dos((getattr(scc, 'eigenvalue')),
                       energies_dftb, 0.09)
        return dos_dftb


def main(rank, world_size):
    """ML training to optimize DFTB H and S matrix."""
    # Initial the model
    setup(rank, world_size)
    dataset_train, dataset_test, data_train_dos = prepare_data(target_run)
    train_data = data_split(rank, world_size, dataset_train, batch_size=n_batch)
    device = torch.device("cpu")
    model = DFTB_DDP()
    ddp_model = DDP(model.to(device))
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=lr)
    loss_list = []
    loss_list.append(0)

    # Training
    for epoch in range(number_of_epochs):
        print('epoch', epoch)
        train_data.sampler.set_epoch(epoch)
        for ibatch, data in enumerate(train_data):
            # Perform the forwards operation
            forward_cal = ddp_model(data)
            loss = loss_fn(forward_cal, data_train_dos, data['idx'])
            optimizer.zero_grad()
            loss.retain_grad()
            # Invoke the autograd engine
            loss.backward()
            # Update the model
            optimizer.step()
            print("rank:", int(rank), "loss:", loss, "idx:", data['idx'])
        loss_list.append(loss.detach())

    # Save the training loss
    f = open('./result/loss' + str(int(rank)) + '.dat', 'w')
    np.savetxt(f, torch.tensor(loss_list[1:]))
    f.close()

    # Testing
    if rank == 0:
        update_para = list(model.parameters())
        print(update_para, file=open('./result/abcd.txt', "w"))
        if test:
            h_feed_p = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=CubicSpline, requires_grad=True)
            s_feed_p = SkFeed.from_database(parameter_db_path, species, 'overlap',
                                  interpolation=CubicSpline, requires_grad=True)
            var = list(model.parameters())
            h_var_u = var[:len(var)//2]
            s_var_u = var[len(var)//2:]
            ii = 0
            for key, val in h_feed_p.off_sites.items():
                val.abcd = h_var_u[ii]
                ii = ii + 1
            ii = 0
            for key, val in s_feed_p.off_sites.items():
                val.abcd = s_var_u[ii]
                ii = ii + 1
            test(0, 1, dataset_test, h_feed_p, s_feed_p)


def test(rank, world_size, test_dataset, h_feed_p, s_feed_p):
    """Test the trained model."""
    try:
        os.mkdir('./result/test')
    except FileExistsError:
        pass

    test_data = DataLoader(test_dataset, batch_size=1)
    energies_test = torch.linspace(-18.0, 5.0, 500)
    dos_pred_tot = []
    hl_pred_tot = []
    dos_dftb_tot = []
    hl_dftb_tot = []

    # Pred
    for ibatch, data in enumerate(test_data):
        scc_pred = dftb_results(data['number'], data['position'],
                                data['latvec'], h_feed_p, s_feed_p)
        hl_pred = getattr(scc_pred, 'homo_lumo').detach()
        hl_pred_tot.append(hl_pred)
        eigval_pred = getattr(scc_pred, 'eigenvalue').detach()
        dos_pred = dos((eigval_pred), energies_test, 0.09)
        f = open('./result/test/Pred_homo_lumo' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, hl_pred)
        f.close()
        pred_dos = torch.cat((energies_test.unsqueeze(-1),
                              dos_pred.unsqueeze(-1).squeeze(0)), -1)
        dos_pred_tot.append(dos_pred)
        f = open('./result/test/Pred_dos' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, pred_dos)
        f.close()
        f = open('./result/test/Pred_eigenvalue' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigval_pred.squeeze(0))
        f.close()

    # Pred mean
    dos_pred_mean = pack(dos_pred_tot).mean(dim=0)
    dos_pred_std = pack(dos_pred_tot).std(dim=0)
    f = open('./result/Pred_dos_mean.dat', 'w')
    np.savetxt(f, torch.cat((energies_test.unsqueeze(-1),
                             dos_pred_mean.unsqueeze(-1).squeeze(0)), -1))
    f.close()
    f = open('./result/Pred_dos_std.dat', 'w')
    np.savetxt(f, dos_pred_std.squeeze(0))
    f.close()
    f = open('./result/Pred_homo_lumo_mean.dat', 'w')
    np.savetxt(f, pack(hl_pred_tot).mean(dim=0).detach())
    f.close()
    f = open('./result/Pred_homo_lumo_std.dat', 'w')
    np.savetxt(f, pack(hl_pred_tot).std(dim=0).detach())
    f.close()

    # DFTB
    # Build new feeds
    h_feed_o = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                                    interpolation=CubicSpline)
    s_feed_o = SkFeed.from_database(parameter_db_path, species, 'overlap',
                                    interpolation=CubicSpline)
    for ibatch, data in enumerate(test_data):
        scc_dftb = dftb_results(data['number'], data['position'],
                                data['latvec'], h_feed_o, s_feed_o)
        hl_dftb = getattr(scc_dftb, 'homo_lumo').detach()
        hl_dftb_tot.append(hl_dftb)
        eigval_dftb = getattr(scc_dftb, 'eigenvalue').detach()
        dos_dftb = dos((eigval_dftb), energies_test, 0.09)
        f = open('./result/test/dftb_homo_lumo' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, hl_dftb)
        f.close()
        dftb_dos = torch.cat((energies_test.unsqueeze(-1),
                              dos_dftb.unsqueeze(-1).squeeze(0)), -1)
        dos_dftb_tot.append(dos_dftb)
        f = open('./result/test/dftb_dos' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, dftb_dos)
        f.close()
        f = open('./result/test/dftb_eigenvalue' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigval_dftb.squeeze(0))
        f.close()

    # DFTB mean
    dos_dftb_mean = pack(dos_dftb_tot).mean(dim=0)
    dos_dftb_std = pack(dos_dftb_tot).std(dim=0)
    f = open('./result/dftb_dos_mean.dat', 'w')
    np.savetxt(f, torch.cat((energies_test.unsqueeze(-1),
                             dos_dftb_mean.unsqueeze(-1).squeeze(0)), -1))
    f.close()
    f = open('./result/dftb_dos_std.dat', 'w')
    np.savetxt(f, dos_dftb_std.squeeze(0))
    f.close()
    f = open('./result/dftb_homo_lumo_mean.dat', 'w')
    np.savetxt(f, pack(hl_dftb_tot).mean(dim=0).detach())
    f.close()
    f = open('./result/dftb_homo_lumo_std.dat', 'w')
    np.savetxt(f, pack(hl_dftb_tot).std(dim=0).detach())
    f.close()


if __name__ == '__main__':
    # The number of processes to spawn.
    world_size = 6

    # Spawning subprocesses
    mp.spawn(main, args=(world_size,),
             nprocs=world_size)
