# -*- coding: utf-8 -*-
from typing import Dict, List
import random

import h5py
import torch
from torch.utils.data import DataLoader as Loader
from torch import Tensor

from tbmalt.common.batch import deflate, pack


class Dataloader(Loader):
    """"""

    def __init__(self, atomic_numbers: Tensor, positions: Tensor,
                 dataset: Dict):
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.dataset = dataset
        size = len(atomic_numbers)
        self.random_idx = random.sample(range(size), size)
        print('self.random_idx', self.random_idx)

    def __getitem__(self, idx):
        """"""
        # Select data with index
        atomic_numbers = pack([self.atomic_numbers[ii] for ii in idx])
        positions = pack([self.positions[ii] for ii in idx])

        # Select different properties
        dataset = {key: pack([val[ii] for ii in idx])
                   for key, val in self.dataset.items()}

        return atomic_numbers, positions, dataset

    @classmethod
    def load_reference(cls, dataset: str, size: int, properties: List):
        """Load reference from h5py type data."""
        data = {pro: [] for pro in properties}
        positions, numbers = [], []

        with h5py.File(dataset, 'r') as f:
            gg = f['global_group']
            molecule_specie = gg.attrs['molecule_specie_global']

            size_mol = int(size / len(molecule_specie))

            # add atom name and atom number
            for imol_spe in molecule_specie:
                g = f[imol_spe]
                g_size = g.attrs['n_molecule']
                size_mol = min(g_size, size_mol)

                ind = random.sample(torch.arange(g_size).tolist(), size_mol)

                # loop for the same molecule specie
                for imol in ind:
                    for ipro in properties:  # loop for each property
                        idata = g[str(imol + 1) + ipro][()]
                        data[ipro].append(torch.from_numpy(idata))

                    positions.append(torch.from_numpy(g[str(imol + 1) + 'position'][()]))
                    numbers.append(torch.from_numpy(g.attrs['numbers']))

        return cls(numbers, positions, data)
