# -*- coding: utf-8 -*-
from typing import Dict, List, Union
import random

from h5py import Group
import torch
from torch.utils.data import DataLoader as Loader
from torch import Tensor

from tbmalt.common.batch import pack


class Dataloader(Loader):
    """A class to read and load data.

    This class will use given targets to read and load data samples from
    h5py files. The '__getitem__()' function, which is inherited from object
    'torch.utils.data.Dataset', supports fetching data with specified indices.
    The indices are from random numbers with defined seeds.

    Arguments:
        dataset: An dictionary which stores all the data.
        labels: A list of label for each sample.
        targets: Names of machine learned targets.
        pbc: Whether read cells.
        seed: Seed to generate random numbers, this will be used to
            generate indices when loading samples.

    """

    def __init__(self, dataset: Dict, labels: List[str], targets, pbc, seed):
        self.targets = targets
        self.dataset = dataset
        self.labels = labels
        self.pbc = pbc

        size = len(dataset['atomic_numbers'])
        random.seed(seed)
        self.random_idx = random.sample(range(size), size)

    def __getitem__(self, idx: Tensor) -> Dict:
        """Get data values according to input indices."""
        atomic_numbers = self.dataset['atomic_numbers']
        positions = self.dataset['positions']
        if self.pbc:
            cells = self.dataset['cells']

        # Select different properties
        data = {target: pack([self.dataset[target][ii] for ii in idx])
                for target in self.targets}

        # Select data with index
        data['atomic_numbers'] = pack([atomic_numbers[ii] for ii in idx])
        data['positions'] = pack([positions[ii] for ii in idx])
        if self.pbc:
            data['cells'] = pack([cells[ii] for ii in idx])

        return data

    @classmethod
    def load_reference(cls, groups: Union[Group, List[Group]],
                       targets: List[str],
                       sizes: Union[List[int], int] = None,
                       pbc: bool = False, seed: int = 1):
        """Load reference from h5py type data.

        Arguments:
            groups: The groups in h5py binary files. The input groups
                can be a single group or a list of groups.
            targets: Loading targets, such as dipole, charge, etc.
            sizes: Loading size of each group.
            pbc: Whether read cells.
            seed: Seed to generate random numbers, this will be used to
                generate indices when loading samples.

        """
        data = {target: [] for target in targets}
        positions, numbers, labels = [], [], []
        if pbc:
            cells = []

        def single_loader(g, size):
            g_size = g.attrs['size']
            this_size = min(g_size, size) if size is not None else g_size

            random.seed(seed)
            ind = random.sample(torch.arange(g_size).tolist(), this_size)
            labels.extend(g.attrs['label'])

            # loop for each property
            for target in targets:
                idata = g[target][()][ind]
                data[target].append(torch.from_numpy(idata))

            positions.append(torch.from_numpy(g['positions'][()][ind]))
            numbers.append(torch.from_numpy(g['atomic_numbers'][()][ind]))
            if pbc:
                cells.append(torch.from_numpy(g['cells'][()][ind]))

        # Read data from specified groups
        if isinstance(groups, list):

            if sizes is not None:
                assert len(groups) == len(sizes),\
                    'len(size) should be equal to len(groups)'

            for ii, g in enumerate(groups):
                size = None if sizes is None else sizes[ii]
                single_loader(g, size)
        else:
            single_loader(groups, sizes)

        data['atomic_numbers'] = pack(numbers).flatten(0, 1)
        data['positions'] = pack(positions).flatten(0, 1)
        if pbc:
            data['cells'] = pack(cells).flatten(0, 1)

        # loop for each property and pack target values
        for target in targets:
            data[target] = pack(data[target]).flatten(0, 1)

        return cls(data, labels, targets, pbc, seed)
