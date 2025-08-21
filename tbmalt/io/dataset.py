# -*- coding: utf-8 -*-
"""Containers to hold data associated the training and testing process."""

from os.path import join
from functools import reduce
import operator
from typing import Union, Dict, Any, Optional, List
import h5py
import torch

import numpy as np


from h5py import Group, File
from tbmalt import Geometry
from tbmalt.common.batch import pack, merge

Tensor = torch.Tensor


class DataSet:
    pass


class DataSetIM(DataSet):
    def __init__(
            self, geometry: Geometry, data: Dict[str, Tensor],
            labels: Optional[List[str]] = None,
            meta: Optional[Dict[str, Any]] = None):
        """In-memory dataset management class.

        A container designed to store the data necessary to perform a training
        or testing operation. Datasets are only intended to hold structural
        information and any required reference data. The `tbmalt.DataSet`
        entities do not inherit from, and are not compatible with, the
        `torch.Dataset` and `torch.DataLoader` objects as they do not support
        the custom objects required by TBMaLT; namely the geometry. Dataset
        objects are designed for more for convenience and as such are not
        strictly necessary.

        Arguments:
            geometry: a single batched geometry object representing the
                systems contained within the dataset.
            data: a dictionary keyed by strings specifying various data-points
                that are to be used during training and/or testing. Values are
                assumed to be torch tensors batched along the first dimension.
                When a dataset is sliced via `dataset[0:3]` then only the
                corresponding subset of each tensor is returned.
            labels: an optional list specifying a label for each system
                present in the dataset. These labels are used only as a visual
                reference point for the user, and as such need not be unique.
                The only restriction is that there must be one label for each
                data-point. These are useful in helping users to quickly and
                visually identify the contents of randomly selected subsets.
            meta: a dictionary that may contain any arbitrary data that the
                user supplies. Data present within the `meta` dictionary is
                assumed to be global and is not sliced like `data`.

        Notes:
            The `meta` attribute is currently in the developmental stage and
            is thus subject to change.

            Support will be added in the future for PyTorch sampling objects
            and for loading meta-data from files.

        """
        self.geometry = geometry
        self.data = data
        self.labels = labels
        self.meta = meta if meta is not None else None

        # If labels have been provided, then ensure that the correct number
        # were specified.
        if self.labels is not None and len(self.labels) != len(self):
            raise ValueError(
                'If labels are provided then the number of labels '
                f'({len(self.labels)}) must match the number of systems'
                f'({len(self)})')

        # Confirm that all values in `data` are tensors of correct size
        # Ensure the contents of `data` conform to requirements.
        for k, v in self.data.items():
            # All values within `data` should be torch.Tensors
            check_1 = isinstance(v, Tensor)
            assert check_1, f'data entry "{k}" must be a torch.tensor'

            # The tensor is expected to be a batch of data, with one entry per
            # system, batched along the first dimension.
            check_2 = v.shape[0] == len(self)
            assert check_2, 'The length of the first dimension of the tensor ' \
                            f'"{k}" (v.shape[0]) does not match the number of ' \
                            f'systems ({len(self)})'

    def __len__(self):
        """Number of systems present in the dataset"""
        return self.geometry.atomic_numbers.shape[0]

    def __getitem__(self, idx):
        """Permit the dataset to be indexed.

        It should be noted that this will fail if the `self.data` attribute
        holds anything that cannot be sliced.
        """

        # Attempting to index with a single integer will cause a single
        # datapoint to be returned. While this is normally desired behaviour
        # it is problematic here. Such situations should be treated as
        # a batch of size 1. Thus a single index `i` is treated as `i:i+1` to
        # prevent accidentally deflating the dimensions of the data.
        if isinstance(idx, int):
            idx = slice(idx, idx+1)

        # The `labels` attribute need only be sliced if it is specified.
        if self.labels is None:
            labels = None
        else:
            # To make life easier it is converted to a numpy array for the
            # duration of the slicing operation.
            labels = list(np.asarray(self.labels, dtype=object)[idx])

        return self.__class__(
            self.geometry[idx],
            {k: v[idx, ...] for k, v in self.data.items()},
            labels=labels, meta=self.meta)

    def __add__(self, other: 'DataSetIM'):
        """Permit datasets to be combined"""
        # Ensure that the two datasets are compatible
        if self.data.keys() != other.data.keys():
            raise KeyError('Cannot combine datasets with differing data')

        if not isinstance(other.labels, type(self.labels)):
            raise ValueError('Labels must be either be defined for both '
                             'datasets or neither.')

        if self.meta != other.meta:
            raise ValueError('Datasets with differing metadata cannot be '
                             'merged')

        if self.labels is None:
            labels = None
        else:
            labels = self.labels + other.labels

        return self.__class__(
            self.geometry + other.geometry,
            {k: merge([self.data[k], other.data[k]])
                for k in self.data.keys()},
            labels=labels,
            meta=self.meta
        )

    @classmethod
    def load_data(
            cls, path: str, sources: List[str],
            targets: Union[List[str], Dict[str, str]], pbc: bool = False,
            device: Optional[torch.device] = None) -> 'DataSetIM':
        """Load a collection of data-points into a dataset instance.

        Arguments:
            path: path to the database.
            sources: a list of paths specifying the groups from which data
                should be loaded; one for each system.
            targets: paths relative to `source` specifying the HDF5 datasets
                to load. Results are loaded from each source group, packed
                together and then placed in the class's `data` attribute.
                The key under which the data is stored can be specified using
                a dictionary of the form `{name: path/to/dataset}`, if a list
                is provided then the path is used as the key.
            pbc: Whether read cells.
            device: Device on which to create any new tensors. [DEFAULT=None]

        Returns:
            dataset: a dataset object holding the requested data.

        Notes:
            Structural information must be present in the group specified by
            `source`. Specifically, an attempt will be made to instantiate
            a `Geometry` instance from a sub-group of the name "geometry",
            failing this it will be assumed that the required date is present
            in the main group itself, i.e. `source`.

        """
        # If `targets` is a list then treat it as a dictionary where the keys
        # & values are the same. Prevents having a list/dict conditional later
        if isinstance(targets, list):
            targets = {i: i for i in targets}

        with h5py.File(path) as database:
            # Load in and combine the geometry objects from all of the
            # source systems.
            geometry = reduce(
                operator.add,
                [_load_structure(database[source], pbc=pbc, device=device)
                 for source in sources])

            # Load and pack the requested target datasets from each system.
            data = {
                target_name: pack([
                    torch.tensor(np.array(database[join(source, target)]),
                                 device=device)
                    for source in sources]
                ) for target_name, target in targets.items()}

            if 'label' in database[sources[0]].attrs:
                labels = [database[source].attrs['label']
                          for source in sources]
            else:
                labels = None

            return cls(geometry, data, labels=labels)

    @classmethod
    def load_data_batch(
            cls, path: str, source: str,
            targets: Union[List[str], Dict[str, str]], pbc: bool = False,
            device: Optional[torch.device] = None) -> 'DataSetIM':
        """Load a batch of data from an HDF5 database into a dataset instance.

        Unlike `load_data`, this method assumes that the data stored within
        the database at the location specified by `source` is pre-batched.

        Arguments:
            path: path to the HDF5 database.
            source: path specifying the group within the database from which
                data should be loaded.
            targets: paths relative to `source` specifying the HDF5 datasets
                to load. Results are placed in the class's `data` attribute.
                The key under which the data is stored can be specified using
                a dictionary of the form `{name: path/to/dataset}`, if a list
                is provided then the path is used as the key.
            pbc: Whether read cells.
            device: Device on which to create any new tensors. [DEFAULT=None]

        Returns:
            dataset: a dataset object holding the requested data.

        Notes:
            Structural information must be present in the group specified by
            `source`. Specifically, an attempt will be made to instantiate
            a `Geometry` instance from a sub-group of the name "geometry",
            failing this it will be assumed that the required date is present
            in the main group itself, i.e. `source`.

            Each of the datasets loaded is assumed to be a packed array with
            with a data-point for each entry in the batch. That is to say if
            there are 100 molecules there should be 100 sets of, for example,
            charges. Furthermore, datasets are assumed to batched along the
            the first dimension.

        """
        # If `targets` is a list then treat it as a dictionary where the keys
        # & values are the same. Prevents having a list/dict conditional later
        if isinstance(targets, list):
            targets = {i: i for i in targets}

        with h5py.File(path) as database:

            # Parse geometry data from the database into a `Geometry` instance
            geometry = _load_structure(database[source], pbc=pbc, device=device)

            # Load in the target data, convert it to a torch tensor and
            # add it to the `data` dictionary.
            data = {
                target_name: torch.tensor(
                    database[join(source, target_path)][()], device=device)
                for target_name, target_path in targets.items()}

            # If a `labels` attribute exists, load it, if not then check for
            # `label`. Otherwise assume that it is not present.
            attrs = database[source].attrs
            labels = attrs.get('labels', attrs.get('label', None))

            # If labels where provided then convert to a list; this is needed
            # because h5py loads it as a numpy array.
            if isinstance(labels, np.ndarray):
                labels = list(labels)

            if labels is not None:
                # Ensure the number of labels matches the number of systems
                assert len(labels) == geometry.atomic_numbers.shape[0]

            # Finally, construct and return the dataset object
            return cls(geometry, data, labels)

    def __repr__(self):
        data = ", ".join(sorted(self.data.keys()))
        return f'{self.__class__.__name__}(n={len(self)}, data=[{data}])'

    def __str__(self):
        return repr(self)


def _load_structure(group: Union[Group, File], pbc, **kwargs):
    # Check to see if the geometry information is stored in a subgroup called
    # "geometry".
    if 'geometry' in group:
        return Geometry.from_hdf5(group['geometry'], **kwargs)
    # If not then perhaps the geometry information is stored in datapoint's
    # root directory.
    elif 'atomic_numbers' in group and 'positions' in group and not pbc:
        return Geometry.from_hdf5(group, **kwargs)
    # With pbc.
    elif 'atomic_numbers' in group and 'positions' in group \
        and 'lattice_vector' in group and pbc:
        return Geometry.from_hdf5(group, **kwargs)
    # If neither is true, then throw an error.
    else:
        raise NameError(f'Could not load geometry information from the group '
                        f'{group}. Could not find either i) a subgroup named '
                        '"geometry" or the ii) datasets "atomic_numbers" and '
                        '"positions" and/or "lattice_vector".')
