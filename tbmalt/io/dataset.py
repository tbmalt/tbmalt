# -*- coding: utf-8 -*-
from typing import List, Literal
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as Loader
from torch import Tensor

from tbmalt.common.batch import pack


class Dataloader(Loader):

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass


    @classmethod
    def load_reference(dataset, size, properties, **kwargs):
        """Load reference from hdf type data."""
        out_type = kwargs.get('output_type', Tensor)
        test_ratio = kwargs.get('test_ratio', 1.0)

        # Choice about how to select data
        choice = kwargs.get('choice', 'squeeze')

        data = {}
        for ipro in properties:
            data[ipro] = []

        positions, numbers, data['cell'] = [], [], []

        with h5py.File(dataset, 'r') as f:
            gg = f['global_group']
            molecule_specie = gg.attrs['geometry_specie_global']
            try:
                data['n_band_grid'] = gg.attrs['n_band_grid']
            except:
                pass

            # Determine each geometry specie size
            if size < len(molecule_specie):
                molecule_specie = molecule_specie[:size]
                _size = 1
            else:
                _size = int(size / len(molecule_specie))

            # add atom name and atom number
            for imol_spe in molecule_specie:
                g = f[imol_spe]
                g_size = g.attrs['n_geometries']
                isize = min(g_size, _size)

                if choice == 'squeeze':
                    start = 0 if test_ratio == 1.0 else int(
                        isize * (1 - test_ratio))

                    # loop for the same molecule specie
                    for imol in range(start, isize):

                        for ipro in properties:  # loop for each property
                            idata = g[str(imol + 1) + ipro][()]
                            try:
                                if isinstance(idata, np.ndarray):
                                    data[ipro].append(
                                        Dataset.to_out_type(idata, out_type))
                                else:
                                    data[ipro].append(idata)
                            except:
                                pass

                        positions.append(Dataset.to_out_type(
                            g[str(imol + 1) + 'position'][()], out_type))
                        numbers.append(Dataset.to_out_type(
                            g.attrs['numbers'], out_type))
                        try:
                            data['cell'].append(Dataset.to_out_type(
                                g[str(imol + 1) + 'cell'][()], out_type))
                        except:
                            pass

                elif choice == 'random':
                    ind = random.sample(torch.arange(g_size).tolist(), isize)
                    # loop for the same molecule specie
                    for imol in ind:
                        for ipro in properties:  # loop for each property
                            idata = g[str(imol + 1) + ipro][()]
                            try:
                                if isinstance(idata, np.ndarray):
                                    data[ipro].append(
                                        Dataset.to_out_type(idata, out_type))
                                else:
                                    data[ipro].append(idata)
                            except:
                                data[ipro].append(idata)

                        positions.append(Dataset.to_out_type(
                            g[str(imol + 1) + 'position'][()], out_type))
                        numbers.append(Dataset.to_out_type(
                            g.attrs['numbers'], out_type))
                        try:
                            data['cell'].append(Dataset.to_out_type(
                                g[str(imol + 1) + 'cell'][()], out_type))
                        except:
                            pass

        if out_type is Tensor:
            for ipro in properties:  # loop for each property
                try:
                    data[ipro] = pack(data[ipro])
                except:
                    data[ipro] = data[ipro]
            try:
                data['cell'] = pack(data['cell'])
            except:
                pass

        return pack(numbers), pack(positions), data
