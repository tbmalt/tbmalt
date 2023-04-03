import os
import urllib

import torch
import h5py
import random
from tbmalt.common.batch import pack
from tbmalt.structures.geometry import batch_chemical_symbols


def dataset_file(input_path: str = 'ani_gdb_s01.h5',
                 output_path: str = 'dataset.h5'):
    """This function works to generate data set for example_02.

    Arguments:
        input_path: Original data set, which could be downloaded for free.
        output_path: Location to where output file should be stored.
    """

    size_split = [[1000, 400], [1000, 400], [1000, 400]]
    seed_list = [[0, 1], [2, 3], [4, 5]]
    name_lay1 = ['run1', 'run2', 'run3']
    name_lay2 = ['train', 'test']

    fa = h5py.File(output_path, 'w')

    with h5py.File(input_path, 'r') as f:

        gg = f['global_group']
        geo_specie = gg.attrs['molecule_specie_global']

        for size, seeds, name1 in zip(size_split, seed_list, name_lay1):

            g1 = fa[name1] if name1 in fa else fa.create_group(name1)

            for isize, seed, name2 in zip(size, seeds, name_lay2):
                # add atom name and atom number
                g2 = g1[name2] if name2 in g1 else g1.create_group(name2)

                _size = int(isize / len(geo_specie))
                data = {}
                for ipro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                    data[ipro] = []

                positions, numbers = [], []

                for imol_spe in geo_specie:
                    random.seed(seed)

                    g = f[imol_spe]

                    g_size = g.attrs['n_molecule']
                    this_size = min(g_size, _size)
                    random_idx = random.sample(range(g_size), this_size)

                    for imol in random_idx:
                        for ipro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                            idata = g[str(imol + 1) + ipro][()]
                            data[ipro].append(torch.from_numpy(idata))

                        positions.append(torch.from_numpy(g[str(imol + 1) + 'position'][()]))
                        numbers.append(torch.from_numpy(g.attrs['numbers']))

                for ipro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                    data[ipro] = pack(data[ipro])

                for ii, (num, pos) in enumerate(zip(numbers, positions)):
                    symbol = batch_chemical_symbols(num)
                    g3 = g2.create_group(''.join(symbol) + str(ii))
                    g3.attrs['label'] = symbol
                    g3.create_dataset('atomic_numbers', data=num)
                    g3.create_dataset('positions', data=pos)

                    for pro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                        g3.create_dataset(pro, data=data[pro][ii])


# Step 1: Download geometry with FHI-aims calculations
link = 'https://seafile.zfn.uni-bremen.de/f/76166cf69acb47b69365/?dl=1'
urllib.request.urlretrieve(link, 'aims_6000_01.hdf')

# Step 2: generate data set which satisfy TBMaLT format
input_path = 'aims_6000_01.hdf'
output_path = 'dataset.h5'
dataset_file(input_path, output_path)

# Step 3: remove the downloaded file
os.system('rm aims_6000_01.hdf')
