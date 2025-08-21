from os.path import join
import numpy as np
import pytest
import h5py
import tempfile
import torch
from tbmalt.io.dataset import DataSetIM
from tbmalt import Geometry
from tbmalt.common.batch import pack


torch.set_default_dtype(torch.float64)

def arange_of_shape(*args, dtype=torch.get_default_dtype(), **kwargs):
    return torch.arange(
        np.prod(args),
        dtype=dtype,
        **kwargs).view(*args)


def test_single_database_io(device):
    # This is not an in-depth test but it will suffice for now.

    with tempfile.TemporaryDirectory() as tempdir:
        path = join(tempdir, 'test_1.h5')


        geometry_1 = Geometry(
            torch.tensor([1, 1], device=device),
            arange_of_shape(2, 3, device=device))

        geometry_2 = Geometry(
            torch.tensor([6, 1, 1, 1, 1], device=device),
            arange_of_shape(5, 3, device=device))

        data_1 = {
            f'dataset_{i}d': arange_of_shape(*j, device=device) for i, j in
            enumerate([(2, 6), (2, 6), (2, 6, 7)], start=1)}

        data_2 = {
            f'dataset_{i}d': arange_of_shape(*j, device=device) for i, j in
            enumerate([(5, 6), (5, 6), (5, 6, 7)], start=1)}

        with h5py.File(path, 'w') as database:
            system_1 = database.create_group('system_1')
            geometry_1.to_hdf5(system_1.create_group('geometry'))

            for k, v in data_1.items():
                system_1.create_dataset(k, data=v.cpu().numpy())

            system_1.create_dataset('no_load', data=arange_of_shape(1))
            system_1.attrs['label'] = '1'

            system_2 = database.create_group('some/other/dir/system_2')
            geometry_2.to_hdf5(system_2.create_group('geometry'))

            for k, v in data_2.items():
                system_2.create_dataset(k, data=v.cpu().numpy())

            system_2.create_dataset('no_load', data=arange_of_shape(1))
            system_2.attrs['label'] = '2'

        dataset = DataSetIM.load_data(
            path, ['system_1', 'some/other/dir/system_2'],
            ['dataset_1d', 'dataset_2d', 'dataset_3d'],
            device=device)

        # Ensure that the geometry information is read in and packed
        # correctly.
        check_1 = dataset.geometry == geometry_1 + geometry_2
        assert check_1, "did not load geometry data correctly"

        # Confirm that the code will not load data that it has not been told
        # to.
        check_2 = 'no_load' not in dataset.data
        assert check_2, "loaded a dataset without instruction"

        # Verify that the data is loaded and packed correctly
        for k, v in dataset.data.items():
            check_3 = torch.all(v == pack([data_1[k], data_2[k]]))
            assert check_3, f"dataset {k} has been mangled"

        # Check that labels are read correctly
        check_4 = ['1', '2'] == dataset.labels
        assert check_4, "labels were not parsed correctly"

        # Check that targets can be specified via dictionary
        dataset = DataSetIM.load_data(
            path, ['system_1', 'some/other/dir/system_2'],
            {'X': 'dataset_1d'}, device=device)

        check_5 = 'X' in dataset.data
        if check_5:
            ref = pack([data_1['dataset_1d'], data_2['dataset_1d']])

            check_5 = check_5 and torch.all(ref == dataset.data['X'])

        assert check_5, "Parsing targets as dictionaries failed"

        check_6 = dataset.geometry.device == device
        assert check_6, 'Geometry was placed on the wrong device'

        for v in dataset.data.values():
            check_7 = v.device == device
            assert check_7, 'data was placed on the wrong device'


def test_batched_dataset_io(device):

    with tempfile.TemporaryDirectory() as tempdir:
        path = join(tempdir, 'test_2.h5')

        geometry = Geometry(
            [
                torch.tensor([1, 1], device=device),
                torch.tensor([6, 1, 1, 1, 1], device=device)
            ], [
                arange_of_shape(2, 3, device=device),
                arange_of_shape(5, 3, device=device)
            ])

        data = {
            f'dataset_{i}d': pack(
                [arange_of_shape(*j, device=device),
                 arange_of_shape(*k, device=device)])
            for i, (j, k) in
            enumerate(zip(
                [(2, 6), (2, 6), (2, 6, 7)],
                [(5, 6), (5, 6), (5, 6, 7)]), start=1)}

        with h5py.File(path, 'w') as database:
            system = database.create_group('batch_1')
            geometry.to_hdf5(system.create_group('geometry'))

            for k, v in data.items():
                system.create_dataset(k, data=v.cpu().numpy())

            system.create_dataset('no_load', data=arange_of_shape(2))
            system.attrs['label'] = ['1', '2']

        dataset = DataSetIM.load_data_batch(
            path, 'batch_1',
            ['dataset_1d', 'dataset_2d', 'dataset_3d'],
            device=device)

        # Ensure that the geometry information is read in correctly.
        check_1 = dataset.geometry == geometry
        assert check_1, "did not load geometry data correctly"

        # Confirm that the code will not load data that it has not been told
        # to.
        check_2 = 'no_load' not in dataset.data
        assert check_2, "loaded a dataset without instruction"

        # Verify that the data is loaded correctly
        for k, v in dataset.data.items():
            check_3 = torch.all(v == data[k])
            assert check_3, f"dataset {k} has been mangled"

        # Check that labels are read correctly
        check_4 = ['1', '2'] == dataset.labels
        assert check_4, "labels were not parsed correctly"

        # Check that targets can be specified via dictionary
        dataset = DataSetIM.load_data_batch(
            path, 'batch_1',
            {'X': 'dataset_1d'}, device=device)

        check_5 = 'X' in dataset.data
        if check_5:
            check_5 = check_5 and torch.all(
                data['dataset_1d'] == dataset.data['X'])

        assert check_5, "Parsing targets as dictionaries failed"


def _check_slice(dataset, idx):
    dataset_sliced = dataset[idx]

    # Check that the geometry is sliced correctly
    check_1 = dataset_sliced.geometry == dataset.geometry[idx]
    assert check_1, 'Geometry object incorrectly sliced when indexing DataSetIM'

    check_2 = dataset_sliced.geometry.device == dataset.geometry.device
    assert check_2, 'Geometry moved device when slicing DataSetIM'

    # Repeat for the data
    for key in dataset.data.keys():
        check_3 = torch.all(dataset_sliced.data[key] == dataset.data[key][idx, ...])
        assert check_3, 'data element incorrectly sliced when indexing DataSetIM'

        check_4 = dataset_sliced.data[key].device == dataset.data[key][idx, ...].device
        assert check_4, 'data moved device when slicing DataSetIM'

    # Finally, check that labels are okay, if they are present.
    if dataset.labels is not None:
        check_3 = dataset_sliced.labels == list(np.array(dataset.labels)[idx])
        assert check_3, 'labels incorrectly sliced when indexing DataSetIM'


def test_dataset_general(device):

    geometry = Geometry(
        arange_of_shape(6, 5, device=device),
        torch.rand(6, 5, 3, device=device)
    )

    labels = ['a', 'b', 'c', 'd', 'e', 'f']

    data = {
        'test_data_1d_data': torch.rand(6, device=device),
        'test_data_2d_data': torch.rand(6, 2, device=device),
        'test_data_3d_data': torch.rand(6, 5, 3, device=device)}

    # Ensure that a simple dataset can be created without error; both with and
    # without labels.
    dataset_a = DataSetIM(geometry, data)
    dataset_b = DataSetIM(geometry, data, labels)

    # Confirm that an error is raised if the number of labels does not match
    # the number of systems.
    with pytest.raises(ValueError, match='If labels are provided then*'):
        DataSetIM(geometry, data, ['x'])

    # Check that the `data` attribute is correctly safety checked.
    with pytest.raises(AssertionError, match='data entry*'):
        DataSetIM(geometry, {'test': np.random.rand(6)})

    with pytest.raises(AssertionError, match='The length of the first *'):
        DataSetIM(geometry, {'test': torch.rand(5, 3)})

    # Verify that the length is correct
    check_1 = len(dataset_a) == len(dataset_a) == 6
    assert check_1, 'calculated length of DataSetIM is in error'

    # Check that the object can be sliced without getting mangled
    for idx in [slice(0, 3), slice(3, -1), [0, 2, 4]]:
        _check_slice(dataset_a, idx)
        _check_slice(dataset_b, idx)

    # Prove that datasets can be combined together
    dataset_a_2 = dataset_a[0:3] + dataset_a[3:]
    dataset_b_2 = dataset_b[0:3] + dataset_b[3:]

    msg = "DataSetIM instances were mangled during merger"

    assert dataset_a.geometry == dataset_a_2.geometry, msg
    assert dataset_b.geometry == dataset_b_2.geometry, msg

    check_2 = dataset_a_2.geometry.device == dataset_a.geometry.device
    assert check_2, 'Geometry moved device when combining datasets'


    for key in dataset_a.data.keys():
        assert torch.all(dataset_a.data[key] == dataset_a_2.data[key]), msg
        assert torch.all(dataset_b.data[key] == dataset_b_2.data[key]), msg

        check_3 = dataset_b.data[key].device == dataset_b_2.data[key].device
        assert check_3, 'data moved device when combining datasets'

    assert dataset_b.labels == dataset_b_2.labels, msg
