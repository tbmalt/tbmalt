# -*- coding: utf-8 -*-
"""Test skf-io operations."""
import os
from contextlib import contextmanager
import pytest
import h5py
import torch
from tbmalt.io.skf import Skf


####################
# Helper Functions #
####################
def skf_files():
    """Returns a generator that loops over skf test files and their contents.

    File 1: Homo-atomic system with a repulsive polynomial & spline.
    File 2: Same as file 1 but includes f orbitals.
    File 3: Hetero-atomic system with a repulsive polynomial & spline.
    File 4: Hetero-atomic system without a repulsive polynomial or spline.
    File 5: Same as file 4 but with some commonly encountered errors.

    Returns:
        path: Path to skf file.
        args:
            has_atomic: True if the file contains parsable atomic data.
            has_r_poly: True if the file contains a valid repulsive polynomial.
            has_r_spline: True if the file contains a repulsive spline.
    """
    path = 'tests/unittests/data/io/skf'
    files = {'File_1_Au-Au.skf': (True, True, True),
             'File_2_Au-Au.skf': (True, True, True),
             'File_3_Au-Au.skf': (False, True, True),
             'File_4_Au-Au.skf': (False, False, False),
             'File_5_Au-Au.skf': (False, False, False)}

    for name, args in files.items():
        yield os.path.join(path, name), args


@contextmanager
def file_cleanup(path_in, path_out):
    """Context manager for cleaning up temporary files once no longer needed."""
    # Loads the file located at `path_in` and saves it to `path_out`
    Skf.read(path_in).write(path_out)
    try:
        yield
    finally:
        # Delete the file once finished
        if os.path.exists(path_out):
            os.remove(path_out)


def _ref_interaction(l1, l2, i_type, device=None):
    """Random looking data for testing electronic integrals with.

    For a given electronic integral, as defined by its azimuthal pair an its
    integral type (i.e. H/S), this function will return some random looking
    data that is unique to that interaction. This prevents having to hard
    code in the expected results.
    """
    l_min = min(l1, l2) + 1
    # Unique noise lookup table
    noise_lut = torch.tensor([97 / 13, 71 / 47, 31 / 89, 11 / 17], device=device)
    c = (noise_lut[l1] + noise_lut[l2]) % 1  # Constant noise offset
    b = (noise_lut[:l_min] % 1)  # Bond order specific noise
    # Generate 3 lines worth of data
    data = (torch.linspace(0, 1, 3 * l_min, device=device) + c
            ).view(3, l_min) + b
    # Invert the sign if this is an overlap matrix
    if i_type.lower() == 's': data = data * -1
    elif i_type.lower() != 'h': raise ValueError('`type` must be "h" or "s".')
    # Append 3 lines worth of 0 to the start of the data & return
    return torch.cat((torch.zeros_like(data, device=device), data), 0).T.squeeze()


def _check_skf_contents(skf, has_atomic, has_r_poly, has_r_spline, device):
    """Helper function to test the contents of an `Skf` instances.

    Arguments:
        has_atomic: True if the file contains parsable atomic data.
        has_r_poly: True if the file contains a valid repulsive polynomial.
        has_r_spline: True if the file contains a repulsive spline.
        device: Device on on which the `Skf` object should be created.
    """
    d = {'device': device}

    def check_it(attrs, src):
        for name, ref in attrs.items():
            n = f'{src.__class__.__qualname__}.{name}'
            pred = src.__getattribute__(name)
            # Check is done this way to ensure that the error message is
            # correctly displayed.
            if dev_check := pred.device == device:
                assert torch.allclose(pred, ref), f'`{n}` is in error'
            else:
                assert dev_check, f'`{n}` is on the wrong device'

    # Check integral grid integrity
    check_grid = torch.allclose(skf.grid, torch.arange(6., **d) + 1.)
    assert check_grid, '`Skf.grid` is in error'

    # Ensure atomic data was parsed
    if skf.atomic:
        check_it({
            'on_sites': torch.arange(len(skf.on_sites), **d) + 0.1,
            'hubbard_us': torch.arange(len(skf.hubbard_us), **d) + 0.2,
            'occupations': torch.arange(len(skf.occupations), **d) + 0.3,
            'mass': torch.tensor(1.234, **d)}, skf)
        if not has_atomic:  # If atomic data is "found" when told there is none
            pytest.fail('Unexpectedly found atomic data')
    elif has_atomic:
        pytest.fail('Failed to locate atomic data')

    # Repulsive polynomial
    if skf.r_poly is not None:
        check_it({'coef': torch.arange(8., **d) + 0.1,
                  'cutoff': torch.tensor(5.321, **d)}, skf.r_poly)
        if not has_r_poly:
            pytest.fail('Unexpectedly found valid repulsive polynomial')
    elif has_r_poly:
        pytest.fail('Failed to locate valid repulsive polynomial')

    # Verify the integrals are read in correctly:
    check_h = all([torch.allclose(v, _ref_interaction(*k, 'h', device))
                   for k, v in skf.hamiltonian.items()])
    check_s = all([torch.allclose(v, _ref_interaction(*k, 's', device))
                   for k, v in skf.overlap.items()])
    check_hs = check_h and check_s
    assert check_hs, 'Electronic integrals are in error'

    # Repulsive spline
    if skf.r_spline is not None:
        check_it({
            'exp_coef': torch.arange(3., **d), 'grid': torch.linspace(0, 2, 6, **d),
            'spline_coef': torch.linspace(0, 1, 20, **d).view(5, 4),
            'tail_coef': torch.arange(6., **d), 'cutoff': torch.tensor(10.0, **d)},
            skf.r_spline)
        if not has_r_spline:
            pytest.fail('Unexpectedly found repulsive spline')
    elif has_r_spline:
        pytest.fail('Failed to locate repulsive spline')


##################
# Test Functions #
##################
def test_skf_from_skf(device):
    """Ensure skf files are correctly read."""
    for path, args in skf_files():
        _check_skf_contents(Skf.from_skf(path, device=device),
                            *args, device)


def test_skf_to_skf(device):
    """Ensure skf files are correctly written.

    Notes:
        This test operates by checking for aberrations after writing a known
        good `Sfk` instance to disc and reading it back in again. As such it
        is dependent on the `Skf.from_skf` method.
    """
    for path, args in skf_files():
        with file_cleanup(path, 'X-X.skf'):
            _check_skf_contents(Skf.from_skf('X-X.skf', device=device),
                                *args, device)


def test_skf_hdf5_io(device):
    """Check `Sfk` instances can be written to and read from HDF5 databases.

    Notes:
        This method is dependent on the `Skf.from_sfk` method.
    """
    for path, args in skf_files():
        with file_cleanup(path, 'skfdb.hdf5'):
            with h5py.File('skfdb.hdf5', 'r') as db:
                _check_skf_contents(
                    Skf.from_hdf5(db['Au-Au'], device=device),
                    *args, device)


def test_read(device):
    """Ensure the `Skf.read` method operates as anticipated."""
    skf, args = next(skf_files())

    # Check 1: ensure read method redirects to from_skf/from_hdf5 and that the
    # device information is passed on. Note that no actual check is performed
    # as previous tests would have failed & upcoming test will fail if did/does
    # not work correctly.

    # Check 2: warning issued when passing in ``atom_pair`` for an skf file.
    with pytest.warns(UserWarning, match='"atom_pair" argument is*'):
        _check_skf_contents(Skf.read(skf, atom_pair=[0, 0], device=device),
                            *args, device)

    with file_cleanup(skf, 'skfdb.hdf5'):
        # Check 3: read should not need the ``atom_pair`` argument to read form an
        # HDF5 database that only has a single system in it.
        _check_skf_contents(Skf.read('skfdb.hdf5', device=device), *args, device)

        # Check 4: an exception should be raise if multiple entries are present in
        # the source HDF5 database but the ``atom_pair`` argument was not given.
        temp = Skf.read(skf, device=device)
        temp.atom_pair = (ap := torch.tensor([6, 6]))
        temp.write('skfdb.hdf5')

        with pytest.raises(ValueError):
            Skf.read('skfdb.hdf5')

        # Check 5: correct pair is returned
        check_4 = (Skf.read('skfdb.hdf5', atom_pair=ap).atom_pair == ap).all()
        assert check_4, 'Wrong atom pair returned'


def test_write():
    """Simple test of the `Skf.write` method's general operation."""
    skf, args = next(skf_files())
    with file_cleanup(skf, 'X-X.skf'), file_cleanup(skf, 'skfdb.hdf5'):
        # Check 1: exception raised when overwriting an skf-file or HDF5-group
        # without the `overwrite` argument set to True
        with pytest.raises(FileExistsError):
            Skf.read(skf).write('X-X.skf')

        with pytest.raises(FileExistsError):
            Skf.read(skf).write('skfdb.hdf5')

        # Check 2: overwriting should be permitted with the `overwrite` argument.
        # No such error should be encountered whe the
        Skf.read(skf).write('X-X.skf', overwrite=True)
        Skf.read(skf).write('skfdb.hdf5', overwrite=True)
