# -*- coding: utf-8 -*-
"""Perform tests on functions which read SKF files.

This test will test precision and device of every parameters from SKF files,
write SKF data to hdf binary and then read parameters from binary.
"""
import os
from itertools import combinations_with_replacement
import torch
import h5py
import numpy as np
from tbmalt.io.skf import Skf
torch.set_default_dtype(torch.float64)


def test_hs_mio_homo(device):
    """Test data from mio SKF files for homo carbon and carbon."""
    path_to_mio = './tests/unittests/data/slko/mio/C-C.skf'

    # Step 1.1: Load  all SKF data
    sk = Skf.read(path_to_mio, 'from_skf', torch.tensor([6, 6]), device=device)

    g_step, n_grids = 0.02, 500
    onsite = torch.tensor([0.0, -0.19435511, -0.50489172], device=device)
    U = torch.tensor([0.341975, 0.387425, 0.3647], device=device)
    occupations = torch.tensor([0.0, 2.0, 2.0], device=device)
    mass, rcut = 12.01, 0.0
    h_20 = torch.tensor([
        0.0, 0.0, 0.0, 0.0, 0.0, -9.857627770306e-01, -1.322146418755e+0,
        0.0, -3.205032176533e-01, -1.149953917193e+00], device=device)
    s_40 = torch.tensor([
        0.0, 0.0, 0.0, 0.0, 0.0, 4.765261147266e-01, 8.052410954677e-01,
        0.0, -2.874596614007e-01, 8.089389614993e-01], device=device)
    r_int, r_cutoff = 48, 4.3
    r_a123 = torch.tensor([2.151029456234113, 3.917667206325493
                           , -0.4605879014976964], device=device)
    r_table_1 = torch.tensor([3.031622, -7.473244053840698, 9.005535480973546,
                              -8.40772837389725], device=device)
    r_grid = torch.linspace(1.2, 3.04, 47, device=device)
    r_long_grid = torch.tensor([3.4, 4.3], device=device)
    r_c_0to5 = torch.tensor([0.016, -0.006590813456982203, -0.02356970905317782,
        -0.09209220073124012, 0.2061755069509315, -0.1001089592255145], device=device)

    # check tolerance for all the parameters
    assert abs(sk.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert abs(sk.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert torch.max(abs(sk.hamiltonian[19] - h_20)) < 1E-14, \
        'hamiltonian tolerance check'
    assert sk.hamiltonian.shape[1] == 10, 'hamiltonian shape error'
    assert sk.overlap.shape[1] == 10, 'overlap shape error'
    assert torch.max(abs(sk.overlap[39] - s_40)) < 1E-14, \
        'overlap tolerance check'
    assert torch.max(abs(sk.onsite - onsite)) < 1E-14, 'onsite tolerance check'
    assert torch.max(abs(sk.U - U)) < 1E-14, 'U tolerance check'
    assert torch.max(abs(sk.occupations - occupations)) < 1E-14, \
        'occupations tolerance check'
    assert abs(sk.mass - mass) < 1E-14, 'mass tolerance check'
    assert abs(sk.rcut - rcut) < 1E-14, 'rcut tolerance check'
    assert abs(sk.r_int - r_int) < 1E-14, 'r_int tolerance check'
    assert abs(sk.r_cutoff - r_cutoff) < 1E-14, 'r_cutoff tolerance check'
    assert torch.max(abs(sk.r_a123 - r_a123)) < 1E-14, 'r_a123 tolerance check'
    assert torch.max(abs(sk.r_table[1] - r_table_1)) < 1E-14, \
        'r_table tolerance check'
    assert torch.max(abs(sk.r_grid[: -1] - r_grid)) < 1E-14, \
        'r_grid tolerance check'
    assert torch.max(abs(sk.r_long_grid - r_long_grid)) < 1E-14, \
        'r_long_grid tolerance check'
    assert torch.max(abs(sk.r_c_0to5 - r_c_0to5)) < 1E-14, \
        'r_c_0to5 tolerance check'

    # check device persistence
    assert sk.hamiltonian.device == device, 'hamiltonian device persistence'
    assert sk.overlap.device == device, 'overlap device persistence'
    assert sk.onsite.device == device, 'onsite device persistence'
    assert sk.U.device == device, 'Hubbert U persistence'
    assert sk.occupations.device == device, 'occupations device persistence'
    assert sk.r_a123.device == device, 'r_a123 device persistence'
    assert sk.r_table.device == device, 'r_table device persistence'
    assert sk.r_grid.device == device, ' r_grid device persistence'
    assert sk.r_long_grid.device == device, 'r_long_grid device persistence'
    assert sk.r_c_0to5.device == device, 'r_c_0to5 device persistence'

    # Step 1.2: Load  all SKF data with mask
    sk12 = Skf.read(path_to_mio, 'from_skf', torch.tensor([6, 6]),
                    mask_hs=True, interactions=_interaction[1], device=device)
    assert abs(sk12.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert abs(sk12.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert sk12.hamiltonian.shape[1] == 4, 'hamiltonian shape error'
    assert sk12.overlap.shape[1] == 4, 'overlap shape error'
    assert torch.max(abs(sk12.overlap[39, 0] - s_40[5])) < 1E-14, \
        'masked overlap tolerance check'
    assert torch.max(abs(sk12.overlap[39, 2] - s_40[8])) < 1E-14, \
        'masked overlap tolerance check'

    assert sk12.hamiltonian.device == device, 'hamiltonian device persistence'
    assert sk12.U.device == device, 'Hubbert U persistence'
    assert sk12.occupations.device == device, 'occupations device persistence'
    assert sk12.r_a123.device == device, 'r_a123 device persistence'

    # Step 2.1: write data to hdf, and then read the hdf
    Skf.to_hdf('cc.hdf', sk)
    sk_b = Skf.read('./cc.hdf', 'from_hdf', torch.tensor([6, 6]), device=device)

    # check tolerance for all the parameters
    assert abs(sk_b.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert torch.max(abs(sk_b.hamiltonian[19] - h_20)) < 1E-14, \
        ' hamiltonian tolerance check'
    assert torch.max(abs(sk_b.overlap[39] - s_40)) < 1E-14, \
        'overlap tolerance check'
    assert torch.max(abs(sk_b.onsite - onsite)) < 1E-14, \
        'onsite tolerance check'
    assert torch.max(abs(sk_b.occupations - occupations)) < 1E-14, \
        'occupations tolerance check'
    assert abs(sk_b.rcut - rcut) < 1E-14, ' rcut tolerance check'
    assert abs(sk_b.r_cutoff - r_cutoff) < 1E-14, 'Tolerance check'
    assert torch.max(abs(sk_b.r_table[1] - r_table_1)) < 1E-14, \
        'r_table tolerance check'
    assert torch.max(abs(sk_b.r_long_grid - r_long_grid)) < 1E-14, \
        'r_long_grid tolerance check'

    # check device persistence
    assert sk_b.hamiltonian.device == device, 'hamiltonian device persistence'
    assert sk_b.overlap.device == device, 'overlap device persistence'
    assert sk_b.onsite.device == device, 'onsite device persistence'
    assert sk_b.occupations.device == device, 'occupations device persistence'
    assert sk_b.r_table.device == device, 'r_table device persistence'
    assert sk_b.r_long_grid.device == device, 'r_long_grid device persistence'

    # Step 2.2: write data to hdf, and then read the hdf with mask
    sk_b2 = Skf.read('./cc.hdf', 'from_hdf', torch.tensor([6, 6]), mask_hs=True,
                    interactions=_interaction[1], device=device)

    # check tolerance for all the parameters
    assert abs(sk_b2.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert torch.max(abs(sk_b2.hamiltonian[19, 1] - h_20[6])) < 1E-14, \
        ' hamiltonian tolerance check'
    assert abs(sk_b2.rcut - rcut) < 1E-14, ' rcut tolerance check'
    assert abs(sk_b2.r_cutoff - r_cutoff) < 1E-14, 'Tolerance check'
    assert torch.max(abs(sk_b2.r_table[1] - r_table_1)) < 1E-14, \
        'r_table tolerance check'
    assert torch.max(abs(sk_b2.r_long_grid - r_long_grid)) < 1E-14, \
        'r_long_grid tolerance check'

    # check device persistence
    assert sk_b2.hamiltonian.device == device, 'hamiltonian device persistence'
    assert sk_b2.overlap.device == device, 'overlap device persistence'
    assert sk_b2.occupations.device == device, 'occupations device persistence'
    assert sk_b2.r_long_grid.device == device, 'r_long_grid device persistence'

    os.remove('cc.hdf')

    # Step 3: write data to hdf, and target is File object, then read data
    target = h5py.File('cc.hdf', 'w')
    Skf.to_hdf(target, sk)
    sk_b2 = Skf.read('./cc.hdf', 'from_hdf', torch.tensor([6, 6]), device=device)

    # check tolerance for all the parameters
    assert abs(sk_b2.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert torch.max(abs(sk_b2.overlap[39] - s_40)) < 1E-14, \
        'overlap tolerance check'
    assert torch.max(abs(sk_b2.U - U)) < 1E-14, 'Hubbert U tolerance check'
    assert abs(sk_b2.mass - mass) < 1E-14, 'mass tolerance check'
    assert abs(sk_b2.r_int - r_int) < 1E-14, 'r_int tolerance check'
    assert torch.max(abs(sk_b2.r_a123 - r_a123)) < 1E-14, \
        'r_a123 tolerance check'
    assert torch.max(abs(sk_b2.r_grid[: -1] - r_grid)) < 1E-14, \
        'r_grid tolerance check'
    assert torch.max(abs(sk_b2.r_c_0to5 - r_c_0to5)) < 1E-14, \
        'r_c_0to5 tolerance check'
    os.remove('cc.hdf')

    # check device persistence
    assert sk.overlap.device == device, 'overlap device persistence'
    assert sk.U.device == device, 'Hubbert U device persistence'
    assert sk.r_a123.device == device, 'r_a123 device persistence'
    assert sk.r_grid.device == device, 'r_grid device persistence'
    assert sk.r_c_0to5.device == device, 'r_c_0to5 device persistence'


def test_hs_mio_hetero(device):
    """Test data from mio SKF files for hetero carbon and hydrogen."""
    path_to_cc_mio = './tests/unittests/data/slko/mio/C-H.skf'

    # Step 1: Load all SKF data
    sk = Skf.read(path_to_cc_mio, 'from_skf', torch.tensor([6, 1]))

    g_step, n_grids = 0.02, 500
    h_37 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         4.274043191660e-01, -5.938433804230e-01])
    s_498 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          3.631580613674e-05, -6.869810941702e-05])
    r_int, r_cutoff = 34, 3.5
    r_a123 = torch.tensor([
        2.198518512629381, 2.147421636649093, -0.1560071349326178])
    r_table_6 = torch.tensor([
        0.242829, -0.8190447866498586, 1.076895288020857, -0.7381405443593912])
    r_long_grid = torch.tensor([2.84, 3.5])
    r_c_0to5 = torch.tensor([
        -0.01, 0.02007634639672507, -0.008500295606269857, 0.1099349367199619,
        -0.2904128801769102, 0.1912556086105955])

    # check tolerance for all the parameters
    assert abs(sk.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert abs(sk.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert torch.max(abs(sk.hamiltonian[36] - h_37)) < 1E-14, \
        'hamiltonian tolerance check'
    assert torch.max(abs(sk.overlap[497] - s_498)) < 1E-14, \
        'overlap tolerance check'
    assert sk.onsite is None, 'hetero do not has onsite'
    assert sk.U is None, 'Hetero do not has Hubbert U'
    assert abs(sk.r_int - r_int) < 1E-14, 'r_int tolerance check'
    assert abs(sk.r_cutoff - r_cutoff) < 1E-14, 'r_cutoff tolerance check'
    assert torch.max(abs(sk.r_a123 - r_a123)) < 1E-14, 'r_a123 tolerance check'
    assert torch.max(abs(sk.r_table[5] - r_table_6)) < 1E-14, \
        'r_table tolerance check'
    assert torch.max(abs(sk.r_long_grid - r_long_grid)) < 1E-14, \
        'r_long_grid tolerance check'
    assert torch.max(abs(sk.r_c_0to5 - r_c_0to5)) < 1E-14, \
        'r_c_0to5 tolerance check'

    # Step 2: do not read hamiltonian and repulsive data
    sk2 = Skf.read(
        path_to_cc_mio, 'from_skf', torch.tensor([6, 1]),
        read_hamiltonian=False, read_overlap=True, read_repulsive=False)

    # check tolerance for all the parameters
    assert abs(sk2.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert abs(sk2.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert torch.max(abs(sk2.overlap[497] - s_498)) < 1E-14, \
        'overlap tolerance check'
    assert sk2.onsite is None, 'Hetero do not has onsite'
    assert sk2.U is None, 'Hetero do not has Hubbert U'

    # Step 2: do not read hamiltonian and overlap data
    sk3 = Skf.read(
        path_to_cc_mio, 'from_skf', torch.tensor([6, 1]),
        read_hamiltonian=False, read_overlap=False, read_repulsive=True)

    # check tolerance for all the parameters
    assert abs(sk3.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert abs(sk3.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert sk3.onsite is None, 'Hetero do not has onsite'
    assert sk3.U is None, 'Hetero do not has Hubbert U'
    assert abs(sk3.r_int - r_int) < 1E-14, 'r_int tolerance check'
    assert abs(sk3.r_cutoff - r_cutoff) < 1E-14, 'r_cutoff tolerance check'
    assert torch.max(abs(sk3.r_a123 - r_a123)) < 1E-14, 'r_a123 tolerance check'
    assert torch.max(abs(sk3.r_table[5] - r_table_6)) < 1E-14, \
        'r_table tolerance check'
    assert torch.max(abs(sk3.r_long_grid - r_long_grid)) < 1E-14, \
        'r_long_grid tolerance check'
    assert torch.max(abs(sk3.r_c_0to5 - r_c_0to5)) < 1E-14, \
        'r_c_0to5 tolerance check'


def test_auorg(device):
    """Test data from mio SKF files for homo Au and Au."""
    path_to_cc_auorg = './tests/unittests/data/slko/auorg/Au-Au.skf'
    hs = torch.from_numpy(np.loadtxt(
        './tests/unittests/data/slko/auorg/hs.dat')).to(torch.get_default_dtype())
    reptab = torch.from_numpy(np.loadtxt(
        './tests/unittests/data/slko/auorg/rep_table.dat')).to(
            torch.get_default_dtype())

    # Load data without mask hamiltonian and overlap
    sk = Skf.read(path_to_cc_auorg, 'from_skf', torch.tensor([79, 79]))

    g_step, n_grids = 0.02, 919
    onsite = torch.tensor([
        -2.531805351853E-01, -2.785941987392E-02, -2.107700668744E-01])
    U = torch.tensor([3.610611525251E-01, 2.556040155551E-01, 2.556040155551E-01])
    occupations = torch.tensor([10.0, 0.0, 1.0])
    mass, rcut = 1.969670000000E+02, 0.0
    r_int, r_cutoff = 51, 7.24486306644
    r_a123 = torch.tensor([4.06611088035, 14.7376297055, -0.0335996827064])
    r_grid_3 = torch.tensor([4.79486306644])
    r_long_grid = torch.tensor([7.19486306644, 7.24486306644])
    r_c_0to5 = torch.tensor([
        -0.000222141403944, 0.00120844133494, -0.0021973106215, 15.0028917489,
        -458.435896775, 3702.64159992])

    # check tolerance for all the parameters
    assert abs(sk.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert abs(sk.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert torch.max(abs(sk.hamiltonian - hs[..., :10])) < 1E-14, \
        'hamiltonian tolerance check'
    assert torch.max(abs(sk.overlap - hs[..., 10:])) < 1E-14, \
        'overlap tolerance check'
    assert torch.max(abs(sk.onsite - onsite)) < 1E-14, 'onsite tolerance check'
    assert torch.max(abs(sk.U - U)) < 1E-14, 'Hubbert U tolerance check'
    assert torch.max(abs(sk.occupations - occupations)) < 1E-14, \
        'occupations tolerance check'
    assert abs(sk.mass - mass) < 1E-14, 'mass tolerance check'
    assert abs(sk.rcut - rcut) < 1E-14, 'rcut tolerance check'
    assert abs(sk.r_int - r_int) < 1E-14, 'r_int tolerance check'
    assert abs(sk.r_cutoff - r_cutoff) < 1E-14, 'r_cutoff tolerance check'
    assert torch.max(abs(sk.r_a123 - r_a123)) < 1E-14, 'r_a123 tolerance check'
    assert torch.max(abs(sk.r_table - reptab[..., 2:])) < 1E-14, \
        'reptab tolerance check'
    assert abs(sk.r_grid[2] - r_grid_3) < 1E-14, 'r_grid tolerance check'
    assert torch.max(abs(sk.r_long_grid - r_long_grid)) < 1E-14, \
        'r_long_grid tolerance check'
    assert torch.max(abs(sk.r_c_0to5 - r_c_0to5)) < 1E-14, \
        'r_c_0to5 tolerance check'


def test_pbc(device):
    """Test pbc SKF file for Si and Si."""
    path_to_cc_auorg = './tests/unittests/data/slko/pbc/Si-Si.skf'

    # Load data without mask hamiltonian and overlap
    sk = Skf.read(path_to_cc_auorg, 'from_skf', torch.tensor([14, 14]))

    g_step, n_grids = 0.02, 520
    onsite = torch.tensor([0.55, -0.15031380, -0.39572506])
    U = torch.tensor([0.247609, 0.247609, 0.247609])
    occupations = torch.tensor([0.0, 2.0, 2.0])
    mass, rcut = 28.086, 4.8
    h_last = torch.tensor([
        0.0, 0.0, 0.0, 0.0, 0.0, 7.307131777146e-05, -3.260140391345e-06,
        0.0, 2.905988812535e-05, -8.476891570534e-06])
    s_479 = torch.tensor([
        0.0, 0.0, 0.0, 0.0, 0.0, -2.475795422332e-04,
        1.082649953055e-05, 0.0, -1.000348796133e-04, 3.407609828781e-05])
    r_int, r_cutoff = 50, 4.8
    r_a123 = torch.tensor([
        2.36722640845175, 3.641618467812913, 0.08316077297802787])
    r_grid_0to3 = torch.tensor([2, 2.056, 2.112, 2.168])
    reptab5 = torch.tensor([0.284314277, -0.4402566278236794,
                            0.5433604448336125, -0.5627516676858922])
    r_long_grid = torch.tensor([4.744, 4.8])
    r_c_0to5 = torch.tensor([
        2.62409366e-06, -0.0001590892294755154, 0.001806881940670694,
        0.05816041366358013, -1.516236786023504, 9.652696752953078])

    # check tolerance for all the parameters
    assert abs(sk.g_step - g_step) < 1E-14, 'g_step tolerance check'
    assert abs(sk.n_grids - n_grids) < 1E-14, 'n_grids tolerance check'
    assert torch.max(abs(sk.hamiltonian[-1] - h_last)) < 1E-14, \
        'hamiltonian tolerance check'
    assert torch.max(abs(sk.overlap[478] - s_479)) < 1E-14, \
        'overlap tolerance check'
    assert torch.max(abs(sk.onsite - onsite)) < 1E-14, 'onsite tolerance check'
    assert torch.max(abs(sk.U - U)) < 1E-14, 'Hubbert U tolerance check'
    assert torch.max(abs(sk.occupations - occupations)) < 1E-14, \
        'occupations tolerance check'
    assert abs(sk.mass - mass) < 1E-14, 'mass tolerance check'
    assert abs(sk.rcut - rcut) < 1E-14, 'rcut tolerance check'
    assert abs(sk.r_int - r_int) < 1E-14, 'r_int tolerance check'
    assert abs(sk.r_cutoff - r_cutoff) < 1E-14, 'r_cutoff tolerance check'
    assert torch.max(abs(sk.r_a123 - r_a123)) < 1E-14, 'r_a123 tolerance check'
    assert torch.max(abs(sk.r_table[4] - reptab5)) < 1E-14, \
        'r_table tolerance check'
    assert torch.max(abs(sk.r_grid[: 4] - r_grid_0to3)) < 1E-14, \
        'r_grid_0to3 tolerance check'
    assert torch.max(abs(sk.r_long_grid - r_long_grid)) < 1E-14, \
        'r_long_grid tolerance check'
    assert torch.max(abs(sk.r_c_0to5 - r_c_0to5)) < 1E-14, \
        'r_c_0to5 tolerance check'


def _get_interaction(lm):
    """Helper function to generate interactions in Slater-Koster files."""
    _int = [(l1, l2, im) for l1, l2 in combinations_with_replacement(lm, 2)
            for im in range(l1, -1, -1)]
    return list(reversed(_int))


_ls, _lp, _ld, _lf = (
    list(range(1)), list(range(2)), list(range(3)), list(range(4)))
_interaction = {0: _get_interaction(_ls), 1: _get_interaction(_lp),
                2: _get_interaction(_ld), 3: _get_interaction(_lf)}
