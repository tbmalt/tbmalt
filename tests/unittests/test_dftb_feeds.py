import os
from os.path import join, dirname
import pytest
import torch
from torch.nn import Parameter, ParameterDict, Module
from typing import List, Type
import numpy as np
from ase.build import molecule

from tbmalt.physics.dftb.feeds import (
    ScipySkFeed, SkFeed, SkfOccupationFeed, HubbardFeed, VcrSkFeed,
    PairwiseRepulsiveEnergyFeed)

from tbmalt import Geometry, OrbitalInfo
from tbmalt.common.batch import pack
from tbmalt.common.maths.interpolation import CubicSpline, PolyInterpU
from tbmalt.ml import Feed
from functools import reduce

torch.set_default_dtype(torch.float64)


def molecules(device) -> List[Geometry]:
    """Returns a selection of `Geometry` entities for testing.

    Currently, returned systems are H2, CH4, and C2H2Au2S3. The last of which
    is designed to ensure most possible interaction types are checked.

    Arguments:
        device: device onto which the `Geometry` objects should be placed.
            [DEFAULT=None]

    Returns:
        geometries: A list of `Geometry` objects.
    """
    H2 = Geometry(torch.tensor([1, 1], device=device),
                  torch.tensor([
                      [+0.00, +0.00, +0.37],
                      [+0.00, +0.00, -0.37]],
                      device=device),
                  units='angstrom')

    CH4 = Geometry(torch.tensor([6, 1, 1, 1, 1], device=device),
                   torch.tensor([
                       [+0.00, +0.00, +0.00],
                       [+0.63, +0.63, +0.63],
                       [-0.63, -0.63, +0.63],
                       [+0.63, -0.63, -0.63],
                       [-0.63, +0.63, -0.63]],
                       device=device),
                   units='angstrom')

    C2H2Au2S3 = Geometry(torch.tensor([1, 6, 16, 79, 16, 79, 16, 6, 1], device=device),
                         torch.tensor([
                             [+0.00, +0.00, +0.00],
                             [-0.03, +0.83, +0.86],
                             [-0.65, +1.30, +1.60],
                             [+0.14, +1.80, +2.15],
                             [-0.55, +0.42, +2.36],
                             [+0.03, +2.41, +3.46],
                             [+1.12, +1.66, +3.23],
                             [+1.10, +0.97, +0.86],
                             [+0.19, +0.93, +4.08]],
                             device=device),
                         units='angstrom')

    return [H2, CH4, C2H2Au2S3]


def hamiltonians(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_H.csv'), delimiter=','),
            device=device))
    return matrices


def overlaps(device):
    matrices = []
    path = join(dirname(__file__), 'data/skfeed')
    for system in ['H2', 'CH4', 'C2H2Au2S3']:
        matrices.append(torch.tensor(np.loadtxt(
            join(path, f'{system}_S.csv'), delimiter=','),
            device=device))
    return matrices


def repulsive_energies(device):
    """Repulsive energies for H2, CH4, and C2H2Au2S3.

    Repulsive energy values are in units of Hartree and are computed by the
    `PairwiseRepulsiveEnergyFeed` class using `DftbpRepulsiveSpline` feeds for each
    interaction pair. Spline data was sourced from the Auorg parameter set.
    """
    # Repulsive energy for H2, CH4, and C2H2Au2S3 in Ha
    return [*torch.tensor([0.0058374104, 0.0130941359, 47.8705446288],
                          device=device)]

#########################################
# tbmalt.physics.dftb.feeds.ScipySkFeed #
#########################################
# Single
def test_scipyskfeed_single(device, skf_file: str):
    """ScipySkFeed matrix single system operability tolerance test"""

    if device.type == "cuda":
        pytest.skip("Scipy splines do not support CUDA.")

    b_def = {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'overlap', device=device)

    for mol, H_ref, S_ref in zip(
            molecules(device), hamiltonians(device), overlaps(device)):
        H = H_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))
        S = S_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))

        check_1 = torch.allclose(H, H_ref, atol=1E-7)
        check_2 = torch.allclose(S, S_ref, atol=1E-7)
        check_3 = H.device == device

        assert check_1, f'ScipySkFeed H matrix outside of tolerance ({mol})'
        assert check_2, f'ScipySkFeed S matrix outside of tolerance ({mol})'
        assert check_3, 'ScipySkFeed.matrix returned on incorrect device'


# Batch
def test_scipyskfeed_batch(device, skf_file: str):
    """ScipySkFeed matrix batch operability tolerance test"""

    if device.type == "cuda":
        pytest.skip("Scipy splines do not support CUDA.")

    H_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'hamiltonian', device=device)
    S_feed = ScipySkFeed.from_database(
        skf_file, [1, 6, 16, 79], 'overlap', device=device)

    mols = reduce(lambda i, j: i+j, molecules(device))
    orbs = OrbitalInfo(mols.atomic_numbers,
                        {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]})

    H = H_feed.matrix(mols, orbs)
    S = S_feed.matrix(mols, orbs)

    check_1 = torch.allclose(H, pack(hamiltonians(device)), atol=1E-7)
    check_2 = torch.allclose(S, pack(overlaps(device)), atol=1E-7)
    check_3 = H.device == device

    # Check that batches of size one do not cause problems
    check_4 = (H_feed.matrix(mols[0:1], orbs[0:1]).ndim == 3)

    assert check_1, 'ScipySkFeed H matrix outside of tolerance (batch)'
    assert check_2, 'ScipySkFeed S matrix outside of tolerance (batch)'
    assert check_3, 'ScipySkFeed.matrix returned on incorrect device'
    assert check_4, 'Failure to operate on batches of size "one"'

# Note that gradient tests are not performed on the ScipySkFeed as it is not
# backpropagatable due to its use of Scipy splines for interpolation.


#########################################
#   tbmalt.physics.dftb.feeds.SkFeed    #
#########################################
@pytest.mark.parametrize("interpolator", [PolyInterpU, CubicSpline])
def test_skfeed_matrix_tolerance_single(
        device, skf_file: str, interpolator: Type[Feed]):
    """SkFeed feed matrix operability tolerance test, single system."""

    def assert_matrix_close(matrix, ref_matrix, matrix_type, mol, interpolator_name):
        """Helper to assert matrix closeness with a detailed message."""
        assert torch.allclose(matrix, ref_matrix, atol=1E-7), (
            f"SkFeed {matrix_type} matrix outside of tolerance for {mol} using {interpolator_name}:\n"
            f"Max diff: {(matrix - ref_matrix).abs().max().item()}\n"
            f"{matrix_type} shape: {matrix.shape}, ref shape: {ref_matrix.shape}\n"
        )

        # Check device consistency
        assert matrix.device == device, (
            f"SkFeed matrix returned on incorrect device:\n"
            f"Expected: {device}, but got: {matrix.device}"
        )

    interpolator_name = interpolator.__class__.__name__
    b_def = {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}

    H_feed = SkFeed.from_database(skf_file, list(b_def.keys()), 'hamiltonian',
                                  interpolator, device=device)
    S_feed = SkFeed.from_database(skf_file, list(b_def.keys()), 'overlap',
                                  interpolator, device=device)

    for mol, H_ref, S_ref in zip(molecules(device), hamiltonians(device), overlaps(device)):
        orb_info = OrbitalInfo(mol.atomic_numbers, b_def)

        H, S = H_feed.matrix(mol, orb_info), S_feed.matrix(mol, orb_info)

        # Perform matrix checks using the helper function
        assert_matrix_close(H, H_ref, 'H', mol, interpolator_name)
        assert_matrix_close(S, S_ref, 'S', mol, interpolator_name)


@pytest.mark.parametrize("interpolator", [PolyInterpU, CubicSpline])
def test_skfeed_matrix_tolerance_batch(
        device, skf_file: str, interpolator: Type[Feed]):
    """SkFeed feed matrix operability tolerance test, batch system."""

    def assert_matrix_close(matrix, ref_matrix, matrix_type, mol, interpolator_name):
        """Helper to assert matrix closeness with a detailed message."""
        assert torch.allclose(matrix, ref_matrix, atol=1E-7), (
            f"SkFeed batch {matrix_type} matrix outside of tolerance for {mol} using {interpolator_name}:\n"
            f"Max diff: {(matrix - ref_matrix).abs().max().item()}\n"
            f"{matrix_type} shape: {matrix.shape}, ref shape: {ref_matrix.shape}\n"
        )

        assert matrix.device == device, (
            f"SkFeed matrix returned on incorrect device:\n"
            f"Expected: {device}, but got: {matrix.device}")

    interpolator_name = interpolator.__class__.__name__
    b_def = {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}

    mols = reduce(lambda i, j: i + j, molecules(device))
    orbs = OrbitalInfo(mols.atomic_numbers, b_def)

    H_feed = SkFeed.from_database(skf_file, list(b_def.keys()), 'hamiltonian',
                                  interpolator, device=device)
    S_feed = SkFeed.from_database(skf_file, list(b_def.keys()), 'overlap',
                                  interpolator, device=device)

    H = H_feed.matrix(mols, orbs)
    S = S_feed.matrix(mols, orbs)

    assert_matrix_close(H, pack(hamiltonians(device)), 'H', mols, interpolator_name)
    assert_matrix_close(S, pack(overlaps(device)), 'S', mols, interpolator_name)

    # Check that batches of size one do not cause problems
    assert (H_feed.matrix(mols[0:1], orbs[0:1]).ndim == 3), (
        "SkFeed failed when running on a batch of size one")


@pytest.mark.parametrize("interpolator", [PolyInterpU, CubicSpline])
def test_skfeed_requires_grad_settings(
        device, skf_file: str, interpolator: Type[Feed]):
    """Ensure `requires_grad` settings in `SkFeed.from_database` are respected."""

    def all_params_require_grad(module: Module):
        return all(param.requires_grad for param in module.parameters())

    def no_params_require_grad(module: Module):
        return not any(param.requires_grad for param in module.parameters())

    args = (skf_file, [1, 6, 79], 'hamiltonian', interpolator)

    # Test with requires_grad_onsite=True and requires_grad_offsite=False
    feed = SkFeed.from_database(
        *args, requires_grad_onsite=True, requires_grad_offsite=False)
    onsite_grad_enabled = all_params_require_grad(feed.on_sites)
    offsite_grad_disabled = no_params_require_grad(feed.off_sites)

    # Test with requires_grad_onsite=False and requires_grad_offsite=True
    feed = SkFeed.from_database(
        *args, requires_grad_onsite=False, requires_grad_offsite=True)
    onsite_grad_disabled = no_params_require_grad(feed.on_sites)
    offsite_grad_enabled = all_params_require_grad(feed.off_sites)

    # Assert that the requires_grad settings are correctly applied
    assert onsite_grad_enabled and onsite_grad_disabled, (
        "`requires_grad_onsite` setting not respected.")
    assert offsite_grad_enabled and offsite_grad_disabled, (
        "`requires_grad_offsite` setting not respected.")

    # On-site terms for the overlap matrix are set to unity and are therefore
    # not valid optimisation targets. Therefore, the `requires_grad_onsite`
    # argument should not be respected when loading an overlap matrix
    feed = SkFeed.from_database(
        skf_file, [1, 6, 79], 'overlap', interpolator,
        requires_grad_onsite=True)
    onsite_grad_disabled_for_overlap = no_params_require_grad(feed.on_sites)

    assert onsite_grad_disabled_for_overlap, (
        "`requires_grad_onsite` setting should be ignored for overlap.")

#########################################
#  tbmalt.physics.dftb.feeds.SkVcrFeed  #
#########################################
# Hamiltonian and overlap data for CH4 and H2O
def reference_data(device):
    H_ref_ch4 = torch.tensor(
        [[-5.048917654803000E-01, 0.000000000000000E+00, 0.000000000000000E+00,
          0.000000000000000E+00, -3.235506247119279E-01, -3.235506247119279E-01,
          -3.235506247119279E-01, -3.235506247119279E-01],
         [0.000000000000000E+00, -1.943551799182000E-01, 0.000000000000000E+00,
          0.000000000000000E+00, -1.644218947194303E-01, 1.644218947194303E-01,
          1.644218947194303E-01, -1.644218947194303E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, -1.943551799182000E-01,
          0.000000000000000E+00, -1.644218947194303E-01, -1.644218947194303E-01,
          1.644218947194303E-01, 1.644218947194303E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
          -1.943551799182000E-01, -1.644218947194303E-01, 1.644218947194303E-01,
          -1.644218947194303E-01, 1.644218947194303E-01],
         [-3.235506247119279E-01, -1.644218947194303E-01, -1.644218947194303E-01,
          -1.644218947194303E-01, -2.386005440483000E-01, -6.680097953235961E-02,
          -6.680097953235961E-02, -6.680097953235961E-02],
         [-3.235506247119279E-01, 1.644218947194303E-01, -1.644218947194303E-01,
          1.644218947194303E-01, -6.680097953235961E-02, -2.386005440483000E-01,
          -6.680097953235961E-02, -6.680097953235961E-02],
         [-3.235506247119279E-01, 1.644218947194303E-01, 1.644218947194303E-01,
          -1.644218947194303E-01, -6.680097953235961E-02, -6.680097953235961E-02,
          -2.386005440483000E-01, -6.680097953235961E-02],
         [-3.235506247119279E-01, -1.644218947194303E-01, 1.644218947194303E-01,
          1.644218947194303E-01, -6.680097953235961E-02, -6.680097953235961E-02,
          -6.680097953235961E-02, -2.386005440483000E-01]],
    device=device)

    S_ref_ch4 = torch.tensor(
        [[1.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
          0.000000000000000E+00, 4.084679411526629E-01, 4.084679411526629E-01,
          4.084679411526629E-01, 4.084679411526629E-01],
         [0.000000000000000E+00, 1.000000000000000E+00, 0.000000000000000E+00,
          0.000000000000000E+00, 2.598016488988212E-01, -2.598016488988212E-01,
          -2.598016488988212E-01, 2.598016488988212E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, 1.000000000000000E+00,
          0.000000000000000E+00, 2.598016488988212E-01, 2.598016488988212E-01,
          -2.598016488988212E-01, -2.598016488988212E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
          1.000000000000000E+00, 2.598016488988212E-01, -2.598016488988212E-01,
          2.598016488988212E-01, -2.598016488988212E-01],
         [4.084679411526629E-01, 2.598016488988212E-01, 2.598016488988212E-01,
          2.598016488988212E-01, 1.000000000000000E+00, 8.616232232870227E-02,
          8.616232232870227E-02, 8.616232232870227E-02],
         [4.084679411526629E-01, -2.598016488988212E-01, 2.598016488988212E-01,
          -2.598016488988212E-01, 8.616232232870227E-02, 1.000000000000000E+00,
          8.616232232870227E-02, 8.616232232870227E-02],
         [4.084679411526629E-01, -2.598016488988212E-01, -2.598016488988212E-01,
          2.598016488988212E-01, 8.616232232870227E-02, 8.616232232870227E-02,
          1.000000000000000E+00, 8.616232232870227E-02],
         [4.084679411526629E-01, 2.598016488988212E-01, -2.598016488988212E-01,
          -2.598016488988212E-01, 8.616232232870227E-02, 8.616232232870227E-02,
          8.616232232870227E-02, 1.000000000000000E+00]],
        device=device)

    H_ref_h2o = torch.tensor(
        [[-8.788325840766993E-01, 0.000000000000000E+00, 0.000000000000000E+00,
          0.000000000000000E+00, -4.882459882956572E-01, -4.882459882956572E-01],
         [0.000000000000000E+00, -3.321317735287993E-01, 0.000000000000000E+00,
          0.000000000000000E+00, -2.760418336133891E-01, 2.760418336133891E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, -3.321317735287993E-01,
          0.000000000000000E+00, 2.156680014519258E-01, 2.156680014519258E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
          -3.321317735287993E-01, 0.000000000000000E+00, 0.000000000000000E+00],
         [-4.882459882956572E-01, -2.760418336133891E-01, 2.156680014519258E-01,
          0.000000000000000E+00, -2.386005440482994E-01, -1.056313505954941E-01],
         [-4.882459882956572E-01, 2.760418336133891E-01, 2.156680014519258E-01,
          0.000000000000000E+00, -1.056313505954941E-01, -2.386005440482994E-01]],
        device=device)

    S_ref_h2o = torch.tensor(
        [[1.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
          0.000000000000000E+00, 4.090587540889245E-01, 4.090587540889245E-01],
         [0.000000000000000E+00, 1.000000000000000E+00, 0.000000000000000E+00,
          0.000000000000000E+00, 3.165866606110211E-01, -3.165866606110211E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, 1.000000000000000E+00,
          0.000000000000000E+00, -2.473451631825645E-01, -2.473451631825645E-01],
         [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00,
          1.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00],
         [4.090587540889245E-01, 3.165866606110211E-01, -2.473451631825645E-01,
          0.000000000000000E+00, 1.000000000000000E+00, 1.554584894565836E-01],
         [4.090587540889245E-01, -3.165866606110211E-01, -2.473451631825645E-01,
          0.000000000000000E+00, 1.554584894565836E-01, 1.000000000000000E+00]],
        device=device)

    return H_ref_ch4, S_ref_ch4, H_ref_h2o, S_ref_h2o


def test_vcrskfeed_single(device, skf_file_vcr):
    """VcrSkFeed matrix single system operability tolerance test"""

    H_ref_ch4, S_ref_ch4, H_ref_h2o, S_ref_h2o = reference_data(device)

    species = [1, 6, 7, 8]
    b_def = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

    # Load the Hamiltonian, overlap feed model
    h_feed = VcrSkFeed.from_database(skf_file_vcr, species, 'hamiltonian',
                                  device=device)
    s_feed = VcrSkFeed.from_database(skf_file_vcr, species, 'overlap',
                                     device=device)

    # Define (wave-function) compression radii, keep s and p the same
    vcrs = [torch.tensor([2.7, 2.5, 2.5, 2.5, 2.5], device=device),
            torch.tensor([2.3, 2.5, 2.5], device=device)]
    H_ref = [H_ref_ch4, H_ref_h2o]
    S_ref = [S_ref_ch4, S_ref_h2o]

    for ii, mol in enumerate([molecule('CH4'), molecule('H2O')]):
        mol = Geometry.from_ase_atoms(mol, device=device)
        h_feed.compression_radii = vcrs[ii]
        s_feed.compression_radii = vcrs[ii]

        H = h_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))
        S = s_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))

        check_1 = torch.allclose(H, H_ref[ii], atol=2E-5)
        check_2 = torch.allclose(S, S_ref[ii], atol=1E-4)
        check_3 = H.device == device

        assert check_1, f'SkFeed H matrix outside of tolerance ({mol})'
        assert check_2, f'SkFeed S matrix outside of tolerance ({mol})'
        assert check_3, 'SkFeed.matrix returned on incorrect device'


# Batch
def test_vcrskfeed_batch(device, skf_file_vcr):
    """VcrSkFeed matrix batch system operability tolerance test"""

    H_ref_ch4, S_ref_ch4, H_ref_h2o, S_ref_h2o = reference_data(device)

    species = [1, 6, 7, 8]

    b_def = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

    # Load the Hamiltonian, overlap feed model
    h_feed = VcrSkFeed.from_database(skf_file_vcr, species, 'hamiltonian',
                                  device=device)
    s_feed = VcrSkFeed.from_database(skf_file_vcr, species, 'overlap',
                                  device=device)

    # Define (wave-function) compression radii, keep s and p the same
    vcrs = torch.tensor([[2.7, 2.5, 2.5, 2.5, 2.5], [2.3, 2.5, 2.5, 0, 0]],
                        device=device)
    H_ref = pack([H_ref_ch4, H_ref_h2o])
    S_ref = pack([S_ref_ch4, S_ref_h2o])

    mol = Geometry.from_ase_atoms(
        [molecule('CH4'), molecule('H2O')], device=device)
    h_feed.compression_radii = vcrs
    s_feed.compression_radii = vcrs

    H = h_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))
    S = s_feed.matrix(mol, OrbitalInfo(mol.atomic_numbers, b_def))

    check_1 = torch.allclose(H, H_ref, atol=2E-5)
    check_2 = torch.allclose(S, S_ref, atol=1E-4)
    check_3 = H.device == device

    assert check_1, f'SkFeed H matrix outside of tolerance ({mol})'
    assert check_2, f'SkFeed S matrix outside of tolerance ({mol})'
    assert check_3, 'SkFeed.matrix returned on incorrect device'


###############################################
# tbmalt.physics.dftb.feeds.SkfOccupationFeed #
###############################################
# General
def test_skfoccupationfeed_general(device, skf_file):

    # Verify that the occupancy feed can be instantiated without issue
    occ_1 = SkfOccupationFeed(
        ParameterDict({"1": torch.tensor([0.]), "6": torch.tensor([1., 2.])})
    )

    # Make sure that use of `torch.ParameterDict` is enforced.
    with pytest.raises(TypeError, match=" must be stored within a*"):
        SkfOccupationFeed({
            "1": torch.tensor([0.]), "6": torch.tensor([1., 2.])})

    # Check 0: ensure that the feed can be constructed from a HDF5 skf database
    # without encountering an error.
    o_feed = SkfOccupationFeed.from_database(skf_file, [1, 6], device=device)

    # Varify that autograd capabilities can be enabled when loading from a database.
    o_feed_grad = SkfOccupationFeed.from_database(
        skf_file, [1, 6], device=device, requires_grad=True)

    example_value = next(iter(o_feed_grad.occupancies.values()))

    check = (isinstance(example_value, Parameter)
             and example_value.requires_grad
             and isinstance(o_feed_grad.occupancies, ParameterDict))

    assert check, "from_database method failed to correctly enable autograd tracking"

    # Check 1: ensure the feed is constructed on the correct device.
    check_1 = o_feed.device == device == list(o_feed.occupancies.values())[0].device
    assert check_1, 'SkfOccupationFeed has been placed on the incorrect device'

    # Check 2: verify that the '.to' method moves the feed and its contents to
    # the specified device as intended. Note that this test cannot be performed
    # if there is no other device present on the system to which a move can be
    # made.

    if torch.cuda.device_count():
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        o_feed_copy = o_feed.to(new_device)

        # In this instance the `.device` property can be trusted as it is
        # calculated ad-hoc from the only pytorch attribute.
        check_2 = o_feed_copy.device == new_device != device

        assert check_2, '".to" method failed to set the correct device'


# Single
def test_skfoccupationfeed_single(device, skf_file):
    o_feed = SkfOccupationFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    # Check 1: verify that results are returned on the correct device.
    check_1 = device == o_feed.forward(
        OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)).device

    assert check_1, 'Results were placed on the wrong device'

    # Check 2: ensure results are within tolerance
    check_2a = torch.allclose(
        o_feed.forward(OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)),
        torch.tensor([1., 1], device=device))

    check_2b = torch.allclose(
        o_feed.forward(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1], device=device), shell_dict)),
        torch.tensor([2., 2/3, 2/3, 2/3, 1, 1, 1, 1], device=device))

    check_2c = torch.allclose(
        o_feed.forward(OrbitalInfo(torch.tensor([1, 1, 8], device=device), shell_dict)),
        torch.tensor([1., 1, 2, 4/3, 4/3, 4/3], device=device))

    check_2 = check_2a and check_2b and check_2c

    assert check_2, 'Predicted occupation values errors exceed allowed tolerance'


# Batch
def test_skfoccupationfeed_batch(device, skf_file):
    o_feed = SkfOccupationFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    orbs = OrbitalInfo(torch.tensor([
        [1, 1, 0, 0, 0],
        [6, 1, 1, 1, 1],
        [1, 1, 8, 0, 0],
    ], device=device), shell_dict)

    reference = torch.tensor([
        [1, 1,   0,   0,   0,   0,   0, 0],
        [2, 2/3, 2/3, 2/3, 1,   1,   1, 1],
        [1, 1,   2,   4/3, 4/3, 4/3, 0, 0]
    ], device=device)

    predicted = o_feed.forward(orbs)

    check_1 = predicted.device == device
    assert check_1, 'Results were placed on the wrong device'

    check_2 = torch.allclose(predicted, reference)
    assert check_2, 'Predicted occupation value errors exceed allowed tolerance'


#########################################
# tbmalt.physics.dftb.feeds.HubbardFeed #
#########################################
# General
def test_hubbardfeed_general(device, skf_file):

    # Verify that the Hubbard-U feed can be instantiated without issue
    _ = HubbardFeed(
        ParameterDict({"1": torch.tensor([0.]), "6": torch.tensor([1., 2.])})
    )

    # Make sure that use of `torch.ParameterDict` is enforced.
    with pytest.raises(TypeError, match="Hubbard-Us must be stored within a*"):
        _ = HubbardFeed(
            {"1": torch.tensor([0.]), "6": torch.tensor([1., 2.])})

    # Check 0: ensure that the feed can be constructed from a HDF5 skf database
    # without encountering an error.
    u_feed = HubbardFeed.from_database(skf_file, [1, 6], device=device)

    # Varify that autograd capabilities can be enabled when loading from a database.
    u_feed_grad = HubbardFeed.from_database(
        skf_file, [1, 6], device=device, requires_grad=True)

    example_value = next(iter(u_feed_grad.hubbard_us.values()))

    check = (isinstance(example_value, Parameter)
             and example_value.requires_grad
             and isinstance(u_feed_grad.hubbard_us, ParameterDict))

    assert check, "from_database method failed to correctly enable autograd tracking"


    # Check 1: ensure the feed is constructed on the correct device.
    check_1 = u_feed.device == device == list(u_feed.hubbard_us.values())[0].device
    assert check_1, 'HubbardFeed has been placed on the incorrect device'

    # Check 2: verify that the '.to' method moves the feed and its contents to
    # the specified device as intended. Note that this test cannot be performed
    # if there is no other device present on the system to which a move can be
    # made.

    if torch.cuda.device_count():
        # Select a device to move to
        new_device = {'cuda': torch.device('cpu'),
                      'cpu': torch.device('cuda:0')}[device.type]
        u_feed_copy = u_feed.to(new_device)

        # In this instance the `.device` property can be trusted as it is
        # calculated ad-hoc from the only pytorch attribute.
        check_2 = u_feed_copy.device == new_device != device

        assert check_2, '".to" method failed to set the correct device'


# Single
def test_hubbardfeed_single(device, skf_file):
    u_feed = HubbardFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    # Check 1: verify that results are returned on the correct device.
    check_1 = device == u_feed.forward(
        OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)).device

    assert check_1, 'Results were placed on the wrong device'

    # Check 2: ensure results are within tolerance
    check_2a = torch.allclose(
        u_feed.forward(OrbitalInfo(torch.tensor([1, 1], device=device), shell_dict)),
        torch.tensor([0.4196174261, 0.4196174261], device=device))

    check_2b = torch.allclose(
        u_feed.forward(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1], device=device), shell_dict)),
        torch.tensor([0.3646664974, 0.4196174261, 0.4196174261,
                      0.4196174261, 0.4196174261], device=device))

    check_2c = torch.allclose(
        u_feed.forward(OrbitalInfo(torch.tensor([1, 1, 8], device=device), shell_dict)),
        torch.tensor([0.4196174261, 0.4196174261, 0.4954041702], device=device))

    check_2 = check_2a and check_2b and check_2c

    assert check_2, 'Predicted hubbard value errors exceed allowed tolerance'


# Batch
def test_hubbardfeed_batch(device, skf_file):
    u_feed = HubbardFeed.from_database(skf_file, [1, 6, 8], device=device)
    shell_dict = {1: [0], 6: [0, 1], 8: [0, 1]}

    orbs = OrbitalInfo(torch.tensor([
        [1, 1, 0, 0, 0],
        [6, 1, 1, 1, 1],
        [1, 1, 8, 0, 0],
    ], device=device), shell_dict)

    reference = torch.tensor([
        [0.4196174261, 0.4196174261, 0,            0,            0],
        [0.3646664974, 0.4196174261, 0.4196174261, 0.4196174261, 0.4196174261],
        [0.4196174261, 0.4196174261, 0.4954041702, 0,            0]
    ], device=device)

    predicted = u_feed.forward(orbs)

    check_1 = predicted.device == device
    assert check_1, 'Results were placed on the wrong device'

    check_2 = torch.allclose(predicted, reference)
    assert check_2, 'Predicted hubbard value errors exceed allowed tolerance'


#################################################
# tbmalt.physics.dftb.feeds.PairwiseRepulsiveEnergyFeed #
#################################################
def test_repulsive_feed_single(device, skf_file: str):
    repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database(
        skf_file, species=[1, 6, 16, 79], device=device)

    for mol, e_ref in zip(molecules(device), repulsive_energies(device)):
        repulsive_energy = repulsive_feed(mol)

        check_1 = repulsive_energy.device == device
        check_2 = torch.allclose(repulsive_energy, e_ref, rtol=0, atol=1E-10)

        assert check_1, 'Results were places on the wrong device'
        assert check_2, f'Repulsive energy outside of tolerance (Geometry: {mol})'


def test_repulsive_feed_batch(device, skf_file: str):
    repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database(
        skf_file, species=[1, 6, 16, 79], device=device)

    mols = reduce(lambda i, j: i+j, molecules(device))

    repulsive_energy = repulsive_feed(mols)

    e_ref = torch.stack(repulsive_energies(device))

    check_1 = repulsive_energy.device == device
    check_2 = torch.allclose(repulsive_energy, e_ref, rtol=0, atol=1E-10)

    assert check_1, 'Results were places on the wrong device'
    assert check_2, 'Repulsive energies outside of tolerance (batch)'
