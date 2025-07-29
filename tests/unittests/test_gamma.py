"""Short gamma unit-tests.

Here the functionality of the Gamma module
is tested."""

import torch
import pytest
from ase.build import molecule
from tbmalt import OrbitalInfo, Geometry
from tbmalt.physics.dftb.gamma import gamma_exponential, build_gamma_matrix
from tbmalt.physics.dftb.coulomb import build_coulomb_matrix
from tbmalt.physics.dftb.feeds import HubbardFeed
from tbmalt.common.batch import pack

torch.set_default_dtype(torch.float64)


@pytest.fixture
def hubbard_feeds(device, skf_file):
    species = [1, 6, 8]
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    return u_feed


def H2(device):
    """Non-shell-resolved gamma for H2."""
    geo = Geometry.from_ase_atoms(molecule('H2'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0]})
    results = {
        'gamma': torch.tensor(
            [[0.419617426124700, 0.377449713796378],
             [0.377449713796378, 0.419617426124700]],
            device=device),
        }

    return geo, orbs, results


def H2O(device):
    """Non-shell-resolved gamma for H2O."""
    geo = Geometry.from_ase_atoms(molecule('H2O'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0], 8: [0, 1]})
    results = {
        'gamma': torch.tensor(
            [[0.495404170221900, 0.372156574146154, 0.372156574146154],
             [0.372156574146154, 0.419617426124700, 0.291272252587970],
             [0.372156574146154, 0.291272252587970, 0.419617426124700]],
            device=device),
        }

    return geo, orbs, results


def CH4(device):
    """Non-shell-resolved gamma for CH4."""
    geo = Geometry.from_ase_atoms(molecule('CH4'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0], 6: [0, 1]})
    results = {
        'gamma': torch.tensor(
            [[0.364666497392500, 0.323400974782220, 0.323400974782220,
              0.323400974782220, 0.323400974782220],
             [0.323400974782220, 0.419617426124700, 0.265356375552299,
              0.265356375552299, 0.265356375552299],
             [0.323400974782220, 0.265356375552299, 0.419617426124700,
              0.265356375552299, 0.265356375552299],
             [0.323400974782220, 0.265356375552299, 0.265356375552299,
              0.419617426124700, 0.265356375552299],
             [0.323400974782220, 0.265356375552299, 0.265356375552299,
              0.265356375552299, 0.419617426124700]],
            device=device),
        }

    return geo, orbs, results


def H2_shell_resolved(device):
    """Shell-resolved gamma for H2."""
    geo = Geometry.from_ase_atoms(molecule('H2'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0]},
                       shell_resolved=True)
    results = {
        'gamma': torch.tensor(
            [[0.419617426124700, 0.377449713796378],
             [0.377449713796378, 0.419617426124700]],
            device=device),
        }

    return geo, orbs, results


def H2O_shell_resolved(device):
    """Shell-resolved gamma for H2O."""
    geo = Geometry.from_ase_atoms(molecule('H2O'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0], 8: [0, 1]},
                       shell_resolved=True)
    results = {
        'gamma': torch.tensor(
            [[0.495404170221900, 0.495404170221900, 0.372156574146154,
              0.372156574146154],
             [0.495404170221900, 0.495404170221900, 0.372156574146154,
              0.372156574146154],
             [0.372156574146154, 0.372156574146154, 0.419617426124700,
              0.291272252587970],
             [0.372156574146154, 0.372156574146154, 0.291272252587970,
              0.419617426124700]],
            device=device),
        }

    return geo, orbs, results


def CH4_shell_resolved(device):
    """Shell-resolved gamma for CH4."""
    geo = Geometry.from_ase_atoms(molecule('CH4'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0], 6: [0, 1]},
                       shell_resolved=True)
    results = {
        'gamma': torch.tensor(
            [[0.364666497392500, 0.364666497392500, 0.323400974782220,
              0.323400974782220, 0.323400974782220, 0.323400974782220],
             [0.364666497392500, 0.364666497392500, 0.323400974782220,
              0.323400974782220, 0.323400974782220, 0.323400974782220],
             [0.323400974782220, 0.323400974782220, 0.419617426124700,
              0.265356375552299, 0.265356375552299, 0.265356375552299],
             [0.323400974782220, 0.323400974782220, 0.265356375552299,
              0.419617426124700, 0.265356375552299, 0.265356375552299],
             [0.323400974782220, 0.323400974782220, 0.265356375552299,
              0.265356375552299, 0.419617426124700, 0.265356375552299],
             [0.323400974782220, 0.323400974782220, 0.265356375552299,
              0.265356375552299, 0.265356375552299, 0.419617426124700]],
            device=device),
        }

    return geo, orbs, results


def H2_pbc(device):
    """Non-shell-resolved gamma for H2 with pbc."""
    latvec = torch.tensor([[1., 1., 0.], [0., 3., 3.], [2., 0., 2.]],
                          device=device)
    positions = torch.tensor([[0., 0., 0.], [0., 2., 0.]], device=device)
    numbers = torch.tensor([1, 1], device=device)
    cutoff = torch.tensor([10.98], device=device)
    geo = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    # geo = Geometry.from_ase_atoms(molecule('H2'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0]})
    results = {
        'gamma': torch.tensor(
            [[-0.261847214945771, -0.379735634924661],
             [-0.379735634924661, -0.261847214945771]],
            device=device),
        }

    return geo, orbs, results


def H2O_pbc(device):
    """Non-shell-resolved gamma for H2O with pbc."""
    latvec = torch.tensor([[4., 0., 0.], [0., 5., 0.], [0., 0., 6.]],
                          device=device)
    positions = torch.tensor([[0.965, 0.075, 0.088],
                              [1.954, 0.047, 0.056],
                              [2.244, 0.660, 0.778]], device=device)
    numbers = torch.tensor([1, 8, 1], device=device)
    cutoff = torch.tensor([10.98], device=device)
    geo = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    # geo = Geometry.from_ase_atoms(molecule('H2O'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0], 8: [0, 1]})
    results = {
        'gamma': torch.tensor(
            [[0.127877340507534, 0.097027326073299, 0.025880007126443],
             [0.097027326073299, 0.204108180123203, 0.082464711188714],
             [0.025880007126443, 0.082464711188714, 0.127877340507534]],
            device=device),
        }

    return geo, orbs, results


def CH4_pbc(device):
    """Non-shell-resolved gamma for CH4 with pbc."""
    latvec = torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]],
                          device=device)
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
        [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    cutoff = torch.tensor([10.98], device=device)
    geo = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    # geo = Geometry.from_ase_atoms(molecule('CH4'), device=device)
    orbs = OrbitalInfo(geo.atomic_numbers, {1: [0], 6: [0, 1]})
    results = {
        'gamma': torch.tensor(
            [[ 0.061049235077047,  0.034043208892575,  0.034043208892576,
              0.034043208892576,  0.034043208892576],
             [ 0.034043208892575,  0.116268871933952,  0.035975526053319,
              0.035975526053319,  0.035975526053319],
             [ 0.034043208892576,  0.035975526053319,  0.116268871933952,
              -0.004648104185804, -0.004648104185804],
             [ 0.034043208892576,  0.035975526053319, -0.004648104185804,
              0.116268871933952, -0.004648104185804],
             [ 0.034043208892576,  0.035975526053319, -0.004648104185804,
              -0.004648104185804,  0.116268871933952]],
            device=device),
        }

    return geo, orbs, results


def merge_systems(device, *systems):
    """Combine multiple test systems into a batch."""

    geometry, orbs, results = systems[0](device)

    results = {k: [v] for k, v in results.items()}

    for system in systems[1:]:
        t_geometry, t_orbs, t_results = system(device)

        geometry += t_geometry
        orbs += t_orbs

        for k, v in t_results.items():
            results[k].append(t_results[k])

    results = {k: pack(v) for k, v in results.items()}

    return geometry, orbs, results


def gamma_helper(gamma_cal, geometry, orbs, results):
    """Helper function"""
    predicted = gamma_cal

    # Check the device
    check1 = gamma_cal.device == results['gamma'].device
    assert check1, 'Gamma returned on incorrect device.'

    # Check the tolerance
    is_close = torch.allclose(predicted, results['gamma'])
    assert is_close, f'Gamma calculation is in error for system {geometry}'


def test_exponential_gamma_single(device, hubbard_feeds):
    """Test non-shell-resolved gamma calculation via the exponential method for
       a single system without pbc."""
    u_feed = hubbard_feeds

    systems = [H2, H2O, CH4]
    for system in systems:
        geometry, orbs, results = system(device)
        gamma_cal = gamma_exponential(geometry, orbs, u_feed(orbs))
        gamma_helper(gamma_cal, geometry, orbs, results)


def test_exponential_gamma_batch(device, hubbard_feeds):
    """Test non-shell-resolved gamma calculation via the exponential method for
       batch systems without pbc."""
    u_feed = hubbard_feeds

    batches = [[H2], [H2, H2O], [H2, H2O, CH4]]
    for batch in batches:
        geometry, orbs, results = merge_systems(device, *batch)
        gamma_cal = gamma_exponential(geometry, orbs, u_feed(orbs))
        gamma_helper(gamma_cal, geometry, orbs, results)


def test_exponential_gamma_single_shell_resolved(device, hubbard_feeds):
    """Test shell-resolved gamma calculation via the exponential method for
       a single system without pbc."""
    u_feed = hubbard_feeds

    systems = [H2_shell_resolved, H2O_shell_resolved, CH4_shell_resolved]
    for system in systems:
        geometry, orbs, results = system(device)
        gamma_cal = gamma_exponential(geometry, orbs, u_feed(orbs))
        gamma_helper(gamma_cal, geometry, orbs, results)


def test_exponential_gamma_batch_shell_resolved(device, hubbard_feeds):
    """Test shell-resolved gamma calculation via the exponential method for
       batch systems without pbc."""
    u_feed = hubbard_feeds

    batches = [[H2_shell_resolved], [H2_shell_resolved, H2O_shell_resolved],
               [H2_shell_resolved, H2O_shell_resolved, CH4_shell_resolved]]
    for batch in batches:
        geometry, orbs, results = merge_systems(device, *batch)
        gamma_cal = gamma_exponential(geometry, orbs, u_feed(orbs))
        gamma_helper(gamma_cal, geometry, orbs, results)


def test_exponential_gamma_single_pbc(device, hubbard_feeds):
    """Test non-shell-resolved gamma calculation via the exponential method for
       a single system with pbc."""
    u_feed = hubbard_feeds

    systems = [H2_pbc, H2O_pbc, CH4_pbc]
    for system in systems:
        geometry, orbs, results = system(device)
        invr = build_coulomb_matrix(geometry, method='search')
        gamma_cal = build_gamma_matrix(geometry, orbs, invr,
                                       u_feed(orbs), 'exponential')
        gamma_helper(gamma_cal, geometry, orbs, results)


def test_exponential_gamma_batch_pbc(device, hubbard_feeds):
    """Test non-shell-resolved gamma calculation via the exponential method for
       batch systems with pbc."""
    u_feed = hubbard_feeds

    batches = [[H2_pbc], [H2_pbc, H2O_pbc], [H2_pbc, H2O_pbc, CH4_pbc]]
    for batch in batches:
        geometry, orbs, results = merge_systems(device, *batch)
        invr = build_coulomb_matrix(geometry, method='search')
        gamma_cal = build_gamma_matrix(geometry, orbs, invr,
                                       u_feed(orbs), 'exponential')
        gamma_helper(gamma_cal, geometry, orbs, results)
