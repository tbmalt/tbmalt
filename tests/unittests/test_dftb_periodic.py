import pytest

import torch
from ase.build import molecule

from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb1, Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units

from tests.test_utils import skf_file

torch.set_default_dtype(torch.float64)

# Todo:
#   - Gradiant tests should be added once backpropagatable feeds have been
#     implemented.
#   - add more tests for DFTB2 calculations, right now only atomic charges


@pytest.fixture
def feeds_nscc(device, skf_file):
    species = [1, 6, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed


@pytest.fixture
def feeds_scc(device, skf_file):
    species = [1, 6, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed, u_feed


def H2(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([1, 1], device=device),
        torch.tensor([
            [+0.000000000000000E+00, +0.000000000000000E+00, 0.696520874048385252],
            [+0.000000000000000E+00, +0.000000000000000E+00, -0.696520874048385252]],
            device=device),
        torch.tensor([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5]],
            device=device), cutoff=cutoff)

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0]})

    results = {
        'q_final_atomic': torch.tensor([
            +1.000000000000000E+00, +1.000000000000000E+00],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def H2_scc(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([1, 1], device=device),
        torch.tensor([
            [+0.000000000000000E+00, +0.000000000000000E+00, 0.696520874048385252],
            [+0.000000000000000E+00, +0.000000000000000E+00, -0.696520874048385252]],
            device=device),
        torch.tensor([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5]],
            device=device), cutoff=cutoff)

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0]})

    results = {
        'q_final_atomic': torch.tensor([
            +1.000000000000000E+00, +1.000000000000000E+00],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def CH4(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([6, 1, 1, 1, 1], device=device),
        torch.tensor([
             [3.0, 3.0, 3.0],
             [3.6, 3.6, 3.6],
             [2.4, 3.6, 3.6],
             [3.6, 2.4, 3.6],
             [3.6, 3.6, 2.4]],
            device=device),
        torch.tensor([
            [4.0, 4.0, 0.0],
            [5.0, 0.0, 5.0],
            [0.0, 6.0, 6.0]],
            device=device), units='a',
        cutoff = cutoff / length_units['angstrom'])

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 6: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            4.7720330516019995, 0.759423433582119, 0.822830097322578,
            0.822841753515823, 0.822871663977470],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def CH4_scc(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([6, 1, 1, 1, 1], device=device),
        torch.tensor([
             [3.0, 3.0, 3.0],
             [3.6, 3.6, 3.6],
             [2.4, 3.6, 3.6],
             [3.6, 2.4, 3.6],
             [3.6, 3.6, 2.4]],
            device=device),
        torch.tensor([
            [4.0, 4.0, 0.0],
            [5.0, 0.0, 5.0],
            [0.0, 6.0, 6.0]],
            device=device), units='a',
        cutoff = cutoff / length_units['angstrom'])

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 6: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            4.6123999558687441, 0.83319744713571420, 0.85270457628532692,
            0.85179367706689479, 0.84990434364332379],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def H2O(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([1, 8, 1], device=device),
        torch.tensor([
            [0.965, 0.075, 0.088],
            [1.954, 0.047, 0.056],
            [2.244, 0.660, 0.778]],
            device=device),
        torch.tensor([
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0]],
            device=device), units='a',
        cutoff = cutoff / length_units['angstrom'])

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 8: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            0.611371087506575, 6.77128317137446, 0.617345741118966],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def H2O_scc(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([1, 8, 1], device=device),
        torch.tensor([
            [0.965, 0.075, 0.088],
            [1.954, 0.047, 0.056],
            [2.244, 0.660, 0.778]],
            device=device),
        torch.tensor([
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0]],
            device=device), units='a',
        cutoff = cutoff / length_units['angstrom'])

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 8: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            0.69168110898393897, 6.6004334231842421, 0.70788546783182138],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def C2H6(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([6, 6, 1, 1, 1, 1, 1, 1], device=device),
        torch.tensor([
            [0.949, 0.084, 0.020],
            [2.469, 0.084, 0.020],
            [0.573, 1.098, 0.268],
            [0.573, -0.638, 0.775],
            [0.573, -0.209, -0.982],
            [2.845, 0.376, 1.023],
            [2.845, 0.805, -0.735],
            [2.845, -0.931, -0.227]],
            device=device),
        torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]],
            device=device), units='a',
        cutoff = cutoff / length_units['angstrom'])

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 6: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            4.244545576729166, 4.244466558934202, 0.918593409005569,
            0.918308601317784, 0.918542436200860, 0.918697416221285,
            0.918015240377124, 0.918830761214014],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def C2H6_scc(device):

    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([6, 6, 1, 1, 1, 1, 1, 1], device=device),
        torch.tensor([
            [0.949, 0.084, 0.020],
            [2.469, 0.084, 0.020],
            [0.573, 1.098, 0.268],
            [0.573, -0.638, 0.775],
            [0.573, -0.209, -0.982],
            [2.845, 0.376, 1.023],
            [2.845, 0.805, -0.735],
            [2.845, -0.931, -0.227]],
            device=device),
        torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0]],
            device=device), units='a',
        cutoff = cutoff / length_units['angstrom'])

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 6: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            4.1871582297233614, 4.1870485301603786, 0.93758157156055788,
            0.93768012543326973, 0.93757380742508800, 0.93772828820761100,
            0.93742072783044328, 0.93780871965928714],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, orbs, results, kwargs


def merge_systems(device, *systems):
    """Combine multiple test systems into a batch."""

    geometry, orbs, results, kwargs = systems[0](device)

    results = {k: [v] for k, v in results.items()}

    for system in systems[1:]:
        t_geometry, t_orbs, t_results, t_kwargs = system(device)

        assert t_kwargs == kwargs, 'Test systems with different settings ' \
                                   'cannot be used together'

        geometry += t_geometry
        orbs += t_orbs

        for k, v in t_results.items():
            results[k].append(t_results[k])

    results = {k: pack(v) for k, v in results.items()}

    return geometry, orbs, results, kwargs


def dftb_helper(calculator, geometry, orbs, results):

    # Trigger the calculation
    _ = calculator(geometry, orbs)

    # Ensure that the `hamiltonian` and `overlap` properties return the correct
    # matrices. We do not need to actually check if the matrices are themselves
    # correct as this is something that is something that is done by the unit
    # tests for those feeds. Furthermore, any errors in said matrix will cause
    # many of the computed properties to be incorrect.

    def check_allclose(i):
        predicted = getattr(calculator, i)
        is_close = torch.allclose(predicted, results[i], atol=1E-10)
        assert is_close, f'Attribute {i} is in error for system {geometry}'
        if isinstance(predicted, torch.Tensor):
            device_check = predicted.device == calculator.device
            assert device_check, f'Attribute {i} was returned on the wrong device'

    check_allclose('q_final_atomic')


def test_dftb1_single(device, feeds_nscc):
    h_feed, s_feed, o_feed = feeds_nscc

    systems = [H2, H2O, CH4, C2H6]

    for system in systems:
        geometry, orbs, results, kwargs = system(device)

        calculator = Dftb1(h_feed, s_feed, o_feed, **kwargs)

        dftb_helper(calculator, geometry, orbs, results)


def test_dftb1_batch(device, feeds_nscc):
    h_feed, s_feed, o_feed = feeds_nscc

    batches = [[H2], [H2, H2O], [H2, H2O, CH4],
               [H2, H2O, CH4, C2H6]]

    for batch in batches:
        geometry, orbs, results, kwargs = merge_systems(device, *batch)

        calculator = Dftb1(h_feed, s_feed, o_feed, **kwargs)
        assert calculator.device == device, 'Calculator is on the wrong device'

        dftb_helper(calculator, geometry, orbs, results)


def test_dftb2_single(device, feeds_scc):
    h_feed, s_feed, o_feed, u_feed = feeds_scc

    systems = [H2_scc, H2O_scc, CH4_scc, C2H6_scc]
    mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2, 'generations': 3, 'tolerance': 1e-10}

    for system in systems:
        geometry, orbs, results, kwargs = system(device)
        kwargs['mix_params'] = mix_params

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)

        dftb_helper(calculator, geometry, orbs, results)


def test_dftb2_batch(device, feeds_scc):
    h_feed, s_feed, o_feed, u_feed = feeds_scc

    batches = [[H2_scc], [H2_scc, H2O_scc], [H2_scc, H2O_scc, CH4_scc],
               [H2_scc, H2O_scc, CH4_scc, C2H6_scc]]

    for batch in batches:
        geometry, orbs, results, kwargs = merge_systems(device, *batch)

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
        assert calculator.device == device, 'Calculator is on the wrong device'

        dftb_helper(calculator, geometry, orbs, results)
