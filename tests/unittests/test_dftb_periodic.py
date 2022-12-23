import pytest

import torch
from ase.build import molecule

from tbmalt import Geometry, Basis
from tbmalt.physics.dftb import Dftb1, Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.batch import pack

from tests.test_utils import skf_file

torch.set_default_dtype(torch.float64)

# Todo:
#   - Gradiant tests should be added once backpropagatable feeds have been
#     implemented.
#   - add more tests for DFTB2 calculations, right now only atomic charges


@pytest.fixture
def feeds_scc(device, skf_file):
    species = [1, 6, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed, u_feed


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

    basis = Basis(geometry.atomic_numbers, {1: [0]})

    results = {
        'q_final_atomic': torch.tensor([
            +1.000000000000000E+00, +1.000000000000000E+00],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, basis, results, kwargs


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
            device=device), units='a', cutoff=cutoff)

    basis = Basis(geometry.atomic_numbers, {1: [0], 6: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            4.6123997141539634, 0.83319689494904103, 0.85270476089961067,
            0.85179394099589589, 0.84990468900149085],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, basis, results, kwargs


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
            device=device), units='a', cutoff=cutoff)

    basis = Basis(geometry.atomic_numbers, {1: [0], 8: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            0.69168111301294599, 6.6004338082456471, 0.70788507874143958],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, basis, results, kwargs


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
            device=device), units='a', cutoff=cutoff)

    basis = Basis(geometry.atomic_numbers, {1: [0], 6: [0, 1]})

    results = {
        'q_final_atomic': torch.tensor([
            4.1871581746672177, 4.1870486075700262, 0.93758159557977583,
            0.93768013260568484, 0.93757384299649438, 0.93772825211822408,
            0.93742073507842505, 0.93780865938414948],
            device=device),
    }

    kwargs = {'filling_scheme': 'fermi', 'filling_temp': 0.001}

    return geometry, basis, results, kwargs


def merge_systems(device, *systems):
    """Combine multiple test systems into a batch."""

    geometry, basis, results, kwargs = systems[0](device)

    results = {k: [v] for k, v in results.items()}

    for system in systems[1:]:
        t_geometry, t_basis, t_results, t_kwargs = system(device)

        assert t_kwargs == kwargs, 'Test systems with different settings ' \
                                   'cannot be used together'

        geometry += t_geometry
        basis += t_basis

        for k, v in t_results.items():
            results[k].append(t_results[k])

    results = {k: pack(v) for k, v in results.items()}

    return geometry, basis, results, kwargs


def dftb2_helper(calculator, geometry, basis, results):

    # Trigger the calculation
    _ = calculator(geometry, basis)

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


def test_dftb2_single(device, feeds_scc):
    h_feed, s_feed, o_feed, u_feed = feeds_scc

    systems = [H2_scc, H2O_scc, CH4_scc, C2H6_scc]
    mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2, 'generations': 3, 'tolerance': 1e-10}

    for system in systems:
        geometry, basis, results, kwargs = system(device)
        kwargs['mix_params'] = mix_params

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)

        dftb2_helper(calculator, geometry, basis, results)


def test_dftb2_batch(device, feeds_scc):
    h_feed, s_feed, o_feed, u_feed = feeds_scc

    batches = [[H2_scc], [H2_scc, H2O_scc], [H2_scc, H2O_scc, CH4_scc],
               [H2_scc, H2O_scc, CH4_scc, C2H6_scc]]

    for batch in batches:
        geometry, basis, results, kwargs = merge_systems(device, *batch)

        calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
        assert calculator.device == device, 'Calculator is on the wrong device'

        dftb2_helper(calculator, geometry, basis, results)
