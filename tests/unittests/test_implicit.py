import pytest
from typing import List
import torch
from tbmalt.io.skf import Skf
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, PairwiseRepulsiveEnergyFeed
from functools import reduce


def molecules(device) -> List[Geometry]:
    """Returns a selection of `Geometry` entities for testing.

    Currently returned systems are H2, CH4, and C2H2Au2S3. The last of which
    is designed to ensure most possible interaction types are checked.

    Arguments:
        device: device onto which the `Geometry` objects should be placed.
            [DEFAULT=None]

    Returns:
        geometries: A list of `Geometry` objects.
    """
    H2 = Geometry(
        torch.tensor([1, 1], device=device),
        torch.tensor([
            [+0.00, +0.00, +0.37],
            [+0.00, +0.00, -0.37]],
            device=device, requires_grad=True),
        units='angstrom')

    H2O = Geometry(
        torch.tensor([8, 1, 1], device=device),
        torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.8, -0.5],
            [0.0, -0.8, -0.5]],
            device=device,requires_grad=True),
        units='angstrom')

    C2H2Au2S3 = Geometry(
        torch.tensor([1, 6, 16, 79, 16, 79, 16, 6, 1], device=device),
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
            device=device, requires_grad=True),
        units='angstrom')

    return [H2, H2O, C2H2Au2S3]


@pytest.fixture
def dftb_calculator(device, skf_file: str):
    species = [1, 8, 6, 16, 79]

    # set up the feeds
    hamiltonian_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    overlap_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    occupation_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    hubbard_feed = HubbardFeed.from_database(skf_file, species, device=device)
    repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database(skf_file, species, device=device)

    # set up the calculator
    return Dftb2(hamiltonian_feed, overlap_feed, occupation_feed,
                 hubbard_feed, r_feed=repulsive_feed, filling_scheme=None)


def implicit_gradient_helper(mol: Geometry, orbs: OrbitalInfo, dftb_calculator, device):
    dftb_calculator.grad_mode = "direct"
    energy_direct = dftb_calculator(mol, orbs)
    forces_direct = -torch.autograd.grad(
        energy_direct, mol.positions,
        grad_outputs=torch.ones_like(energy_direct))[0]

    dftb_calculator.grad_mode = "implicit"
    energy_imp = dftb_calculator(mol, orbs)
    forces_imp = - torch.autograd.grad(
        energy_imp, mol.positions,
        grad_outputs=torch.ones_like(energy_imp))[0]

    check_1 = torch.allclose(
        forces_direct.detach().cpu(), forces_imp.detach().cpu(),
        atol=1E-10, rtol=1E-5)

    assert check_1, f"Implicit gradient forces do not match direct gradient forces for {mol}."


def test_implicit_gradient_single(device, dftb_calculator):
    b_def = {1: [0], 6: [0, 1],8: [0,1], 16: [0, 1, 2], 79: [0, 1, 2]}
    for mol in molecules(device):
        orbs = OrbitalInfo(mol.atomic_numbers, b_def, shell_resolved=False)
        implicit_gradient_helper(mol, orbs, dftb_calculator, device)


def test_implicit_gradient_batch(dftb_calculator, device):
    b_def = {1: [0], 6: [0, 1], 8: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    H2, H2O, C2H2Au2S3 = molecules(device)
    mol = H2 + H2O + C2H2Au2S3
    orbs = OrbitalInfo(mol.atomic_numbers, b_def, shell_resolved=False)
    implicit_gradient_helper(mol, orbs, dftb_calculator, device)
