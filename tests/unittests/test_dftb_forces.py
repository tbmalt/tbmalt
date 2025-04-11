import pytest
from typing import List
import torch
from tbmalt.io.skf import Skf
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from functools import reduce

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=15, sci_mode=False, linewidth=200, profile="full")


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
    H2 = Geometry(torch.tensor([1, 1], device=device),
                  torch.tensor([
                      [+0.00, +0.00, +0.37],
                      [+0.00, +0.00, -0.37]],
                      device=device),
                  units='angstrom')

    H2O = Geometry(torch.tensor([8, 1, 1], device=device),
               torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.8, -0.5],
                             [0.0, -0.8, -0.5]], 
                            device=device,),
               units='angstrom')
    return [H2, H2O]

species = [1, 8]

#Reference values
reference_nonscc_H2 = torch.tensor([[[ 0.000000000,  0.000000000,  0.001977038],
         [ 0.000000000,  0.000000000, -0.001977038]]])

reference_nonscc_H2O = torch.tensor([[[    -0.000000000,      0.000000000,      0.049182873],
         [     0.000000000,      0.022087108,     -0.024591436],
         [    -0.000000000,     -0.022087108,     -0.024591436]]])

references_nonscc = [reference_nonscc_H2, reference_nonscc_H2O]

reference_nonscc_batch = torch.tensor([[[     0.000000000,      0.000000000,      0.001977038],
         [     0.000000000,      0.000000000,     -0.001977038],
         [     0.000000000,      0.000000000,      0.000000000]],

        [[     0.000000000,      0.000000000,      0.049182873],
         [    -0.000000000,      0.022087108,     -0.024591436],
         [     0.000000000,     -0.022087108,     -0.024591436]]])


reference_scc_H2 = reference_nonscc_H2

reference_scc_H2O = torch.tensor([[[    -0.000000000,     -0.000000000,      0.036591558],
         [    -0.000000000,      0.011168830,     -0.018295779],
         [     0.000000000,     -0.011168830,     -0.018295779]]])

references_scc = [reference_scc_H2, reference_scc_H2O]

reference_scc_batch = torch.tensor([[[     0.000000000,      0.000000000,      0.001977038],
         [     0.000000000,      0.000000000,     -0.001977038],
         [     0.000000000,      0.000000000,      0.000000000]],

        [[     0.000000000,      0.000000000,      0.036591558],
         [    -0.000000000,      0.011168830,     -0.018295779],
         [     0.000000000,     -0.011168830,     -0.018295779]]])

# Single
#non-scc
def test_forces_single_nonscc(skf_file: str, device):

    b_def = {1: [0], 8: [0, 1]}
    
    # setup the feeds
    hamiltonian_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    overlap_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    occupation_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    repulsive_feed = RepulsiveSplineFeed.from_database(skf_file, species, device=device)

    # setup the calculator
    dftb_calculator = Dftb1(hamiltonian_feed, overlap_feed, occupation_feed, r_feed=repulsive_feed)
    
    for mol, force_ref in zip(molecules(device), references_nonscc):
        orbital_info = OrbitalInfo(mol.atomic_numbers, b_def, shell_resolved=False)
        # run dftb calculation
        _ = dftb_calculator(mol, orbital_info)
        
        #calculate forces
        forces = dftb_calculator.forces

        check_1 = forces.device == device 
        check_2 = torch.allclose(forces.detach().cpu(), force_ref, rtol=0, atol=1E-9)
        check_3 = forces.dim() == 2

        assert check_1, 'Results were places on the wrong device'
        assert check_2, f'Forces outside of tolerance (Geometry: {mol}, Forces: {forces})'
        assert check_3, f'Forces for single system do not have single dimension'

#scc
def test_forces_single_scc(skf_file: str, device):

    b_def = {1: [0], 8: [0, 1]}
    
    # setup the feeds
    hamiltonian_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    overlap_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    occupation_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    hubbard_feed = HubbardFeed.from_database(skf_file, species, device=device)
    repulsive_feed = RepulsiveSplineFeed.from_database(skf_file, species, device=device)

    # setup the calculator
    dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)

    for mol, force_ref in zip(molecules(device), references_scc):
        orbital_info = OrbitalInfo(mol.atomic_numbers, b_def, shell_resolved=False)
        # run dftb calculation
        _ = dftb_calculator(mol, orbital_info)
        
        #calculate forces
        forces = dftb_calculator.forces
        
        check_1 = forces.device == device 
        check_2 = torch.allclose(forces.detach().cpu(), force_ref, rtol=0, atol=1E-9)
        check_3 = forces.dim() == 2

        assert check_1, 'Results were places on the wrong device'
        assert check_2, f'Forces outside of tolerance (Geometry: {mol})'
        assert check_3, f'Forces for single system do not have single dimension'


# Batch
#non-scc
def test_forces_batch_nonscc(skf_file: str, device):
    mols = reduce(lambda i, j: i+j, molecules(device))

    b_def = {1: [0], 8: [0, 1]}
    
    # setup the feeds
    hamiltonian_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    overlap_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    occupation_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    repulsive_feed = RepulsiveSplineFeed.from_database(skf_file, species, device=device)

    # setup the calculator
    dftb_calculator = Dftb1(hamiltonian_feed, overlap_feed, occupation_feed, r_feed=repulsive_feed)

    orbital_info = OrbitalInfo(mols.atomic_numbers, b_def, shell_resolved=False)
    # run dftb calculation
    _ = dftb_calculator(mols, orbital_info)
#calculate forces
    forces = dftb_calculator.forces

    check_1 = forces.device == device
    check_2 = torch.allclose(forces.detach().cpu(), reference_nonscc_batch, atol=1e-9, rtol=0)
    check_3 = forces.dim() == 3

    assert check_1, 'Results were places on the wrong device'
    assert check_2, f'Batch forces difference to reference outside of tolerance'
    assert check_3, f'Forces for batch system do not have batch dimension'

#scc
def test_forces_batch_scc(skf_file: str, device):
    mols = reduce(lambda i, j: i+j, molecules(device))

    b_def = {1: [0], 8: [0, 1]}
    
    # setup the feeds
    hamiltonian_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    overlap_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    occupation_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    hubbard_feed = HubbardFeed.from_database(skf_file, species, device=device)
    repulsive_feed = RepulsiveSplineFeed.from_database(skf_file, species, device=device)

    # setup the calculator
    dftb_calculator = Dftb2(hamiltonian_feed, overlap_feed, occupation_feed, hubbard_feed, r_feed=repulsive_feed)

    orbital_info = OrbitalInfo(mols.atomic_numbers, b_def, shell_resolved=False)
    # run dftb calculation
    _ = dftb_calculator(mols, orbital_info)

    #calculate forces
    forces = dftb_calculator.forces

    check_1 = forces.device == device
    check_2 = torch.allclose(forces.detach().cpu(), reference_scc_batch, atol=1e-9, rtol=0)
    check_3 = forces.dim() == 3

    assert check_1, 'Results were places on the wrong device'
    assert check_2, f'Batch forces difference to reference outside of tolerance'
    assert check_3, f'Forces for batch system do not have batch dimension'

