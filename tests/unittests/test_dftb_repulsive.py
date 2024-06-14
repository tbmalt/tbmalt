import pytest
from typing import List
import torch
from tbmalt.physics.dftb.feeds import RepulsiveSplineFeed
from tbmalt import Geometry, OrbitalInfo
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

references = [0.0058374104, 0.0130941359, 47.8705446288] #H2, CH4, C2H2Au2S3 in Hartree

# General
def test_repulsivefeed_general(skf_file: str, device):

    # Check 0: ensure that the feed can be constructed fro a HDF5 skf database
    # without encountering an error.
    repulsive_feed = RepulsiveSplineFeed.from_database(skf_file, species=[1, 6, 16, 79], device=device)

    # Check 1: ensure that the feed is contructed on the correct device.
    check_1 = repulsive_feed.device == device == list(repulsive_feed.spline_data.values())[0].grid.device
    assert check_1, f'RepulsiveSplineFeed has been placed on the incorrect device'

# Single
def test_repulsivefeed_single(skf_file: str, device):

    b_def = {1: [0], 6: [0, 1], 16: [0, 1, 2], 79: [0, 1, 2]}
    repulsive_feed = RepulsiveSplineFeed.from_database(skf_file, species=[1, 6, 16, 79], device=device)

    for mol, repulsive_ref in zip(molecules(device), references):
        repulsive_energy = repulsive_feed(mol)
        
        check_1 = repulsive_energy.device == device 
        check_2 = torch.allclose(repulsive_energy.detach().cpu(), torch.tensor([repulsive_ref]), rtol=0, atol=1E-10)

        assert check_1, 'Results were places on the wrong device'
        assert check_2, f'RepulsiveSplineFeed repulsive energy outside of tolerance (Geometry: {mol}, Energy: {repulsive_energy}, Reference: {repulsive_ref})'

# Batch
def test_repulsivefeed_batch(skf_file: str, device):
    repulsive_feed = RepulsiveSplineFeed.from_database(skf_file, species=[1, 6, 16, 79], device=device)
    mols = reduce(lambda i, j: i+j, molecules(device))

    repulsive_energy = repulsive_feed(mols)
    
    repulsive_ref_single = torch.tensor([])
    for mol in molecules(device):
        repulsive_ref_single = torch.cat((repulsive_ref_single, repulsive_feed(mol).detach().cpu()))

    check_1 = repulsive_energy.device == device
    check_2 = torch.allclose(repulsive_energy.detach().cpu(), repulsive_ref_single, atol=1e-9, rtol=0)

    assert check_1, 'Results were places on the wrong device'
    assert check_2, f'RepulsiveSplineFeed batch energy difference to single calculation outside of tolerance (Batch: {repulsive_energy}, Single: {repulsive_ref_single}) '



