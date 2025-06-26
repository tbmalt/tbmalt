import pytest
from typing import List
import torch
from tbmalt.io.skf import Skf
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2, Dftb1
from tbmalt.physics.dftb.feeds import (
    SkFeed, SkfOccupationFeed, HubbardFeed, PairwiseRepulsiveEnergyFeed)
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
    repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database(skf_file, species, device=device)

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
    repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database(skf_file, species, device=device)

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
    repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database(skf_file, species, device=device)

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
    repulsive_feed = PairwiseRepulsiveEnergyFeed.from_database(skf_file, species, device=device)

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


# def gamma_exponential_gradient(geometry: Geometry, orbs: OrbitalInfo, hubbard_Us: Tensor
#                       ) -> Tensor:
#     """Construct the gradient of the gamma matrix via the exponential method.
#
#     Arguments:
#         geometry: `Geometry` object of the system(s) whose gamma matrix is to
#             be constructed.
#         orbs: `OrbitalInfo` instance associated with the target system.
#         hubbard_Us: Hubbard U values. one value should be specified for each
#             atom or shell depending if the calculation being performed is atom
#             or shell resolved.
#
#     Returns:
#         gamma_grad: gradient of gamma matrix.
#
#     Examples:
#         >>> from tbmalt import OrbitalInfo, Geometry
#         >>> from tbmalt.physics.dftb.gamma import gamma_exponential
#         >>> from ase.build import molecule
#
#         # Preparation of system to calculate
#         >>> geo = Geometry.from_ase_atoms(molecule('CH4'))
#         >>> orbs = OrbitalInfo(geo.atomic_numbers,
#                                shell_dict= {1: [0], 6: [0, 1]})
#         >>> hubbard_U = torch.tensor([0.3647, 0.4196, 0.4196, 0.4196, 0.4196])
#
#         # Build the gamma matrix
#         >>> gamma = gamma_exponential(geo, orbs, hubbard_U)
#         >>> print(gamma)
#         tensor([[0.3647, 0.3234, 0.3234, 0.3234, 0.3234],
#                 [0.3234, 0.4196, 0.2654, 0.2654, 0.2654],
#                 [0.3234, 0.2654, 0.4196, 0.2654, 0.2654],
#                 [0.3234, 0.2654, 0.2654, 0.4196, 0.2654],
#                 [0.3234, 0.2654, 0.2654, 0.2654, 0.4196]])
#
#     """
#
#     # Build the Slater type gamma in second-order term.
#     U = hubbard_Us
#     r = geometry.distances
#     z = geometry.atomic_numbers
#
#     # normed version of the distance vectors
#     normed_distance_vectors = geometry.distance_vectors / geometry.distances.unsqueeze(-1)
#     normed_distance_vectors[normed_distance_vectors.isnan()] = 0
#
#     dtype, device = r.dtype, r.device
#
#     if orbs.shell_resolved:  # and expand it if this is shell resolved calc.
#         def dri(t, ind):  # Abstraction of lengthy double interleave operation
#             return t.repeat_interleave(ind, -1).repeat_interleave(ind, -2)
#
#         # Get № shells per atom & determine batch status, then expand.
#         batch = (spa := orbs.shells_per_atom).ndim >= 2
#         r = pack([dri(i, j) for i, j in zip(r, spa)]) if batch else dri(r, spa)
#
#         z = prepeat_interleave(z, orbs.n_shells_on_species(z))
#
#     # Construct index list for upper triangle gather operation
#     ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 0))
#     distance_tr = r[..., ut[0], ut[1]]
#     an1 = z[..., ut[0]]
#     an2 = z[..., ut[1]]
#
#     # build the whole gamma, shortgamma (without 1/R) and triangular gamma
#     gamma = torch.zeros(r.shape, dtype=dtype, device=device)
#     gamma_tr = torch.zeros(distance_tr.shape, dtype=dtype, device=device)
#     #build the gamma gradient matrix
#     gamma_grad = torch.ones(gamma.size() + (3,), dtype=dtype, device=device)
#
#     # diagonal values is so called chemical hardness Hubbard
#     gamma_tr[..., ut[0] == ut[1]] = 0
#
#     # off-diagonal values of on-site part for shell resolved calc
#     if orbs.shell_resolved:
#         mask_shell = (ut[0] != ut[1]).to(device) * distance_tr.eq(0)
#         ua, ub = U[..., ut[0]][mask_shell], U[..., ut[1]][mask_shell]
#         mask_diff = (ua - ub).abs() < 1E-8
#         gamma_shell = torch.zeros_like(ua, dtype=dtype, device=device)
#         gamma_shell[mask_diff] = -0.5 * (ua[mask_diff] + ub[mask_diff])
#         if torch.any(~mask_diff):
#             ta, tb = 3.2 * ua[~mask_diff], 3.2 * ub[~mask_diff]
#             gamma_shell[~mask_diff] = 0.0
#         gamma_tr[mask_shell] = gamma_shell
#
#     mask_homo = (an1 == an2) * distance_tr.ne(0)
#     mask_hetero = (an1 != an2) * distance_tr.ne(0)
#     alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
#     r_homo = 1.0 / distance_tr[mask_homo]
#     r_hetero = 1.0 / distance_tr[mask_hetero]
#
#     # homo Hubbard
#     aa, dd_homo = alpha[mask_homo], distance_tr[mask_homo]
#     tau_r = aa * dd_homo
#     e_fac = torch.exp(-tau_r) / 48.0 * r_homo**2
#     gamma_tr[mask_homo] = \
#         -(48.0 + 48.0 * tau_r + 24.0 * tau_r**2 + 7.0 * tau_r**3 + tau_r**4) * e_fac
#
#     # hetero Hubbard
#     aa, bb = alpha[mask_hetero], beta[mask_hetero]
#     dd_hetero = distance_tr[mask_hetero]
#     aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
#     bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
#     rab, rba = 1 / (aa2 - bb2), 1 / (bb2 - aa2)
#     exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
#     val_ab = exp_a * ( ((bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero**2) -
#                       aa * (0.5 * aa * bb4 * rab ** 2 -
#                       (bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero) )
#     val_ba = exp_b * ( ((aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero**2) -
#                       bb *(0.5 * bb * aa4 * rba ** 2 -
#                       (aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero) )
#     gamma_tr[mask_hetero] = val_ab + val_ba
#
#     # to make sure gamma values symmetric
#     gamma[..., ut[0], ut[1]] = gamma_tr
#     gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]
#
#     gamma = gamma.squeeze()
#
#     # Subtract the gamma matrix from the inverse distance to get the final
#     # result.
#     r[r != 0.0] = 1.0 / r[r != 0.0]
#     gamma = -r**2 - gamma
#     gamma_grad = normed_distance_vectors * gamma.unsqueeze(-1)
#
#     return gamma_grad
