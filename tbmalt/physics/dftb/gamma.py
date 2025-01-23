# -*- coding: utf-8 -*-
"""Short gamma calculations."""
from typing import Literal
import torch
from torch import Tensor
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb.feeds import Feed
import numpy as np
from tbmalt.common.batch import pack, prepeat_interleave
from tbmalt.data import gamma_cutoff, gamma_element_list


def gamma_exponential(geometry: Geometry, orbs: OrbitalInfo, hubbard_Us: Tensor
                      ) -> Tensor:
    """Construct the gamma matrix via the exponential method.

    Arguments:
        geometry: `Geometry` object of the system(s) whose gamma matrix is to
            be constructed.
        orbs: `OrbitalInfo` instance associated with the target system.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.

    Returns:
        gamma: gamma matrix.

    Examples:
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.gamma import gamma_exponential
        >>> from ase.build import molecule

        # Preparation of system to calculate
        >>> geo = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs = OrbitalInfo(geo.atomic_numbers,
                               shell_dict= {1: [0], 6: [0, 1]})
        >>> hubbard_U = torch.tensor([0.3647, 0.4196, 0.4196, 0.4196, 0.4196])

        # Build the gamma matrix
        >>> gamma = gamma_exponential(geo, orbs, hubbard_U)
        >>> print(gamma)
        tensor([[0.3647, 0.3234, 0.3234, 0.3234, 0.3234],
                [0.3234, 0.4196, 0.2654, 0.2654, 0.2654],
                [0.3234, 0.2654, 0.4196, 0.2654, 0.2654],
                [0.3234, 0.2654, 0.2654, 0.4196, 0.2654],
                [0.3234, 0.2654, 0.2654, 0.2654, 0.4196]])

    """
    gamma_gradient = gamma_exponential_gradient(geometry, orbs, hubbard_Us)

    # Build the Slater type gamma in second-order term.
    U = hubbard_Us
    r = geometry.distances
    z = geometry.atomic_numbers

    dtype, device = r.dtype, r.device

    if orbs.shell_resolved:  # and expand it if this is shell resolved calc.
        def dri(t, ind):  # Abstraction of lengthy double interleave operation
            return t.repeat_interleave(ind, -1).repeat_interleave(ind, -2)

        # Get № shells per atom & determine batch status, then expand.
        batch = (spa := orbs.shells_per_atom).ndim >= 2
        r = pack([dri(i, j) for i, j in zip(r, spa)]) if batch else dri(r, spa)
        z = prepeat_interleave(z, orbs.n_shells_on_species(z))

    # Construct index list for upper triangle gather operation
    ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 0))
    distance_tr = r[..., ut[0], ut[1]]
    an1 = z[..., ut[0]]
    an2 = z[..., ut[1]]

    # build the whole gamma, shortgamma (without 1/R) and triangular gamma
    gamma = torch.zeros(r.shape, dtype=dtype, device=device)
    gamma_tr = torch.zeros(distance_tr.shape, dtype=dtype, device=device)

    # diagonal values is so called chemical hardness Hubbard
    gamma_tr[..., ut[0] == ut[1]] = -U

    # off-diagonal values of on-site part for shell resolved calc
    if orbs.shell_resolved:
        mask_shell = (ut[0] != ut[1]).to(device) * distance_tr.eq(0)
        ua, ub = U[..., ut[0]][mask_shell], U[..., ut[1]][mask_shell]
        mask_diff = (ua - ub).abs() < 1E-8
        gamma_shell = torch.zeros_like(ua, dtype=dtype, device=device)
        gamma_shell[mask_diff] = -0.5 * (ua[mask_diff] + ub[mask_diff])
        if torch.any(~mask_diff):
            ta, tb = 3.2 * ua[~mask_diff], 3.2 * ub[~mask_diff]
            gamma_shell[~mask_diff] = -0.5 * ((ta * tb) / (ta + tb) +
                                              (ta * tb) ** 2 / (ta + tb) ** 3)
        gamma_tr[mask_shell] = gamma_shell

    mask_homo = (an1 == an2) * distance_tr.ne(0)
    mask_hetero = (an1 != an2) * distance_tr.ne(0)
    alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
    r_homo = 1.0 / distance_tr[mask_homo]
    r_hetero = 1.0 / distance_tr[mask_hetero]

    # homo Hubbard
    aa, dd_homo = alpha[mask_homo], distance_tr[mask_homo]
    tau_r = aa * dd_homo
    e_fac = torch.exp(-tau_r) / 48.0 * r_homo
    gamma_tr[mask_homo] = \
        (48.0 + 33.0 * tau_r + 9.0 * tau_r ** 2 + tau_r ** 3) * e_fac

    # hetero Hubbard
    aa, bb = alpha[mask_hetero], beta[mask_hetero]
    dd_hetero = distance_tr[mask_hetero]
    aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
    bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
    rab, rba = 1 / (aa2 - bb2), 1 / (bb2 - aa2)
    exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
    val_ab = exp_a * (0.5 * aa * bb4 * rab ** 2 -
                      (bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero)
    val_ba = exp_b * (0.5 * bb * aa4 * rba ** 2 -
                      (aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero)
    gamma_tr[mask_hetero] = val_ab + val_ba

    # to make sure gamma values symmetric
    gamma[..., ut[0], ut[1]] = gamma_tr
    gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]

    gamma = gamma.squeeze()

    # Subtract the gamma matrix from the inverse distance to get the final
    # result.
    r[r != 0.0] = 1.0 / r[r != 0.0]
    gamma = r - gamma

    return gamma.squeeze()


def gamma_gaussian(geometry: Geometry, orbs: OrbitalInfo, hubbard_Us: Tensor
                   ) -> Tensor:
    r"""Constructs the gamma matrix via the Gaussian method.

    Elements of the gamma matrix are calculated via the following equation:

    .. math::

        \gamma_{ij}(R_{ij}) = \frac{\text{erf}(C_{ij}R_{ij})}{R_{ij}}

    Where the coefficients :math:`C` are calculated as

    .. math::

        C_{ij} = \sqrt{\frac{4\ln{2}}{\text{FWHM}_{i}^2 + \text{FWHM}_{j}^2}}

    and the full width half max values, FWHM, like so

    .. math::
        \text{FWHM}_{i} = \sqrt{\frac{8\ln{2}}{\pi}}\frac{1}{U_i}

    Where U and R are the hubbard U and distance values respectively.

    Arguments:
        geometry: `Geometry` object of the system whose gamma matrix is to be
            constructed.
        orbs: `OrbitalInfo` instance associated with the target system.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.

    Returns:
        gamma: gamma matrix.

    Notes:
        One Hubbard U value must be specified for each atom or each shell based
        on whether an atom or shell resolved calculation is being performed.
        Note that this must be consistent with the `shell_resolved` attribute
        of the supplied `OrbitalInfo` instance ``orbs``.

        Currently the Hubbard U values must be specified manually, however the
        option to supply a feed object will be added at a later data. This will
        facilitate the automated construction of the Hubbard U tensor and shall
        permit the use of environmentally dependent U values if desired.

    Examples:
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.gamma import gamma_gaussian
        >>> from ase.build import molecule

        # Preparation of system to calculate
        >>> geo = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs = OrbitalInfo(geo.atomic_numbers,
                               shell_dict= {1: [0], 6: [0, 1]})
        >>> hubbard_U = torch.tensor([0.3647, 0.4196, 0.4196, 0.4196, 0.4196])

        # Build the gamma matrix
        >>> gamma = gamma_gaussian(geo, orbs, hubbard_U)
        >>> print(gamma)
        tensor([[0.3647, 0.3326, 0.3326, 0.3326, 0.3326],
                [0.3326, 0.4196, 0.2745, 0.2745, 0.2745],
                [0.3326, 0.2745, 0.4196, 0.2745, 0.2745],
                [0.3326, 0.2745, 0.2745, 0.4196, 0.2745],
                [0.3326, 0.2745, 0.2745, 0.2745, 0.4196]])

    """
    # Developers Notes: At some point in the future this function will be
    # changed to allow dictionaries and UFeeds to be supplied. This will allow
    # for more flexible and concise code. It would be nice to make the r matrix
    # expansion operation batch agnostic. This could be achieved by manually
    # computing the distance matrix or using gather operations.

    if geometry.positions.ndim > 2:
        raise NotImplementedError(
            'The Gaussian gamma matrix construction scheme does not currently '
            'support batch operability.')


    # Equations are in reference to Koskinen (2009). Comput. Mater. Sci.,
    # 47(1), 237–253.

    # Calculate the full width half max values (eq. 29). Note that masks must
    # be used to avoid division by zero which will break the gradient.
    mask = (*hubbard_Us.nonzero().T,)
    fwhm = torch.full_like(hubbard_Us, np.sqrt((8. * np.log(2.)) / np.pi))
    fwhm[mask] = fwhm[mask] / hubbard_Us[mask]

    r = geometry.distances  # Get the distance matrix,

    dtype, device = r.dtype, r.device

    if orbs.shell_resolved:  # and expand it if this is shell resolved calc.
        def dri(t, ind):  # Abstraction of lengthy double interleave operation
            return t.repeat_interleave(ind, -1).repeat_interleave(ind, -2)

        # Get № shells per atom & determine batch status, then expand.
        batch = (spa := orbs.shells_per_atom).ndim >= 2
        r = pack([dri(i, j) for i, j in zip(r, spa)]) if batch else dri(r, spa)

    # Build the "C" coefficients (eq. 27)
    c = torch.sqrt((4. * np.log(2.)) /
                   (fwhm.unsqueeze(-2)**2 + fwhm.unsqueeze(-1)**2))

    # Construct the gamma matrix then set the diagonals.
    mask = (*r.nonzero().T,)
    gamma = torch.erf(c * r)  # <──(eq. 26)──┐
    gamma[mask] = gamma[mask] / r[mask]  # <─┘
    gamma.diagonal(0, -2, -1)[:] = hubbard_Us[:]  # (eq. 31)

    # off-diagonal values of on-site part for shell resolved calc
    if orbs.shell_resolved:
        mask_shell = torch.ones_like(gamma, dtype=dtype, device=device).bool()
        mask_shell.diagonal(0, -2, -1)[:] = False
        mask_shell = mask_shell * r.eq(0)
        ua, ub = hubbard_Us[..., mask_shell.nonzero()[..., 0]], \
            hubbard_Us[..., mask_shell.nonzero()[..., 1]]
        mask_diff = (ua - ub).abs() < 1E-8
        gamma_shell = torch.zeros_like(ua, dtype=dtype, device=device)
        gamma_shell[mask_diff] = 0.5 * (ua[mask_diff] + ub[mask_diff])
        if torch.any(~mask_diff):
            ta, tb = 3.2 * ua[~mask_diff], 3.2 * ub[~mask_diff]
            gamma_shell[~mask_diff] = 0.5 * ((ta * tb) / (ta + tb) +
                                              (ta * tb) ** 2 / (ta + tb) ** 3)
        gamma[mask_shell] = gamma_shell

    return gamma


def gamma_exponential_pbc(geometry: Geometry, orbs: OrbitalInfo,
                          invr: Tensor, hubbard_Us: Tensor) -> Tensor:
    """Build the gamma matrix with pbc via the exponential method.

    Arguments:
        geometry: `Geometry` object of the system(s) whose gamma matrix is to
            be constructed.
        orbs: `OrbitalInfo` instance associated with the target system.
        invr: the 1/R matrix.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.

    Returns:
        gamma: gamma matrix.

    Notes:
        Currently shell resolved calculation is not supported yet for periodic
        systems.
        To avoid loops to calculate cutoff distances of short range part for
        atom pairs, some of the cutoff values are pre-calculated and stored in
        data module, which can be used to construct gamma matrix. For atom pairs
        which have not been stored, the built-in function will search for the
        cutoff values.

    Examples:
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.gamma import gamma_exponential_pbc

        # Preparation of system to calculate
        >>> cell = torch.tensor([[2., 0., 0.], [0., 4., 0.], [0., 0., 2.]])
        >>> pos = torch.tensor([[0., 0., 0.], [0., 2., 0.]])
        >>> num = torch.tensor([1, 1])
        >>> cutoff = torch.tensor([9.98])
        >>> geo = Geometry(num, pos, cell, units='a', cutoff=cutoff)
        >>> orbs = OrbitalInfo(geo.atomic_numbers,
                               shell_dict= {1: [0]})
        >>> hubbard_U = torch.tensor([0.4196, 0.4196])

        # 1/R matrix.calculated from coulomb module
        >>> invr = torch.tensor([[-0.4778, -0.2729],
                                 [-0.2729, -0.4778]])

        # Build the gamma matrix
        >>> gamma = gamma_exponential_pbc(geo, orbs, invr, hubbard_U)
        >>> print(gamma)
        tensor([[-0.1546, -0.3473],
                [-0.3473, -0.1546]])

    """

    # Read geometry information
    r = geometry.periodicity.periodic_distances
    U = torch.clone(hubbard_Us).repeat(r.size(-3), 1, 1).transpose(0, 1)
    z = geometry.atomic_numbers
    U = U.squeeze(0) if z.ndim == 1 else U

    dtype, device = r.dtype, r.device

    # Shell resolved calc is not supported yet. For this, coulomb also
    # needs shell resolved calc.
    if orbs.shell_resolved:
        raise NotImplementedError('Not implement shell resolved for pbc yet.')

    # Construct index list for upper triangle gather operation
    ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 0))
    distance_tr = r[..., ut[0], ut[1]]
    distance_tr[distance_tr == 0] = 99

    an1 = z[..., ut[0]]
    an2 = z[..., ut[1]]

    # build the whole gamma, shortgamma (without 1/R) and triangular gamma
    gamma = torch.zeros(r.shape, dtype=dtype, device=device)
    gamma_tr = torch.zeros(distance_tr.shape, dtype=dtype, device=device)

    mask_homo = an1 == an2
    mask_hetero = an1 != an2
    alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2

    # expcutoff for different atom pairs
    # construct the cutoff tensor from pre-calculated values
    if sum(an1 == i for i in gamma_element_list).bool().all() and sum(
            an2 == i for i in gamma_element_list).bool().all():
        if z.ndim == 1:  # -> Single
            expcutoff = torch.cat([
                gamma_cutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')].to(device)
                for ii, jj in zip(an1, an2)]
                ).unsqueeze(-2).repeat_interleave(r.size(-3), -2)
        else:  # -> Batch
            expcutoff = torch.stack([torch.cat([
                gamma_cutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')].to(device)
                for ii, jj in zip(an1[ibatch], an2[ibatch])])
                for ibatch in range(an1.size(0))]
                ).unsqueeze(-2).repeat_interleave(alpha.size(-2), dim=-2)
    # construct the cutoff tensor by searching
    else:
        expcutoff = _expgamma_cutoff(alpha.transpose(-1, -2)[..., 0].unsqueeze(-2),
                                     beta.transpose(-1, -2)[..., 0].unsqueeze(-2),
                                     torch.clone(gamma_tr).transpose(
                                         -1, -2)[..., 0].unsqueeze(-2))
        expcutoff = expcutoff.repeat_interleave(r.size(-3), -2)

    # new masks of homo or hetero Hubbard
    mask_cutoff = distance_tr < expcutoff
    mask_homo = mask_homo if z.ndim == 1 else mask_homo.unsqueeze(-2)
    mask_hetero = mask_hetero if z.ndim == 1 else mask_hetero.unsqueeze(-2)
    mask_homo = mask_homo & mask_cutoff
    mask_hetero = mask_hetero & mask_cutoff

    # triangular gamma values
    gamma_tr = _expgamma(distance_tr, alpha, beta, mask_homo, mask_hetero,
                         gamma_tr)

    # symmetric gamma values
    gamma[..., ut[0], ut[1]] = gamma_tr
    gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]
    gamma = gamma.sum(-3)

    # diagonal values is so called chemical hardness Hubbard
    _onsite = -U[0] if z.ndim == 1 else -U[:, 0]
    _tem = gamma.diagonal(0, -2, -1) + _onsite
    gamma.diagonal(0, -2, -1)[:] = _tem[:]

    # Subtract the gamma matrix from the inverse distance to get the final
    # result.
    gamma = invr - gamma

    return gamma


def _expgamma_cutoff(alpha: Tensor, beta: Tensor, gamma_tem: Tensor,
                     minshortgamma=1e-10, tolshortgamma=1e-10) -> Tensor:
    """The cutoff distance for short range part.

    Arguments:
        alpha: 16/5 * U for the first atom.
        beta: 16/5 * U for the second atom.
        gamma_tem: gamma values.

    Returns:
        mincutoff: The cutoff distance where the short range part goes to zero
            for certain atom pairs.

    Examples:
        >>> from tbmalt.physics.dftb.gamma import _expgamma_cutoff
        >>> ua = torch.tensor([0.3647, 0.4196]) # Hubbard U values of C, H
        >>> ub = torch.tensor([0.4196, 0.4196]) # Hubbard U values of H, H
        >>> cutoff = _expgamma_cutoff(ua * 3.2, ub * 3.2,
                                      torch.zeros_like(ua))

        # Cutoff distances for C-H and H-H in a.u.
        >>> print(cutoff)
        tensor([22.0375, 20.0250])

    """

    # initial distance
    rab = torch.ones_like(alpha)

    # mask of homo or hetero Hubbard in triangular gamma
    mask_homo, mask_hetero = alpha == beta, alpha != beta
    mask_homo[alpha.eq(0)], mask_hetero[alpha.eq(0)] = False, False
    mask_homo[beta.eq(0)], mask_hetero[beta.eq(0)] = False, False

    # mask for batch calculation
    gamma_init = _expgamma(rab, alpha, beta, mask_homo, mask_hetero,
                           torch.clone(gamma_tem))
    mask = gamma_init > minshortgamma

    # determine rab
    while True:
        rab[mask] = 2.0 * rab[mask]
        gamma_init[mask] = _expgamma(rab[mask], alpha[mask], beta[mask],
                                     mask_homo[mask], mask_hetero[mask],
                                     torch.clone(gamma_tem)[mask])
        mask = gamma_init > minshortgamma
        if (~mask).all() == True:
            break

    # bisection search for expcutoff
    mincutoff = rab + 0.1
    maxcutoff = 0.5 * rab - 0.1
    cutoff = maxcutoff + 0.1
    maxgamma = _expgamma(maxcutoff, alpha, beta, mask_homo, mask_hetero,
                         torch.clone(gamma_tem))
    mingamma = _expgamma(mincutoff, alpha, beta, mask_homo, mask_hetero,
                         torch.clone(gamma_tem))
    lowergamma = torch.clone(mingamma)
    gamma = _expgamma(cutoff, alpha, beta, mask_homo, mask_hetero,
                      torch.clone(gamma_tem))

    # mask for batch calculation
    mask2 = (gamma - lowergamma) > tolshortgamma
    while True:
        maxcutoff = 0.5 * (cutoff + mincutoff)
        mask_search = (maxgamma >= mingamma) == (
            minshortgamma >= _expgamma(
                maxcutoff, alpha, beta, mask_homo, mask_hetero, torch.clone(
                    gamma_tem)))
        mask_a = mask2 & mask_search
        mask_b = mask2 & (~mask_search)
        mincutoff[mask_a] = maxcutoff[mask_a]
        lowergamma[mask_a] = _expgamma(mincutoff[mask_a], alpha[mask_a],
                                       beta[mask_a], mask_homo[mask_a],
                                       mask_hetero[mask_a],
                                       torch.clone(gamma_tem)[mask_a])
        cutoff[mask_b] = maxcutoff[mask_b]
        gamma[mask_b] = _expgamma(cutoff[mask_b], alpha[mask_b],
                                  beta[mask_b], mask_homo[mask_b],
                                  mask_hetero[mask_b],
                                  torch.clone(gamma_tem)[mask_b])
        mask2 = (gamma - lowergamma) > tolshortgamma
        if (~mask2).all() == True:
            break

    return mincutoff


def _expgamma(distance_tr: Tensor, alpha: Tensor, beta: Tensor,
              mask_homo: Tensor, mask_hetero: Tensor, gamma_tem: Tensor
              ) -> Tensor:
    """Calculate the value of short range gamma.

    Arguments:
        distance_tr: triangular distance matrix.
        alpha: 16/5 * U for the first atom.
        beta: 16/5 * U for the second atom.
        mask_homo: A mask for homo Hubbard values.
        mask_hetero: A mask for hetero Hubbard values.
        gamma_tem: gamma values.

    Returns:
        gamma_tem: gamma values.

    """

    r_homo = 1.0 / distance_tr[mask_homo]
    r_hetero = 1.0 / distance_tr[mask_hetero]

    # homo Hubbard
    aa, dd_homo = alpha[mask_homo], distance_tr[mask_homo]
    tau_r = aa * dd_homo
    e_fac = torch.exp(-tau_r) / 48.0 * r_homo
    gamma_tem[mask_homo] = \
        (48.0 + 33.0 * tau_r + 9.0 * tau_r ** 2 + tau_r ** 3) * e_fac

    # hetero Hubbard
    aa, bb = alpha[mask_hetero], beta[mask_hetero]
    dd_hetero = distance_tr[mask_hetero]
    aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
    bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
    rab, rba = 1 / (aa2 - bb2), 1 / (bb2 - aa2)
    exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
    val_ab = exp_a * (0.5 * aa * bb4 * rab ** 2 -
                      (bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero)
    val_ba = exp_b * (0.5 * bb * aa4 * rba ** 2 -
                      (aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero)
    gamma_tem[mask_hetero] = val_ab + val_ba

    return gamma_tem


def build_gamma_matrix(
        geometry: Geometry, orbs: OrbitalInfo, invr: Tensor,
        hubbard_Us: Tensor, scheme: Literal['exponential', 'gaussian'] =
        'exponential') -> Tensor:
    """Construct the gamma matrix

    Arguments:
        geometry: `Geometry` object of the system(s) whose gamma matrix is to
            be constructed.
        orbs: `OrbitalInfo` instance associated with the target system.
        invr: 1/R matrix.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.
        scheme: scheme used to construct the gamma matrix. This may be either
            either "exponential" or "gaussian". [DEFAULT="exponential"]

    Return:
        gamma_matrix: the resulting gamma matrix.

    Examples:
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.gamma import build_gamma_matrix
        >>> from ase.build import molecule

        # Preparation of system to calculate
        >>> geo = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs = OrbitalInfo(geo.atomic_numbers,
                               shell_dict= {1: [0], 6: [0, 1]})
        >>> hubbard_U = torch.tensor([0.3647, 0.4196, 0.4196, 0.4196, 0.4196])
        >>> r = geo.distances
        >>> r[r != 0.0] = 1.0 / r[r != 0.0]

        # Build the gamma matrix
        >>> gamma = build_gamma_matrix(geo, orbs, r, hubbard_U, 'exponential')
        >>> print(gamma)
        tensor([[0.3647, 0.3234, 0.3234, 0.3234, 0.3234],
                [0.3234, 0.4196, 0.2654, 0.2654, 0.2654],
                [0.3234, 0.2654, 0.4196, 0.2654, 0.2654],
                [0.3234, 0.2654, 0.2654, 0.4196, 0.2654],
                [0.3234, 0.2654, 0.2654, 0.2654, 0.4196]])

    """

    if geometry.periodicity is None:
        if scheme == 'exponential':
            return gamma_exponential(geometry, orbs, hubbard_Us)

        elif scheme == 'gaussian':
            return gamma_gaussian(geometry, orbs, hubbard_Us)
        else:
            raise (NotImplementedError(
                   f'Gamma constructor method {scheme} is unknown'))
    else:
        if scheme == 'exponential':
            return gamma_exponential_pbc(geometry, orbs, invr, hubbard_Us)
        elif scheme == 'gaussian':
            raise NotImplementedError('Not implement gaussian for pbc yet.')

def gamma_exponential_gradient(geometry: Geometry, orbs: OrbitalInfo, hubbard_Us: Tensor
                      ) -> Tensor:
    """Construct the gradient of the gamma matrix via the exponential method.

    Arguments:
        geometry: `Geometry` object of the system(s) whose gamma matrix is to
            be constructed.
        orbs: `OrbitalInfo` instance associated with the target system.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.

    Returns:
        gamma_grad: gradient of gamma matrix.

    Examples:
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.gamma import gamma_exponential
        >>> from ase.build import molecule

        # Preparation of system to calculate
        >>> geo = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs = OrbitalInfo(geo.atomic_numbers,
                               shell_dict= {1: [0], 6: [0, 1]})
        >>> hubbard_U = torch.tensor([0.3647, 0.4196, 0.4196, 0.4196, 0.4196])

        # Build the gamma matrix
        >>> gamma = gamma_exponential(geo, orbs, hubbard_U)
        >>> print(gamma)
        tensor([[0.3647, 0.3234, 0.3234, 0.3234, 0.3234],
                [0.3234, 0.4196, 0.2654, 0.2654, 0.2654],
                [0.3234, 0.2654, 0.4196, 0.2654, 0.2654],
                [0.3234, 0.2654, 0.2654, 0.4196, 0.2654],
                [0.3234, 0.2654, 0.2654, 0.2654, 0.4196]])

    """

    # Build the Slater type gamma in second-order term.
    U = hubbard_Us
    r = geometry.distances
    z = geometry.atomic_numbers

    # normed version of the distance vectors
    normed_distance_vectors = geometry.distance_vectors / geometry.distances.unsqueeze(-1)
    normed_distance_vectors[normed_distance_vectors.isnan()] = 0

    dtype, device = r.dtype, r.device

    if orbs.shell_resolved:  # and expand it if this is shell resolved calc.
        def dri(t, ind):  # Abstraction of lengthy double interleave operation
            return t.repeat_interleave(ind, -1).repeat_interleave(ind, -2)

        # Get № shells per atom & determine batch status, then expand.
        batch = (spa := orbs.shells_per_atom).ndim >= 2
        r = pack([dri(i, j) for i, j in zip(r, spa)]) if batch else dri(r, spa)

        z = prepeat_interleave(z, orbs.n_shells_on_species(z))

    # Construct index list for upper triangle gather operation
    ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 0))
    distance_tr = r[..., ut[0], ut[1]]
    an1 = z[..., ut[0]]
    an2 = z[..., ut[1]]

    # build the whole gamma, shortgamma (without 1/R) and triangular gamma
    gamma = torch.zeros(r.shape, dtype=dtype, device=device)
    gamma_tr = torch.zeros(distance_tr.shape, dtype=dtype, device=device)
    #build the gamma gradient matrix
    gamma_grad = torch.ones(gamma.size() + (3,), dtype=dtype, device=device)

    # diagonal values is so called chemical hardness Hubbard
    gamma_tr[..., ut[0] == ut[1]] = 0

    # off-diagonal values of on-site part for shell resolved calc
    if orbs.shell_resolved:
        mask_shell = (ut[0] != ut[1]).to(device) * distance_tr.eq(0)
        ua, ub = U[..., ut[0]][mask_shell], U[..., ut[1]][mask_shell]
        mask_diff = (ua - ub).abs() < 1E-8
        gamma_shell = torch.zeros_like(ua, dtype=dtype, device=device)
        gamma_shell[mask_diff] = -0.5 * (ua[mask_diff] + ub[mask_diff])
        if torch.any(~mask_diff):
            ta, tb = 3.2 * ua[~mask_diff], 3.2 * ub[~mask_diff]
            gamma_shell[~mask_diff] = 0.0
        gamma_tr[mask_shell] = gamma_shell

    mask_homo = (an1 == an2) * distance_tr.ne(0)
    mask_hetero = (an1 != an2) * distance_tr.ne(0)
    alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
    r_homo = 1.0 / distance_tr[mask_homo]
    r_hetero = 1.0 / distance_tr[mask_hetero]

    # homo Hubbard
    aa, dd_homo = alpha[mask_homo], distance_tr[mask_homo]
    tau_r = aa * dd_homo
    e_fac = torch.exp(-tau_r) / 48.0 * r_homo**2
    gamma_tr[mask_homo] = \
        -(48.0 + 48.0 * tau_r + 24.0 * tau_r**2 + 7.0 * tau_r**3 + tau_r**4) * e_fac

    # hetero Hubbard
    aa, bb = alpha[mask_hetero], beta[mask_hetero]
    dd_hetero = distance_tr[mask_hetero]
    aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
    bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
    rab, rba = 1 / (aa2 - bb2), 1 / (bb2 - aa2)
    exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
    val_ab = exp_a * ( ((bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero**2) -
                      aa * (0.5 * aa * bb4 * rab ** 2 -
                      (bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero) )
    val_ba = exp_b * ( ((aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero**2) -
                      bb *(0.5 * bb * aa4 * rba ** 2 -
                      (aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero) )
    gamma_tr[mask_hetero] = val_ab + val_ba

    # to make sure gamma values symmetric
    gamma[..., ut[0], ut[1]] = gamma_tr
    gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]

    gamma = gamma.squeeze()

    # Subtract the gamma matrix from the inverse distance to get the final
    # result.
    r[r != 0.0] = 1.0 / r[r != 0.0]
    gamma = -r**2 - gamma
    gamma_grad = normed_distance_vectors * gamma.unsqueeze(-1)

    return gamma_grad


