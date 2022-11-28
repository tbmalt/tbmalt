# -*- coding: utf-8 -*-
"""Short gamma calculations."""
from typing import Literal
import torch
from torch import Tensor
from tbmalt import Geometry, Basis
from tbmalt.physics.dftb.feeds import Feed
import numpy as np
from tbmalt.common.batch import pack, prepeat_interleave
from tbmalt.data import gamma_cutoff, gamma_element_list


def gamma_exponential(geometry: Geometry, basis: Basis, hubbard_Us: Tensor):
    """Construct the gamma matrix via the exponential method.

    Arguments:
        geometry: `Geometry` object of the system(s) whose gamma matrix is to
            be constructed.
        basis: `Basis` instance associated with the target system.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.

    Returns:
        gamma: gamma matrix.

    """
    """Build the Slater type gamma in second-order term."""
    U = hubbard_Us
    r = geometry.distances
    z = geometry.atomic_numbers

    dtype, device = r.dtype, r.device

    if basis.shell_resolved:  # and expand it if this is shell resolved calc.
        def dri(t, ind):  # Abstraction of lengthy double interleave operation
            return t.repeat_interleave(ind, -1).repeat_interleave(ind, -2)

        # Get № shells per atom & determine batch status, then expand.
        batch = (spa := basis.shells_per_atom).ndim >= 2
        r = pack([dri(i, j) for i, j in zip(r, spa)]) if batch else dri(r, spa)

        z = prepeat_interleave(z, basis.n_shells_on_species(z))

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


def gamma_gaussian(geometry: Geometry, basis: Basis, hubbard_Us: Tensor
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
        basis: `Basis` instance associated with the target system.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.

    Returns:
        gamma: gamma matrix.

    Notes:
        One Hubbard U value must be specified for each atom or each shell based
        on whether an atom or shell resolved calculation is being performed.
        Note that this must be consistent with the `shell_resolved` attribute
        of the supplied `Basis` instance ``basis``.

        Currently the Hubbard U values must be specified manually, however the
        option to supply a feed object will be added at a later data. This will
        facilitate the automated construction of the Hubbard U tensor and shall
        permit the use of environmentally dependent U values if desired.

    """
    # Developers Notes: At some point in the future this function will be
    # changed to allow dictionaries and UFeeds to be supplied. This will allow
    # for more flexible and concise code. It would be nice to make the r matrix
    # expansion operation batch agnostic. This could be achieved by manually
    # computing the distance matrix or using gather operations.

    if geometry.positions.ndim >= 2:
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
    if basis.shell_resolved:  # and expand it if this is shell resolved calc.
        def dri(t, ind):  # Abstraction of lengthy double interleave operation
            return t.repeat_interleave(ind, -1).repeat_interleave(ind, -2)

        # Get № shells per atom & determine batch status, then expand.
        batch = (spa := basis.shells_per_atom).ndim >= 2
        r = pack([dri(i, j) for i, j in zip(r, spa)]) if batch else dri(r, spa)

    # Build the "C" coefficients (eq. 27)
    c = torch.sqrt((4. * np.log(2.)) /
                   (fwhm.unsqueeze(-2)**2 + fwhm.unsqueeze(-1)**2))

    # Construct the gamma matrix then set the diagonals.
    mask = (*r.nonzero().T,)
    gamma = torch.erf(c * r)  # <──(eq. 26)──┐
    gamma[mask] = gamma[mask] / r[mask]  # <─┘
    gamma.diagonal(0, -2, -1)[:] = hubbard_Us[:]  # (eq. 31)

    return gamma


def gamma_exponential_pbc(geometry, basis, invr, hubbard_Us):
    """Build the Slater type gamma in second-order term with pbc."""

    r = geometry.periodic.periodic_distances
    U = torch.clone(hubbard_Us).repeat(r.size(-3), 1, 1).transpose(0, 1)
    z = geometry.atomic_numbers
    U = U.squeeze(0) if z.ndim == 1 else U

    dtype, device = r.dtype, r.device

    # TODO: shell resolved calc is missing. For this, coulomb also needs shell
    # resolved calc.
    if basis.shell_resolved:
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

    # new masks of homo or hetero Hubbert
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

    # diagonal values is so called chemical hardness Hubbert
    _onsite = -U[0] if z.ndim == 1 else -U[:, 0]
    _tem = gamma.diagonal(0, -2, -1) + _onsite
    gamma.diagonal(0, -2, -1)[:] = _tem[:]

    # Subtract the gamma matrix from the inverse distance to get the final
    # result.
    gamma = invr - gamma

    return gamma


def _expgamma_cutoff(alpha, beta, gamma_tem,
                     minshortgamma=1.0e-10, tolshortgamma=1.0e-10):
    """The cutoff distance for short range part."""
    # initial distance
    rab = torch.ones_like(alpha)

    # mask of homo or hetero Hubbert in triangular gamma
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


def _expgamma(distance_tr, alpha, beta, mask_homo, mask_hetero, gamma_tem):
    """Calculate the value of short range gamma."""
    r_homo = 1.0 / distance_tr[mask_homo]
    r_hetero = 1.0 / distance_tr[mask_hetero]

    # homo Hubbert
    aa, dd_homo = alpha[mask_homo], distance_tr[mask_homo]
    tau_r = aa * dd_homo
    e_fac = torch.exp(-tau_r) / 48.0 * r_homo
    gamma_tem[mask_homo] = \
        (48.0 + 33.0 * tau_r + 9.0 * tau_r ** 2 + tau_r ** 3) * e_fac

    # hetero Hubbert
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
        geometry: Geometry, basis: Basis, invr: Tensor,
        hubbard_Us: Tensor, scheme: Literal['exponential', 'gaussian'] =
        'exponential'):
    """Construct the gamma matrix

    Arguments:
        geometry: `Geometry` object of the system(s) whose gamma matrix is to
            be constructed.
        basis: `Basis` instance associated with the target system.
        invr: 1/R matrix.
        hubbard_Us: Hubbard U values. one value should be specified for each
            atom or shell depending if the calculation being performed is atom
            or shell resolved.
        scheme: scheme used to construct the gamma matrix. This may be either
            either "exponential" or "gaussian". [DEFAULT="exponential"]

    Return:
        gamma_matrix: the resulting gamma matrix.

    """

    if geometry.periodic is None:
        if scheme == 'exponential':
            return gamma_exponential(geometry, basis, hubbard_Us)

        elif scheme == 'gaussian':
            return gamma_gaussian(geometry, basis, hubbard_Us)
        else:
            raise(NotImplemented(
                f'Gamma constructor method {scheme} is unknown'))
    else:
        if scheme == 'exponential':
            return gamma_exponential_pbc(geometry, basis, invr, hubbard_Us)
        elif scheme == 'gaussian':
            raise NotImplementedError('Not implement gaussian for pbc yet.')
