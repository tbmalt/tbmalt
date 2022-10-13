# -*- coding: utf-8 -*-
"""Short gamma calculations."""
from typing import Literal
import torch
from torch import Tensor
from tbmalt import Geometry, Basis
from tbmalt.physics.dftb.feeds import Feed
import numpy as np
from tbmalt.common.batch import pack, prepeat_interleave

def gamma_exponential(geometry: Geometry, basis: Basis, hubbard_Us: Tensor):
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


def build_gamma_matrix(geometry, basis, hubbard_Us, form='exponential'):
    """Construct the gamma matrix

    This is mostly just a placeholder function.

    """

    assert geometry.cells is None, "Gamma construct does not yet support PBC"

    if form == 'exponential':
        gamma_matrix = gamma_exponential(geometry, basis, hubbard_Us)

    elif form == 'gaussian':
        gamma_matrix = gamma_gaussian(geometry, basis, hubbard_Us)
    else:
        raise(NotImplemented(f'Gamma constructor methods {form} is not known'))

    return gamma_matrix
