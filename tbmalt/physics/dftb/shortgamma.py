# -*- coding: utf-8 -*-
"""Short gamma calculations."""
from typing import Literal
import torch
from torch import Tensor
from tbmalt import Geometry, Basis
from tbmalt.physics.dftb.feeds import HubbardFeed


class ShortGamma:
    """Calculate the short gamma in second-order DFTB.

    Arguments:
        u_feed: Feed for Hubbard U.
        gamma_type: Short gamma calculation method, right now only support
            Slater type basis based exponential method.

    Attributes:
        u_feed: Feed for Hubbard U.
        gamma_type: Short gamma calculation method, right now only support
            Slater type basis based exponential method.

    TODO:
        Add periodic conditions and gaussian method.
    """

    def __init__(self, u_feed: HubbardFeed, orbital_resolve: bool = False,
                 gamma_type: Literal['exponential', 'gaussian'] = 'exponential'):
        self.u_feed = u_feed
        self.orbital_resolve = orbital_resolve
        self.gamma_type = gamma_type
        if self.orbital_resolve:
            raise NotImplementedError('do not support orbital_resolve')

    def __call__(self, geometry: Geometry, basis: Basis):
        r"""Calculate the short gamma in second-order DFTB.

        Arguments:
            geometry: `Geometry` object in TBMaLT.
            basis: `Basis` object in TBMaLT.

        Return:
            short_gamma: Calculated short gamma for single or batch systems.

        """
        U = self.u_feed(basis, self.orbital_resolve)
        atomic_numbers = geometry.atomic_numbers
        distances = geometry.distances
        return getattr(ShortGamma, self.gamma_type)(U, atomic_numbers, distances)

    @staticmethod
    def exponential(U: Tensor, atomic_numbers, distances: Tensor) -> Tensor:
        """Build the Slater type gamma in second-order term."""
        dtype, device = distances.dtype, distances.device

        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 0))
        distance_tr = distances[..., ut[0], ut[1]]
        an1 = atomic_numbers[..., ut[0]]
        an2 = atomic_numbers[..., ut[1]]

        # build the whole gamma, shortgamma (without 1/R) and triangular gamma
        gamma = torch.zeros(distances.shape, dtype=dtype, device=device)
        gamma_tr = torch.zeros(distance_tr.shape, dtype=dtype, device=device)

        # diagonal values is so called chemical hardness Hubbert
        gamma_tr[..., ut[0] == ut[1]] = -U

        mask_homo = (an1 == an2) * distance_tr.ne(0)
        mask_hetero = (an1 != an2) * distance_tr.ne(0)
        alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
        r_homo = 1.0 / distance_tr[mask_homo]
        r_hetero = 1.0 / distance_tr[mask_hetero]

        # homo Hubbert
        aa, dd_homo = alpha[mask_homo], distance_tr[mask_homo]
        tau_r = aa * dd_homo
        e_fac = torch.exp(-tau_r) / 48.0 * r_homo
        gamma_tr[mask_homo] = \
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
        gamma_tr[mask_hetero] = val_ab + val_ba

        # to make sure gamma values symmetric
        gamma[..., ut[0], ut[1]] = gamma_tr
        gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]

        return gamma.squeeze()

    def gaussian(self):
        """Build the Gaussian type gamma in second-order term."""
        raise NotImplementedError('Not implement gaussian yet.')