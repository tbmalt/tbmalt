# -*- coding: utf-8 -*-
"""Functions relating to the calculation of physical properties.

This module contains functions pertaining to the calculation of general
properties.
"""
import torch
import numpy as np
from numbers import Real
from typing import Optional, Tuple, Union
from tbmalt.common.batch import pack

Tensor = torch.Tensor


##################################
# Density of States Related Code #
##################################
def _generate_broadening(energies: Tensor, eps: Tensor,
                         sigma: Union[Real, Tensor] = 0.0) -> Tensor:
    """Construct the gaussian broadening terms.

    This is used to calculate the gaussian broadening terms used when
    calculating the DoS/PDoS.

    Arguments:
        energies: Energy values to evaluate the DoS/PDoS at.
        eps: Energy eigenvalues (epsilon).
        sigma: Smearing width for gaussian broadening function. [DEFAULT=0]

    Returns:
        g: Gaussian broadening terms.

    """

    def _gaussian_broadening(energy_in: Tensor, eps_in: Tensor, sigma: Real
                             ) -> Tensor:
        """Gaussian broadening factor used when calculating the DoS/PDoS."""
        return torch.erf((energy_in[..., :, None] - eps_in[..., None, :]).T
                         / (np.sqrt(2) * sigma)).T

    # Construct gaussian smearing terms.
    de = energies[..., 1] - energies[..., 0]
    ga = _gaussian_broadening((energies.T - (de / 2)).T, eps, sigma)
    gb = _gaussian_broadening((energies.T + (de / 2)).T, eps, sigma)
    return ((gb - ga).T / (2.0 * de)).T


def dos(eps: Tensor, energies: Tensor, sigma: Union[Real, Tensor] = 0.0,
        offset: Optional[Union[Real, Tensor]] = None,
        mask: Optional[Tensor] = None, scale: bool = False) -> Tensor:
    r"""Calculates the density of states for one or more systems.

    This calculates and returns the Density of States (DoS) for one or more
    systems. If desired, all but a selection of specific states can be masked
    out via the ``mask`` argument.

    Arguments:
        eps: Energy eigenvalues (epsilon).
        energies: Energy values to evaluate the DoS at. These are assumed to
            be relative to the ``offset`` value, if it is specified.
        sigma: Smearing width for gaussian broadening function. [DEFAULT=0]
        offset: Indicates that ``energies`` are given with respect to a offset
            value, e.g. the fermi energy.
        mask: Used to control which states are used in constricting the DoS.
            Only unmasked (True) states will be used, all others are ignored.
        scale: Scales the DoS to have a maximum value of 1. [DEFAULT=False]

    Returns:
        dos: The densities of states.

    Notes:
        The DoS is calculated via an equation equivalent to:

        .. math::
            g(E)=\delta(E-\epsilon_{i})

        Where g(E) is the density of states at an energy value E, and δ(E-ε)
        is the smearing width calculated as:

        .. math::
            \delta(E-\epsilon)=\frac{
                erf\left(\frac{E-\epsilon+\frac{\Delta E}{2}}{\sqrt{2}\sigma}\right)-
                erf\left(\frac{E-\epsilon-\frac{\Delta E}{2}}{\sqrt{2}\sigma}\right)}
                {2\Delta E}

        Where ΔE is the difference in energy between neighbouring points.
        It may be useful, such as in the creation of a cost function, to have
        only specific states (i.e. HOMO, HOMO-1, etc.) used to construct the
        PDoS. State selection can be achieved via the ``mask`` argument. This
        should be a boolean tensor with a shape matching that of ``eps``. Only
        states whose mask value is True will be included in the DoS, e.g.

            mask = torch.Tensor([True, False, False, False])

        would only use the first state when constructing the DoS.

    Warnings:
        It is imperative that padding values are masked out when operating on
        batches of systems! Failing to do so will result in the emergence of
        erroneous state occupancies. Care must also be taken to ensure that
        the number of sample points is the same for each system; i.e. all
        systems have the same number of elements in ``energies``. As padding
        values will result in spurious behaviour.

    Examples:
        Density of states constructed for an H2 molecule using test data:

        >>> from tests.unittests.data.properties.dos import H2
        >>> from tbmalt.physics.dftb.properties import dos
        >>> import matplotlib.pyplot as plt
        >>> eps = H2['eigenvalues']
        >>> energies = H2['dos']['energy']
        >>> sigma = H2['sigma']
        >>> dos_values = dos(eps, energies, sigma)
        >>> plt.plot(energies, dos_values)
        >>> plt.xlabel('Energy [Ha]')
        >>> plt.ylabel('DoS [Ha]')
        >>> plt.show()

    """
    if mask is not None:  # Mask out selected eigen-states if requested.
        eps = eps.clone()  # Use a clone to prevent altering the original
        # Move masked states towards +inf
        eps[~mask] = torch.finfo(eps.dtype).max

    if offset is not None:  # Apply the offset, if applicable.
        # Offset must be applied differently for batches.
        if isinstance(offset, (Tensor, np.ndarray)) and len(offset.shape) > 0:
            eps = eps - offset[:, None]
        else:
            eps = eps - offset

    g = _generate_broadening(energies, eps, sigma)  # Gaussian smearing terms.
    distribution = torch.sum(g, -1)  # Compute the densities of states

    # Rescale the DoS so that it has a maximum peak value of 1.
    if scale:
        distribution = distribution / distribution.max(-1, keepdim=True)[0]

    return distribution


def band_pass_state_filter(eps: Tensor, n_homo: Union[int, Tensor],
                           n_lumo: Union[int, Tensor],
                           fermi: Union[Real, Tensor]) -> Tensor:
    """Generates masks able to filter out states too far from the fermi level.

     This function returns a mask for each ``eps`` system that can filter out
     all but a select number of states above and below the fermi level.

    Arguments:
        eps: Eigenvalues.
        n_homo: n states below the fermi level to retain, including the HOMO.
        n_lumo: n states above the fermi level to retain.
        fermi: Fermi level.

    Returns:
        mask: A boolean mask which is True for selected states.

    Notes:
        For each system, a ``n_homo``, ``n_lumo``, and ``fermi`` value must be
        provided. This assumes that all states, ``eps`` are ordered from
        lowest to highest.

    Warnings:
        It is down to the user to ensure that the number of requested HOMOs &
        LUMOs for each system is valid. This cannot be done safely here due to
        the effects of zero-padded packing; i.e. this function sees padding
        zeros as valid LUMO states.

    Raises:
        RuntimeError: If multiple systems have been provided but not multiple
            ``n_homo``, ``n_lumo``, and ``fermi`` values.

    Examples:
        Here, all but three states below and two states above the fermi level
        are masked out:

        >>> from tbmalt.physics.dftb.properties import band_pass_state_filter
        >>> eps = torch.arange(-4., 6.)
        >>> mask = band_pass_state_filter(eps, 3, 2, 0.)
        >>> print(eps)
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
        >>> print(mask)
        tensor([False, False, True,  True,  True,  True,  True,  False,
                False, False])

    """
    def index_list(eps_in, fermi_in):
        """Generate a state index list offset relative to the HOMO level.

        e.g: [-n, ..., -2, -1, 0, 1, 2, ..., +n], where 0 is at the index of
        the HOMO. An issue is encountered when operating on zero-padded packed
        data, as the padding 0s can get miss-identified as the homo. We want:
        [True, True, True, False, False, True, True]
                 this ↑,   but not these: ↑     ↑, to be the zero point. Thus,
        a more involved method must be used to get the HOMO state's index.
        """
        le_fermi = torch.where(eps_in <= fermi_in)[0]  # Find values ≤ fermi
        if len(le_fermi) == 1:  # If there's 1 value; then it is homo e.g. H2.
            homo = le_fermi[0]
        else:
            # Construct a difference array to highlight non-sequential states.
            # The Homo will be the last entry of the first sequential block.
            diff = le_fermi[1:] - le_fermi[:-1]
            homo = le_fermi[torch.unique_consecutive(diff, return_counts=True)[1][0]]


        # Generate and return the index list
        return torch.arange(eps.shape[-1], device=eps.device) - homo

    # If multiple systems were specified then ensure that multiple n_homo,
    # n_lumo, and fermi values were also specified.
    if eps.dim() == 2:
        if not all([isinstance(i, Tensor) and i.dim() != 0 for i in
                    [n_homo, n_lumo, fermi]]):
            raise RuntimeError('n_homo, n_lumo, and fermi values must be '
                               'provided for each system.')

    # Build relative index list, batch & non-batch must be handled differently.
    if eps.dim() == 1:
        ril = index_list(eps, fermi)
    else:
        ril = torch.stack([index_list(e, f) for e, f in zip(eps, fermi)])

    # Create & return a mash that masks out states outside of the band filter
    return ((-n_homo < ril.T) & (ril.T <= n_lumo)).T
