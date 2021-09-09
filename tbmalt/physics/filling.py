# -*- coding: utf-8 -*-
"""Code associated with electronic/finite temperature."""
from typing import Union, Tuple, Optional, Callable
from numbers import Real
from numpy import sqrt, pi
import torch
from torch import Tensor

from tbmalt import ConvergenceError
from tbmalt import Basis
from tbmalt.common import float_like
from tbmalt.common.batch import psort

_Scheme = Callable[[Tensor, Tensor, float_like], Tensor]


def fermi_entropy(eigenvalues: Tensor, fermi_energy: Tensor, kT: float_like,
                  e_mask: Optional[Union[Tensor, Basis]] = None) -> Tensor:
    r"""Calculates the electronic entropy term for Fermi-Dirac smearing.

        Calculate a system's electronic entropy term. The entropy term is required
        when calculating various properties, most notably the Mermin free energy;
        which is used in place of the total system energy when finite temperature
        (electronic broadening) is active.

        .. math::

            TS = -k_B\sum_{i}f_i \; ln(f_i) + (1 - f_i)\; ln(1 - f_i))

        Arguments:
            eigenvalues: Eigen-energies, i.e. orbital-energies.
            fermi_energy: The Fermi energy.
            kT: Electronic temperature.
            e_mask: Padding mask see :func:`fermi_search` for more information.
                [DEFAULT=None]

        Returns:
            ts: The entropy term(s).

        """
    # If a Basis instance was given as a mask then convert it to a tensor
    if isinstance(e_mask, Basis):
        e_mask = e_mask.on_atoms != -1

    # Shape of eigenvalue tensor in which k-points & spin-channels (with
    # common fermi-energies) are flattened out.
    shp = (*fermi_energy.shape, -1)
    eigenvalues = eigenvalues.view(shp)
    if e_mask is not None and e_mask.ndim != 0:
        e_mask = e_mask.view(shp)

    # Get fractional orbital occupancies
    fo = fermi_smearing(eigenvalues, fermi_energy, kT)
    # Log form is used as it avoids having to cull error inducing 1's/0's
    s = -torch.log(fo ** fo * (1 - fo) ** (1 - fo))

    if e_mask is not None:  # Mask out *fake* padding states as appropriate
        # pylint: disable=E1130
        s[~e_mask] = 0.0

    return s.sum(-1) * kT


def gaussian_entropy(eigenvalues: Tensor, fermi_energy: Tensor, kT: float_like,
                     e_mask: Optional[Union[Tensor, Basis]] = None) -> Tensor:
    r"""Calculates the electronic entropy term for Gaussian bases smearing.

        .. math::

            TS = \frac{k_B}{2 \sqrt{\pi}} \sum_{i} exp \left(- \left(
                    \frac{\epsilon_i E_f}{kT} \right)^2 \right)

        Arguments:
            eigenvalues: Eigen-energies, i.e. orbital-energies.
            fermi_energy: The Fermi energy.
            kT: Electronic temperature.
            e_mask: Padding mask see :func:`fermi_search` for more information.
                [DEFAULT=None]

        Returns:
            ts: The entropy term(s).

        """
    # If a Basis instance was given as a mask then convert it to a tensor
    if isinstance(e_mask, Basis):
        e_mask = e_mask.on_atoms != -1

    # Shape of eigenvalue tensor in which k-points & spin-channels (with
    # common fermi-energies) are flattened out.
    shp = (*fermi_energy.shape, -1)
    eigenvalues = eigenvalues.view(shp)
    if e_mask is not None and e_mask.ndim != 0:
        e_mask = e_mask.view(shp)

    # kT/fermi_energy must be correctly shaped for division to work
    fermi_energy, kT = _smearing_preprocessing(eigenvalues, fermi_energy, kT)

    # Calculate xi values then return the entropy
    xi = (eigenvalues - fermi_energy) / kT
    s = torch.exp(-xi ** 2) * 0.5 / sqrt(pi)

    if e_mask is not None:  # Mask out *fake* padding states as appropriate
        # pylint: disable=E1130
        s[~e_mask] = 0.0

    s = s.sum(-1)
    if kT.nelement() == s.nelement():
        kT = kT.view_as(s)

    return s * kT


def _smearing_preprocessing(
        eigenvalues: Tensor, fermi_energy: Tensor, kT: float_like
        ) -> Tuple[Tensor, Tensor]:
    """Abstracts repetitive code from the smearing functions.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        fermi_energy: The Fermi energy.
        kT: Electronic temperature.

    Returns:
        fermi_energy: Processed fermi energy tensor.
        kT: Processed kT tensor.

    """
    # These must be tensors for code to be batch agnostic & device safe
    assert isinstance(eigenvalues, Tensor), 'eigenvalues must be a tensor'
    assert isinstance(fermi_energy, Tensor), 'fermi_energy must be a tensor'

    # Shape fermi_energy so that there is one entry per row (repeat for kT).
    if fermi_energy.ndim == 1 and len(fermi_energy) != 1:
        # fermi_energy = fermi_energy.view(-1, 1)
        fermi_energy = fermi_energy.view(-1, *[1] * (eigenvalues.ndim - 1))

    # Ensure kT is a tensor & is shaped correctly if multiple values passed
    if not isinstance(kT, Tensor):
        kT = torch.tensor(kT, dtype=eigenvalues.dtype,
                          device=eigenvalues.device)

    if kT.ndim >= 1 and len(kT) != 1:
        kT = kT.view(-1, *[1] * (eigenvalues.ndim - 1))

    # kT cannot be allowed to be true zero, otherwise nan's will occur.
    kT = torch.max(torch.tensor(torch.finfo(eigenvalues.dtype).tiny), kT)

    return fermi_energy, kT


def fermi_smearing(eigenvalues: Tensor, fermi_energy: Tensor, kT: float_like
                   ) -> Tensor:
    r"""Fractional orbital occupancies due to Fermi-Dirac smearing.

    Using Fermi-Dirac smearing, orbital occupancies are calculated via:

    .. math::

        f_i = \frac{1}{1 + exp\left ( \frac{\epsilon_i - E_f}{kT}\right )}

    where ε, :math:`E_f` & :math:`kT` are the eigenvalues, fermi-energies and
    electronic temperatures respectively.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        fermi_energy: The Fermi energy.
        kT: Electronic temperature.

    Returns:
        occupancies: Occupancies of the orbitals, or total electron count if
            total=True.

    Notes:
        ``eigenvalues`` may be a single value, an array of values or a tensor.
        If a tensor is passed then multiple ``fermi_energy`` and ``kT`` values
        may be passed if desired.

        If multiple systems are passed, smearing will be applied to all eigen
        values present, irrespective of whether they are real or fake (caused
        by packing).

    Warnings:
        Gradients resulting from this function can be ill defined, i.e. nan.
    """
    # Developers Notes: it might be worth trying to resolve the gradient
    # stability issue associated with this function.
    fermi_energy, kT = _smearing_preprocessing(eigenvalues, fermi_energy, kT)
    # Calculate and return the occupancies values via the Fermi-Dirac method
    return 1.0 / (1.0 + torch.exp((eigenvalues - fermi_energy) / kT))


def gaussian_smearing(eigenvalues: Tensor, fermi_energy: Tensor,
                      kT: float_like) -> Tensor:
    r"""Fractional orbital occupancies due to Gaussian smearing.

    Using Gaussian smearing, orbital occupancies are calculated via:

    .. math::

        f_i = frac{\textit{erfc}\left( \frac{\epsilon_i - E_f}{kT} \right)}{2}

    where ε, :math:`E_f` & :math:`kT` are the eigenvalues, fermi-energies and
    electronic temperatures respectively.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        fermi_energy: The Fermi energy.
        kT: Electronic temperature.

    Returns:
        occupancies: Occupancies of the orbitals.

    Notes:
        ``eigenvalues`` may be a single value, an array of values or a tensor.
        If a tensor is passed then multiple ``fermi_energy`` and ``kT`` values
        may be passed if desired.

        If multiple systems are passed, smearing will be applied to all eigen
        values present, irrespective of whether they are real or fake (caused
        by packing).

    Warnings:
        Gradients will become unstable if a `kT` value of zero is used.

    """
    fermi_energy, kT = _smearing_preprocessing(eigenvalues, fermi_energy, kT)
    # Calculate and return the occupancies values via the Gaussian method
    return torch.erfc((eigenvalues - fermi_energy) / kT) / 2


@torch.no_grad()
def fermi_search(
        eigenvalues: Tensor, n_electrons: float_like,
        kT: Optional[float_like] = None, scheme: _Scheme = fermi_smearing,
        tolerance: Optional[Real] = None, max_iter: int = 200,
        e_mask: Optional[Union[Tensor, Basis]] = None,
        k_weights: Optional[Tensor] = None) -> Tensor:
    r"""Determines the Fermi-energy of a system or batch thereof.

    Calculates the Fermi-energy with or without finite temperature. Finite
    temperature can be enabled by specifying a ``kT`` value. Note that this
    function will always operate outside of any graph.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies. This may have up to
            4 dimensions, 3 of which are optional, so long as the following
            order is satisfied [batch, spin, k-points, eigenvalues].
        n_electrons: Total number of (valence) electrons.
        kT: Electronic temperature. By default finite temperature is not
            active, i.e. ``kT`` = None. [DEFAULT=None]
        scheme: Finite temperature broadening function to be used, TBMaLT
            natively supports two broadening methods:

                - Fermi-Dirac broadening :func:`fermi_smearing`
                - Gaussian broadening :func:`gaussian_smearing`

            Only used when ``kT`` is not None. [DEFAULT=`fermi_smearing`]

        tolerance: Tolerance to which e⁻ count is converged during the search;
            defaults to 1E-10/5/2 for 64/32/16 bit floats respectively. Not
            used when finite temperature is disabled. [DEFAULT=None]
        max_iter: Maximum permitted number of fermi search cycles; ignored
            when finite temperature is disabled. [DEFAULT=200]
        e_mask: Provides info required to distinguish "real" ``eigenvalues``
            from "fake" ones. This is Mandatory when using smearing on batched
            systems. This may be a `Tensor` that is `True` for real states or
            a `Basis` object. [DEFAULT=None]
        k_weights: If periodic systems are supplied then k-point wights can be
            given via this argument.

    Returns:
        fermi_energy: Fermi energy value(s).

    Warnings:
        It is imperative to ensure that ``e_mask`` is specified when, and only
        when, a batch of systems is provided. Failing to satisfy this condition
        will cause **spurious and hard to diagnose errors**.

        This function operates outside of the pytorch autograd graph and is
        therefore **not** back-propagatable!

    Notes:
        The eigenvalues should be ordered from lowest to highest, with padding
        values located at the end.

        Smearing is disabled if ``kT`` = `None`, causing Fermi energy to resolve
        to the HOMO & LUMO midpoint. Whereas a value of 0 carries out smearing
        with a temperature of 0. Values can be specified on a system by system
        basis or a single value can be provided which is used by all. However,
        smearing cannot be applied selectively; i.e. finite temperature can be
        enabled for some systems but not others during a single call.

        This function will run outside of any torch graph & is therefore not
        back-propagatable. This avoids introducing unnecessary and expensive
        clutter into the graph.

        This code is based on the DFTB+ etemp module. [1]_ However, unlike the
        DFTB+ implementation, no final Newton-Raphson step is performed.

    Raises:
        ConvergenceFailure: If the fermi level search fails to converge
            within the permitted number of iterations.
        ValueError: If the tolerance value is too tight for the specified
            dtype, a negative ``kT`` value is encountered, the number of
            electrons is to zero or exceeds the number of available states.

    References:
        .. [1] Hourahine, B., Aradi, B., Blum, V., Frauenheim, T. et al.,
               (2020). DFTB+, a software package for efficient approximate
               density functional theory based atomistic simulations. The
               Journal of Chemical Physics, 152(12), 124101.
    """
    def middle_gap_approximation():
        """Returns the midpoint between the HOMO and LUMO."""
        # Flatten & sort the eigenvalues so there's 1 row per-system; the
        # spin dimension is only flattened when the spin channels share a
        # common fermi-energy.
        ev_flat, srt = psort(
            e_vals.view(shp),
            None if e_mask is None else e_mask.view(shp))

        # Maximum occupation of each eigenstate, sorted and flattened.
        occ = (torch.ones_like(ev_flat) * scale_factor).gather(-1, srt)
        # Locate HOMO index, via the transition between under/over filled
        i_homo = occ.cumsum(-1).T.le(n_elec).T.diff().nonzero().T[-1].view(shp)
        # Return the Fermi value
        return ev_flat.gather(-1, torch.cat((i_homo, i_homo + 1), -1)).mean(-1)

    # __Setup__
    # Arguments, tolerance, eigenvalues, n_electrons aliased here for brevity.
    e_vals, n_elec, tol = eigenvalues, n_electrons, tolerance
    dtype, dev = e_vals.dtype, e_vals.device

    # Convert n_elec & kT into tensors to make them easier to work with.
    if not isinstance(n_elec, Tensor):
        n_elec = torch.tensor(n_elec, dtype=dtype, device=dev)
    if kT is not None and not isinstance(kT, Tensor):
        kT = torch.tensor(kT, dtype=dtype, device=dev)

    # Scaling factor is the max № of electrons that can occupancy each state;
    # 2/1 for restricted/unrestricted molecular systems. For periodic systems
    # this is then multiplied by the k-point weights.
    pf = 5 - e_vals.ndim - [k_weights, e_mask].count(None)
    scale_factor = pf if k_weights is None else pf * k_weights

    # Shape of Ɛ tensor where k-points & spin-channels have been flattened out.
    # Note that only spin-channels with common fermi energies get flattened.
    shp = torch.Size([*n_elec.shape, -1])

    # If a Basis instance was given as a mask then convert it to a tensor
    if isinstance(e_mask, Basis):
        e_mask = e_mask.on_atoms != -1

    # __Error Checking__
    eps = torch.finfo(dtype).eps
    if tol is None:  # auto-assign if no tolerance was given
        tol = {torch.float64: 1E-10, torch.float32: 1E-5,
               torch.float16: 1E-2}[dtype]
    elif tol < eps:  # Ensure tolerance value is viable
        raise ValueError(f'Tolerance {tol:7.1E} too tight for "{dtype}", '
                         f'the minimum permitted value is: {eps:7.1E}.')

    if kT is not None and (kT < 0.0).any():  # Negative kT catch
        raise ValueError(f'kT must be positive or None ({kT})')

    if torch.lt(n_elec.abs(), eps).any():  # A system has no electrons
        raise ValueError('Number of elections cannot be zero.')

    # A system has too many electrons
    if torch.any((n_elec / pf).gt(e_vals.view(shp).shape[-1] if e_mask is None
                               else e_mask.view(shp).count_nonzero(-1))):
        raise ValueError('Number of electrons cannot exceed 2 * n states')

    # __Finite Temperature Disabled__
    # Set the fermi energy to the mid point between the HOMO and LUMO.
    if kT is None:
        return middle_gap_approximation()

    # __Finite Temperature Enabled__
    # Perform a fermi level search via the bisection method
    else:
        # e_fermi holds results & c_mask tracks which systems have converged.
        e_fermi = torch.zeros_like(n_elec, device=dev, dtype=dtype)
        c_mask = torch.full_like(n_elec, False, dtype=torch.bool, device=dev)

        def elec_count(f, m=...):
            """Makes a call to the smearing function & returns the sum.

            This limits the messy masking operations needed by the smearing
            function to one place. kT's mask are treated differently as 0d &
            1d are both valid shapes of kT in both batch & single system mode.
            """
            res = scheme(e_vals[m], f[m], kT[m if kT.ndim != 0 else ...])
            if e_mask is not None:  # Cull "fake" states caused by padding
                res[~e_mask[m]] = 0.0

            # Sum up over all axes apart from the batch dimension
            return (res * scale_factor).T.sum_to_size(n_elec[m].shape)

        # If there's an even, integer number of e⁻; try setting e_fermi to
        # the middle gap, i.e. fill according to the Aufbau principle.
        if (mask := abs((n_elec / 2) - (n_elec / 2).round()) <= tol).any():
            # Store fermi value, recalculate № of e⁻ & identity of convergence
            e_fermi[mask] = middle_gap_approximation()[mask]
            c_mask[mask] = abs(elec_count(e_fermi)[mask] - n_elec[mask]) < tol

        # If all systems converged then just return the results now
        if c_mask.all():
            return e_fermi.view_as(n_elec)

        # __Setup Bounds for Bisection Search__
        # Identify upper (e_up) & lower (e_lo) search bounds; fermi level should
        # be between the highest & lowest eigenvalues, so start there.
        e_lo = e_vals.view(shp).min(-1).values
        e_up = e_vals.view(shp).max(-1).values
        ne_lo, ne_up = elec_count(e_lo), elec_count(e_up)

        # Bounds may fail on large kT or full band structures; if too many e⁻
        # are present at the e_lo then decrease it & recalculate. If too few e⁻
        # present at the e_up, then it's too low so increase it & recalculate
        # the number of elections there.
        while (mask := ne_lo > n_elec).any():
            e_lo[mask] += 2.0 * (e_lo[mask] - e_up[mask])
            ne_lo[mask] = elec_count(e_lo, mask)

        while (mask := ne_up < n_elec).any():
            e_up[mask] += 2.0 * (e_up[mask] - e_lo[mask])
            ne_up[mask] = elec_count(e_up, mask)

        # Set the fermi energy to the mid point between the two bounds.
        e_fermi[~c_mask] = (0.5 * (e_up + e_lo))[~c_mask]
        ne_fermi = elec_count(e_fermi)

        # __Perform the Bisection Search__
        n_steps = 0
        # Continue squeezing e_up & e_lo together until the delta between the
        # actual & predicted number of e⁻ is less than "tolerance".
        while (mask := ~c_mask).any():
            n_steps += 1

            # Move e_lo to mid-point if `e_lo & e_up haven't crossed` ≡ `mid-point is
            # below the fermi level`; otherwise move e_up up to the mid-point.
            if (m_up := ((ne_up > ne_lo) == (n_elec > ne_fermi)) & mask).any():
                e_lo[m_up], ne_lo[m_up] = e_fermi[m_up], ne_fermi[m_up]
            if (m_down := mask & ~m_up).any():
                e_up[m_down], ne_up[m_down] = e_fermi[m_down], ne_fermi[m_down]

            # Recompute mid-point & its electron count then update the c_mask
            e_fermi[mask] = 0.5 * (e_up + e_lo)[mask]
            ne_fermi[mask] = elec_count(e_fermi, mask)
            c_mask[mask] = abs(ne_fermi - n_elec)[mask] <= tol

            # If maximum allowed number of iterations reached: raise and error.
            if n_steps > max_iter:
                raise ConvergenceError('Fermi search failed to converge',
                                       ~c_mask)

        # Return the fermi energy
        return e_fermi
