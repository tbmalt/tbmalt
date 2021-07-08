# -*- coding: utf-8 -*-
"""Code associated with electronic/finite temperature."""
from typing import Union, Tuple, Literal, Optional
from numbers import Real
from numpy import sqrt, pi
import torch
from torch import Tensor

from tbmalt import ConvergenceError
from tbmalt import Basis

Schemes = Literal['fermi', 'gaussian']


def get_smearing_function(scheme_name: Schemes) -> callable:
    """Yields a smearing function based on the supplied name.

    Takes a name as an argument & yields the corresponding smearing function.

    Arguments:
        scheme_name: Target smearing function's name, implemented schemes are:

            - 'fermi': Fermi-Dirac smearing method
            - 'gaussian': Gaussian smearing method

    Returns:
        smearing_function: A callable smearing function.

    Notes:
        This exists to reduce the number of functions that must be altered
        when implementing a new smearing method.

    Raises:
       KeyError: If ``scheme_name`` is unknown.
    """
    # Dictionary containing all available smearing schemes
    schemes = {'fermi': fermi_smearing,
               'gaussian': gaussian_smearing}

    # Select & return requested scheme. A try/except clause is used here as it
    # is marginally faster than a manual check.
    try:
        return schemes[scheme_name]
    except KeyError as e:
        _ = '\n\t'  # <- Workaround for f-string SyntaxError when using "\n"
        raise KeyError(
            f'Unknown smearing scheme used ({scheme_name}), known schemes:\n\t-'
            f'{f"{_}".join(schemes.keys())}') from e


def smearing_entropy(eigenvalues: Tensor, fermi_energy: Tensor,
                     kT: Union[Real, Tensor], scheme: Schemes,
                     e_mask: Optional[Union[Tensor, Basis]] = None) -> Tensor:
    r"""Calculates the electronic entropy term.

    Calculate a system's electronic entropy term. The entropy term is required
    when calculating various properties, most notably the Mermin free energy;
    which is used in place of the total system energy when finite temperature
    (electronic broadening) is active.

    Entropy is calculated as:

    .. math::

        TS = -k_B\sum_{i}f_i \; ln(f_i) + (1 - f_i)\; ln(1 - f_i))

    when using fermi smearing, and:

    .. math::

        TS = \frac{k_B}{2 \sqrt{\pi}} \sum_{i} exp \left(- \left(
                \frac{\epsilon_i E_f}{kT} \right)^2 \right)

    when using gaussian smearing.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        fermi_energy: The Fermi energy.
        kT: Electronic temperature.
        scheme: Finite temperature broadening scheme used (name of):

                - "gaussian": Gaussian broadening.
                - "fermi": Fermi-Dirac broadening.

        e_mask: Provides the information required to distinguish the "*real*"
            ``eigenvalues`` from "*fake*" ones. This is mandatory for batched
            systems but optional for single systems. This may be a `Tensor`
            that is `True` for real states or a `Basis` object. [DEFAULT=None]

    Returns:
        ts: The entropy term(s).

    Raises:
        RuntimeError: ``e_mask`` not provided when operating on multiple
            systems.
    """
    # If a Basis instance was given as a mask then convert it to a tensor
    if isinstance(e_mask, Basis):
        e_mask = e_mask.on_atoms != -1

    if eigenvalues.ndim == 2 and e_mask is None:  # Mask absent in batch mode
        raise RuntimeError('A mask is required when in batch-mode!')

    if scheme == 'fermi':
        # Get fractional orbital occupancies
        fo = fermi_smearing(eigenvalues, fermi_energy, kT)
        # Log form is used as it avoids having to cull error inducing 1's/0's
        s = -torch.log(fo ** fo * (1 - fo) ** (1 - fo))

    elif scheme == 'gaussian':
        # kT/fermi_energy must be correctly shaped for division to work
        fermi_energy, kT = _smearing_preprocessing(eigenvalues, fermi_energy, kT)
        # Calculate xi values then return the entropy
        xi = (eigenvalues - fermi_energy) / kT
        s = torch.exp(-xi ** 2) * 0.5 / sqrt(pi)

    else:  # If any other or an unknown method was requested
        raise Exception(f'Unknown smearing scheme provided "{scheme}"')

    if e_mask is not None:  # Mask out *fake* padding states as appropriate
        # pylint: disable=E1130
        s[~e_mask] = 0.0

    return s.sum(-1) * kT


def _smearing_preprocessing(
        eigenvalues: Tensor, fermi_energy: Tensor, kT: Union[Real, Tensor]
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
        fermi_energy = fermi_energy.view(-1, 1)

    # Ensure kT is a tensor & is shaped correctly if multiple values passed
    if not isinstance(kT, Tensor):
        kT = torch.tensor(kT, dtype=eigenvalues.dtype,
                          device=eigenvalues.device)

    if kT.ndim == 1 and len(kT) != 1:
        kT = kT.view(-1, 1)

    # kT cannot be allowed to be true zero, otherwise nan's will occur.
    kT = torch.max(torch.tensor(torch.finfo(eigenvalues.dtype).tiny), kT)

    return fermi_energy, kT


def gaussian_smearing(eigenvalues: Tensor, fermi_energy: Tensor,
                      kT: Union[Real, Tensor]) -> Tensor:
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


def fermi_smearing(eigenvalues: Tensor, fermi_energy: Tensor,
                   kT: Union[Real, Tensor]) -> Tensor:
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


@torch.no_grad()
def fermi_search(eigenvalues: Tensor, n_electrons: Union[Real, Tensor],
                 kT: Optional[Union[Real, Tensor]] = None,
                 scheme: Schemes = 'fermi', tolerance: Optional[Real] = None,
                 max_iter: int = 200,
                 e_mask: Optional[Union[Tensor, Basis]] = None) -> Tensor:
    r"""Determines the Fermi-energy of a system or batch thereof.

    Calculates the Fermi-energy with or without finite temperature. Finite
    temperature can be enabled by specifying a ``kT`` value. Note that this
    function will always operate outside of any graph.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        n_electrons: Total number of (valence) electrons.
        kT: Electronic temperature. By default finite temperature is not
            active, i.e. ``kT`` = None. [DEFAULT=None]
        scheme: Finite temperature broadening scheme to be used:

                - "gaussian": Gaussian broadening
                - "fermi": Fermi-Dirac broadening [DEFAULT='fermi']

            Only used when ``kT`` is not None.

        tolerance: Tolerance to which e⁻ count is converged during the search;
            defaults to 1E-10/5/2 for 64/32/16 bit floats respectively. Not
            used when finite temperature is disabled. [DEFAULT=None]
        max_iter: Maximum permitted number of fermi search cycles; ignored
            when finite temperature is disabled. [DEFAULT=200]
        e_mask: Provides info required to distinguish "real" ``eigenvalues``
            from "fake" ones. This is Mandatory when using smearing on batched
            systems. This may be a `Tensor` that is `True` for real states or
            a `Basis` object. [DEFAULT=None]

    Returns:
        fermi_energy: Fermi energy value(s).

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

    Warnings:
        This function operates outside of the pytorch autograd graph and is
        therefore **not** back-propagatable!

    Raises:
        ConvergenceFailure: If the fermi level search fails to converge
            within the permitted number of iterations.
        RuntimeError: A ``e_mask`` is not provided when operating on multiple
            systems.
        ValueError: If the tolerance value is too tight for the specified
            dtype, a negative ``kT`` value is encountered, the number of
            electrons is to zero or exceeds the number of available states.


    References:
        .. [1] Hourahine, B., Aradi, B., Blum, V., Frauenheim, T. et al.,
               (2020). DFTB+, a software package for efficient approximate
               density functional theory based atomistic simulations. The
               Journal of Chemical Physics, 152(12), 124101.
    """
    # __Setup__
    # Arguments, tolerance, eigenvalues, n_electrons aliased here for brevity.
    e_vals, n_elec, tol = eigenvalues, n_electrons, tolerance
    dtype, dev = e_vals.dtype, e_vals.device

    # Convert n_elec & kT into tensors to make them easier to work with.
    if not isinstance(n_elec, Tensor):
        n_elec = torch.tensor(n_elec, dtype=dtype, device=dev)
    if kT is not None and not isinstance(kT, Tensor):
        kT = torch.tensor(kT, dtype=dtype, device=dev)

    # If a Basis instance was given as a mask then convert it to a tensor
    if isinstance(e_mask, Basis):
        e_mask = e_mask.on_atoms != -1

    # __Error Checking__
    eps = torch.finfo(dtype).eps
    if tol is None:  # auto-assign if no tolerance was given
        tol = {torch.float64: 1E-10, torch.float32: 1E-5,
               torch.float16: 1E-2}[dtype]
    elif abs(tol) < eps:  # Ensure tolerance value is viable
        raise ValueError(f'Tolerance {tol:7.1E} too tight for "{dtype}", '
                         f'the minimum permitted value is: {eps:7.1E}.')
    elif tol < 0.0:  # Negative tolerance value
        raise ValueError('Tolerance value cannot be negative')

    if kT is not None and (kT < 0.0).any():  # Negative kT catch
        raise ValueError(f'kT must be positive or None ({kT})')

    if torch.lt(n_elec.abs(), eps).any():  # A system has no electrons
        raise ValueError('Number of elections cannot be zero.')

    # A system has too many electrons
    if torch.any(n_elec / 2 > (len(e_vals) if e_mask is None
                               else e_mask.count_nonzero(-1))):
        raise ValueError('Number of electrons cannot exceed 2 * n states')

    # Mask absent when batch mode and using broadening.
    if e_vals.ndim == 2 and e_mask is None and kT is not None:
        raise RuntimeError('A mask is required when in batch-mode!')

    # __Finite Temperature Disabled__
    # Set the fermi energy to the mid point between the HOMO and LUMO.
    if kT is None:
        n = (n_elec / 2).ceil().long()
        r = None if n.ndim == 0 else range(len(n))  # <- rows
        return (e_vals[r, n] + e_vals[r, n - 1]) / 2

    # __Finite Temperature Enabled__
    # Perform a fermi level search via the bisection method
    else:

        # e_fermi holds results & c_mask tracks which systems have converged.
        e_fermi = torch.zeros_like(n_elec, device=dev, dtype=dtype)
        c_mask = torch.full_like(n_elec, False, dtype=torch.bool, device=dev)

        # Select smearing method & create a helper function to allow for more
        # concise calls to be made when counting e⁻.
        smear_func = get_smearing_function(scheme)

        def elec_count(f, m=...):
            """Makes a call to the smearing function & returns the sum.

            This limits the messy masking operations needed by the smearing
            function to one place. kT's mask are treated differently as 0d &
            1d are both valid shapes of kT in both batch & single system mode.
            """
            res = smear_func(e_vals[m], f[m], kT[m if kT.ndim != 0 else ...])
            if e_mask is not None:
                # Blank out "fake" states caused by padding
                res[~e_mask[m]] = 0.0
            return res.sum(-1) * 2

        # If there's an integer number of e⁻; try setting e_fermi to
        # the middle gap, i.e. fill according to the Aufbau principle.
        if (mask := abs(n_elec - n_elec.round()) <= tol).any():
            n = (n_elec[mask] / 2).ceil().long()
            e_fermi[mask] = (e_vals[mask, n] + e_vals[mask, n - 1]) / 2
            n_elec_fermi = elec_count(e_fermi)
            c_mask[mask] = abs(n_elec_fermi[mask] - n_elec[mask]) < tol

        # If all systems converged then just return the results now
        if c_mask.all():
            return e_fermi.view_as(n_elec)

        # __Setup Bounds for Bisection Search__
        # Identify upper (ub) & lower (lb) search bounds; fermi level should
        # be between the highest & lowest eigenvalues, so start there.
        lb, ub = e_vals.min(-1).values, e_vals.max(-1).values
        ne_lb, ne_ub = elec_count(lb), elec_count(ub)

        # Bounds may fail on large kT or full band structures; if too many e⁻
        # are present at the lb then decrease it & recalculate. If too few e⁻
        # present at the ub, then it's too low so increase it & recalculate
        # the number of elections there.
        while (mask := ne_lb > n_elec).any():
            lb[mask] += 2.0 * (lb[mask] - ub[mask])
            ne_lb[mask] = elec_count(lb, mask)

        while (mask := ne_ub < n_elec).any():
            ub[mask] += 2.0 * (ub[mask] - lb[mask])
            ne_ub[mask] = elec_count(ub, mask)

        # Set the fermi energy to the mid point between the two bounds.
        e_fermi[~c_mask] = (0.5 * (ub + lb))[~c_mask]
        ne_fermi = elec_count(e_fermi)

        # __Perform the Bisection Search__
        n_steps = 0
        # Continue squeezing ub & lb together until the delta between the
        # actual & predicted number of e⁻ is less than "tolerance".
        while (mask := ~c_mask).any():
            n_steps += 1

            # Move lb to mid-point if `lb & ub haven't crossed` XNOR `mid-point
            # is below the fermi level`; otherwise move ub up to the mid-point.
            if (m_up := ~ ((ne_ub > ne_lb) ^ (n_elec > ne_fermi)) & mask).any():
                lb[m_up], ne_lb[m_up] = e_fermi[m_up], ne_fermi[m_up]
            if (m_down := mask & ~m_up).any():
                ub[m_down], ne_ub[m_down] = e_fermi[m_down], ne_fermi[m_down]

            # Recompute mid-point & its electron count then update the c_mask
            e_fermi[mask] = 0.5 * (ub + lb)[mask]
            ne_fermi[mask] = elec_count(e_fermi, mask)
            c_mask[mask] = abs(ne_fermi - n_elec)[mask] <= tol

            # If maximum allowed number of iterations reached: raise and error.
            if n_steps > max_iter:
                raise ConvergenceError('Fermi search failed to converge',
                                       ~c_mask)

        # Return the fermi energy
        return e_fermi
