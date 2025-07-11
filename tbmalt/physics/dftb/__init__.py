# -*- coding: utf-8 -*-
"""Code associated with carrying out DFTB calculations."""
import torch

from typing import Optional, Dict, Any, Literal, Union, Tuple, Callable
import warnings

from tbmalt.ml.module import Calculator
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt import OrbitalInfo
from tbmalt.physics.dftb.feeds import Feed, SkfOccupationFeed
from tbmalt.physics.filling import (
    fermi_search, fermi_smearing, gaussian_smearing, entropy_term,
    aufbau_filling, Scheme)
from tbmalt.common.maths import eighb
from tbmalt.physics.dftb.coulomb import build_coulomb_matrix
from tbmalt.physics.dftb.gamma import build_gamma_matrix
from tbmalt.physics.dftb.properties import dos
from tbmalt.common.batch import prepeat_interleave
from tbmalt.common import float_like
from tbmalt.common.maths.mixers import Simple, Anderson, Mixer
from tbmalt.data.units import energy_units
from tbmalt import ConvergenceError

from torch import Tensor


# Issues:
#   - There is an issue with how thins are currently being dealt with; according
#     to DFTB+ Eband, E0, TS, & fillings should be doubled for spin unpolarised
#     calculations. For k-point dependant calculations there should be a loop
#     over k-points like so `Ef(:) = Ef + EfTmp(:nSpinHams) * kWeights(iK)`.
#     Common Fermi level across two colinear spin channels Ef(2) = Ef(1), and
#     for ! Fixed value of the Fermi level for each spin channel `things are
#     just left as they are`.

# Notes:
#   - Do we really want q_zero, q_final, etc to be the number of electrons or
#     the charge?

# This method really could benefit from a refactoring. It should be more
# clear in its intent and operation. This function will be made private
# until it is cleared up.
def _mulliken(
        rho: Tensor, S: Tensor, orbs: Optional[OrbitalInfo] = None,
        resolution: Optional[Literal['atom', 'shell', 'orbital']] = None
) -> Tensor:
    r"""Mulliken population analysis.

    By default, orbital resolved populations are returned, however, passing the
    associated orbs instance will produce atom or shell resolved populations
    depending on the orbs instance's `shell_resolved` attribute (the behavior
    of which can be overridden via the ``resolution`` argument).

    Arguments:
        rho: density matrix.
        S: overlap matrix.
        orbs: a `OrbitalInfo` instance may be specified to enable atom/shell
            resolved populations. If omitted, populations will be orbital
            resolved. [DEFAULT=None]
        resolution: can be specified to override the degree of resolution
            defined by the ``orbs`` instance, available options are:

                - "atom": atom resolved
                - "shell": shell resolved
                - "orbital": orbital resolved

            If unspecified, this will default to the resolution defined by the
            `orbs.shell_resolved` attribute. This is only valid when ``orbs``
            is also specified. [DEFAULT=None]

    Returns:
        q: mulliken populations.

    Raises:
        TypeError: if ``resolution`` is specified in absence of ``orbs``.

    """

    if resolution is not None and orbs is None:
        raise TypeError(
            '"resolution" overrides default behaviour associated with the '
            '"orbs" object\'s "shell_resolved" attribute. Thus it cannot be '
            'specified in absence of the "orbs" argument.')

    q = (rho * S).sum(-1)  # Calculate the per-orbital Mulliken populations
    if orbs is not None:  # Resolve to per-shell/atom if instructed to
        if resolution is None:
            # TODO: Change orbs to have a res_matrix_shape property.
            size, ind = orbs.res_matrix_shape, orbs.on_res
        elif resolution == 'atom':
            size, ind = orbs.atomic_matrix_shape, orbs.on_atoms
        elif resolution == 'shell':
            size, ind = orbs.shell_matrix_shape, orbs.on_shells
        else:
            raise NotImplementedError("Unknown resolution")

        q = torch.zeros(size[:-1], device=rho.device, dtype=rho.dtype
                        ).scatter_add_(-1, ind.clamp(min=0), q)

    return q


class Dftb1(Calculator):
    """
    Non-self-consistent-charge density-functional tight-binding method
    (non-SCC DFTB).

    Arguments:
        h_feed: this feed provides the Slater-Koster based integrals used to
            construct Hamiltonian matrix.
        s_feed: this feed provides the Slater-Koster based integrals used to
            construct overlap matrix.
        o_feed: this feed provides the angular-momenta resolved occupancies
            for the requested species.
        r_feed: this feed describes the repulsive interaction.[DEFAULT=`None`]
        filling_temp: Electronic temperature used to calculate Fermi-energy.
            [DEFAULT=0.0]
        filling_scheme: The scheme used for finite temperature broadening.
            There are two broadening methods, Fermi-Dirac broadening and
            Gaussian broadening, supported in TBMaLT. [DEFAULT="fermi"]

    Attributes:
        rho: density matrix.
        eig_values: eigen values.
        eig_vectors: eigen vectors.

    Notes:
        Currently energies and occupancies are not scaled correctly. Occupancies
        and band energies need to be scaled based on whether or not they are i)
        spin-polarised or spin-unpolarised, ii) have a fixed fermi level for
        each spin channel or a common one across two colinear spin channels,
        iii) whether k-points are present or not. See the subroutine named
        "getFillingsAndBandEnergies" of the file "dftb/lib_dftbplus/main.F90"
        in DFTB+ for examples.

    Examples:
        >>> import torch
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed
        >>> from tbmalt.physics.dftb import Dftb1
        >>> from tbmalt.tools.downloaders import download_dftb_parameter_set
        >>> from ase.build import molecule
        >>> torch.set_default_dtype(torch.float64)
        # Download the auorg-1-1 parameter set
        >>> url = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
        >>> path = "auorg.h5"
        >>> download_dftb_parameter_set(url, path)

        # Preparation of system to calculate

        # Single system
        >>> geos = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs_s = OrbitalInfo(geos.atomic_numbers, shell_dict={1: [0], 6: [0, 1]})

        # Batch systems
        >>> geob = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
        >>> orbs_b = OrbitalInfo(geob.atomic_numbers, shell_dict={
        ...     1: [0], 6: [0, 1], 8: [0, 1]})

        # Definition of feeds
        >>> h_feed = SkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
        >>> s_feed = SkFeed.from_database(path, [1, 6, 8], 'overlap')
        >>> o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])

        # Run DFTB1 calculation
        >>> dftb = Dftb1(h_feed, s_feed, o_feed, filling_temp=0.0036749324)
        >>> dftb(geos, orbs_s)
        >>> print(dftb.q_final_atomic)
        tensor([4.3591, 0.9102, 0.9102, 0.9102, 0.9102])
        >>> dftb(geob, orbs_b)
        >>> print(dftb.q_final_atomic)
        tensor([[6.7552, 0.6224, 0.6224, 0.0000, 0.0000],
                [4.3591, 0.9102, 0.9102, 0.9102, 0.9102]])

    """
    def __init__(
            self, h_feed: IntegralFeed, s_feed: IntegralFeed, o_feed: Feed,
            r_feed: Optional[Feed] = None, filling_temp: Optional[float] = 0.0,
            filling_scheme: Optional[str] = 'fermi', **kwargs):

        super().__init__(h_feed.dtype, h_feed.device)

        # Calculator Feeds
        self.h_feed = h_feed
        self.s_feed = s_feed
        self.o_feed = o_feed
        self.r_feed = r_feed

        device_list = [d.device for d in [h_feed, s_feed, o_feed, r_feed]
                       if hasattr(d, "device") and d is not None]

        if not len(set(device_list)) == 1:
            raise ValueError('All `Feeds` must be on the same device')

        self._overlap: Optional[Tensor] = None
        self._hamiltonian: Optional[Tensor] = None
        self.rho: Optional[Tensor] = None
        self.eig_values: Optional[Tensor] = None
        self.eig_vectors: Optional[Tensor] = None

        # Calculator Settings
        self.filling_temp = filling_temp
        self.filling_scheme = {
            'fermi': fermi_smearing, 'gaussian': gaussian_smearing,
            None: None
        }[filling_scheme]

        # Optional keyword arguments can be passed through to the `eighb`
        # solver via the `_solver_settings` dictionary argument.
        self._solver_settings = kwargs.get('eigen_solver_settings', {})

    @property
    def overlap(self):
        """Overlap matrix"""

        # Check to see if the overlap matrix has already been constructed. If
        # not then construct it.
        if self._overlap is None:
            self._overlap = self.s_feed.matrix_from_calculator(self)

        # Return the cached overlap matrix.
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._overlap = value

    @property
    def hamiltonian(self):
        """Hamiltonian matrix"""

        # See `Dftb1.overlap` for an explanation of what this code does.
        if self._hamiltonian is None:
            self._hamiltonian = self.h_feed.matrix_from_calculator(self)

        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value):
        self._hamiltonian = value

    @property
    def q_zero(self):
        """Initial orbital populations"""
        return self.o_feed.forward(self.orbs)

    @property
    def q_final(self):
        """Final orbital populations"""
        return _mulliken(self.rho, self.overlap)

    @property
    def q_delta(self):
        """Delta orbital populations"""
        return self.q_final - self.q_zero

    @property
    def q_zero_shells(self):
        """Initial shell-wise populations"""
        return torch.zeros(
            self.orbs.shell_matrix_shape[:-1],
            device=self.device, dtype=self.dtype).scatter_add_(
            -1, self.orbs.on_shells.clamp(min=0), self.q_zero)

    @property
    def q_final_shells(self):
        """Final shell-wise populations"""
        return _mulliken(self.rho, self.overlap, self.orbs, 'shell')

    @property
    def q_delta_shells(self):
        """Delta shell-wise populations"""
        return self.q_final_shells - self.q_zero_shells

    @property
    def q_zero_atomic(self):
        """Initial atomic populations"""
        return torch.zeros(
            self.orbs.atomic_matrix_shape[:-1],
            device=self.device, dtype=self.dtype).scatter_add_(
            -1, self.orbs.on_atoms.clamp(min=0), self.q_zero)

    @property
    def q_final_atomic(self):
        """Final atomic populations"""
        return _mulliken(self.rho, self.overlap, self.orbs, 'atom')

    @property
    def q_delta_atomic(self):
        """Delta atomic populations"""
        return self.q_final_atomic - self.q_zero_atomic

    @property
    def q_zero_res(self):
        """Initial charges, atom or shell resolved according to `OrbitalInfo`"""
        if self.orbs.shell_resolved:
            return self.q_zero_shells
        else:
            return self.q_zero_atomic

    @property
    def dipole(self) -> Tensor:
        """Return dipole moments."""
        return torch.sum(
            self.q_delta_atomic.unsqueeze(-1) * self.geometry.positions, -2
        )

    @property
    def n_electrons(self):
        """Number of electrons"""
        return self.q_zero.sum(-1)

    @property
    def occupancy(self):
        """Occupancies of each state"""

        # Note that this scale factor assumes spin-restricted and will need to
        # be refactored when implementing spin-unrestricted calculations.
        scale_factor = 2.0

        # If finite temperature is active then use the appropriate smearing
        # method.
        if self.filling_temp is not None and self.filling_scheme is not None:
            return self.filling_scheme(
                self.eig_values, self.fermi_energy, self.filling_temp,
                e_mask=self.orbs if self.is_batch else None) * scale_factor
        # Otherwise just fill according to the Aufbau principle
        else:
            return aufbau_filling(
                self.eig_values, self.n_electrons,
                e_mask=self.orbs if self.is_batch else None) * scale_factor

    @property
    def fermi_energy(self):
        """Fermi energy"""
        return fermi_search(
            self.eig_values, self.n_electrons, self.filling_temp,
            self.filling_scheme,
            # Pass the e_mask argument, but only if required.
            e_mask=self.orbs if self.is_batch else None)

    @property
    def band_energy(self):
        """Band structure energy"""
        return torch.einsum('...i,...i->...', self.eig_values, self.occupancy)

    @property
    def band_free_energy(self):
        """Band free energy; i.e. E_band-TS"""
        # Note that this scale factor assumes spin-restricted and will need to
        # be refactored when implementing spin-unrestricted calculations.
        scale_factor = 2.0
        energy = self.band_energy
        if self.filling_scheme is not None and self.filling_temp is not None:
            # The function `entropy_term` yields the "TS" term
            energy -= scale_factor * entropy_term(
                self.filling_scheme, self.eig_values, self.fermi_energy,
                self.filling_temp, e_mask=self.orbs if self.is_batch else None)
        return energy

    @property
    def repulsive_energy(self):
        """Repulsive energy; zero in the absence of a repulsive feed"""
        return 0.0 if self.r_feed is None else self.r_feed(self.geometry)

    @property
    def total_energy(self):
        """Total system energy"""
        return self.band_energy + self.repulsive_energy

    @property
    def mermin_energy(self):
        """Mermin free energy; i.e. E_total-TS"""
        return self.band_free_energy + self.repulsive_energy

    @property
    def homo_lumo(self):
        """Highest occupied and lowest unoccupied energy level in unit hartree"""
        # Number of occupied states
        nocc = (~(self.occupancy - 0 < 1E-10)).long().sum(-1)

        # Check if HOMO&LUMO well defined
        if self.occupancy.size(dim=-1) <= nocc.max():
            raise ValueError('Warning: HOMO&LUMO are not defined properly!')
        else:
            # Mask of HOMO and LUMO
            mask = torch.zeros_like(
                self.occupancy, device=self.device, dtype=self.dtype).scatter_(
                -1, nocc.unsqueeze(-1), 1).scatter_(
                    -1, nocc.unsqueeze(-1) - 1, 1).bool()
            homo_lumo = self.eig_values[mask] if self.occupancy.ndim == 1 else\
                self.eig_values[mask].view(self.occupancy.size(0), -1)

        return homo_lumo

    @property
    def dos_energy(self, ext=energy_units['ev'], grid=1000):
        """Energy distribution of (p)DOS in unit hartree"""
        e_min = torch.min(self.eig_values.detach(), dim=-1).values - ext
        e_max = torch.max(self.eig_values.detach(), dim=-1).values + ext
        dos_energy = torch.linspace(
            e_min, e_max, grid, device=self.device, dtype=self.dtype) if\
            self.occupancy.ndim == 1 else torch.stack([torch.linspace(
                imin, imax, grid, device=self.device, dtype=self.dtype
                ) for imin, imax in zip(e_min, e_max)])

        return dos_energy

    @property
    def dos(self):
        """Electronic density of states"""
        # Mask to remove padding values.
        mask = torch.where(self.eig_values == 0, False, True)
        sigma = 0.1 * energy_units['ev']
        return dos(self.eig_values, self.dos_energy, sigma=sigma, mask=mask)

    @property
    def forces(self) -> Tensor:
        """Atomic forces"""
        if not self.geometry.positions.requires_grad:
            raise RuntimeError(
                "Forces are computed via the PyTorch auto-grad engine, thus "
                "the positions tensor must be differentiable, i.e. "
                "\"Geometry.positions.requires_grad = True\".")

        gradient, *_ = torch.autograd.grad(
            self.total_energy.sum(), self.geometry.positions, create_graph=True)

        return -gradient

    def reset(self):
        """Reset all attributes and cached properties."""
        self._overlap = None
        self._hamiltonian = None
        self.rho = None
        self.eig_values = None
        self.eig_vectors = None

    def forward(self, cache: Optional[Dict[str, Any]] = None):
        """Execute the non-SCC DFTB calculation.

        This method triggers the execution of the non-self-consistent-charge
        density functional tight binding theory calculation. Once complete
        this will return the total system energy.

        Arguments:
            cache: Currently, the `Dftb1` calculator does not make use of the
                `cache` argument.

        Returns:
            total_energy: total energy of the target system(s). If repulsive
                interactions are not considered, i.e. the repulsion feed is
                omitted, then this will be the band structure energy. If finite
                temperature is active then this will technically be the Mermin
                energy.

        """
        # Construct the Hamiltonian & overlap matrices then perform the eigen
        # decomposition to get the eigen values and vectors.
        self.eig_values, self.eig_vectors = eighb(
            self.hamiltonian, self.overlap, **self._solver_settings)

        # Then construct the density matrix.
        s_occs = torch.einsum(  # Scaled occupancy values
            '...i,...ji->...ji', torch.sqrt(self.occupancy), self.eig_vectors)
        self.rho = s_occs @ s_occs.transpose(-1, -2).conj()

        # Calculate and return the total system energy, taking into account
        # the entropy term as and when necessary.
        return self.mermin_energy


class Dftb2(Calculator):
    """Self-consistent-charge density-functional tight-binding method
    (SCC-DFTB).

    Arguments:
        h_feed: this feed provides the Slater-Koster based integrals used to
            construct Hamiltonian matrix.
        s_feed: this feed provides the Slater-Koster based integrals used to
            construct overlap matrix.
        o_feed: this feed provides the angular-momenta resolved occupancies
            for the requested species.
        u_feed: this feed provides the Hubbard-U values as needed to construct
            the gamma matrix. This must be a `Feed` type object which when
            provided with a `OrbitalInfo` object returns the Hubbard-U values
            for the target system.
        r_feed: this feed describes the repulsive interaction.[DEFAULT=`None`]
        filling_temp: Electronic temperature used to calculate Fermi-energy.
            [DEFAULT=0.0]
        filling_scheme: The scheme used for finite temperature broadening.
            There are two broadening methods, Fermi-Dirac broadening and
            Gaussian broadening, supported in TBMaLT. [DEFAULT="fermi"]
        grad_mode: controls gradient mode used when running the SCC
            cycle. Available options are [DEFAULT="last_step"]:

                - "direct": run the SCC cycle without any special treatment.
                - "last_step": run the SCC cycle outside the purview of the
                    PyTorch graph to obtain the converged charges. Then run a
                    single SCC step within the graph using the converged
                    charges as the initial "guess". Gradients obtained via the
                    "last_step" approach are inexact. While such gradients are
                    commonly a good enough approximation for many cases they
                    are not as accurate as either the "direct" or "implicit"
                    gradient modes.
                - "implicit": uses implicit function theorem to accurately
                    and memory efficiently compute the derivative. This
                    requires the use of an iterative solver. By default,
                    the solver will employ the same mixer provided for
                    the SCC cycle, `DFTB2.mixer`. It is critical to note
                    that the accuracy of the gradients produced by the
                    implicit method is directly tied to the tolerance
                    and stability of the mixer. As such a different
                    dedicated mixer can be supplied via the
                    ``implicit_mixer`` argument.

        implicit_mixer: Dedicated mixer to be used by the implicit solver if
            using the "implicit" ``grad_mode`` option. This is only used
            when the "implicit" option is set via ``grad_mode``. If not set
            then the standard SCC mixer will be used.
        max_scc_iter: maximum permitted number of SCC iterations. If one or
            more system fail to converge within ``max_scc_iter`` cycles then a
            convergence error will be raised; unless the ``suppress_scc_error``
            flag has been set. [DEFAULT=200]
        mixer: specifies the charge mixing scheme to be used. Providing the
            strings "simple" and "anderson" will result in their respectively
            named mixing schemes being used. Initialised `Mixer` class objects
            may also be provided directly.

    Keyword Arguments:
        suppress_scc_error: if True, convergence errors will be suppressed and
            the calculation will proceed with as normal. This is of use during
            fitting when operating on large batches. This way if most systems
            converge but one does not then it can just be ignored rather than
            ending the program. Unconverged systems can be identified via the
            ``converged`` attribute. [DEFAULT=False]
        gamma_scheme: scheme used to construct the gamma matrix. This may be
            either "exponential" or "gaussian". [DEFAULT="exponential"]
        coulomb_scheme: scheme used to construct the coulomb matrix. This may
            be either "search" or "experience". [DEFAULT="search"]

    Attributes:
        overlap: overlap matrix as constructed by the supplied `s_feed`.
        core_hamiltonian: first order core Hamiltonian matrix as built by the
            `h_feed` entity.
        gamma: the gamma matrix, this is constructed via the specified scheme
            and uses the Hubbard-U values produced by the `u_feed`.
        invr: the 1/R matrix.
        hamiltonian: second order Hamiltonian matrix as produced via the SCC
            cycle.
        scc_energy: energy contribution from charge fluctuation via the SCC
            cycle.
        converged: a tensor of booleans indicating which systems have and
            have not converged (True if converged). This can be used during
            training, alongside `suppress_scc_error`, to allow unconverged
            systems to be omitted from the final loss calculation; as so to
            prevent introducing unnecessary instabilities.
        mixer: a `Mixer` type class instance used during the SCC cycle to
            perform charge mixing.
        rho: density matrix.
        eig_values: eigen values.
        eig_vectors: eigen vectors.

    Examples:
        >>> import torch
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.feeds import HubbardFeed, SkFeed, SkfOccupationFeed
        >>> from tbmalt.physics.dftb import Dftb2
        >>> from tbmalt.tools.downloaders import download_dftb_parameter_set
        >>> from ase.build import molecule
        >>> torch.set_default_dtype(torch.float64)
        # Download the auorg-1-1 parameter set
        >>> url = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
        >>> path = "auorg.h5"
        >>> download_dftb_parameter_set(url, path)

        # Preparation of system to calculate

        # Single system
        >>> geos = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs_s = OrbitalInfo(geos.atomic_numbers, shell_dict={1: [0], 6: [0, 1]})

        # Batch systems
        >>> geob = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
        >>> orbs_b = OrbitalInfo(geob.atomic_numbers, shell_dict={
        ...     1: [0], 6: [0, 1], 8: [0, 1]})

        # Single system with pbc
        >>> geop = Geometry(
        ...     torch.tensor([6, 1, 1, 1, 1]),
        ...     torch.tensor([[3.0, 3.0, 3.0],
        ...                   [3.6, 3.6, 3.6],
        ...                   [2.4, 3.6, 3.6],
        ...                   [3.6, 2.4, 3.6],
        ...                   [3.6, 3.6, 2.4]]),
        ...     torch.tensor([[4.0, 4.0, 0.0],
        ...                   [5.0, 0.0, 5.0],
        ...                   [0.0, 6.0, 6.0]]),
        ...     units='a', cutoff=torch.tensor([9.98]))
        >>> orbs_p = OrbitalInfo(geop.atomic_numbers, shell_dict={1: [0], 6: [0, 1]})

        # Definition of feeds
        >>> h_feed = SkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
        >>> s_feed = SkFeed.from_database(path, [1, 6, 8], 'overlap')
        >>> o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])
        >>> u_feed = HubbardFeed.from_database(path, [1, 6, 8])

        # Run DFTB2 calculation
        >>> mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2,
        ...               'generations': 3, 'tolerance': 1e-10}
        >>> dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed,
        ...               filling_temp=0.0036749324, mix_params=mix_params)
        >>> dftb2(geos, orbs_s)
        >>> print(dftb2.q_final_atomic)
        tensor([4.3054, 0.9237, 0.9237, 0.9237, 0.9237])
        >>> dftb2(geob, orbs_b)
        >>> print(dftb2.q_final_atomic)
        tensor([[6.5856, 0.7072, 0.7072, 0.0000, 0.0000],
                [4.3054, 0.9237, 0.9237, 0.9237, 0.9237]])
        >>> dftb2(geop, orbs_p)
        >>> print(dftb2.q_final_atomic)
        tensor([4.6124, 0.8332, 0.8527, 0.8518, 0.8499])

    """

    def __init__(
            self, h_feed: IntegralFeed, s_feed: IntegralFeed, o_feed: Feed,
            u_feed: Feed, r_feed: Optional[Feed] = None,
            filling_temp: Optional[float] = 0.0,
            filling_scheme: Optional[str] = 'fermi',
            grad_mode: Literal["direct", "last_step", "implicit"] | str = "last_step",
            implicit_mixer: Optional[Mixer] = None,
            max_scc_iter: int = 200,
            mixer: Union[Mixer, Literal['anderson', 'simple']] = 'anderson',
            **kwargs):

        super().__init__(h_feed.dtype, h_feed.device)

        # Calculator Feeds
        self.h_feed = h_feed
        self.s_feed = s_feed
        self.o_feed = o_feed
        self.u_feed = u_feed
        self.r_feed = r_feed

        device_list = [
            d.device for d in [h_feed, s_feed, o_feed, u_feed, r_feed]
            if hasattr(d, "device") and d is not None]

        if not len(set(device_list)) == 1:
            raise ValueError('All `Feeds` must be on the same device')

        self._overlap: Optional[Tensor] = None
        self._hamiltonian: Optional[Tensor] = None
        self._core_hamiltonian: Optional[Tensor] = None
        self.rho: Optional[Tensor] = None
        self.eig_values: Optional[Tensor] = None
        self.eig_vectors: Optional[Tensor] = None
        self._gamma: Optional[Tensor] = None
        self._invr: Optional[Tensor] = None
        self.converged: Optional[Tensor] = None

        # Calculator Settings
        self.filling_temp = filling_temp
        self.filling_scheme = {
            'fermi': fermi_smearing, 'gaussian': gaussian_smearing,
            None: None
        }[filling_scheme]

        self.grad_mode = grad_mode
        self.implicit_mixer = implicit_mixer
        self.max_scc_iter = max_scc_iter
        self.suppress_scc_error = kwargs.get('suppress_scc_error', False)
        self.gamma_scheme = kwargs.get('gamma_scheme', 'exponential')
        self.coulomb_scheme = kwargs.get('coulomb_scheme', 'search')

        # If no pre-initialised was provided then construct one.
        if isinstance(mixer, str):
            mixer = {
                'anderson': Anderson, 'simple': Simple}[
                mixer.lower()](False, **kwargs.get('mix_params', {}))

        self.mixer = mixer

        # Optional keyword arguments can be passed through to the `eighb`
        # solver via the `_solver_settings` dictionary argument.
        self._solver_settings = kwargs.get('eigen_solver_settings', {})

        if grad_mode not in ["direct", "last_step", "implicit"]:
            raise ValueError(
                f"\"{grad_mode}\" does not correspond to a known gradient mode."
                " Valid options are \"direct\", \"last_step\", \"implicit\"")

    @property
    def overlap(self):
        """Overlap matrix"""

        # Check to see if the overlap matrix has already been constructed. If
        # not then construct it.
        if self._overlap is None:
            self._overlap = self.s_feed.matrix_from_calculator(self)

        # Return the cached overlap matrix.
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._overlap = value

    @property
    def hamiltonian(self):
        """Second order Hamiltonian matrix as produced via the SCC cycle"""
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value):
        self._hamiltonian = value

    @property
    def core_hamiltonian(self):
        """First order core Hamiltonian matrix as built by the `h_feed`
        entity"""
        if self._core_hamiltonian is None:
            self._core_hamiltonian = self.h_feed.matrix_from_calculator(self)

        return self._core_hamiltonian

    @core_hamiltonian.setter
    def core_hamiltonian(self, value):
        self._core_hamiltonian = value

    @property
    def gamma(self):
        """Gamma matrix as constructed using the `u_feed`"""
        if self._gamma is None:
            self._gamma = build_gamma_matrix(
                self.geometry, self.orbs, self.invr,
                self.u_feed.forward(self.orbs), self.gamma_scheme)
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def q_zero(self):
        """Initial orbital populations"""
        return self.o_feed.forward(self.orbs)

    @property
    def q_final(self):
        """Final orbital populations"""
        return _mulliken(self.rho, self.overlap)

    @property
    def q_delta(self):
        """Delta orbital populations"""
        return self.q_final - self.q_zero

    @property
    def q_zero_shells(self):
        """Initial shell-wise populations"""
        return torch.zeros(
            self.orbs.shell_matrix_shape[:-1],
            device=self.device, dtype=self.dtype).scatter_add_(
            -1, self.orbs.on_shells.clamp(min=0), self.q_zero)

    @property
    def q_final_shells(self):
        """Final shell-wise populations"""
        return _mulliken(self.rho, self.overlap, self.orbs, 'shell')

    @property
    def q_delta_shells(self):
        """Delta shell-wise populations"""
        return self.q_final_shells - self.q_zero_shells

    @property
    def q_zero_atomic(self):
        """Initial atomic populations"""
        return torch.zeros(
            self.orbs.atomic_matrix_shape[:-1],
            device=self.device, dtype=self.dtype).scatter_add_(
            -1, self.orbs.on_atoms.clamp(min=0), self.q_zero)

    @property
    def q_final_atomic(self):
        """Final atomic populations"""
        return _mulliken(self.rho, self.overlap, self.orbs, 'atom')

    @property
    def q_delta_atomic(self):
        """Delta atomic populations"""
        return self.q_final_atomic - self.q_zero_atomic

    @property
    def q_zero_res(self):
        """Initial charges, atom or shell resolved according to `OrbitalInfo`"""
        if self.orbs.shell_resolved:
            return self.q_zero_shells
        else:
            return self.q_zero_atomic

    @property
    def dipole(self) -> Tensor:
        """Return dipole moments."""
        return torch.sum(
            self.q_delta_atomic.unsqueeze(-1) * self.geometry.positions, -2
        )

    @property
    def n_electrons(self):
        """Number of electrons"""
        return self.q_zero.sum(-1)

    @property
    def occupancy(self):
        """Occupancies of each state"""

        # Note that this scale factor assumes spin-restricted and will need to
        # be refactored when implementing spin-unrestricted calculations.
        scale_factor = 2.0

        # If finite temperature is active then use the appropriate smearing
        # method.
        if self.filling_temp is not None and self.filling_scheme is not None:
            return self.filling_scheme(
                self.eig_values, self.fermi_energy, self.filling_temp,
                e_mask=self.orbs if self.is_batch else None) * scale_factor
        # Otherwise just fill according to the Aufbau principle
        else:
            return aufbau_filling(
                self.eig_values, self.n_electrons,
                e_mask=self.orbs if self.is_batch else None) * scale_factor

    @property
    def fermi_energy(self):
        """Fermi energy"""
        return fermi_search(
            self.eig_values, self.n_electrons, self.filling_temp,
            self.filling_scheme,
            # Pass the e_mask argument, but only if required.
            e_mask=self.orbs if self.is_batch else None)

    @property
    def band_energy(self):
        """Band structure energy, including SCC contributions"""
        return torch.einsum('...i,...i->...', self.eig_values, self.occupancy)

    @property
    def core_band_energy(self):
        """Core band structure energy, excluding SCC contributions"""
        return ((self.rho * self.core_hamiltonian).sum(-1).sum(-1))

    @property
    def band_free_energy(self):
        """Band free energy including SCC contributions; i.e. E_band-TS"""
        return self.band_energy * self._get_entropy_term()

    @property
    def core_band_free_energy(self):
        """Core band free energy, excluding SCC contributions"""
        return self.core_band_energy * self._get_entropy_term()

    def _get_entropy_term(self):
        # Note that this scale factor assumes spin-restricted and will need to
        # be refactored when implementing spin-unrestricted calculations.
        scale_factor = 2.0
        if self.filling_scheme is not None and self.filling_temp is not None:
            # The function `entropy_term` yields the "TS" term
            return scale_factor * entropy_term(
                self.filling_scheme, self.eig_values, self.fermi_energy,
                self.filling_temp, e_mask=self.orbs if self.is_batch else None)
        else:
            return 1.0

    @property
    def scc_energy(self):
        """Energy contribution from charge fluctuation"""
        q_delta = _mulliken(self.rho, self.overlap, self.orbs) - self.q_zero_res
        shifts = torch.einsum('...i,...ij->...j', q_delta, self.gamma)
        return .5 * (shifts * q_delta).sum(-1)

    @property
    def repulsive_energy(self):
        """Repulsive energy; zero in the absence of a repulsive feed"""
        return 0.0 if self.r_feed is None else self.r_feed(self.geometry)

    @property
    def total_energy(self):
        """Total system energy"""
        return self.core_band_energy + self.scc_energy + self.repulsive_energy

    @property
    def mermin_energy(self):
        """Mermin free energy; i.e. E_total-TS"""
        return self.core_band_free_energy + self.scc_energy + self.repulsive_energy

    @property
    def homo_lumo(self):
        """Highest occupied and lowest unoccupied energy level in unit hartree"""
        # Number of occupied states
        nocc = (~(self.occupancy - 0 < 1E-10)).long().sum(-1)

        # Check if HOMO&LUMO well defined
        if self.occupancy.size(dim=-1) <= nocc.max():
            raise ValueError('Warning: HOMO&LUMO are not defined properly!')
        else:
            # Mask of HOMO and LUMO
            mask = torch.zeros_like(
                self.occupancy, device=self.device, dtype=self.dtype).scatter_(
                -1, nocc.unsqueeze(-1), 1).scatter_(
                    -1, nocc.unsqueeze(-1) - 1, 1).bool()
            homo_lumo = self.eig_values[mask] if self.occupancy.ndim == 1 else\
                self.eig_values[mask].view(self.occupancy.size(0), -1)

        return homo_lumo

    @property
    def dos_energy(self, ext=energy_units['ev'], grid=1000):
        """Energy distribution of (p)DOS in unit hartree"""
        e_min = torch.min(self.eig_values.detach(), dim=-1).values - ext
        e_max = torch.max(self.eig_values.detach(), dim=-1).values + ext
        dos_energy = torch.linspace(
            e_min, e_max, grid, device=self.device, dtype=self.dtype) if\
            self.occupancy.ndim == 1 else torch.stack([torch.linspace(
                imin, imax, grid, device=self.device, dtype=self.dtype
                ) for imin, imax in zip(e_min, e_max)])

        return dos_energy

    @property
    def dos(self):
        """Electronic density of states"""
        # Mask to remove padding values.
        mask = torch.where(self.eig_values == 0, False, True)
        sigma = 0.1 * energy_units['ev']
        return dos(self.eig_values, self.dos_energy, sigma=sigma, mask=mask)

    @property
    def invr(self):
        """1/R matrix"""
        if self._invr is None:
            if self.geometry.is_periodic:
                self._invr = build_coulomb_matrix(self.geometry,
                                                  method=self.coulomb_scheme)
            else:
                r = self.geometry.distances
                r[r != 0.0] = 1.0 / r[r != 0.0]
                self._invr = r

        return self._invr

    @invr.setter
    def invr(self, value):
        self._invr = value

    @property
    def forces(self) -> Tensor:
        """Atomic forces"""
        if not self.geometry.positions.requires_grad:
            raise RuntimeError(
                "Forces are computed via the PyTorch auto-grad engine, thus "
                "the positions tensor must be differentiable, i.e. "
                "\"Geometry.positions.requires_grad = True\".")

        gradient, *_ = torch.autograd.grad(
            self.total_energy.sum(), self.geometry.positions, create_graph=True)

        return -gradient

    def forward(
            self, cache: Optional[Dict[str, Any]] = None) -> Tensor:
        """Execute the SCC-DFTB calculation.

        Invoking this will trigger the execution of the self-consistent-charge
        density functional tight binding theory calculation.

        Arguments:
            cache: This stores any information which can be used to bootstrap
                the calculation. Currently supported values are:

                    - "q_initial": initial starting guess for the SCC cycle.

        Returns:
            total_energy: total energy for the target systems this will include
                both the repulsive and entropy terms, where appropriate.
        """

        # Step 1: Initial Setup
        # Gather keyword arguments relevant to the SCC cycle
        kwargs_in = {
            "filling_temp": self.filling_temp,
            "filling_scheme": self.filling_scheme,
            "max_scc_iter": self.max_scc_iter,
            "mixer": self.mixer,
            "suppress_scc_error": self.suppress_scc_error,
            "eigen_solver_settings": self._solver_settings}

        # If insert any supplied cache data into the keyword argument
        # dictionary. Currently, the SCC cycle only makes use of the
        # "q_initial" value. This can be used to specify the initial
        # charge guess used at the start of the SCC cycle.
        if cache is not None:
            kwargs_in.update(cache)

        # Core Hamiltonian, overlap, and gamma matrices
        hsg_args = (self.core_hamiltonian, self.overlap, self.gamma)

        # Step 2: SCC Cycle
        # The following block performs the actual SCC cycle. However, different
        # approaches are needed depending on what gradient mode is to be used.
        # The "direct" approach carries out the SCC cycle without any special
        # handling. Therefor the gradient will pass through the full SCC cycle.
        if self.grad_mode == "direct":
            (q_out, self._hamiltonian, self.eig_values, self.eig_vectors,
             self.rho) = Dftb2.scc_cycle(
                self.q_zero_res, self.orbs, *hsg_args, **kwargs_in)

        elif self.grad_mode == "last_step" or self.grad_mode == "implicit":

            # In the "last step" approach the SCC cycle is performed outside the
            # purview of the graph and is used only to obtain the converged
            # charges. Following this, an additional "SCC step" is performed within
            # the PyTorch graph. The emulates what would happen if one were to
            # perform the SCC cycle using a highly accurate initial guess for the
            # charges. The "implicit" approach adds a correction factor to the
            # gradient.

            # The arguments to be supplied to the `scc_step` function are
            # first aggregated here into `step_args`. This is not done only for
            # the sake of brevity, but to ensure the first call made to these
            # properties is not done so from within a `torch.no_grad` context.
            # If this is not done then the cached properties will not be
            # compatible with the auto-grad graph.
            step_args = (self.q_zero_res, *hsg_args, self.orbs,
                         self.n_electrons)

            with torch.no_grad():
                q_converged, *_ = Dftb2.scc_cycle(
                    self.q_zero_res, self.orbs, *hsg_args, **kwargs_in)

            if self.grad_mode == "implicit":
                # The implicit method requires the parameter-to-charge gradient
                # path. Thus, an initial single SCC step must be performed to
                # "re-attach" the charges. Note that this is in addition to the
                # final re-attachment call made later on which connects the
                # other side of the path, i.e. charge-to-properties.
                q_converged, *_ = Dftb2.scc_step(q_converged, *step_args, **kwargs_in)

                # Construct a second isolated graph which can be used to
                # compute dF/dq.
                q0 = q_converged.clone().detach().requires_grad_()
                f0, *_ = Dftb2.scc_step(q0, *step_args, **kwargs_in)

                # Create and register the gradient callback hook. This will
                # compute and apply the gradient correction during the
                # backwards pass. This will use the same mixer as the SCC
                # cycle unless the user provides a dedicated mixer.
                implicit_mixer = (self.mixer if self.implicit_mixer is None
                                  else self.implicit_mixer)
                q_converged.register_hook(self._implicit_solver_hook(
                    f0, q0, implicit_mixer, self.max_scc_iter,
                    self.suppress_scc_error))

            # Perform a single SCC step to reconnect q_converged to the graph
            # so that the auto-grad can pass through the SCC operation.
            (q_out, self.hamiltonian, self.eig_values, self.eig_vectors,
             self.rho) = Dftb2.scc_step(q_converged, *step_args, **kwargs_in)

        else:
            raise ValueError(
                f"\"{self.grad_mode}\" does not correspond to a known gradient mode."
                " Valid options are \"direct\", \"last_step\", \"implicit\"")

        # Calculate and return the total system energy, taking into account
        # the entropy term as and when necessary.
        return self.mermin_energy


    @staticmethod
    def _implicit_solver_hook(
            outputs: Tensor, inputs: Tensor, mixer: Mixer, max_iter: int,
            suppress_scc_error: bool) -> Callable[[Tensor], Tensor]:
        r"""Autograd hook that applies implicit-function gradient correction.

        The returned callable is meant to be registered on the converged
        charge tensor via ``Tensor.register_hook``. During the backward
        pass the hook

            1. uses `torch.autograd.grad(outputs, inputs, v)` to obtain the
               vector–Jacobian product :math:`J^{T}\cdot v`,
            2. iteratively solves the linear system
               :math:`(I − J^{T}) g = \text{incoming_grad}` with the supplied
               ``mixer``,
            3. returns the converged `g`, thereby replacing the naive
               gradient with the fully corrected one.

        Arguments:
            outputs: Tensor representing a single SCC step evaluated at the
                converged charges; provides the *outputs* side of the vector-
                Jacobian product (VJP) `torch.autograd.grad` call.
            inputs: Leaf tensor holding the converged charges; supplies the
                *inputs* side of the VJP.
            mixer: `Mixer` instance used as the fixed-point accelerator for
                the implicit solver. It will be reset each time the hook is
                invoked.
            max_iter: Maximum number of iterations allowed for the implicit
                solver before convergence is deemed to have failed.
            suppress_scc_error: If `True`, the hook will silently return the
                last iterate when the iteration budget is exhausted; otherwise
                a `ConvergenceError` is raised.

        Returns:
            autograd_hook: A callable `hook(grad) -> Tensor` suitable for
                registration with `Tensor.register_hook`.

        Raises:
            ConvergenceError: If the implicit solver fails to converge within
                ``max_iter`` steps and ``suppress_scc_error`` is `False`.
        """

        is_batch = outputs.ndim <= 2

        def autograd_hook(grad: Tensor) -> Tensor:
            # Reset the mixer & ensure batch-mode is set appropriately
            mixer.reset()
            mixer.is_batch = is_batch

            # The variable `grad_current` is used to stor the starting point
            # for the iterative solver and the mixed result at the end of each
            # step of an unconverged system.
            grad_current = grad

            # Loop the self-consistent gradient solver until convergence is
            # achieved or the iteration limit is reached.
            for step in range(1, max_iter + 1):

                # Compute the new updated gradient
                grad_new = torch.autograd.grad(outputs, inputs, grad_current,
                                               retain_graph=True)[0] + grad

                # Check which systems have converged
                converged = grad_new.sub(grad_current).abs().le(
                    mixer.tolerance).all(dim=grad_new.dim_order()[1:])

                # If the gradients for all systems have been converged then
                # return the new corrected gradients.
                if converged.all():
                    # Unlike the SCC cycle function, this does not sequentially
                    # cull individual systems within as batch as they converge.
                    # The cost of no doing this should be competitively low
                    # with minimal adverse effects.
                    return grad_new

                # If convergence has not yet been achieved, then mix & continue
                grad_current = mixer(grad_new, grad_current)

            # If the iteration limit has been reached, an exception should be
            # raised, unless otherwise instructed.
            else:
                if not suppress_scc_error:
                    raise ConvergenceError(
                        "Implicit solver did not converge; "
                        "iteration limit reached.")

        return autograd_hook

    def reset(self):
        """Reset all attributes and cached properties."""
        self._overlap = None
        self._core_hamiltonian = None
        self._hamiltonian = None
        self._gamma = None
        self._invr = None
        self.converged = None

        self.rho = None
        self.eig_values = None
        self.eig_vectors = None

    @staticmethod
    def scc_step(
            q_in: Tensor, q_zero: Tensor, core_hamiltonian: Tensor,
            overlap: Tensor, gamma: Tensor, orbs: OrbitalInfo,
            n_electrons: float_like, filling_temp: float_like = 0.0,
            filling_scheme: Optional[Scheme] = fermi_smearing, **kwargs
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform a single self-consistent charge cycle step.

        Arguments:
            q_in: the input charges. This is normally the charge values from the
                previous iteration, or from an initial guess if this is the
                first step. Care must be taken to use either shell or atom
                resolved charges where appropriate.
            q_zero: the neutral reference charges. Again care must be taken to
                ensure that these are provided with the correct resolution,
                i.e. atom vs shell resolved.
            core_hamiltonian: the core, non-scc, hamiltonian matrix.
            overlap: overlap matrix.
            gamma: gamma matrix.
            orbs: `OrbitalInfo` object for the associated systems.
            n_electrons: total number of electrons.
            filling_temp: electronic temperature used to calculate Fermi-energy.
                [DEFAULT=0.0]
            filling_scheme: scheme used for finite temperature broadening.
                There are two broadening methods, Fermi-Dirac broadening and
                Gaussian broadening, supported in TBMaLT.
                [DEFAULT=`fermi_smearing`]

        Keyword Arguments:
            eigen_solver_settings: a dictionary storing advanced settings to
                be provided to the eigen solver.

        Returns:
            q_out: the resulting output charges.
            hamiltonian: the second order hamiltonian matrix. When ``q_in`` is
                equal to ``q_out`` this will be the "true" self-consistent
                charge hamiltonian matrix.
            eig_values: corresponding eigenvalues.
            eig_vectors: associated eigenvectors.
            rho: density matrix.
        """
        # Construct the shift matrix
        shifts = torch.einsum(
            '...i,...ij->...j', q_in - q_zero, gamma)

        shifts = prepeat_interleave(shifts, orbs.orbs_per_res)
        shifts = (shifts[..., None] + shifts[..., None, :])

        # Compute the second order Hamiltonian matrix
        hamiltonian = core_hamiltonian + .5 * overlap * shifts

        # Obtain the eigen-values/vectors via an eigen decomposition
        eigen_solver_settings = kwargs.get('eigen_solver_settings', {})
        eig_values, eig_vectors = eighb(hamiltonian, overlap, **eigen_solver_settings)

        # Calculate Occupancies
        # ---------------------
        # Note that this scale factor assumes spin-restricted and will need to
        # be refactored when implementing spin-unrestricted calculations.
        scale_factor = 2.0

        e_mask = orbs if orbs.atomic_numbers.dim() == 2 else None

        # Compute the new occupancy values, performing smearing as necessary.
        if filling_scheme is not None and filling_temp is not None:
            fermi_energy = fermi_search(eig_values, n_electrons, filling_temp,
                                        filling_scheme, e_mask=e_mask)

            occupancy = filling_scheme(eig_values, fermi_energy, filling_temp,
                                       e_mask=e_mask) * scale_factor

        else:
            occupancy = aufbau_filling(eig_values, n_electrons,
                                       e_mask=e_mask) * scale_factor

        # The density matrix is constructed via a two-step approach, similar to
        # that used in DFTB+. This is done to avoid gradient instabilities that
        # arise as a result of taking the square root of zeros within the eigen
        # vector scaling step. While an occupancy value of zero is valid, the
        # derivative of square root is not. Thus, padding must be applied to these
        # values during evaluation, and compensated for later on.

        # Somewhat arbitrary offset needed to ensure occupancies are non-zero
        smallest_occupancy = torch.min(occupancy)
        offset = smallest_occupancy - 0.1

        # Scaled occupancy values (including offset)
        s_occs = torch.einsum(
            '...i,...ji->...ji', torch.sqrt(occupancy - offset), eig_vectors)

        # First the density matrix without occupancy scaling is computed. This is
        # then added to the scaled density matrix, which includes the offset.
        rho = eig_vectors @ eig_vectors.transpose(-1, -2).conj()
        rho = (s_occs @ s_occs.transpose(-1, -2).conj()) + offset * rho

        # Compute the new charges
        q_out = _mulliken(rho, overlap, orbs)

        return q_out, hamiltonian, eig_values, eig_vectors, rho

    @staticmethod
    def scc_cycle(
            q_zero: Tensor, orbs: OrbitalInfo, core_hamiltonian: Tensor,
            overlap: Tensor, gamma: Tensor, filling_temp: float_like = 0.0,
            filling_scheme: Scheme = fermi_smearing, max_scc_iter: int = 200,
            mixer: Mixer = Anderson(True),
            **kwargs):
        """Perform the self-consistent charge cycle.

        This method runs the full self-consistent charge cycle to compute the
        converged charge(s). This will also return the second order hamiltonian
        matrix along with other associated data.

        Arguments:
            q_zero: the neutral reference charges. Care must be taken to use
                either shell or atom resolved charges where appropriate.
            orbs: `OrbitalInfo` object for the associated systems.
            core_hamiltonian: the core, non-scc, hamiltonian matrix.
            overlap: overlap matrix.
            gamma: gamma matrix.
            filling_temp: electronic temperature used to calculate Fermi-energy.
                [DEFAULT=0.0]
            filling_scheme: scheme used for finite temperature broadening.
                There are two broadening methods, Fermi-Dirac broadening and
                Gaussian broadening, supported in TBMaLT.
                [DEFAULT=`fermi_smearing`]
            max_scc_iter: maximum permitted number of SCC iterations. If one or
                more system fail to converge within ``max_scc_iter`` cycles
                then a convergence error will be raised; unless the
                ``suppress_scc_error`` flag has been set. [DEFAULT=200]
            mixer: the `Mixer` instance with which to mix the charges during
                the SCC cycle. [DEFAULT=`Anderson`]

        Keyword Arguments:
            suppress_scc_error: if True, convergence errors will be suppressed
                and the calculation will proceed with as normal. This is of use
                during fitting when operating on large batches. This way if
                most systems converge but one does not, then it can just be
                ignored rather than ending the program. [DEFAULT=False]
            eigen_solver_settings: a dictionary storing advanced settings to
                be provided to the eigen solver.

        Returns:
            q_converged: the converged charges.
            hamiltonian: the second order self-consistent charge hamiltonian
                matrix.
            eig_values: resulting eigenvalues.
            eig_vectors: associated eigenvectors.
            rho: density matrix.

        Raises:
            ConvergenceFailure: if the charge convergence is not reached within
                the permitted number of iterations as specified by the argument
                ``max_scc_iter``. The ``suppress_scc_error` flag can be used to
                suppress this error.
        """

        was_non_batch = orbs.atomic_numbers.ndim == 1
        # Implementation of a batch agnostic self-consistent charge function is
        # somewhat challenging. While one could create a pair of functions, one
        # for the batch case and another for the non-batch case this would
        # require maintaining two almost identical functions. Instead, single
        # system cases are converted to a batch of size one.
        if was_non_batch:
            orbs = OrbitalInfo(
                orbs.atomic_numbers.unsqueeze(0), orbs.shell_dict,
                shell_resolved=orbs.shell_resolved)

            q_zero = q_zero.unsqueeze(0)
            core_hamiltonian = core_hamiltonian.unsqueeze(0)
            overlap = overlap.unsqueeze(0)
            gamma = gamma.unsqueeze(0)

        # Reset the mixer and ensure it is in batch-mode
        mixer.reset()
        mixer.is_batch = True

        # Pull out ancillary settings from the keyword arguments
        suppress_scc_error = kwargs.get('suppress_scc_error', False)
        eigen_solver_settings = kwargs.get('eigen_solver_settings', {})

        # Check if an initial guess has been provided for the charges. If not
        # then default to q-zero. This is also use to store the mixed charges
        # at the end of each step for the unconverged systems.
        q_current = kwargs.get("q_initial", q_zero)

        n_electrons = q_zero.sum(-1)

        # Tensors in which the final results are to be stored. Results are
        # appended to results tensors as each system converges.
        hamiltonian_out = torch.zeros_like(core_hamiltonian)
        rho_out = torch.zeros_like(core_hamiltonian)
        q_new_out = torch.zeros_like(q_zero)
        eig_values_out = torch.zeros_like(core_hamiltonian[:, :, 0])
        eig_vectors_out = torch.zeros_like(core_hamiltonian)

        # Tensor to track currently which systems have not yet converged.
        system_indices = torch.arange(len(n_electrons), device=q_zero.device)

        # Enter the self-consistent charge cycle. This will continue looping until
        # either the maximum permitted number of iterations has been reached or
        # convergence has been archived.
        for step in range(1, max_scc_iter + 1):
            # Perform a single step of the SCC cycle to generate the new charges.
            # The SCC step function will also return other important properties
            # that were calculated during the step.
            q_new, hamiltonian, eig_values, eig_vectors, rho = Dftb2.scc_step(
                q_current, q_zero, core_hamiltonian, overlap, gamma,
                orbs, n_electrons, filling_temp, filling_scheme,
                **eigen_solver_settings)

            # Check if the deviation between the current and new charges are
            # within tolerance. Check is done manually here rather than via
            # `Mixer.converged`. This because i) said attribute will be removed
            # and ii) the previous approach required mixing to perform the
            # tolerance check.
            converged = torch.le((q_new - q_current).abs(), mixer.tolerance).all(
                dim=q_new.dim_order()[1:])

            # Identify which, if any, systems have converged. Copy over data for
            # the converged systems into the results tensor.
            if converged.any():
                idxs = system_indices[converged]

                n_orbs = torch.max(orbs.n_orbitals)
                n_res = orbs.res_matrix_shape[-1]

                # Copy over converged values to the output tensors
                hamiltonian_out[idxs, :n_orbs, :n_orbs] = hamiltonian[converged, ...]
                rho_out[idxs, :n_orbs, :n_orbs] = rho[converged, ...]
                q_new_out[idxs, :n_res] = q_new[converged, ...]
                eig_values_out[idxs, :n_orbs] = eig_values[converged, ...]
                eig_vectors_out[idxs, :n_orbs, :n_orbs] = eig_vectors[converged, ...]

            # If all systems have converged, then discontinue the scc cycle.
            if converged.all():
                break

            # Otherwise just mix the charges ready for the next step and remove
            # converged systems from the working tensors.
            elif step < max_scc_iter and not converged.all():

                # Remove converged systems from the working tensors
                mask = ~converged

                orbs = orbs[mask]
                n_orbs = torch.max(orbs.n_orbitals)
                n_res = orbs.res_matrix_shape[-1]

                # When the orbital info entity is sliced, it is also
                # automatically squeezed to remove any unnecessary padding
                # values. This must be accounted for when culling the arrays.
                q_new = q_new[mask, :n_res]
                q_zero = q_zero[mask, :n_res]
                q_current = q_current[mask, :n_res]
                gamma = gamma[mask, :n_res, :n_res]
                core_hamiltonian = core_hamiltonian[mask, :n_orbs, :n_orbs]
                overlap = overlap[mask, :n_orbs, :n_orbs]
                n_electrons = n_electrons[mask]
                system_indices = system_indices[mask]

                # Cull the converged systems from the mixer.
                if step != 1:
                    # No history to cull in the first step as not mixing has taken
                    # place yet.
                    mixer.cull(converged, new_size=[n_res])

                q_current = mixer(q_new, q_current)

            # However, if the iteration limit has been reached, an exception
            # should be raised instead.
            elif not suppress_scc_error:
                raise ConvergenceError(
                    "SCC cycle failed to converge; iteration limit reached")

            # Unless instructed to ignore the convergence limit. In which case the
            # current date for the unconverged systems should be copied into the
            # output tensors.
            else:

                n_orbs = torch.max(orbs.n_orbitals)
                n_res = orbs.res_matrix_shape[-1]

                hamiltonian_out[system_indices, :n_orbs, :n_orbs] = hamiltonian
                rho_out[system_indices, :n_orbs, :n_orbs] = rho
                q_new_out[system_indices, :n_res] = q_new
                eig_values_out[system_indices, :n_orbs] = eig_values
                eig_vectors_out[system_indices, :n_orbs] = eig_vectors

        if was_non_batch:
            q_new_out = q_new_out.squeeze(0)
            hamiltonian_out = hamiltonian_out.squeeze(0)
            eig_values_out = eig_values_out.squeeze(0)
            eig_vectors_out = eig_vectors_out.squeeze(0)
            rho_out = rho_out.squeeze(0)

        return q_new_out, hamiltonian_out, eig_values_out, eig_vectors_out, rho_out
