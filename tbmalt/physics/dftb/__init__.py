# -*- coding: utf-8 -*-
"""Code associated with carrying out DFTB calculations."""
import numpy as np
import torch

from typing import Optional, Dict, Any, Literal, Union

from tbmalt.ml.module import Calculator, requires_args, call_with_required_args
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt import Basis
from tbmalt.physics.dftb.feeds import Feed, SkfOccupationFeed
from tbmalt.physics.filling import (
    fermi_search, fermi_smearing, gaussian_smearing, entropy_term, aufbau_filling)
from tbmalt.common.maths import eighb
from tbmalt.physics.dftb.gamma import build_gamma_matrix
from tbmalt.common.batch import prepeat_interleave
from tbmalt.common.maths.mixers import Simple, Anderson, _Mixer
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
        rho: Tensor, S: Tensor, basis: Optional[Basis] = None,
        resolution: Optional[Literal['atom', 'shell', 'orbital']] = None
) -> Tensor:
    r"""Mulliken population analysis.

    By default, orbital resolved populations are returned, however, passing the
    associated basis instance will produce atom or shell resolved populations
    depending on the basis instance's `shell_resolved` attribute (the behavior
    of which can be overridden via the ``resolution`` argument).

    Arguments:
        rho: density matrix.
        S: overlap matrix.
        basis: a `Basis` instance may be specified to enable atom/shell
            resolved populations. If omitted, populations will be orbital
            resolved. [DEFAULT=None]
        resolution: can be specified to override the degree of resolution
            defined by the ``basis`` instance, available options are:

                - "atom": atom resolved
                - "shell": shell resolved
                - "orbital": orbital resolved

            If unspecified, this will default to the resolution defined by the
            `basis.shell_resolved` attribute. This is only valid when ``basis``
            is also specified. [DEFAULT=None]

    Returns:
        q: mulliken populations.

    Raises:
        TypeError: if ``resolution`` is specified in absence of ``basis``.

    """

    if resolution is not None and basis is None:
        raise TypeError(
            '"resolution" overrides default behaviour associated with the '
            '"basis" object\'s "shell_resolved" attribute. Thus it cannot be '
            'specified in absence of the "basis" argument.')

    q = (rho * S).sum(-1)  # Calculate the per-orbital Mulliken populations
    if basis is not None:  # Resolve to per-shell/atom if instructed to
        if resolution is None:
            # TODO: Change basis to have a res_matrix_shape property.
            size, ind = basis.res_matrix_shape, basis.on_res
        elif resolution == 'atom':
            size, ind = basis.atomic_matrix_shape, basis.on_atoms
        elif resolution == 'shell':
            size, ind = basis.shell_matrix_shape, basis.on_shells
        else:
            raise NotImplementedError("Unknown resolution")

        q = torch.zeros(size[:-1], device=rho.device, dtype=rho.dtype
                        ).scatter_add_(-1, ind.clamp(min=0), q)

    return q


class Dftb1(Calculator):
    """

    Developers Notes:
        Currently energies and occupancies are not scaled correctly. Occupancies
        and band energies need to be scaled based on whether or not they are i)
        spin-polarised or spin-unpolarised, ii) have a fixed fermi level for
        each spin channel or a common one across two colinear spin channels,
        iii) whether k-points are present or not. See the subroutine named
        "getFillingsAndBandEnergies" of the file "dftb/lib_dftbplus/main.F90"
        in DFTB+ for examples.
    """
    def __init__(
            self, h_feed: IntegralFeed, s_feed: IntegralFeed, o_feed: Feed,
            r_feed: Optional[Feed] = None, filling_temp: Optional[float] = None,
            filling_scheme: str = 'fermi', **kwargs):

        super().__init__(
            h_feed.dtype, h_feed.device, kwargs.get('mass', None))

        # Calculator Feeds
        self.h_feed = h_feed
        self.s_feed = s_feed
        self.o_feed = o_feed
        self.r_feed = r_feed

        device_list = [d.device for d in [h_feed, s_feed, o_feed, r_feed]
                       if d is not None]
        common_device = len(set(device_list)) == 1
        assert common_device, 'All `Feeds` must be on the same device'

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
            # Check if special arguments are required by the s_feed.matrix method.
            if requires_args(self.s_feed.matrix):
                # If so then make the call via `call_with_required_args`
                self._overlap = call_with_required_args(self.s_feed.matrix, self)
            else:
                # Otherwise just make the default call
                self._overlap = self.s_feed.matrix(self.geometry, self.basis)

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
            if requires_args(self.h_feed.matrix):
                self._hamiltonian = call_with_required_args(self.h_feed.matrix, self)
            else:
                self._hamiltonian = self.h_feed.matrix(self.geometry, self.basis)

        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value):
        self._hamiltonian = value

    @property
    def q_zero(self):
        """Initial orbital populations"""
        return self.o_feed(self.basis)

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
            self.basis.shell_matrix_shape[:-1],
            device=self.device, dtype=self.dtype).scatter_add_(
            -1, self.basis.on_shells.clamp(min=0), self.q_zero)

    @property
    def q_final_shells(self):
        """Final shell-wise populations"""
        return _mulliken(self.rho, self.overlap, self.basis, 'shell')

    @property
    def q_delta_shells(self):
        """Delta shell-wise populations"""
        return self.q_final_shells - self.q_zero_shells

    @property
    def q_zero_atomic(self):
        """Initial atomic populations"""
        return torch.zeros(
            self.basis.atomic_matrix_shape[:-1],
            device=self.device, dtype=self.dtype).scatter_add_(
            -1, self.basis.on_atoms.clamp(min=0), self.q_zero)

    @property
    def q_final_atomic(self):
        """Final atomic populations"""
        return _mulliken(self.rho, self.overlap, self.basis, 'atom')

    @property
    def dipole(self) -> Tensor:
        """Return dipole moments."""
        return torch.sum(
            -self.q_delta_atomic.unsqueeze(-1) * self.geometry.positions, -2
        )

    @property
    def q_delta_atomic(self):
        """Delta atomic populations"""
        return self.q_final_atomic - self.q_zero_atomic

    @property
    def q_zero_res(self):
        if self.basis.shell_resolved:
            return self.q_zero_shells
        else:
            return self.q_zero_atomic

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
        if self.filling_temp is not None:
            return self.filling_scheme(
                self.eig_values, self.fermi_energy, self.filling_temp,
                e_mask=self.basis if self.is_batch else None) * scale_factor
        # Otherwise just fill according to the Aufbau principle
        else:
            return aufbau_filling(
                self.eig_values, self.n_electrons,
                e_mask=self.basis if self.is_batch else None) * scale_factor

    @property
    def fermi_energy(self):
        """Fermi energy"""
        return fermi_search(
            self.eig_values, self.n_electrons, self.filling_temp,
            self.filling_scheme,
            # Pass the e_mask argument, but only if required.
            e_mask=self.basis if self.is_batch else None)

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
                self.filling_temp, e_mask=self.basis if self.is_batch else None)
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

        Args:
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


class Dftb2(Dftb1):
    """Self-consistent-charge density-functional tight-binding method (SCC-DFTB).

    Arguments:
        u_feed: this feed provides the Hubbard-U values as needed to construct
            the gamma matrix. This must be a `Feed` type object which when
            provided with a `Basis` object returns the Hubbard-U values for
            the target system.
        mixer: specifies the charge mixing scheme to be used. Providing the
            strings "simple" and "anderson" will result in their respectively
            named mixing schemes being used. Initialised `Mixer` class objects
            may also be provided directly.
        gamma_scheme: scheme used to construct the gamma matrix. This may be
            either "exponential" or "gaussian". [DEFAULT="exponential"]
        max_scc_iter: maximum permitted number of SCC iterations. If one or
            more system fail to converge within ``max_scc_iter`` cycles then a
            convergence error will be raise; unless the ``suppress_SCF_error``
            flag has been set. [DEFAULT=200]
        suppress_SCF_error: if True, convergence errors will be suppressed and
            the calculation will proceed with as normal. This is of use during
            fitting when operating on large batches. This way if most systems
            converge but one does not then it can just be ignored rather than
            ending the program. Unconverged systems can be identified via the
            ``converged`` attribute. [DEFAULT=False]

    Attributes:
        overlap: overlap matrix as constructed by the supplied `s_feed`.
        core_hamiltonian: first order core Hamiltonian matrix as built by the
            `h_feed` entity.
        gamma: the gamma matrix, this is constructed via the specified scheme
            and uses the Hubbard-U values produced by the `u_feed`
        hamiltonian: second order Hamiltonian matrix as produced via the SCC
            cycle.
        converged: a tensor of booleans indicating which systems have and
            have not converged (True if converged). This can be used during
            training, along side `suppress_SCF_error`, to allow unconverged
            systems to be omitted from the final loss calculation; as so to
            prevent introducing unnecessary instabilities.
        mixer: a `Mixer` type class instance used during the SCC cycle to
            perform charge mixing.


    """

    def __init__(
            self, h_feed: IntegralFeed, s_feed: IntegralFeed, o_feed: Feed,
            u_feed: Feed, r_feed: Optional[Feed] = None,
            max_scc_iter: int = 200,
            mixer: Union[_Mixer, Literal['anderson', 'simple']] = 'anderson',
            **kwargs):

        super().__init__(
            h_feed, s_feed, o_feed, r_feed=r_feed, **kwargs)

        # DFTB2 specific calculator feeds
        self.u_feed = u_feed

        self._core_hamiltonian: Optional[Tensor] = None
        self._gamma: Optional[Tensor] = None
        self.converged: Optional[Tensor] = None

        # Calculator Settings
        self.max_scc_iter = max_scc_iter
        self.suppress_SCF_error = kwargs.get('supress_SCF_error', False)
        self._gamma_scheme = kwargs.get('gamma_scheme', 'exponential')

        # If no pre-initialised was provided then construct one.
        if isinstance(mixer, str):
            mixer = {
                'anderson': Anderson, 'simple': Simple}[
                mixer.lower()](False, **kwargs.get('mix_params', {}))

        self.mixer = mixer

    @property
    def hamiltonian(self):
        """Second order Hamiltonian matrix"""
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value):
        self._hamiltonian = value

    @property
    def core_hamiltonian(self):
        """Core Hamiltonian matrix"""
        if self._core_hamiltonian is None:
            if requires_args(self.h_feed.matrix):
                self._core_hamiltonian = call_with_required_args(
                    self.h_feed.matrix, self)
            else:
                self._core_hamiltonian = self.h_feed.matrix(
                    self.geometry, self.basis)

        return self._core_hamiltonian

    @core_hamiltonian.setter
    def core_hamiltonian(self, value):
        self._core_hamiltonian = value

    @property
    def gamma(self):
        """Gamma matrix"""
        if self._gamma is None:
            self._gamma = build_gamma_matrix(
                self.geometry, self.basis, self.u_feed(self.basis),
                self._gamma_scheme)
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def forward(self, cache: Optional[Dict[str, Any]] = None
                , **kwargs) -> Tensor:
        """Execute the SCC-DFTB calculation.

        Invoking this will trigger the execution of the self-consistent-charge
        density functional tight binding theory calculation.

        Args:
            cache: This stores any information which can be used to boot-strap
                the calculation. Currently supported values are:
                    - "q_initial": initial starting guess for the SCC cycle.

        Returns:
            total_energy: total energy for the target systems this will include
                both the repulsive and entropy terms, where appropriate.

        """
        # Step 1: Initialisation

        # Reset the mixer and inform it whether it will be operating on a batch.
        # This must be explicitly defined each time as it cannot be inferred
        # from context.
        self.mixer.reset()
        self.mixer._is_batch = self.is_batch
        scc_grad = kwargs.get('scc_grad', True)

        # Set the initial starting guess for the charges.
        q_current = self.q_zero_res
        if cache is not None:
            q_current = cache.get('q_initial', q_current)

        # Array in which the final converged charges of each system are stored.
        # Results are assigned to `q_converged` as systems coverage during the
        # initial non-gradient-tracked SCC cycle. The values are then used as
        # the "initial guesses" for the final single shot SCC cycle which takes
        # place within the graph to reconnect the gradients.
        q_converged = torch.zeros_like(q_current)

        # Calls are made to the various cached properties to ensure that they
        # are constructed within the purview of the graph.
        self.overlap, self.core_hamiltonian, self.gamma

        # Step 2: Preliminary SCC cycle
        # A preliminary SCC cycle is performed outside of the gradient and acts
        # only to get the converged charges to be used in the second cycle.
        if not scc_grad:
            with torch.no_grad():
                q_converged = self.scc(q_current, q_converged)

        else:
            q_converged = self.scc(q_current, q_converged)

        # Step 3: Final SCC cycle
        # A single shot SCC cycle is now performed using the converged charges
        # as the initial starting guess. As this is done within view of the
        # auto-grad engine it will allow for gradients to be computed. This two
        # step approach allows for gradients to be computed without having to
        # track them through the full SCC cycle.
        self._scc_cycle(q_converged)

        # Calculate and return the total system energy, taking into account
        # the entropy term as and when necessary.
        return self.mermin_energy

    def scc(self, q_current, q_converged):
        # Non-batch systems are treated separately for the sake of clarity
        # as special treatment is required for the batch case.
        if not self.is_batch:
            # Begin the SCC cycle
            for step in range(1, self.max_scc_iter + 1):

                # Perform a single SCC step and apply the mixing algorithm.
                q_current = self.mixer(self._scc_cycle(q_current), q_current)

                # If the system has converged then assign the `q_converged`
                # values and break out of the SCC cycle.
                if self.mixer.converged:
                    q_converged[:] = q_current[:]
                    self.converged = torch.tensor(True)
                    break

            # If the maximum permitted number of iterations is exceeded then
            # then raise an exception; unless explicitly instructed not to.
            else:
                self.converged = torch.tensor(False)
                if not self.suppress_SCF_error:
                    raise ConvergenceError(
                        "SCC cycle failed to converge; "
                        "iteration limit reached")

        else:
            # For the batch case, systems will be culled as and when they
            # converge. This process involves modifying attributes such as
            # `geometry`, `basis`, `overlap`, etc. Doing so allows all the
            # existing code within the methods and properties to be used.
            # However, this requires that copies of the original objects
            # are saved and restored at the end of the batch SCC cycle.
            # Note that a copy of the second order hamiltonian matrix is not
            # required as it is regenerated in full in the second SCC cycle.
            c_geometry, c_basis = self.geometry, self.basis
            c_overlap, c_gamma = self.overlap, self.gamma
            c_hamiltonian_copy = self.core_hamiltonian

            # Todo:
            #  Implement a method that can identify which properties do and
            #  do not need to be fully destroyed by __restore.

            # `system_indices` provides the indices of each system and is
            # culled along with the other arrays so that one can identify
            # which systems remain.
            system_indices = torch.arange(self.geometry._n_batch)

            # Used to help the user track which systems have converged.
            self.converged = torch.full(system_indices.shape, False)

            for step in range(1, self.max_scc_iter + 1):
                q_current = self.mixer(self._scc_cycle(q_current), q_current)

                if (c_mask := self.mixer.converged).any():

                    idxs = system_indices[c_mask]
                    q_converged[idxs, :q_current.shape[-1]] = q_current[c_mask, :]
                    self.converged[idxs] = True

                    # If all systems have converged then the end of the SCC
                    # cycle has been reached.
                    if torch.all(c_mask):
                        break
                    # Otherwise there are still systems left to converge. Thus
                    # the converged systems will now be culled to avoid over-
                    # converging them.
                    else:
                        # The order in which things are done here matters
                        # Cull calculator attributes
                        self.__cull(c_mask)
                        # Cull local variables
                        n_res = self.basis.res_matrix_shape[-1]
                        system_indices = system_indices[~c_mask]
                        q_current = q_current[~c_mask, :n_res]
                        # Cull mixer
                        self.mixer.cull(c_mask, new_size=[n_res])

            else:
                self.converged = torch.tensor(False)
                if not self.suppress_SCF_error:
                    # Here a restore is performed before the error being
                    # raised to help with debugging.
                    self._geometry, self._basis = c_geometry, c_basis
                    self.overlap, self.gamma = c_overlap, c_gamma
                    self.core_hamiltonian = c_hamiltonian_copy

                    raise ConvergenceError(
                        "SCC cycle failed to converge; "
                        "iteration limit reached", self.converged)

            # Restore the calculator back to its state prior to culling.
            # Properties like `rho` and `eig_values` are not reset as it is
            # assumed that they will be overridden in the next stage.
            self._geometry, self._basis = c_geometry, c_basis
            self.overlap, self.gamma = c_overlap, c_gamma
            self.core_hamiltonian = c_hamiltonian_copy

        return q_converged

    def __cull(self, mask: Tensor):
        """Cull converged systems from the calculator instance.

        Calling this method will strip a selection of systems from various
        components of the associated calculator instance. This is intended to
        be used to temporarily remove converged systems during the SCC cycle.
        However, as this only filters some, but not all, attributes its ill
        advised to use this anywhere other than the `.forward` method.

        Args:
            mask: A tensor of Booleans indicating which systems have converged.
                Systems which have converged will be masked partially out.

        Warnings:
            Do not invoke this function manually unless you are sure that you
            know what you are doing!
        """
        self._basis = self.basis[~mask]
        self._geometry = self.geometry[~mask]
        n_orbs = torch.max(self.basis.n_orbitals)
        n_res = self.basis.res_matrix_shape[-1]
        self._overlap = self._overlap[~mask, :n_orbs, :n_orbs]
        self._core_hamiltonian = self._core_hamiltonian[~mask, :n_orbs, :n_orbs]
        self._gamma = self._gamma[~mask, :n_res, :n_res]

    def _scc_cycle(self, q_in: Tensor) -> Tensor:
        """Perform a single self-consistent charge cycle.

        This method performs a single Self-Consistent Charge cycle (SCC). Using
        ``q_in`` as the initial guess during the first cycle and as the mixed
        charges in all subsequent cycles.

        Args:
            q_in: Input charges are provided via the ``q_in`` argument. This
                can be viewed as the initial guess for the first cycle and as
                the mixed charges from the previous step in all subsequent
                cycles.

        Returns:
            q_out: New updated charges, as computed following the SCC step.

        Notes:
            It is important to note that this method is intended to modify and
            update the class and its attributes during the SCC cycle. The newly
            computed charges are only returned to facilitate ease of use.

            The charges ``q_in`` and ``q_out`` may be either shell or atom
            resolved, but must match up with that as defined by the basis
            attribute `shell_resolved`.
        """

        # Construct the shift matrix
        shifts = torch.einsum(
            '...i,...ij->...j', q_in - self.q_zero_res, self.gamma)
        shifts = prepeat_interleave(shifts, self.basis.orbs_per_res)
        shifts = (shifts[..., None] + shifts[..., None, :])

        # Compute the second order Hamiltonian matrix
        self._hamiltonian = self.core_hamiltonian + .5 * self.overlap * shifts

        # Obtain the eigen-values/vectors via an eigen decomposition
        self.eig_values, self.eig_vectors = eighb(
            self.hamiltonian, self.overlap, **self._solver_settings)

        # Scaled occupancy values
        s_occs = torch.einsum(
            '...i,...ji->...ji', torch.sqrt(self.occupancy), self.eig_vectors)

        # Density matrix
        self.rho = s_occs @ s_occs.transpose(-1, -2).conj()

        # Compute and return the new
        return _mulliken(self.rho, self.overlap, self.basis)

    def reset(self):
        """Reset all attributes and cached properties."""
        self._overlap = None
        self._core_hamiltonian = None
        self._hamiltonian = None
        self._gamma = None
        self.converged = None

        self.rho = None
        self.eig_values = None
        self.eig_vectors = None


if __name__ == '__main__':
    from tbmalt.physics.dftb.feeds import ScipySkFeed
    from tbmalt import Basis, Geometry
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(15)
    from ase.build import molecule

    geom = Geometry.from_ase_atoms(molecule('CH4'))
    basis = Basis(geom.atomic_numbers, shell_dict={1: [0], 6: [0, 1]})

    path = '../../../examples/example_01/example_dftb_parameters.h5'
    h_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
    s_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'overlap')

    o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])

    dftb = Dftb1(h_feed, s_feed, o_feed, filling_temp=0.0036749324)
    dftb(geom, basis)

    # DFTB2
    from tbmalt.physics.dftb.feeds import SkFeed, HubbardFeed
    h_feed = SkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
    s_feed = SkFeed.from_database(path, [1, 6, 8], 'overlap')

    o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])
    u_feed = HubbardFeed.from_database(path, [1, 6, 8])

    ch4 = torch.tensor([4.30537894059011, 0.92365526485247, 0.92365526485247,
                        0.92365526485247, 0.92365526485247])
    h2o = torch.tensor([6.58558984371061, 0.70720507814469, 0.70720507814469])


    geos = Geometry.from_ase_atoms(molecule('CH3O'))
    geob = Geometry.from_ase_atoms([
        molecule('H2O'), molecule('CH4'), molecule('CH3O'), molecule('OCHCHO'),
        molecule('CH3CHO'), molecule('CH3CH2OCH3'), molecule('bicyclobutane')])
    basiss = Basis(geos.atomic_numbers, shell_dict={1: [0], 6: [0, 1], 8: [0, 1]})
    basisb = Basis(geob.atomic_numbers, shell_dict={1: [0], 6: [0, 1], 8: [0, 1]})

    mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2, 'generations': 3, 'tolerance': 1e-10}
    dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, filling_temp=0.0036749324, mix_params=mix_params)
    dftb2(geos, basiss)
