# -*- coding: utf-8 -*-
"""Code associated with carrying out DFTB calculations."""
import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Literal, Tuple

from tbmalt.ml.module import Calculator, requires_args, call_with_required_args
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt import Basis
from tbmalt.physics.dftb.feeds import Feed, SkfOccupationFeed
from tbmalt.physics.filling import (
    fermi_search, fermi_smearing, gaussian_smearing, entropy_term, aufbau_filling)
from tbmalt.common.maths import eighb
from tbmalt.physics.dftb.shortgamma import ShortGamma
from tbmalt.common.batch import pack
from tbmalt.common.maths.mixers import Simple, Anderson

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
        resolution: Optional[Literal['atom', 'shell']] = None,
        mask: Tuple[Tensor, Tensor, Tensor] = None) -> Tensor:
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

            If unspecified, this will default to the resolution defined by the
            `basis.shell_resolved` attribute. This is only valid when ``basis``
            is also specified. [DEFAULT=None]
        mask: mask of `Basis` indices in batch calculations. The first is mask
            of batch, the second is maximum size of orbitals, the third will be
            maximum size of orbitals if `resolution` is `shell`, else maximum
             size of atoms.

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
            size = basis.shell_matrix_shape if basis.shell_resolved else basis.atomic_matrix_shape
            ind = basis.on_res
        elif resolution == 'atom':
            size, ind = basis.atomic_matrix_shape, basis.on_atoms
        elif resolution == 'shell':
            size, ind = basis.shell_matrix_shape, basis.on_shells
        else:
            raise NotImplementedError("Unknown resolution")

        if mask is not None:
            size = torch.Size((len(q), mask[1]))
            q = torch.zeros(size, device=rho.device, dtype=rho.dtype
                            ).scatter_add_(-1, ind[mask[0]][..., :mask[2]].clamp(min=0), q)
        else:
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

        self.mask_batch = ...
        self.mask_obrs = ...
        self._mask_batch = ...
        self._mask_obrs = ...
        self.max_atoms = None
        self.max_orbs = None

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
    def q_delta_atomic(self):
        """Delta atomic populations"""
        return self.q_final_atomic - self.q_zero_atomic

    @property
    def n_electrons(self):
        """Number of electrons"""
        return self.q_zero.sum(-1)  #[self.mask_batch]

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
                self.eig_values[self.mask_batch, :self.max_orbs], self.fermi_energy,
                self.filling_temp,
                e_mask=self.basis.on_atoms[self.mask_batch, :self.max_orbs]
                if self.is_batch else None) * scale_factor
        # Otherwise just fill according to the Aufbau principle
        else:
            return aufbau_filling(
                self.eig_values, self.n_electrons,
                e_mask=self.basis if self.is_batch else None) * scale_factor

    @property
    def fermi_energy(self):
        """Fermi energy"""
        return fermi_search(
            self.eig_values[self.mask_batch, :self.max_orbs],
            self.n_electrons[self.mask_batch], self.filling_temp,
            self.filling_scheme,
            # Pass the e_mask argument, but only if required.
            e_mask=self.basis.on_atoms[self.mask_batch, :self.max_orbs]
            if self.is_batch else None)

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
        self.overlap = None
        self.hamiltonian = None
        self.rho = None
        self.eig_values = None
        self.eig_vectors = None
        self.max_orbs = None
        self.max_atoms = None

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
        # Non-SCC do not really use the following variables, just to reduce
        # duplicated code for batch SCC-DFTB calculations
        self.max_atoms = torch.max(self.geometry.n_atoms[self.mask_batch])
        self.max_orbs = torch.max(self.basis.n_orbitals[self.mask_batch])

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
    """Self-consistent-charge density-functional tight-binding method (SCC-DFTB)."""

    def __init__(
            self, h_feed: IntegralFeed, s_feed: IntegralFeed,
            o_feed: Feed, u_feed: Feed, r_feed: Optional[Feed] = None,
            filling_temp: Optional[float] = None,
            filling_scheme: str = 'fermi', **kwargs):

        super().__init__(
            h_feed=h_feed, s_feed=s_feed, o_feed=o_feed, r_feed=r_feed,
            filling_temp=filling_temp, filling_scheme=filling_scheme, **kwargs)

        self.max_scc_step = kwargs.get('max_scc_step', 60)

        # Setting parameters for mixers
        _mixer = kwargs.get('mixer', 'Anderson')
        _mixer_list = {'Anderson': Anderson, 'Simple': Simple}
        self._mixer_settings = kwargs.get('mixer_settings',
                                          {'init_mix_param': 0.2,
                                           'mix_param': 0.2,
                                           'generations': 6})
        self._mixer = _mixer_list[_mixer](is_batch=None, **self._mixer_settings)

        # Hubbard U initialization
        self.u_feed = u_feed
        _gamma_type = kwargs.get('gamma', 'exponential')
        self.gamma = ShortGamma(self.u_feed, gamma_type=_gamma_type)

    def forward(self, cache: Optional[Dict[str, Any]] = None, **kwargs):
        """Execute the SCC DFTB calculation.

        This method triggers the execution of the self-consistent-charge
        density functional tight binding theory calculation. Once complete
        this will return the total system energy.

        Args:
            cache: Currently, the `Dftb2` calculator does not make use of the
                `cache` argument.

        Returns:
            total_energy: total energy of the target system(s). If repulsive
                interactions are not considered, i.e. the repulsion feed is
                omitted, then this will be the band structure energy. If finite
                temperature is active then this will technically be the Mermin
                energy.

        """
        # Reset parameters which may inherit from last calculator
        _gamma = self.short_gamma()
        self._mixer._step_number = 0
        self._mixer._is_batch = self.is_batch

        # to reset all parameters associated to converged
        self.mask_batch = ...
        self.mask_obrs = ...

        # to store not converged information of this SCC loop
        self._mask_batch = ...
        self._mask_obrs = ...
        charge = self.q_zero_atomic.clone()

        self.max_atoms = torch.max(self.geometry.n_atoms[self.mask_batch])
        self.max_orbs = torch.max(self.basis.n_orbitals[self.mask_batch])

        # Loop for DFTB2
        for step in range(self.max_scc_step):

            _shift_mat, H, S = self._second_order_ham(charge, _gamma)
            self._loop_scc(step, H, S, charge)

            if self._mixer.converged.all() or step + 1 == self.max_scc_step:
                break

        # Reset parameters so that each property will give full information
        self.mask_batch = ...
        self.max_atoms = torch.max(self.geometry.n_atoms[self.mask_batch])
        self.max_orbs = torch.max(self.basis.n_orbitals[self.mask_batch])

    def _loop_scc(self, step, H, S, charge):
        """Perform each single SCC-DFTB loop."""
        if step == 0 or not self.is_batch:
            eig_values, eig_vectors = eighb(H, S, **self._solver_settings)
            self.eig_values = eig_values
            self.eig_vectors = eig_vectors
        else:
            eig_values, eig_vectors = eighb(H, S, **self._solver_settings)
            self.eig_values[self.mask_batch, :self.max_orbs] = eig_values
            self.eig_vectors[self.mask_batch, :self.max_orbs, :self.max_orbs] = eig_vectors

        s_occs = torch.einsum(  # Scaled occupancy values
            '...i,...ji->...ji', torch.sqrt(self.occupancy), eig_vectors)

        rho = s_occs @ s_occs.transpose(-1, -2).conj()

        # Calculate batch or single Mulliken charge
        mask_q = None if step == 0 or not self.is_batch else (
            self.mask_batch, self.max_atoms, self.max_orbs)
        q_new = _mulliken(rho, S, self.basis, resolution='atom', mask=mask_q)

        if step == 0:
            charge_mix = self._mixer(q_new, x_old=self.q_zero_atomic)
            self.rho = rho
        else:
            charge_mix = self._mixer(q_new)
            self.rho[self.mask_batch, :self.max_orbs, :self.max_orbs] = rho

        charge[self.mask_batch, :self.max_atoms] = charge_mix
        converge = self._mixer.converged

        # to update parameters associated with converge and cull converged
        # systems for next SCC-DFTB loop
        self.cull(step, converge)

    def cull(self, step, converge):
        """Purge select systems form the Dftb2 or Dftb3 calculators."""
        if step == 0:
            self._mask_batch = ~converge
            if self.is_batch:
                self.mask_batch = self._mask_batch.clone()
        else:
            self._mask_batch = ~converge
            if self.is_batch:
                self.mask_batch[self.mask_batch.clone()] = self._mask_batch

        if self.is_batch and not converge.all():
            self._mixer.cull(converge, torch.max(self.geometry.n_atoms[self.mask_batch]))

            self.max_atoms = torch.max(self.geometry.n_atoms[self.mask_batch])
            self.max_orbs = torch.max(self.basis.n_orbitals[self.mask_batch])

    def _inv_distance(self):
        """Return inverse distance."""
        dist = self.geometry.distances
        inv_distance = torch.zeros(*dist.shape)
        inv_distance[dist.ne(0.0)] = 1.0 / dist[dist.ne(0.0)]
        return inv_distance

    def _second_order_ham(self, charge, shortgamma):
        """Build second order Gamma and Fock."""
        shift = self._update_shift(charge, shortgamma)

        if self.is_batch:
            shift_mat = torch.stack(
                [torch.unsqueeze(ishift, 1) + ishift for ishift in shift])
        else:
            shift_mat = shift.unsqueeze(1) + shift

        # Return masked H & S
        S = self.overlap[self.mask_batch, :self.max_orbs, :self.max_orbs]
        H = self.hamiltonian[self.mask_batch, :self.max_orbs, :self.max_orbs]\
            + 0.5 * S * shift_mat

        return shift_mat, H, S

    def _update_shift(self, charge, shortgamma):
        """Update shift."""
        if self.is_batch:
            atomic_orbitals = self.basis.orbs_per_atom[self.mask_batch]
            shift = torch.einsum(
                "...j, ...jk-> ...k", (charge - self.q_zero_atomic)[self.mask_batch],
                shortgamma[self.mask_batch]
            )
            return pack([torch.repeat_interleave(sh, ao)
                         for sh, ao in zip(shift, atomic_orbitals)])
        else:
            shift = (charge - self.q_zero_atomic) @ shortgamma
            return torch.repeat_interleave(shift, self.basis.orbs_per_atom)

    def short_gamma(self, format: str = 'atomic') -> Tensor:
        """Return second order short gamma term.

        Arguments:
            format: `atomic` or `orbital`, suggesting that short gamma are
                atomic resolution or orbital resolved.

        """
        assert format in ('atomic', 'orbital'), \
            f'{format} is not valid, please set atomic or orbital'

        if not self.is_periodic:
            inv_dist = self._inv_distance()
        else:
            raise NotImplementedError('PBC is not implemented®')

        if format == 'atomic':
            return inv_dist - self.gamma(self.geometry, self.basis)
        else:
            raise NotImplementedError('orbital resolved gamma is not implemented')


if __name__ == '__main__':
    from tbmalt.physics.dftb.feeds import ScipySkFeed
    from tbmalt import Basis, Geometry
    torch.set_default_dtype(torch.float64)
    from ase.build import molecule

    geom = Geometry.from_ase_atoms(molecule('CH4'))
    basis = Basis(geom.atomic_numbers, shell_dict={1: [0], 6: [0, 1]}, shell_resolved=True)

    path = '../../../tests/unittests/data/io/skfdb.hdf5'
    h_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
    s_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'overlap')

    o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])
    dftb = Dftb1(h_feed, s_feed, o_feed, filling_temp=0.0036749324)
    dftb(geom, basis)

    from tbmalt.physics.dftb.feeds import SkFeed, HubbardFeed

    ch4 = torch.tensor([4.30537894059011, 0.92365526485247, 0.92365526485247,
                        0.92365526485247, 0.92365526485247])
    h2o = torch.tensor([6.58558984371061, 0.70720507814469, 0.70720507814469])

    geos = Geometry.from_ase_atoms(molecule('H2O'))
    geob = Geometry.from_ase_atoms([molecule('CH4'), molecule('H2O')])
    basiss = Basis(geos.atomic_numbers, shell_dict={1: [0], 6: [0, 1], 8: [0, 1]}, shell_resolved=True)
    basisb = Basis(geob.atomic_numbers, shell_dict={1: [0], 6: [0, 1], 8: [0, 1]}, shell_resolved=True)
    h_feed = SkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
    s_feed = SkFeed.from_database(path, [1, 6, 8], 'overlap')

    o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])
    u_feed = HubbardFeed.from_database(path, [1, 6, 8])
    dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, filling_temp=0.0036749324)
    dftb2(geos, basiss)
    assert torch.allclose(dftb2.q_final_atomic, h2o)

    dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, filling_temp=0.0036749324)
    dftb2(geos, basiss)

    dftb2(geob, basisb)
    assert torch.allclose(dftb2.q_final_atomic[1, :3], h2o)
    assert torch.allclose(dftb2.q_final_atomic[0, :5], ch4)
    dftb2(geob, basisb)
