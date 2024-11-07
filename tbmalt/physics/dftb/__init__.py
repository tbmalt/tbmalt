# -*- coding: utf-8 -*-
"""Code associated with carrying out DFTB calculations."""
import copy
import numpy as np
import torch

from typing import Optional, Dict, Any, Literal, Union

from tbmalt.ml.module import Calculator
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt import OrbitalInfo
from tbmalt.physics.dftb.feeds import Feed, SkfOccupationFeed
from tbmalt.physics.filling import (
    fermi_search, fermi_smearing, gaussian_smearing, entropy_term,
    aufbau_filling)
from tbmalt.common.maths import eighb
from tbmalt.physics.dftb.coulomb import build_coulomb_matrix
from tbmalt.physics.dftb.gamma import build_gamma_matrix, gamma_exponential_gradient
from tbmalt.physics.dftb.properties import dos
from tbmalt.common.batch import prepeat_interleave
from tbmalt.common.maths.mixers import Simple, Anderson, _Mixer
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
        r_feed: this feed describes the repulsive interaction. [DEFAULT: None]
        filling_temp: Electronic temperature used to calculate Fermi-energy.
            [DEFAULT: None]
        filling_scheme: The scheme used for finite temperature broadening.
            There are two broadening methods, Fermi-Dirac broadening and
            Gaussian broadening, supported in TBMaLT. [DEFAULT: fermi]

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
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.feeds import ScipySkFeed, SkfOccupationFeed
        >>> from tbmalt.physics.dftb import Dftb1
        >>> from tbmalt.io.skf import Skf
        >>> from ase.build import molecule
        >>> import urllib
        >>> import tarfile
        >>> from os.path import join
        >>> torch.set_default_dtype(torch.float64)

        # Link to the auorg-1-1 parameter set
        >>> link = \
        'https://dftb.org/fileadmin/DFTB/public/slako/auorg/auorg-1-1.tar.xz'

        # Preparation of sk file
        >>> elements = ['H', 'C', 'O', 'Au', 'S']
        >>> tmpdir = './'
        >>> urllib.request.urlretrieve(
                link, path := join(tmpdir, 'auorg-1-1.tar.xz'))
        >>> with tarfile.open(path) as tar:
                tar.extractall(tmpdir)
        >>> skf_files = [join(tmpdir, 'auorg-1-1', f'{i}-{j}.skf')
                         for i in elements for j in elements]
        >>> for skf_file in skf_files:
                Skf.read(skf_file).write(path := join(tmpdir, 'auorg.hdf5'))

        # Preparation of system to calculate
        # Single system
        >>> geos = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs_s = OrbitalInfo(geos.atomic_numbers, shell_dict={1: [0], 6: [0, 1]})
        # Batch systems
        >>> geob = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
        >>> orbs_b = OrbitalInfo(geob.atomic_numbers, shell_dict={1: [0], 6: [0, 1],
                                                            8: [0, 1]})

        # Definition of feeds
        >>> h_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
        >>> s_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'overlap')
        >>> o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])

        # Run DFTB1 calculation
        >>> dftb = Dftb1(h_feed, s_feed, o_feed, filling_temp=0.0036749324)
        >>> _ = dftb(geos, orbs_s)
        >>> getattr(dftb, 'q_final_atomic')
        tensor([4.3591, 0.9102, 0.9102, 0.9102, 0.9102])
        >>> _ = dftb(geob, orbs_b)
        >>> getattr(dftb, 'q_final_atomic')
        tensor([[6.7552, 0.6224, 0.6224, 0.0000, 0.0000],
                [4.3591, 0.9102, 0.9102, 0.9102, 0.9102]])

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
        self._scc_energy: Optional[Tensor] = None
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
        return self.o_feed(self.orbs)

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
    def dipole(self) -> Tensor:
        """Return dipole moments."""
        return torch.sum(
            self.q_delta_atomic.unsqueeze(-1) * self.geometry.positions, -2
        )

    @property
    def q_delta_atomic(self):
        """Delta atomic populations"""
        return self.q_final_atomic - self.q_zero_atomic

    @property
    def q_zero_res(self):
        if self.orbs.shell_resolved:
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
    def h0_energy(self):
        """H0 energy"""
        _h0 = self.h_feed.matrix_from_calculator(self)
        return ((self.rho * _h0).sum(-1).sum(-1))

    @property
    def h2_energy(self):
        """SCC energy"""
        return 0.0 if self._scc_energy is None else self._scc_energy

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
        return self.h0_energy + self.h2_energy + self.repulsive_energy

    @property
    def mermin_energy(self):
        """Mermin free energy; i.e. E_total-TS"""
        return self.band_free_energy + self.repulsive_energy

    @property
    def eigenvalue(self):
        """Eigenvalue in unit hartree"""
        return self.eig_values

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
            homo_lumo = self.eigenvalue[mask] if self.occupancy.ndim == 1 else\
                self.eigenvalue[mask].view(self.occupancy.size(0), -1)

        return homo_lumo

    @property
    def dos_energy(self, ext=energy_units['ev'], grid=1000):
        """Energy distribution of (p)DOS in unit hartree"""
        e_min = torch.min(self.eigenvalue.detach(), dim=-1).values - ext
        e_max = torch.max(self.eigenvalue.detach(), dim=-1).values + ext
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
        mask = torch.where(self.eigenvalue == 0, False, True)
        sigma = 0.1 * energy_units['ev']
        return dos(self.eigenvalue, self.dos_energy, sigma=sigma, mask=mask)

    @property
    def forces(self):
        """Forces acting on the atoms"""

        doverlap, dh0 = self._finite_diff_overlap_h0()
        # Use the already calculated density matrix rho_mu,nu
        density = self.rho
        #print('#####################')
        #print('Density:', density)
        #print('#####################')
        # Calculate energy weighted density matrix
        temp_dens = torch.einsum(  # Scaled occupancy values
            '...i,...ji->...ji', torch.sqrt(self.occupancy), self.eig_vectors)
        #TODO This is currently a workaround to include the energy (eigenvalues) but should be solved in a better way
        temp_dens_weighted = torch.einsum(  # Scaled occupancy values
            '...i,...ji->...ji', self.eig_values * torch.sqrt(self.occupancy), self.eig_vectors)
        
        rho_weighted = temp_dens_weighted @ temp_dens.transpose(-1, -2).conj()
        #print('Rho weighted:', rho_weighted)

        #TODO something in this summation seems to be wrong (it returns non padded tensor for first batch)
        force = - torch.einsum('...nm,...acmn->...ac', density, dh0) + torch.einsum('...nm,...acmn->...ac', rho_weighted, doverlap) - self.r_feed.dErep
        #force = self.r_feed.dErep

        return force

    def _finite_diff_overlap_h0(self, delta=1.0e-6):
        """Calculates the gradient of the overlap using finite differences
        
        Arguments:
            delta: step size for finite differences

        Returns:
            doverlap: gradients of the overlap matrix for each atom and corresponding coordinates.
                The returned Tensor has the dimensions [ num_batches, num_atoms, coords, 1st overlap dim, 2nd overlap dim ].
                The atoms for each batch are ordered in the same way as given by geomytry.atomic_numbers.
        """
        # Instantiate Tensor for overlapp diff with dim: [ num_batches, num_atoms, coords, 1st overlap dim, 2nd overlap dim ]
        overlap_dim = self.overlap.size()[-2::]
        h0_dim = self.hamiltonian.size()[-2::]
        postions_dim = self.geometry._positions.size()
        doverlap_dim = postions_dim + overlap_dim
        dh0_dim = postions_dim + h0_dim

        doverlap = torch.zeros(doverlap_dim, device=self.device, dtype=self.dtype)
        dh0 = torch.zeros(dh0_dim, device=self.device, dtype=self.dtype)

        for atom_idx in range(self.geometry.atomic_numbers.size(-1)*3):
            # Make full copy of original geometry and change position
            dgeometry1 = copy.deepcopy(self.geometry)
            dgeometry2 = copy.deepcopy(self.geometry)
            # The following changes the atom_idx-nth coordinate of the geometry for each batch
            temp_pos1 = dgeometry1._positions.flatten()
            temp_pos1[atom_idx::3*postions_dim[-2]] += delta
            
            temp_pos2 = dgeometry2._positions.flatten()
            temp_pos2[atom_idx::3*postions_dim[-2]] -= delta
            # Set the changed positions for the dgeometry
            dgeometry1._positions = temp_pos1.unflatten(dim=0, sizes=postions_dim)
            dgeometry2._positions = temp_pos2.unflatten(dim=0, sizes=postions_dim)
            # Calculate temporary overlap matrix with the shifted geometry then finite difference
            temp_overlap1 = self.s_feed.matrix(dgeometry1, self.orbs)
            temp_overlap2 = self.s_feed.matrix(dgeometry2, self.orbs)

            temp_h01 = self.h_feed.matrix(dgeometry1, self.orbs)
            temp_h02 = self.h_feed.matrix(dgeometry2, self.orbs)
            
            doverlap[..., int(atom_idx / 3), atom_idx % 3, :, :] = (temp_overlap1 - temp_overlap2) / (2*delta)
            dh0[..., int(atom_idx / 3), atom_idx % 3, :, :] = (temp_h01 - temp_h02) / (2*delta)

        return doverlap, dh0
    
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


class Dftb2(Dftb1):
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
        r_feed: this feed describes the repulsive interaction. [DEFAULT: None]
        filling_temp: Electronic temperature used to calculate Fermi-energy.
            [DEFAULT: None]
        max_scc_iter: maximum permitted number of SCC iterations. If one or
            more system fail to converge within ``max_scc_iter`` cycles then a
            convergence error will be raise; unless the ``suppress_SCF_error``
            flag has been set. [DEFAULT=200]
        mixer: specifies the charge mixing scheme to be used. Providing the
            strings "simple" and "anderson" will result in their respectively
            named mixing schemes being used. Initialised `Mixer` class objects
            may also be provided directly.

    Keyword Arguments:
        suppress_SCF_error: if True, convergence errors will be suppressed and
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
            training, along side `suppress_SCF_error`, to allow unconverged
            systems to be omitted from the final loss calculation; as so to
            prevent introducing unnecessary instabilities.
        mixer: a `Mixer` type class instance used during the SCC cycle to
            perform charge mixing.

    Examples:
        >>> from tbmalt import OrbitalInfo, Geometry
        >>> from tbmalt.physics.dftb.feeds import ScipySkFeed,\
            SkfOccupationFeed, HubbardFeed
        >>> from tbmalt.physics.dftb import Dftb2
        >>> from tbmalt.io.skf import Skf
        >>> from ase.build import molecule
        >>> import urllib
        >>> import tarfile
        >>> from os.path import join
        >>> torch.set_default_dtype(torch.float64)

        # Link to the auorg-1-1 parameter set
        >>> link = \
        'https://dftb.org/fileadmin/DFTB/public/slako/auorg/auorg-1-1.tar.xz'

        # Preparation of sk file
        >>> elements = ['H', 'C', 'O', 'Au', 'S']
        >>> tmpdir = './'
        >>> urllib.request.urlretrieve(
                link, path := join(tmpdir, 'auorg-1-1.tar.xz'))
        >>> with tarfile.open(path) as tar:
                tar.extractall(tmpdir)
        >>> skf_files = [join(tmpdir, 'auorg-1-1', f'{i}-{j}.skf')
                         for i in elements for j in elements]
        >>> for skf_file in skf_files:
                Skf.read(skf_file).write(path := join(tmpdir, 'auorg.hdf5'))

        # Preparation of system to calculate
        # Single system
        >>> geos = Geometry.from_ase_atoms(molecule('CH4'))
        >>> orbs_s = OrbitalInfo(geos.atomic_numbers, shell_dict={1: [0], 6: [0, 1]})
        # Batch systems
        >>> geob = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
        >>> orbs_b = OrbitalInfo(geob.atomic_numbers, shell_dict={1: [0], 6: [0, 1],
                                                            8: [0, 1]})
        # Single system with pbc
        >>> geop = Geometry(
                torch.tensor([6, 1, 1, 1, 1]),
                torch.tensor([[3.0, 3.0, 3.0],
                              [3.6, 3.6, 3.6],
                              [2.4, 3.6, 3.6],
                              [3.6, 2.4, 3.6],
                              [3.6, 3.6, 2.4]]),
                torch.tensor([[4.0, 4.0, 0.0],
                              [5.0, 0.0, 5.0],
                              [0.0, 6.0, 6.0]]),
                units='a', cutoff=torch.tensor([9.98]))
        >>> orbs_p = OrbitalInfo(geop.atomic_numbers, shell_dict={1: [0], 6: [0, 1]})


        # Definition of feeds
        >>> h_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'hamiltonian')
        >>> s_feed = ScipySkFeed.from_database(path, [1, 6, 8], 'overlap')
        >>> o_feed = SkfOccupationFeed.from_database(path, [1, 6, 8])
        >>> u_feed = HubbardFeed.from_database(path, [1, 6, 8])

        # Run DFTB2 calculation
        >>> mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2,
                          'generations': 3, 'tolerance': 1e-10}
        >>> dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed,
                          filling_temp=0.0036749324, mix_params=mix_params)
        >>> _ = dftb2(geos, orbs_s)
        >>> getattr(dftb2, 'q_final_atomic')
        tensor([4.3054, 0.9237, 0.9237, 0.9237, 0.9237])
        >>> _ = dftb2(geob, orbs_b)
        >>> getattr(dftb2, 'q_final_atomic')
        tensor([[6.5856, 0.7072, 0.7072, 0.0000, 0.0000],
                [4.3054, 0.9237, 0.9237, 0.9237, 0.9237]])
        >>> _ = dftb2(geop, orbs_p)
        >>> getattr(dftb2, 'q_final_atomic')
        tensor([4.6124, 0.8332, 0.8527, 0.8518, 0.8499])

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
        self._invr: Optional[Tensor] = None
        self._scc_energy: Optional[Tensor] = None
        self.converged: Optional[Tensor] = None

        # Calculator Settings
        self.max_scc_iter = max_scc_iter
        self.suppress_SCF_error = kwargs.get('supress_SCF_error', False)
        self.gamma_scheme = kwargs.get('gamma_scheme', 'exponential')
        self.coulomb_scheme = kwargs.get('coulomb_scheme', 'search')

        # If no pre-initialised was provided then construct one.
        if isinstance(mixer, str):
            mixer = {
                'anderson': Anderson, 'simple': Simple}[
                mixer.lower()](False, **kwargs.get('mix_params', {}))

        self.mixer = mixer

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
    def invr(self):
        """1/R matrix"""
        if self._invr is None:
            if self.geometry.periodicity is not None:
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
    def gamma(self):
        """Gamma matrix as constructed using the `u_feed`"""
        if self._gamma is None:
            self._gamma = build_gamma_matrix(
                self.geometry, self.orbs, self.invr,
                self.u_feed(self.orbs), self.gamma_scheme)
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def scc_energy(self):
        """Energy contribution from charge fluctuation"""
        return self._scc_energy

    @scc_energy.setter
    def scc_energy(self, value):
        self._scc_energy = value

    @property
    def forces(self):
        """Forces acting on the atoms"""

        doverlap, dh0 = self._finite_diff_overlap_h0()
        # Use the already calculated density matrix rho_mu,nu
        density = self.rho
        # Calculate energy weighted density matrix
        temp_dens = torch.einsum(  # Scaled occupancy values
            '...i,...ji->...ji', torch.sqrt(self.occupancy), self.eig_vectors)
        #TODO This is currently a workaround to include the energy (eigenvalues) but should be solved in a better way
        temp_dens_weighted = torch.einsum(  # Scaled occupancy values
            '...i,...ji->...ji', self.eig_values * torch.sqrt(self.occupancy), self.eig_vectors)
        
        rho_weighted = temp_dens_weighted @ temp_dens.transpose(-1, -2).conj()
        
        #Non-scc Forces
        force = - torch.einsum('...nm,...acmn->...ac', density, dh0) + torch.einsum('...nm,...acmn->...ac', rho_weighted, doverlap) - self.r_feed.dErep

        #Scc corrcections
        
        # Construct the shift matrix
        shifts = torch.einsum(
            '...i,...ij->...j', self.q_final_atomic - self.q_zero_res, self.gamma)
        shifts = prepeat_interleave(shifts, self.orbs.orbs_per_res)
        shifts = (shifts[..., None] + shifts[..., None, :])

        print('force shifts:', shifts)

        # Compute the h1 hamiltonian matrix
        h1 = .5 * shifts
        print('H1:', h1)
        
        h1_correction = torch.einsum('...nm,...nm,...acmn->...ac', density, h1, doverlap)
        print('H1 correction:', h1_correction)

        # Gamma gradient correction
        gamma_grad = gamma_exponential_gradient(self.geometry, self.orbs, self.u_feed(self.orbs))
        print('Gamma gradient:', gamma_grad)
        gamma_correction = torch.einsum('...a,...abc,...b->...ac', self.q_delta_atomic, gamma_grad, self.q_delta_atomic)
        print('Gamma correction:', gamma_correction)

        force = force - h1_correction - gamma_correction

        return force

    def _finite_diff_overlap_h0(self, delta=1.0e-6):
        """Calculates the gradient of the overlap using finite differences
        
        Arguments:
            delta: step size for finite differences

        Returns:
            doverlap: gradients of the overlap matrix for each atom and corresponding coordinates.
                The returned Tensor has the dimensions [ num_batches, num_atoms, coords, 1st overlap dim, 2nd overlap dim ].
                The atoms for each batch are ordered in the same way as given by geomytry.atomic_numbers.
        """
        # Instantiate Tensor for overlapp diff with dim: [ num_batches, num_atoms, coords, 1st overlap dim, 2nd overlap dim ]
        overlap_dim = self.overlap.size()[-2::]
        h0_dim = self.core_hamiltonian.size()[-2::]
        postions_dim = self.geometry._positions.size()
        doverlap_dim = postions_dim + overlap_dim
        dh0_dim = postions_dim + h0_dim

        doverlap = torch.zeros(doverlap_dim, device=self.device, dtype=self.dtype)
        dh0 = torch.zeros(dh0_dim, device=self.device, dtype=self.dtype)

        for atom_idx in range(self.geometry.atomic_numbers.size(-1)*3):
            # Make full copy of original geometry and change position
            dgeometry1 = copy.deepcopy(self.geometry)
            dgeometry2 = copy.deepcopy(self.geometry)
            # The following changes the atom_idx-nth coordinate of the geometry for each batch
            temp_pos1 = dgeometry1._positions.flatten()
            temp_pos1[atom_idx::3*postions_dim[-2]] += delta
            
            temp_pos2 = dgeometry2._positions.flatten()
            temp_pos2[atom_idx::3*postions_dim[-2]] -= delta
            # Set the changed positions for the dgeometry
            dgeometry1._positions = temp_pos1.unflatten(dim=0, sizes=postions_dim)
            dgeometry2._positions = temp_pos2.unflatten(dim=0, sizes=postions_dim)
            # Calculate temporary overlap matrix with the shifted geometry then finite difference
            temp_overlap1 = self.s_feed.matrix(dgeometry1, self.orbs)
            temp_overlap2 = self.s_feed.matrix(dgeometry2, self.orbs)

            temp_h01 = self.h_feed.matrix(dgeometry1, self.orbs)
            temp_h02 = self.h_feed.matrix(dgeometry2, self.orbs)
            
            doverlap[..., int(atom_idx / 3), atom_idx % 3, :, :] = (temp_overlap1 - temp_overlap2) / (2*delta)
            dh0[..., int(atom_idx / 3), atom_idx % 3, :, :] = (temp_h01 - temp_h02) / (2*delta)

        return doverlap, dh0
 

    def forward(self, cache: Optional[Dict[str, Any]] = None
                , **kwargs) -> Tensor:
        """Execute the SCC-DFTB calculation.

        Invoking this will trigger the execution of the self-consistent-charge
        density functional tight binding theory calculation.

        Arguments:
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
        self.overlap, self.core_hamiltonian, self.invr, self.gamma

        # Step 2: Preliminary SCC cycle
        # A preliminary SCC cycle is performed outside of the gradient and acts
        # only to get the converged charges to be used in the second cycle.
        with torch.no_grad():

            # Non-batch systems are treated separately for the sake of clarity
            # as special treatment is required for the batch case.
            if not self.is_batch:
                # Begin the SCC cycle
                for step in range(1, self.max_scc_iter + 1):

                    # Perform a single SCC step and apply the mixing algorithm.
                    q_current = self.mixer(self._scc_cycle(q_current),
                                           q_current)

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
                # `geometry`, `orbs`, `overlap`, etc. Doing so allows all the
                # existing code within the methods and properties to be used.
                # However, this requires that copies of the original objects
                # are saved and restored at the end of the batch SCC cycle.
                # Note that a copy of the second order hamiltonian matrix is not
                # required as it is regenerated in full in the second SCC cycle.
                c_geometry, c_orbs = self.geometry, self.orbs
                c_overlap, c_gamma = self.overlap, self.gamma
                c_invr, c_hamiltonian_copy = self.invr, self.core_hamiltonian

                # Todo:
                #  Implement a method that can identify which properties do and
                #  do not need to be fully destroyed by __restore.

                # `system_indices` provides the indices of each system and is
                # culled along with the other arrays so that one can identify
                # which systems remain.
                system_indices = torch.arange(self.geometry._n_batch, device=self.device)

                # Used to help the user track which systems have converged.
                self.converged = torch.full(system_indices.shape, False, device=self.device)

                for step in range(1, self.max_scc_iter + 1):
                    q_current = self.mixer(self._scc_cycle(q_current),
                                           q_current)

                    if (c_mask := self.mixer.converged).any():

                        idxs = system_indices[c_mask]
                        q_converged[idxs, :q_current.shape[-1]] = q_current[
                            c_mask, :]
                        self.converged[idxs] = True

                        # If all systems have converged then the end of the SCC
                        # cycle has been reached.
                        if torch.all(c_mask):
                            break
                        # Otherwise there are still systems left to converge.
                        # Thus, the converged systems will now be culled to avoid
                        # over-converging them.
                        else:
                            # The order in which things are done here matters
                            # Cull calculator attributes
                            self.__cull(c_mask)
                            # Cull local variables
                            n_res = self.orbs.res_matrix_shape[-1]
                            system_indices = system_indices[~c_mask]
                            q_current = q_current[~c_mask, :n_res]
                            # Cull mixer
                            self.mixer.cull(c_mask, new_size=[n_res])

                else:
                    self.converged = torch.tensor(False)
                    if not self.suppress_SCF_error:
                        # Here a restore is performed before the error being
                        # raised to help with debugging.
                        self._geometry, self._orbs = c_geometry, c_orbs
                        self.overlap, self.gamma = c_overlap, c_gamma
                        self.invr, self.core_hamiltonian = c_invr,\
                            c_hamiltonian_copy

                        raise ConvergenceError(
                            "SCC cycle failed to converge; "
                            "iteration limit reached", self.converged)

                # Restore the calculator back to its state prior to culling.
                # Properties like `rho` and `eig_values` are not reset as it is
                # assumed that they will be overridden in the next stage.
                self._geometry, self._orbs = c_geometry, c_orbs
                self.overlap, self.gamma = c_overlap, c_gamma
                self.invr, self.core_hamiltonian = c_invr, c_hamiltonian_copy

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

    def __cull(self, mask: Tensor):
        """Cull converged systems from the calculator instance.

        Calling this method will strip a selection of systems from various
        components of the associated calculator instance. This is intended to
        be used to temporarily remove converged systems during the SCC cycle.
        However, as this only filters some, but not all, attributes its ill
        advised to use this anywhere other than the `.forward` method.

        Arguments:
            mask: A tensor of Booleans indicating which systems have converged.
                Systems which have converged will be masked partially out.

        Warnings:
            Do not invoke this function manually unless you are sure that you
            know what you are doing!
        """
        self._orbs = self.orbs[~mask]
        self._geometry = self.geometry[~mask]
        n_orbs = torch.max(self.orbs.n_orbitals)
        n_res = self.orbs.res_matrix_shape[-1]
        self._overlap = self._overlap[~mask, :n_orbs, :n_orbs]
        self._core_hamiltonian = self._core_hamiltonian[~mask, :n_orbs, :n_orbs]
        self._gamma = self._gamma[~mask, :n_res, :n_res]
        self._invr = self._invr[~mask, :n_res, :n_res]

    def _scc_cycle(self, q_in: Tensor) -> Tensor:
        """Perform a single self-consistent charge cycle.

        This method performs a single Self-Consistent Charge cycle (SCC). Using
        ``q_in`` as the initial guess during the first cycle and as the mixed
        charges in all subsequent cycles.

        Arguments:
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
            resolved, but must match up with that as defined by the orbs
            attribute `shell_resolved`.
        """

        # Construct the shift matrix
        shifts = torch.einsum(
            '...i,...ij->...j', q_in - self.q_zero_res, self.gamma)
        self._scc_energy = .5 * (shifts * (q_in - self.q_zero_res)).sum(-1)
        shifts = prepeat_interleave(shifts, self.orbs.orbs_per_res)
        shifts = (shifts[..., None] + shifts[..., None, :])
        
        print('-----------------------------------------')
        print('Shifts:', shifts)

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
        return _mulliken(self.rho, self.overlap, self.orbs)

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
        self._scc_energy = None
