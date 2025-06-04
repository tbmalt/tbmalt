# -*- coding: utf-8 -*-
"""Slater-Koster integral feed objects.

This contains all Slater-Koster integral feed objects. These objects are
responsible for generating the Slater-Koster integrals and for constructing
the associated Hamiltonian and overlap matrices.
"""
from __future__ import annotations
import warnings
import re
import numpy as np
from numpy import ndarray as Array
from itertools import combinations_with_replacement
from typing import List, Literal, Optional, Dict, Tuple, Union, Type
from scipy.interpolate import CubicSpline as ScipyCubicSpline
import torch
from torch import Tensor
from torch.nn import Parameter, ParameterDict, ModuleDict, Module

from tbmalt import Geometry, OrbitalInfo, Periodicity
from tbmalt.structures.geometry import atomic_pair_distances
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt.io.skf import Skf, VCRSkf
from tbmalt.physics.dftb.slaterkoster import sub_block_rot
from tbmalt.data.elements import chemical_symbols
from tbmalt.ml import Feed
from tbmalt.common.batch import pack, prepeat_interleave, bT, bT2
from tbmalt.common.maths.interpolation import PolyInterpU, BicubInterpSpl
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.common import unique

# Todo:
#   - Need to determine why this is so slow for periodic systems.

interp_dict = {'polynomial': PolyInterpU, 'spline': CubicSpline,
               'bicubic': BicubInterpSpl}


class ScipySkFeed(IntegralFeed):
    r"""Slater-Koster based Scipy integral feed for testing DFTB calculations.

    This feed uses Scipy splines & Slater-Koster transformations to construct
    Hamiltonian and overlap matrices via the traditional DFTB method. It is
    important to note that, due to the use of Scipy splines, this class and
    its methods are not backpropagatable. Thus, this feed should not be used
    for training.

    Arguments:
        on_sites: On-site integrals presented as a dictionary keyed by atomic
            numbers & valued by a tensor specifying all of associated the on-
            site integrals; i.e. one for each orbital.
        off_sites: Off-site integrals; dictionary keyed by tuples of the form
            (z₁, z₂, s₁, s₂), where zᵢ & sᵢ are the atomic & shell numbers of
            the interactions, & valued by Scipy `CubicSpline` entities. Note
            that z₁ must be less than or equal to z₂, see the notes section
            for further information.
        device: Device on which the feed object and its contents resides.
        dtype: dtype used by feed object.

    Notes:
        The splines contained within the ``off_sites`` argument must return
        all relevant bond order integrals; e.g. a s-s interaction should only
        return a single value for the σ interaction whereas a d-d interaction
        should return three values when interpolated (σ,π & δ).

        Furthermore it is critical that no duplicate interactions are present.
        That is to say if there is a (1, 6, 0, 0) (H[s]-C[s]) key present then
        there must not also be a (6, 1, 0, 0) (H[s]-C[s]) key present as they
        are the same interaction. To help prevent this the class will raise an
        error if the second atomic number is greater than the first; e.g. the
        key (6, 1, 0, 0) will raise an error but (1, 6, 0, 0) will not.

    Warnings:
        This integral feed is not backpropagatable as Scipy splines are used
        to interpolate the Slater-Koster tables. This is primarily indented to
        be used for testing purposes.

        `CubicSpline` instances should not attempt to extrapolate, but rather
        return NaNs, i.e. 'extrapolate=False'. When interpolating `ScipySkFeed`
        instances will identify and set all NaNs to zero.

    """

    # TODO: Remove this class

    def __init__(self, on_sites: Dict[int, Tensor],
                 off_sites: Dict[Tuple[int, int, int, int], ScipyCubicSpline],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):

        # Ensure that the on_sites are torch tensors
        if not isinstance(temp := list(on_sites.values())[0], Tensor):
            on_sites = {k: torch.tensor(v, dtype=dtype, device=device)
                        for k, v in on_sites.items()}

        # Validate the off-site keys
        if any(map(lambda k: k[0] > k[1], off_sites.keys())):
            ValueError('Lowest Z must be given first in off_site keys')

        # Pass the dtype and device to the ABC, if none are given the default
        super().__init__(temp.dtype if dtype is None else dtype,
                         temp.device if device is None else device)

        self.on_sites = on_sites
        self.off_sites = off_sites

    def _off_site_blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
                         geometry: Geometry, orbs: OrbitalInfo) -> Tensor:
        """Compute atomic interaction blocks (off-site only).

        Constructs the off-site atomic blocks using Slater-Koster integral
        tables.

        Arguments:
              atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
              atomic_idx_2: Indices of the 2'nd atom associated with each
                  desired interaction block.
              geometry: The systems to which the atomic indices relate.
              orbs: Orbital information associated with said systems.

          Returns:
              blocks: Requested atomic interaction sub-blocks.
        """
        # Identify atomic numbers associated with the interaction
        z_1 = int(geometry.atomic_numbers[*bT(atomic_idx_1[0: 1])])
        z_2 = int(geometry.atomic_numbers[*bT(atomic_idx_2[0: 1])])

        # Get the species' shell lists (basically a list of azimuthal numbers)
        shells_1, shells_2 = orbs.shell_dict[z_1], orbs.shell_dict[z_2]

        # Inter-atomic distance and distance vector calculator.
        dist_vec = (geometry.positions[*bT2(atomic_idx_2)]
                    - geometry.positions[*bT2(atomic_idx_1)])
        dist = torch.linalg.norm(dist_vec, dim=-1)
        u_vec = (dist_vec.T / dist).T

        # Work out the width of each sub-block then use it to get the row and
        # column index slicers for placing sub-blocks into their atom-blocks.
        rws, cws = np.array(shells_1) * 2 + 1, np.array(shells_2) * 2 + 1
        rows = [slice(i - j, i) for i, j in zip(rws.cumsum(), rws)]
        cols = [slice(i - j, i) for i, j in zip(cws.cumsum(), cws)]

        # Tensor to hold the resulting atomic-blocks.
        blks = torch.zeros(atomic_idx_1.shape[0], rws.sum(), cws.sum(),
                           dtype=self.dtype, device=self.device)

        # Loop over the 1st species' shells; where i & l_1 are the shell's index
        # & azimuthal numbers respectively. Then over the 2nd species' shells,
        # but ignore sub-blocks in the lower triangle of homo-atomic blocks as
        # they can be constructed via symmetrisation.
        for i, l_1 in enumerate(shells_1):
            o = i if z_1 == z_2 else 0
            for j, l_2 in enumerate(shells_2[o:], start=o):
                # Retrieve/interpolate the integral spline, remove any NaNs
                # due to extrapolation then convert to a torch tensor.
                inte = self.off_sites[(z_1, z_2, i, j)](dist.detach().cpu())
                inte[inte != inte] = 0.0
                inte = torch.tensor(inte, dtype=self.dtype, device=self.device)

                # Apply the Slater-Koster transformation
                inte = sub_block_rot(torch.tensor([l_1, l_2]), u_vec, inte)

                # Add the sub-blocks into their associated atom-blocks
                blks[:, rows[i], cols[j]] = inte

                # Add symmetrically equivalent sub-blocks (homo-atomic only)
                if z_1 == z_2 and i != j:
                    sign = (-1) ** (l_1 + l_2)
                    blks.transpose(-1, -2)[:, rows[i], cols[j]] = inte * sign

        return blks

    def _pe_blocks(self, atomic_idx_1: Array, atomic_idx_2: Array,
                   geometry: Geometry, orbs: OrbitalInfo,
                   periodic: Periodicity, **kwargs) -> Tensor:
        """Compute atomic interaction blocks (on-site and off-site) with pbc.

        Constructs the on-site and off-site atomic blocks using Slater-Koster
        integral tables for periodicity systems.

        Arguments:
              atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
              atomic_idx_2: Indices of the 2'nd atom associated with each
                  desired interaction block.
              geometry: The systems to which the atomic indices relate.
              orbs: Orbital information associated with said systems.
              periodic: Distance matrix and position vectors including periodicity
                  images.

          Returns:
              blocks: Requested atomic interaction sub-blocks.
        """

        # Check whether on-site block
        onsite = kwargs.get('onsite', False)

        # Check whether batch
        n_batch = (len(periodic.neighbour_vector)
                   if periodic.neighbour_vector.ndim == 5 else None)

        # Identify atomic numbers associated with the interaction
        z_1 = int(geometry.atomic_numbers[*bT(atomic_idx_1[0: 1])])
        z_2 = int(geometry.atomic_numbers[*bT(atomic_idx_2[0: 1])])

        # Get the species' shell lists (basically a list of azimuthal numbers)
        shells_1, shells_2 = orbs.shell_dict[z_1], orbs.shell_dict[z_2]

        # Inter-atomic distance and distance vector calculator.
        if n_batch is None:  # -> single
            dist_vec = periodic.neighbour_vector[:, atomic_idx_1, atomic_idx_2]
        else:  # -> batch
            # Split the atomic index due to batch
            sys_idx, idx = np.unique(atomic_idx_1[:, 0], return_index=True)
            _idx_1_split = np.split(atomic_idx_1[:, 1], idx[1:])
            _idx_2_split = np.split(atomic_idx_2[:, 1], idx[1:])

            # Distance vectors are packed for batch
            dist_vec, mask_pack = pack([
                    periodic.neighbour_vector[
                        sys_idx[ibatch]][:, _idx_1_split[ibatch], _idx_2_split[
                            ibatch]]
                    for ibatch in range(sys_idx.size)],
                        value=1e3, return_mask=True)

            # Reduce the dimension of batch
            dist_vec = dist_vec.flatten(-4, -3)

            # Mask to select items before padding with correct size
            mask_pack = mask_pack[..., 0][:, 0]

        # Number of images
        ncell = periodic.neighbour_vector.size(-4)

        # Distance matrix
        dist = torch.linalg.norm(dist_vec, dim=-1)

        # Mask for zero-distance terms in on-site block
        dist[dist == 0] = 99

        # Reduce the dimension of image
        dist_vec = dist_vec.flatten(-3, -2)
        dist = dist.flatten(-2, -1)
        u_vec = (dist_vec.T / dist).T

        # Work out the width of each sub-block then use it to get the row and
        # column index slicers for placing sub-blocks into their atom-blocks.
        rws, cws = np.array(shells_1) * 2 + 1, np.array(shells_2) * 2 + 1
        rows = [slice(i - j, i) for i, j in zip(rws.cumsum(), rws)]
        cols = [slice(i - j, i) for i, j in zip(cws.cumsum(), cws)]

        # Tensor to hold the resulting atomic-blocks.
        blks = torch.zeros(atomic_idx_1.shape[0], rws.sum(), cws.sum(),
                           dtype=self.dtype, device=self.device)

        # Loop over the 1st species' shells; where i & l_1 are the shell's index
        # & azimuthal numbers respectively. Then over the 2nd species' shells,
        # but ignore sub-blocks in the lower triangle of homo-atomic blocks as
        # they can be constructed via symmetrisation.
        for i, l_1 in enumerate(shells_1):
            o = i if z_1 == z_2 else 0
            for j, l_2 in enumerate(shells_2[o:], start=o):
                # Retrieve/interpolate the integral spline, remove any NaNs
                # due to extrapolation then convert to a torch tensor.
                inte = self.off_sites[(z_1, z_2, i, j)](dist.detach().cpu())
                inte[inte != inte] = 0.0
                inte = torch.tensor(inte, dtype=self.dtype, device=self.device)

                # Apply the Slater-Koster transformation
                inte = sub_block_rot(torch.tensor([l_1, l_2]), u_vec, inte)

                # Reshape the integral for images and sum together
                if n_batch is None:
                    _inte = inte.view(ncell, -1, inte.size(-2),
                                      inte.size(-1)).sum(-4)
                else:
                    _inte = inte.view(sys_idx.size, ncell, -1, inte.size(-2),
                                      inte.size(-1)).sum(-4)[mask_pack]

                # Add the sub-blocks into their associated atom-blocks
                blks[:, rows[i], cols[j]] = _inte

                # Add symmetrically equivalent sub-blocks (homo-atomic only)
                if z_1 == z_2 and i != j:
                    if onsite:
                        blks.transpose(-1, -2)[:, rows[i], cols[j]] = _inte
                    else:
                        sign = (-1) ** (l_1 + l_2)
                        blks.transpose(-1, -2)[:, rows[i], cols[j]] = _inte * sign

        return blks

    def blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
               geometry: Geometry, orbs: OrbitalInfo, **kwargs) -> Tensor:
        r"""Compute atomic interaction blocks using SK-integral tables.

          Returns the atomic blocks associated with the atoms in ``atomic_idx_1``
          interacting with those in ``atomic_idx_2`` splines and Slater-Koster
          transformations. This is the base method used in DFTB calculations.

          Note that The № of interaction blocks returned will be equal to the
          length of the two index lists; i.e. *not* one for every combination.

          Arguments:
              atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
              atomic_idx_2: Indices of the 2'nd atom associated with each
                  desired interaction block.
              geometry: The systems to which the atomic indices relate.
              orbs: Orbital information associated with said systems.

          Returns:
              blocks: Requested atomic interaction sub-blocks.

          Warnings:
              This is not backpropagatable.

        """
        # Get the atomic numbers of the atoms
        zs = geometry.atomic_numbers
        zs_1 = zs[*bT2(atomic_idx_1)]
        zs_2 = zs[*bT2(atomic_idx_2)]

        # Ensure all interactions are between identical species pairs.
        if len(zs_1.unique()) != 1:
            raise ValueError('Atoms in atomic_idx_1 must be the same species')

        if len(zs_2.unique()) != 1:
            raise ValueError('Atoms in atomic_idx_2 must be the same species')

        # Atomic numbers of the species in list 1 and 2
        z_1, z_2 = zs_1[0], zs_2[0]

        # C-N and N-C are the same interaction: choice has been made to have
        # only one set of splines for each species pair. Thus, the two lists
        # may need to be swapped.
        if z_1 > z_2:
            atomic_idx_1, atomic_idx_2 = atomic_idx_2, atomic_idx_1
            z_1, z_2 = z_2, z_1
            flip = True
        else:
            flip = False

        # Construct the tensor into which results are to be placed
        n_rows, n_cols = orbs.n_orbs_on_species(torch.stack((z_1, z_2)))
        blks = torch.empty(len(atomic_idx_1), n_rows, n_cols, dtype=self.dtype,
                           device=self.device)

        # Identify which are on-site blocks and which are off-site
        on_site = self._partition_blocks(atomic_idx_1, atomic_idx_2)
        mask_shell = torch.zeros_like(self.on_sites[int(z_1)]).bool()
        mask_shell[:(torch.arange(len(orbs.shell_dict[int(z_1)]))
                     * 2 + 1).sum()] = True

        if any(on_site):  # Construct the on-site blocks (if any are present)
            blks[on_site] = torch.diag(self.on_sites[int(z_1)][mask_shell])

            # Interactions between images need to be considered for on-site
            # blocks with pbc.
            if geometry.periodicity is not None:
                _on_site = self._pe_blocks(
                    atomic_idx_1[on_site], atomic_idx_2[on_site],
                    geometry, orbs, geometry.periodicity, onsite=True)
                blks[on_site] = blks[on_site] + _on_site

        if any(~on_site):  # Then the off-site blocks
            if geometry.periodicity is None:
                blks[~on_site] = self._off_site_blocks(
                    atomic_idx_1[~on_site], atomic_idx_2[~on_site],
                    geometry, orbs)
            else:
                blks[~on_site] = self._pe_blocks(
                    atomic_idx_1[~on_site], atomic_idx_2[~on_site],
                    geometry, orbs, geometry.periodicity)

        if flip:  # If the atoms were switched, then a transpose is required.
            blks = blks.transpose(-1, -2)

        return blks

    @classmethod
    def from_database(
            cls, path: str, species: List[int],
            target: Literal['hamiltonian', 'overlap'],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> 'ScipySkFeed':
        r"""Instantiate instance from an HDF5 database of Slater-Koster files.

        Instantiate a `ScipySkFeed` instance for the specified elements using
        integral tables contained within a Slater-Koster HDF5 database.

        Arguments:
            path: Path to the HDF5 file from which integrals should be taken.
            species: Integrals will only be loaded for the requested species.
            target: Specifies which integrals should be loaded options are
                "hamiltonian" and "overlap".
            device: Device on which the feed object and its contents resides.
            dtype: dtype used by feed object.

        Returns:
            sk_feed: A `ScipySkFeed` instance with the requested integrals.

        Notes:
            This method interpolates off-site integrals with `CubicSpline`
            instances.

            This method will not instantiate `ScipySkFeed` instances directly
            from human-readable skf files, or a directory thereof. Thus, any
            such files must first be converted into their binary equivalent.
            This reduces overhead & file format error instabilities. The code
            block provide below shows how this can be done:

            >>> from tbmalt.tools.downloaders import download_dftb_parameter_set
            >>> url = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
            >>> path = "auorg.h5"
            >>> download_dftb_parameter_set(url, path)

        Examples:
            >>> from tbmalt import OrbitalInfo, Geometry
            >>> from tbmalt.physics.dftb.feeds import ScipySkFeed
            >>> from tbmalt.io.skf import Skf
            >>> from tbmalt.tools.downloaders import download_dftb_parameter_set
            >>> from ase.build import molecule
            >>> import torch
            >>> torch.set_default_dtype(torch.float64)

            # Download the auorg-1-1 parameter set
            >>> url = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
            >>> path = "auorg.h5"
            >>> download_dftb_parameter_set(url, path)

            # Preparation of system to calculate
            >>> geo = Geometry.from_ase_atoms(molecule('H2'))
            >>> orbs = OrbitalInfo(geo.atomic_numbers,
                                     shell_dict={1: [0]})

            # Definition of feeds
            >>> h_feed = ScipySkFeed.from_database(path, [1], 'hamiltonian')
            >>> s_feed = ScipySkFeed.from_database(path, [1], 'overlap')

            # Matrix elements
            >>> H = h_feed.matrix(geo, orbs)
            >>> S = s_feed.matrix(geo, orbs)
            >>> print(H)
            tensor([[-0.2386, -0.3211],
                    [-0.3211, -0.2386]])
            >>> print(S)
            tensor([[1.0000, 0.6433],
                    [0.6433, 1.0000]])

        """
        # As C-H & C-H interactions are the same only one needs to be loaded.
        # Thus, only off-site interactions where z₁≤z₂ are generated. However,
        # integrals are split over the two Slater-Koster tables, thus both must
        # be loaded.

        def clip(x, y):
            # Removes leading zeros from the sk data which may cause errors
            # when fitting the CubicSpline.
            start = torch.nonzero(y.sum(0), as_tuple=True)[0][0]
            return x[start:], y[:, start:].T  # Transpose here to save effort

        if device and device.type == 'cuda':
            raise TypeError(
                "`ScipySkFeed` instances do not offer CUDA support as they are"
                " backed by Scipy splines which are cpu only.5")

        # Ensure a valid target is selected
        if target not in ['hamiltonian', 'overlap']:
            ValueError('Invalid target selected; '
                       'options are "hamiltonian" or "overlap"')

        on_sites, off_sites = {}, {}

        # The species list must be sorted to ensure that the lowest atomic
        # number comes first in the species pair.
        for pair in combinations_with_replacement(sorted(species), 2):
            skf = Skf.read(path, pair, device=device)
            # Loop over the off-site interactions & construct the splines.
            for key, value in skf.__getattribute__(target).items():
                off_sites[pair + key] = ScipyCubicSpline(
                    *clip(skf.grid, value), extrapolate=False)

            # The X-Y.skf file may not contain all information. Thus some info
            # must be loaded from its Y-X counterpart.
            if pair[0] != pair[1]:
                skf_2 = Skf.read(path, tuple(reversed(pair)), device=device)
                for key, value in skf_2.__getattribute__(target).items():
                    if key[0] < key[1]:
                        off_sites[pair + (*reversed(key),)] = ScipyCubicSpline(
                            *clip(skf_2.grid, value), extrapolate=False)

            else:  # Construct the onsite interactions
                # Repeated so there's 1 value per orbital not just per shell.
                on_sites_vals = skf.on_sites.repeat_interleave(
                    torch.arange(len(skf.on_sites)) * 2 + 1).to(device)

                if target == 'overlap':  # use an identify matrix for S
                    on_sites_vals = torch.ones_like(on_sites_vals, device=device)

                on_sites[pair[0]] = on_sites_vals

        return cls(on_sites, off_sites, dtype, device)

    def __str__(self):
        elements = ', '.join([
            chemical_symbols[i]for i in sorted(self.on_sites.keys())])
        return f'{self.__class__.__name__}({elements})'

    def __repr__(self):
        return str(self)


class SkFeed(IntegralFeed):
    r"""Slater-Koster based integral feed for DFTB calculations.

    This feed uses polynomial and cubic-spline interpolators in tandem with
    Slater-Koster transformations to construct Hamiltonian and overlap matrices
    in line with the traditional DFTB method.

    Arguments:
        on_sites: A torch `ParameterDict` where keys represent atomic numbers
            as strings, and values are torch parameters specifying the on-site
            energies for each orbital.
        off_sites: A torch `ModuleDict` containing the off-site integrals
            required for constructing Hamiltonian and/or overlap matrices.
            The keys are strings representing tuples in the format
            `"(z₁, z₂, s₁, s₂)"`, where `z₁` & `z₂` are the atomic numbers
            of the interacting atoms (with `z₁ ≤ z₂`), and `s₁` & `s₂` are
            their respective shell numbers. Keys must exactly match the
            string obtained from converting a tuple to a string, including
            the parentheses and spaces; for example, `"(1, 6, 0, 0)"`. The
            values are interpolation modules, such as `CubicSpline`
            instances, that provide the bond order integrals for the
            specified interactions.
        device: Device on which the feed object and its contents resides.
        dtype: dtype used by feed object.

    Notes:
        The splines contained within the ``off_sites`` argument must return
        all relevant bond order integrals; e.g. an s-s interaction should only
        return a single value for the σ interaction whereas a d-d interaction
        should return three values when interpolated (σ, π & δ).

        Furthermore, it is critical that no duplicate interactions are present.
        That is to say if there is a (1, 6, 0, 0) (H[s]-C[s]) key present then
        there must not also be a (6, 1, 0, 0) (H[s]-C[s]) key present as they
        are the same interaction. To help prevent this the class will raise an
        error if the second atomic number is greater than the first; e.g. the
        key (6, 1, 0, 0) will raise an error but (1, 6, 0, 0) will not.

    Warnings:
        `CubicSpline` instances should not attempt to extrapolate, but rather
        return NaNs, i.e. 'extrapolate=False'. When interpolating `SkFeed`
        instances will identify and set all NaNs to zero.

    """
    def __init__(self, on_sites: ParameterDict[str, Parameter],
                 off_sites: ModuleDict[str, Module],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):

        # Pass the dtype and device to the ABC, if none are given the default
        temp = next(iter(on_sites.values()))
        super().__init__(temp.dtype if dtype is None else dtype,
                         temp.device if device is None else device)

        if isinstance(on_sites, dict):
            raise TypeError(
                "An instance of `torch.nn.ParameterDict` was expected for the "
                "attribute `on_sites`, but a standard Python `dict` was "
                "received.")

        if not isinstance(off_sites, ModuleDict):
            raise TypeError(
                f"An instance of `torch.nn.ModuleDict` was expected for the "
                f"attribute `off_sites`, but a/an `{type(off_sites)}` was "
                "received.")

        for key in off_sites.keys():
            self.__validate_key(key)

        self._on_sites: ParameterDict[str, Parameter] = on_sites
        self._off_sites = off_sites

    @property
    def on_sites(self) -> ParameterDict[str, Parameter]:
        """Parameter dictionary of on-site values."""
        return self._on_sites

    @on_sites.setter
    def on_sites(self, value: ParameterDict[str, Parameter]):

        # Ensure that the on-site terms are supplied via a `ParameterDict` and
        # not a standard Python `dict`. Using a standard `dict` would modify
        # the behaviour of the feed in unexpected ways.
        if isinstance(value, dict):
            raise TypeError(
                "An instance of `torch.nn.ParameterDict` was expected for the "
                "attribute `on_sites`, but a standard Python `dict` was "
                "received.")

        self._on_sites = value

    @property
    def off_sites(self) -> ModuleDict[str, Module]:
        """Module dictionary of off-site feed modules."""
        return self._off_sites

    @off_sites.setter
    def off_sites(self, value: ModuleDict[str, Module]):

        if not isinstance(value, ModuleDict):
            raise TypeError(
                f"An instance of `torch.nn.ModuleDict` was expected for the "
                f"attribute `off_sites`, but a/an `{type(value)}` was "
                "received.")

        for key in value.keys():
            self.__validate_key(key)

        self._off_sites = value

    def _off_site_blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
                         geometry: Geometry, orbs: OrbitalInfo,
                         shift_vec: Optional[Tensor] = None, **kwargs
                         ) -> Tensor:
        """Compute atomic interaction blocks (off-site only).

        Constructs the off-site atomic blocks using Slater-Koster integral
        tables.

        Arguments:
              atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
              atomic_idx_2: Indices of the 2'nd atom associated with each
                  desired interaction block.
              geometry: The systems to which the atomic indices relate.
              orbs: Orbital information associated with said systems.
              shift_vec: Vector to shift the distance vector by. [DEFAULT=`None`]
          Returns:
              blocks: Requested atomic interaction sub-blocks.
        """
        # Identify atomic numbers associated with the interaction
        z_1 = int(geometry.atomic_numbers[*bT(atomic_idx_1[0: 1])])
        z_2 = int(geometry.atomic_numbers[*bT(atomic_idx_2[0: 1])])

        # Get the species' shell lists (basically a list of azimuthal numbers)
        shells_1, shells_2 = orbs.shell_dict[z_1], orbs.shell_dict[z_2]

        # Inter-atomic distance and distance vector calculator.
        dist_vec = (geometry.positions[*bT2(atomic_idx_2)]
                    - geometry.positions[*bT2(atomic_idx_1)])
        if shift_vec is not None:
            dist_vec = dist_vec + shift_vec
        dist = torch.linalg.norm(dist_vec, dim=-1)
        u_vec = (dist_vec.T / dist).T

        # Work out the width of each sub-block then use it to get the row and
        # column index slicers for placing sub-blocks into their atom-blocks.
        rws, cws = np.array(shells_1) * 2 + 1, np.array(shells_2) * 2 + 1
        rows = [slice(i - j, i) for i, j in zip(rws.cumsum(), rws)]
        cols = [slice(i - j, i) for i, j in zip(cws.cumsum(), cws)]

        # Tensor to hold the resulting atomic-blocks.
        blks = torch.zeros(atomic_idx_1.shape[0], rws.sum(), cws.sum(),
                           dtype=self.dtype, device=self.device)

        # Loop over the 1st species' shells; where i & l_1 are the shell's index
        # & azimuthal numbers respectively. Then over the 2nd species' shells,
        # but ignore sub-blocks in the lower triangle of homo-atomic blocks as
        # they can be constructed via symmetrisation.
        for i, l_1 in enumerate(shells_1):
            o = i if z_1 == z_2 else 0
            for j, l_2 in enumerate(shells_2[o:], start=o):
                # Retrieve/interpolate the integral spline, remove any NaNs
                # due to extrapolation then convert to a torch tensor.
                inte = self._off_sites[str((z_1, z_2, i, j))].forward(dist)

                # Apply the Slater-Koster transformation
                inte = sub_block_rot(torch.tensor([l_1, l_2]), u_vec, inte)

                # Add the sub-blocks into their associated atom-blocks
                blks[:, rows[i], cols[j]] = inte

                # Add symmetrically equivalent sub-blocks (homo-atomic only)
                if z_1 == z_2 and i != j:
                    sign = (-1) ** (l_1 + l_2)
                    blks.transpose(-1, -2)[:, rows[i], cols[j]] = inte * sign

        return blks

    def _pe_blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
                   geometry: Geometry, orbs: OrbitalInfo,
                   periodic: Periodicity, onsite: bool = False) -> Tensor:
        """Compute atomic interaction blocks (on-site and off-site) with pbc.

        Constructs the on-site and off-site atomic blocks using Slater-Koster
        integral tables for periodicity systems.

        Arguments:
            atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
            atomic_idx_2: Indices of the 2'nd atom associated with each
                desired interaction block.
            geometry: The systems to which the atomic indices relate.
            orbs: Orbital information associated with said systems.
            periodic: Periodic object containing distance matrix and position
                vectors for periodic images.
            onsite: Used to signal that the provided blocks represent
                on-site interactions. [DEFAULT=`False`]

          Returns:
              blocks: Requested atomic interaction sub-blocks.
        """

        # Check whether batch
        n_batch = (len(periodic.neighbour_vector)
                   if periodic.neighbour_vector.ndim == 5 else None)

        # Identify atomic numbers associated with the interaction
        z_1 = int(geometry.atomic_numbers[*bT(atomic_idx_1[0: 1])])
        z_2 = int(geometry.atomic_numbers[*bT(atomic_idx_2[0: 1])])

        # Get the species' shell lists (basically a list of azimuthal numbers)
        shells_1, shells_2 = orbs.shell_dict[z_1], orbs.shell_dict[z_2]

        # Inter-atomic distance and distance vector calculator.
        if n_batch is None:  # -> single
            dist_vec = periodic.neighbour_vector[:, atomic_idx_1, atomic_idx_2]
        else:  # -> batch
            # Split the atomic index due to batch
            sys_idx, idx = unique(atomic_idx_1[:, 0], return_index=True)
            # Convert locations at which a split should be made into bucket
            # size values, as needed by PyTorch.
            idx = idx.diff().cpu().tolist()
            idx.append(atomic_idx_2.shape[-2] - sum(idx))
            _idx_1_split = torch.split(atomic_idx_1[:, 1], idx)
            _idx_2_split = torch.split(atomic_idx_2[:, 1], idx)

            # Distance vectors are packed for batch
            dist_vec, mask_pack = pack([
                    periodic.neighbour_vector[
                        sys_idx[ibatch]][:, _idx_1_split[ibatch], _idx_2_split[
                            ibatch]]
                    for ibatch in range(len(sys_idx))],
                        value=1e3, return_mask=True)

            # Reduce the dimension of batch
            dist_vec = dist_vec.flatten(-4, -3)

            # Mask to select items before padding with correct size
            mask_pack = mask_pack[..., 0][:, 0]

        # Number of images
        ncell = periodic.neighbour_vector.size(-4)

        # Distance matrix
        dist = torch.linalg.norm(dist_vec, dim=-1)

        # Mask for zero-distance terms in on-site block
        dist[dist == 0] = 99

        # Reduce the dimension of image
        dist_vec = dist_vec.flatten(-3, -2)
        dist = dist.flatten(-2, -1)
        u_vec = (dist_vec.T / dist).T

        # Work out the width of each sub-block then use it to get the row and
        # column index slicers for placing sub-blocks into their atom-blocks.
        rws, cws = np.array(shells_1) * 2 + 1, np.array(shells_2) * 2 + 1
        rows = [slice(i - j, i) for i, j in zip(rws.cumsum(), rws)]
        cols = [slice(i - j, i) for i, j in zip(cws.cumsum(), cws)]

        # Tensor to hold the resulting atomic-blocks.
        blks = torch.zeros(atomic_idx_1.shape[0], rws.sum(), cws.sum(),
                           dtype=self.dtype, device=self.device)

        # Loop over the 1st species' shells; where i & l_1 are the shell's index
        # & azimuthal numbers respectively. Then over the 2nd species' shells,
        # but ignore sub-blocks in the lower triangle of homo-atomic blocks as
        # they can be constructed via symmetrisation.
        for i, l_1 in enumerate(shells_1):
            o = i if z_1 == z_2 else 0
            for j, l_2 in enumerate(shells_2[o:], start=o):
                # Retrieve/interpolate the integral spline, remove any NaNs
                # due to extrapolation then convert to a torch tensor.
                inte = self._off_sites[str((z_1, z_2, i, j))].forward(dist)
                inte[inte != inte] = 0.0

                # Apply the Slater-Koster transformation
                inte = sub_block_rot(torch.tensor([l_1, l_2]), u_vec, inte)

                # Reshape the integral for images and sum together
                if n_batch is None:
                    _inte = inte.view(ncell, -1, inte.size(-2),
                                      inte.size(-1)).sum(-4)
                else:
                    _inte = inte.view(len(sys_idx), ncell, -1, inte.size(-2),
                                      inte.size(-1)).sum(-4)[mask_pack]

                # 5, ncell, -1, 1, 1
                # Add the sub-blocks into their associated atom-blocks
                blks[:, rows[i], cols[j]] = _inte

                # Add symmetrically equivalent sub-blocks (homo-atomic only)
                if z_1 == z_2 and i != j:
                    if onsite:
                        blks.transpose(-1, -2)[:, rows[i], cols[j]] = _inte
                    else:
                        sign = (-1) ** (l_1 + l_2)
                        blks.transpose(-1, -2)[:, rows[i], cols[j]] = _inte * sign

        return blks

    def blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
               geometry: Geometry, orbs: OrbitalInfo, **kwargs) -> Tensor:
        r"""Compute atomic interaction blocks using SK-integral tables.

        Returns the atomic blocks associated with the atoms in ``atomic_idx_1``
        interacting with those in ``atomic_idx_2`` splines and Slater-Koster
        transformations. This is the base method used in DFTB calculations.
        Note that The № of interaction blocks returned will be equal to the
        length of the two index lists; i.e. *not* one for every combination.

        Arguments:
            atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
            atomic_idx_2: Indices of the 2'nd atom associated with each
                desired interaction block.
            geometry: The systems to which the atomic indices relate.
            orbs: Orbital information associated with said systems.

        Returns:
            blocks: Requested atomic interaction sub-blocks.

        """
        # Get the atomic numbers of the atoms
        zs = geometry.atomic_numbers
        zs_1 = zs[*bT2(atomic_idx_1)]
        zs_2 = zs[*bT2(atomic_idx_2)]

        # Ensure all interactions are between identical species pairs.
        if len(zs_1.unique()) != 1:
            raise ValueError('Atoms in atomic_idx_1 must be the same species')

        if len(zs_2.unique()) != 1:
            raise ValueError('Atoms in atomic_idx_2 must be the same species')

        # Atomic numbers of the species in list 1 and 2
        z_1, z_2 = zs_1[0], zs_2[0]

        # C-N and N-C are the same interaction: choice has been made to have
        # only one set of splines for each species pair. Thus, the two lists
        # may need to be swapped.
        if z_1 > z_2:
            atomic_idx_1, atomic_idx_2 = atomic_idx_2, atomic_idx_1
            z_1, z_2 = z_2, z_1
            flip = True
        else:
            flip = False

        # Construct the tensor into which results are to be placed
        n_rows, n_cols = orbs.n_orbs_on_species(torch.stack((z_1, z_2)))
        blks = torch.zeros(len(atomic_idx_1), n_rows, n_cols, dtype=self.dtype,
                           device=self.device)

        # Identify which are on-site blocks and which are off-site
        on_site = self._partition_blocks(atomic_idx_1, atomic_idx_2)
        mask_shell = torch.zeros_like(self.on_sites[str(z_1.item())]).bool()
        mask_shell[:(torch.arange(len(orbs.shell_dict[z_1.item()]))
                     * 2 + 1).sum()] = True

        # Construct the on-site blocks (if any are present)
        if any(on_site):
            blks[on_site] = torch.diag(self.on_sites[str(z_1.item())][mask_shell])

            # Interactions between images need to be considered for on-site
            # blocks with pbc.
            if geometry.periodicity is not None:
                _on_site = self._pe_blocks(
                    atomic_idx_1[on_site], atomic_idx_2[on_site],
                    geometry, orbs, geometry.periodicity, onsite=True)
                blks[on_site] = blks[on_site] + _on_site

        if any(~on_site):  # Then the off-site blocks
            if geometry.periodicity is None:
                blks[~on_site] = self._off_site_blocks(
                    atomic_idx_1[~on_site], atomic_idx_2[~on_site],
                    geometry, orbs)
            else:
                blks[~on_site] = self._pe_blocks(
                    atomic_idx_1[~on_site], atomic_idx_2[~on_site],
                    geometry, orbs, geometry.periodicity)

        if flip:  # If the atoms were switched, then a transpose is required.
            blks = blks.transpose(-1, -2)

        return blks

    @classmethod
    def from_database(
            cls, path: str, species: List[int],
            target: Literal['hamiltonian', 'overlap'],
            interpolation: Type[Feed] = PolyInterpU,
            requires_grad_onsite: bool = False,
            requires_grad_offsite: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> 'SkFeed':
        r"""Instantiate instance from an HDF5 database of Slater-Koster files.

        Instantiate a `SkFeed` instance for the specified elements using
        integral tables contained within a Slater-Koster HDF5 database.

        Arguments:
            path: Path to the HDF5 file from which integrals should be taken.
            species: Integrals will only be loaded for the requested species.
            target: Specifies which integrals should be loaded, options are
                "hamiltonian" and "overlap".
            interpolation: The `Feed` derived class that should be used to
                interpolate the off-site Slater-Koster parameters. This may
                be `CubicSpline`, `PolyInterpU` or any other `Feed` derived
                class that can perform univarient interpolation. If this is
                not manually specified then it will default to `PolyInterpU`
                which is a PyTorch implimentation of the interpolation method
                used in DFTB+. [DEFAULT=`PolyInterpU`]
            requires_grad_onsite: When set to `True` gradient tracking will be
                enabled for the on-site parameters. This flag is ignored for
                the overlap matrix case as its on-site terms are always unity.
                [DEFAULT=`False`]
            requires_grad_offsite: When set to `True` gradient tracking will
                be enabled for the off-site parameters. [DEFAULT=`False`]
            device: Device on which the feed object and its contents resides.
            dtype: dtype used by feed object.

        Returns:
            sk_feed: A `SkFeed` instance with the requested integrals.

        Notes:
            This method will not instantiate `SkFeed` instances directly
            from human-readable skf files, or a directory thereof. Thus, any
            such files must first be converted into their binary equivalent.
            This reduces overhead & file format error instabilities. The code
            block provide below shows how this can be done:

            >>> from tbmalt.io.skf import Skf
            >>> Zs = ['H', 'C', 'Au', 'S']
            >>> for file in [f'{i}-{j}.skf' for i in Zs for j in Zs]:
            ...     Skf.read(file).write('my_skf.hdf5')

        Examples:
            >>> from tbmalt import OrbitalInfo, Geometry
            >>> from tbmalt.physics.dftb.feeds import SkFeed
            >>> from tbmalt.io.skf import Skf
            >>> from ase.build import molecule
            >>> import urllib
            >>> import tarfile
            >>> from os.path import join
            >>> import torch
            >>> torch.set_default_dtype(torch.float64)

            # Link to the auorg-1-1 parameter set
            >>> link = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'

            # Preparation of sk file
            >>> elements = ['H', 'C', 'O', 'Au', 'S']
            >>> tmpdir = './'
            >>> urllib.request.urlretrieve(
            ...     link, path := join(tmpdir, 'auorg-1-1.tar.xz'))
            >>> with tarfile.open(path) as tar:
            ...     tar.extractall(tmpdir)
            >>> skf_files = [join(tmpdir, 'auorg-1-1', f'{i}-{j}.skf')
            ...              for i in elements for j in elements]
            >>> for skf_file in skf_files:
            ...     Skf.read(skf_file).write(path := join(tmpdir,
            ...                                           'auorg.hdf5'))

            # Preparation of system to calculate
            >>> geo = Geometry.from_ase_atoms(molecule('H2'))
            >>> orbs = OrbitalInfo(geo.atomic_numbers,
            ...                    shell_dict={1: [0]})

            # Definition of feeds
            >>> h_feed = SkFeed.from_database(path, [1], 'hamiltonian')
            >>> s_feed = SkFeed.from_database(path, [1], 'overlap')

            # Matrix elements
            >>> H = h_feed.matrix(geo, orbs)
            >>> S = s_feed.matrix(geo, orbs)
            >>> print(H)
            tensor([[-0.2386, -0.3211],
                    [-0.3211, -0.2386]])
            >>> print(S)
            tensor([[1.0000, 0.6433],
                    [0.6433, 1.0000]])

        """
        # As C-H & C-H interactions are the same only one needs to be loaded.
        # Thus, only off-site interactions where z₁≤z₂ are generated. However,
        # integrals are split over the two Slater-Koster tables, thus both must
        # be loaded.

        def construct_interpolator(
                grid, interaction_value, interpolation_type,
                grad, **extra_kwargs):
            # Clip the knot values so that any leading zeros are removed from
            # the Slater-Koster data. This is done to prevent some issues that
            # results from fitting splines to the knots with leading zeros.
            start = torch.nonzero(interaction_value.sum(0), as_tuple=True)[0][0]
            grid = grid[start:]
            interaction_value = interaction_value[:, start:].transpose(0, 1)

            # Convert the y-knot values to a `torch.nn.Parameter`
            # enable gradient tracking as required.
            interaction_value = Parameter(
                interaction_value, requires_grad=grad)

            # Finally construct and return the interpolator
            return interpolation_type(grid, interaction_value, **extra_kwargs)

        # Ensure a valid target is selected
        if target not in ['hamiltonian', 'overlap']:
            ValueError('Invalid target selected; '
                       'options are "hamiltonian" or "overlap"')

        on_sites = ParameterDict()
        off_sites = ModuleDict()
        params = {'extrapolate': False} if interpolation is ScipyCubicSpline else {}

        # The species list must be sorted to ensure that the lowest atomic
        # number comes first in the species pair.
        for pair in combinations_with_replacement(sorted(species), 2):

            skf = Skf.read(path, pair, device=device, dtype=dtype)

            # Loop over the off-site interactions & construct the splines.
            for key, value in skf.__getattribute__(target).items():
                off_sites[str(pair + key)] = construct_interpolator(
                    skf.grid, value, interpolation, requires_grad_offsite,
                    **params)

            # The X-Y.skf file may not contain all information. Thus, some info
            # must be loaded from its Y-X counterpart.
            if pair[0] != pair[1]:
                skf_2 = Skf.read(path, tuple(reversed(pair)), device=device, dtype=dtype)
                for key, value in skf_2.__getattribute__(target).items():
                    if key[0] < key[1]:
                        name = str(pair + (*reversed(key),))
                        off_sites[name] = construct_interpolator(
                            skf_2.grid, value, interpolation, requires_grad_offsite,
                            **params)

            else:  # Construct the onsite interactions
                # Repeated so there's 1 value per orbital not just per shell.
                on_sites_vals = Parameter(
                    skf.on_sites.repeat_interleave(
                        torch.arange(len(skf.on_sites), device=device) * 2 + 1),
                    requires_grad=requires_grad_onsite)

                if target == 'overlap':  # use an identity matrix for S
                    # Auto-grad tracking is always disabled as the values can
                    # never be anything other than one.
                    on_sites_vals = Parameter(
                        torch.ones_like(on_sites_vals, dtype=dtype, device=device),
                        requires_grad=False)

                on_sites[str(pair[0])] = on_sites_vals

        return cls(on_sites, off_sites, dtype, device)

    def __str__(self):
        elements = ', '.join([
            chemical_symbols[i] for i in
            sorted([int(j) for j in self.on_sites.keys()])
        ])
        return f'{self.__class__.__name__}({elements})'

    def __repr__(self):
        return str(self)

    @staticmethod
    def __validate_key(key_string):
        """Validate format of an off-site module-dictionary key.

        Validates that the provided string matches the expected format of a tuple
        converted to a string, i.e., "(z₁, z₂, s₁, s₂)".

        The string must:
        - Start with a left parenthesis '('
        - End with a right parenthesis ')'
        - Contain exactly four comma-separated integers
        - Have exactly one space after each comma
        - Ensure that z₁ ≤ z₂

        Arguments:
            key_string: The string to validate.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        errors = []

        # Check if the string starts with '(' and ends with ')'
        if not (key_string.startswith('(') and key_string.endswith(')')):
            errors.append("The string must start with '(' and end with ')'.")
        else:
            # Remove the parentheses
            content = key_string[1:-1]

            # Define the expected pattern
            pattern = r'^(\d+), (\d+), (\d+), (\d+)$'
            match = re.match(pattern, content)
            if not match:
                errors.append(
                    "The string must contain exactly four comma-separated integers, "
                    "with exactly one space after each comma.")
            else:
                z1, z2, s1, s2 = map(int, match.groups())

                # Check if z₁ ≤ z₂
                if z1 > z2:
                    errors.append(
                        "The first atomic number z₁ must be less than or equal to "
                        "the second atomic number z₂.")

        if errors:
            errors.insert(
                0,
                f"The string \"{key_string}\" used as a key in the `off_sites` "
                f"dictionary does not match the expected format. It should "
                f"represent a tuple converted to a string, like '(1, 6, 0, 0)'.")

            raise ValueError('\n'.join(errors))


class VcrSkFeed(IntegralFeed):
    r"""Variable compression radius based DFTB Slater-Koster integral feed.

    This feed is similar in behaviour to the `SkFeed` but with the ability to
    dynamically change the compression radius in an ad-hoc manner; effectively
    allowing one to smoothly glide from one "standard" parameter set to
    another by adjusting the compression radius.

    This relies upon the existence of a variable compression radius reference
    database. Such as database stores a collection of parameter sets for each
    interaction computed with varying compression radii.

    Arguments:
        on_sites: A torch `ParameterDict` where keys represent atomic numbers
            as strings, and values are torch parameters specifying the on-site
            energies for each orbital.
        off_sites: A torch `ModuleDict` containing the off-site integrals
            required for constructing Hamiltonian and/or overlap matrices.
            The keys are strings representing tuples in the format
            `"(z₁, z₂, s₁, s₂)"`, where `z₁` & `z₂` are the atomic numbers
            of the interacting atoms (with `z₁ ≤ z₂`), and `s₁` & `s₂` are
            their respective shell numbers. Keys must exactly match the
            string obtained from converting a tuple to a string, including
            the parentheses and spaces; for example, `"(1, 6, 0, 0)"`. The
            values are `BicubInterpSpl` instances, that provide the bond
            order integrals for the specified interactions as a function of
            not only distance but compression radius too.
        device: Device on which the feed object and its contents resides.
        dtype: dtype used by feed object.

    Attributes:
        compression_radii: A torch `Parameter` instance specifying the
            compression radius of each atom in the target system. Note that
            this must be manually set for each target system before each call.
        is_local_onsite: `is_local_onsite` allows for constructing chemical
            environment dependent on-site energies.

    Warnings:
        It is critical to note that this class is a work in progress with many
        rough edges. Currently, this feed cannot operate automatically.
        Before this feed is invoked one must set the `compression_radii`
        attribute for the target system. This torch `Parameter` instance should
        store the compression radii for each atom in the target system. This
        will not automatically be set or updated at this time and thus must be
        done manually before every call. Additionally, this feed does not
        support periodic systems. These issues will be addressed later on
        down the line, time permitting.

    Notes:

        The splines contained within the ``off_sites`` argument must return
        all relevant bond order integrals; e.g. an s-s interaction should only
        return a single value for the σ interaction whereas a d-d interaction
        should return three values when interpolated (σ, π & δ).

        Furthermore, it is critical that no duplicate interactions are present.
        That is to say if there is a (1, 6, 0, 0) (H[s]-C[s]) key present then
        there must not also be a (6, 1, 0, 0) (H[s]-C[s]) key present as they
        are the same interaction. To help prevent this the class will raise an
        error if the second atomic number is greater than the first; e.g. the
        key (6, 1, 0, 0) will raise an error but (1, 6, 0, 0) will not.

    """
    def __init__(self, on_sites: ParameterDict[str, Parameter],
                 off_sites: ModuleDict[str, Module],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):

        # Pass the dtype and device to the ABC, if none are given the default
        temp = next(iter(on_sites.values()))
        super().__init__(temp.dtype if dtype is None else dtype,
                         temp.device if device is None else device)

        if isinstance(on_sites, dict):
            raise TypeError(
                "An instance of `torch.nn.ParameterDict` was expected for the "
                "attribute `on_sites`, but a standard Python `dict` was "
                "received.")

        if not isinstance(off_sites, ModuleDict):
            raise TypeError(
                f"An instance of `torch.nn.ModuleDict` was expected for the "
                f"attribute `off_sites`, but a/an `{type(off_sites)}` was "
                "received.")

        for key in off_sites.keys():
            self.__validate_key(key)

        self._on_sites: ParameterDict[str, Parameter] = on_sites
        self._off_sites = off_sites

        self.compression_radii = Optional[Parameter]
        self.is_local_onsite = False

    @property
    def on_sites(self) -> ParameterDict[str, Parameter]:
        """Parameter dictionary of on-site values."""
        return self._on_sites

    @on_sites.setter
    def on_sites(self, value: ParameterDict[str, Parameter]):

        # Ensure that the on-site terms are supplied via a `ParameterDict` and
        # not a standard Python `dict`. Using a standard `dict` would modify
        # the behaviour of the feed in unexpected ways.
        if isinstance(value, dict):
            raise TypeError(
                "An instance of `torch.nn.ParameterDict` was expected for the "
                "attribute `on_sites`, but a standard Python `dict` was "
                "received.")

        self._on_sites = value

    @property
    def off_sites(self) -> ModuleDict[str, Module]:
        """Module dictionary of off-site feed modules."""
        return self._off_sites

    @off_sites.setter
    def off_sites(self, value: ModuleDict[str, Module]):

        if not isinstance(value, ModuleDict):
            raise TypeError(
                f"An instance of `torch.nn.ModuleDict` was expected for the "
                f"attribute `off_sites`, but a/an `{type(value)}` was "
                "received.")

        for key in value.keys():
            self.__validate_key(key)

        self._off_sites = value

    def _off_site_blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
                         geometry: Geometry, orbs: OrbitalInfo, **kwargs) -> Tensor:
        """Compute atomic interaction blocks (off-site only).

        Constructs the off-site atomic blocks using Slater-Koster integral
        tables.

        Arguments:
              atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
              atomic_idx_2: Indices of the 2'nd atom associated with each
                  desired interaction block.
              geometry: The systems to which the atomic indices relate.
              orbs: Orbital information associated with said systems.
          Returns:
              blocks: Requested atomic interaction sub-blocks.
        """
        # Identify atomic numbers associated with the interaction
        z_1 = int(geometry.atomic_numbers[*bT(atomic_idx_1[0: 1])])
        z_2 = int(geometry.atomic_numbers[*bT(atomic_idx_2[0: 1])])

        # Get the species' shell lists (basically a list of azimuthal numbers)
        shells_1, shells_2 = orbs.shell_dict[z_1], orbs.shell_dict[z_2]

        # Inter-atomic distance and distance vector calculator.
        dist_vec = (geometry.positions[*bT2(atomic_idx_2)]
                    - geometry.positions[*bT2(atomic_idx_1)])
        dist = torch.linalg.norm(dist_vec, dim=-1)
        u_vec = (dist_vec.T / dist).T

        # Compression radii for the associated atoms
        compression_radii = torch.stack([
            self.compression_radii[*bT2(atomic_idx_1)],
            self.compression_radii[*bT2(atomic_idx_2)]]).T
        # print(compression_radii)

        # Work out the width of each sub-block then use it to get the row and
        # column index slicers for placing sub-blocks into their atom-blocks.
        rws, cws = np.array(shells_1) * 2 + 1, np.array(shells_2) * 2 + 1
        rows = [slice(i - j, i) for i, j in zip(rws.cumsum(), rws)]
        cols = [slice(i - j, i) for i, j in zip(cws.cumsum(), cws)]

        # Tensor to hold the resulting atomic-blocks.
        blks = torch.zeros(atomic_idx_1.shape[0], rws.sum(), cws.sum(),
                           dtype=self.dtype, device=self.device)

        # Loop over the 1st species' shells; where i & l_1 are the shell's index
        # & azimuthal numbers respectively. Then over the 2nd species' shells,
        # but ignore sub-blocks in the lower triangle of homo-atomic blocks as
        # they can be constructed via symmetrisation.
        for i, l_1 in enumerate(shells_1):
            o = i if z_1 == z_2 else 0
            for j, l_2 in enumerate(shells_2[o:], start=o):
                # Retrieve/interpolate the integral spline, remove any NaNs
                # due to extrapolation then convert to a torch tensor.
                inte = self._off_sites[str((z_1, z_2, i, j))].forward(
                    compression_radii, dist)

                # Apply the Slater-Koster transformation
                inte = sub_block_rot(torch.tensor([l_1, l_2]), u_vec, inte)

                # Add the sub-blocks into their associated atom-blocks
                blks[:, rows[i], cols[j]] = inte

                # Add symmetrically equivalent sub-blocks (homo-atomic only)
                if z_1 == z_2 and i != j:
                    sign = (-1) ** (l_1 + l_2)
                    blks.transpose(-1, -2)[:, rows[i], cols[j]] = inte * sign

        return blks

    def blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
               geometry: Geometry, orbs: OrbitalInfo, **kwargs) -> Tensor:
        r"""Compute atomic interaction blocks using SK-integral tables.

        Returns the atomic blocks associated with the atoms in ``atomic_idx_1``
        interacting with those in ``atomic_idx_2`` splines and Slater-Koster
        transformations. This is the base method used in DFTB calculations.
        Note that The № of interaction blocks returned will be equal to the
        length of the two index lists; i.e. *not* one for every combination.

        Arguments:
            atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
            atomic_idx_2: Indices of the 2'nd atom associated with each
                desired interaction block.
            geometry: The systems to which the atomic indices relate.
            orbs: Orbital information associated with said systems.

        Returns:
            blocks: Requested atomic interaction sub-blocks.

        """

        if self.compression_radii is None:
            raise AttributeError(
                "The `compression_radii` attribute must be set for the target "
                "system before this method can be called.")

        # Get the atomic numbers of the atoms
        zs = geometry.atomic_numbers
        zs_1 = zs[*bT2(atomic_idx_1)]
        zs_2 = zs[*bT2(atomic_idx_2)]

        # Ensure all interactions are between identical species pairs.
        if len(zs_1.unique()) != 1:
            raise ValueError('Atoms in atomic_idx_1 must be the same species')

        if len(zs_2.unique()) != 1:
            raise ValueError('Atoms in atomic_idx_2 must be the same species')

        # Atomic numbers of the species in list 1 and 2
        z_1, z_2 = zs_1[0], zs_2[0]

        # C-N and N-C are the same interaction: choice has been made to have
        # only one set of splines for each species pair. Thus, the two lists
        # may need to be swapped.
        if z_1 > z_2:
            atomic_idx_1, atomic_idx_2 = atomic_idx_2, atomic_idx_1
            z_1, z_2 = z_2, z_1
            flip = True
        else:
            flip = False

        # Construct the tensor into which results are to be placed
        n_rows, n_cols = orbs.n_orbs_on_species(torch.stack((z_1, z_2)))
        blks = torch.zeros(len(atomic_idx_1), n_rows, n_cols, dtype=self.dtype,
                           device=self.device)

        # Identify which are on-site blocks and which are off-site
        on_site = self._partition_blocks(atomic_idx_1, atomic_idx_2)
        mask_shell = torch.zeros_like(self.on_sites[str(z_1.item())]).bool()
        mask_shell[:(torch.arange(len(orbs.shell_dict[z_1.item()]))
                     * 2 + 1).sum()] = True

        # Construct the on-site blocks (if any are present)
        if any(on_site):
            if not self.is_local_onsite:
                blks[on_site] = torch.diag(self.on_sites[str(z_1.item())][mask_shell])
            else:
                blks[on_site] = torch.diag_embed(
                    self.on_sites[str(z_1.item())], dim1=-2, dim2=-1)

            # Interactions between images need to be considered for on-site
            # blocks with pbc.
            if geometry.periodicity is not None:
                _on_site = self._pe_blocks(
                    atomic_idx_1[on_site], atomic_idx_2[on_site],
                    geometry, orbs, geometry.periodicity, onsite=True)
                blks[on_site] = blks[on_site] + _on_site

        if any(~on_site):  # Then the off-site blocks
            if geometry.periodicity is None:
                blks[~on_site] = self._off_site_blocks(
                    atomic_idx_1[~on_site], atomic_idx_2[~on_site],
                    geometry, orbs)
            else:
                blks[~on_site] = self._pe_blocks(
                    atomic_idx_1[~on_site], atomic_idx_2[~on_site],
                    geometry, orbs, geometry.periodicity)

        if flip:  # If the atoms were switched, then a transpose is required.
            blks = blks.transpose(-1, -2)

        return blks

    @classmethod
    def from_database(
            cls, path: str, species: List[int],
            target: Literal['hamiltonian', 'overlap'],
            requires_grad_onsite: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> 'VcrSkFeed':
        r"""Instantiate instance from an HDF5 database of Slater-Koster files.

        Instantiate a `SkFeed` instance for the specified elements using
        integral tables contained within a Slater-Koster HDF5 database.

        Arguments:
            path: Path to the HDF5 file from which integrals should be taken.
            species: Integrals will only be loaded for the requested species.
            target: Specifies which integrals should be loaded, options are
                "hamiltonian" and "overlap".
            requires_grad_onsite: When set to `True` gradient tracking will be
                enabled for the on-site parameters. This flag is ignored for
                the overlap matrix case as its on-site terms are always unity.
                [DEFAULT=`False`]
            device: Device on which the feed object and its contents resides.
            dtype: dtype used by feed object.

        Returns:
            vcrsk_feed: A `VcrSkFeed` instance with the requested integrals.

        """
        # As C-H & C-H interactions are the same only one needs to be loaded.
        # Thus, only off-site interactions where z₁≤z₂ are generated. However,
        # integrals are split over the two Slater-Koster tables, thus both must
        # be loaded.

        # Ensure a valid target is selected
        if target not in ['hamiltonian', 'overlap']:
            ValueError('Invalid target selected; '
                       'options are "hamiltonian" or "overlap"')

        on_sites = ParameterDict()
        off_sites = ModuleDict()

        # The species list must be sorted to ensure that the lowest atomic
        # number comes first in the species pair.
        for pair in combinations_with_replacement(sorted(species), 2):

            skf = VCRSkf.read(path, pair, device=device, dtype=dtype)

            # Loop over the off-site interactions & construct the splines.
            for key, value in skf.__getattribute__(target).items():
                off_sites[str(pair + key)] = BicubInterpSpl(
                    skf.compression_radii.to(device), value.transpose(0, 1), skf.grid)

            # The X-Y.skf file may not contain all information. Thus, some info
            # must be loaded from its Y-X counterpart.
            if pair[0] != pair[1]:
                skf_2 = VCRSkf.read(path, tuple(reversed(pair)), device=device, dtype=dtype)
                for key, value in skf_2.__getattribute__(target).items():
                    if key[0] < key[1]:
                        name = str(pair + (*reversed(key),))
                        off_sites[name] = BicubInterpSpl(
                            skf_2.compression_radii.to(device), value.transpose(0, 1), skf.grid)

            else:  # Construct the onsite interactions
                # Repeated so there's 1 value per orbital not just per shell.
                on_sites_vals = Parameter(
                    skf.on_sites.repeat_interleave(
                        torch.arange(len(skf.on_sites), device=device) * 2 + 1),
                    requires_grad=requires_grad_onsite)

                if target == 'overlap':  # use an identity matrix for S
                    # Auto-grad tracking is always disabled as the values can
                    # never be anything other than one.
                    on_sites_vals = Parameter(
                        torch.ones_like(on_sites_vals, dtype=dtype, device=device),
                        requires_grad=False)

                on_sites[str(pair[0])] = on_sites_vals

        return cls(on_sites, off_sites, dtype, device)

    def __str__(self):
        elements = ', '.join([
            chemical_symbols[i] for i in
            sorted([int(j) for j in self.on_sites.keys()])
        ])
        return f'{self.__class__.__name__}({elements})'

    def __repr__(self):
        return str(self)

    @staticmethod
    def __validate_key(key_string):
        """Validate format of an off-site module-dictionary key.

        Validates that the provided string matches the expected format of a tuple
        converted to a string, i.e., "(z₁, z₂, s₁, s₂)".

        The string must:
        - Start with a left parenthesis '('
        - End with a right parenthesis ')'
        - Contain exactly four comma-separated integers
        - Have exactly one space after each comma
        - Ensure that z₁ ≤ z₂

        Arguments:
            key_string: The string to validate.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        errors = []

        # Check if the string starts with '(' and ends with ')'
        if not (key_string.startswith('(') and key_string.endswith(')')):
            errors.append("The string must start with '(' and end with ')'.")
        else:
            # Remove the parentheses
            content = key_string[1:-1]

            # Define the expected pattern
            pattern = r'^(\d+), (\d+), (\d+), (\d+)$'
            match = re.match(pattern, content)
            if not match:
                errors.append(
                    "The string must contain exactly four comma-separated integers, "
                    "with exactly one space after each comma.")
            else:
                z1, z2, s1, s2 = map(int, match.groups())

                # Check if z₁ ≤ z₂
                if z1 > z2:
                    errors.append(
                        "The first atomic number z₁ must be less than or equal to "
                        "the second atomic number z₂.")

        if errors:
            errors.insert(
                0,
                f"The string \"{key_string}\" used as a key in the `off_sites` "
                f"dictionary does not match the expected format. It should "
                f"represent a tuple converted to a string, like '(1, 6, 0, 0)'.")

            raise ValueError('\n'.join(errors))


class SkfOccupationFeed(Feed):
    """Occupations feed entity that derives its data from a skf file.

    Arguments:
        occupancies: A PyTorch parameter dictionary specifying the angular-
            -momenta resolved occupancies, keyed by atomic numbers (as
            strings) and valued by parameters. Dictionary keys must be strings
            as required by the PyTorch `ParameterDict` structure.

    Examples:
        >>> from tbmalt.physics.dftb.feeds import SkfOccupationFeed
        >>> l_resolved = SkfOccupationFeed(  #     fs, fp, fd
        ...     ParameterDict({"79": torch.tensor([1., 0., 10.])}))

    Notes:
        Note that this method discriminates between orbitals based only on
        the azimuthal number of the orbital & the species to which it belongs.
    """

    # Developer's Notes:
    # This class will be abstracted and extended to allow for specification
    # via shell number which will avoid the current limits which only allow
    # for minimal orbs sets.

    def __init__(self, occupancies: ParameterDict[str, Parameter]):
        super().__init__()

        self.occupancies = occupancies

        # Occupancy values must be supplied via a PyTorch `ParameterDict`.
        if not isinstance(occupancies, ParameterDict):
            raise TypeError(
                "Occupancies must be stored within a `torch.nn.ParameterDict` "
                "entity. This allows PyTorch to a automatically detect valid "
                "optimisation targets as and when necessary."
            )

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by `SkfOccupationFeed` object."""
        return list(self.occupancies.values())[0].dtype

    @property
    def device(self) -> torch.device:
        """The device on which the `SkfOccupationFeed` object resides."""
        return list(self.occupancies.values())[0].device

    def to(self, device: torch.device) -> SkfOccupationFeed:
        """Return a copy of the `SkfOccupationFeed` on the specified device.

        Arguments:
            device: Device to which all associated tensor should be moved.

        Returns:
            occupancy_feed: A copy of the `SkfOccupationFeed` instance placed
                on the specified device.

        """
        return self.__class__(ParameterDict({
            k: v.to(device=device) for k, v in self.occupancies.items()}))

    def forward(self, orbs: OrbitalInfo) -> Tensor:
        """Shell resolved occupancies.

        This returns the shell resolved occupancies for the neutral atom in the
        ground state. The resulting values are derived from static occupancy
        parameters read in from an SKF formatted file.

        Arguments:
            orbs: orbs objects for the target systems.

        Returns:
            occupancies: shell resolved occupancies.
        """

        # Construct a pair of arrays, 'zs' & `ls`, that can be used to look up
        # the species and shell number for each orbital.
        z_list, l_list = orbs.atomic_numbers, orbs.shell_ls
        zs = prepeat_interleave(z_list, orbs.n_orbs_on_species(z_list), -1)
        ls = prepeat_interleave(l_list, orbs.orbs_per_shell, -1)

        # Tensor into which the results will be placed
        occupancies = torch.zeros_like(zs, dtype=self.dtype)

        # Loop over all available occupancy information
        for z, occs in self.occupancies.items():
            # As the atomic number keys in the occupancies dictionaries are
            # stored as strings, for PyTorch compatability reasons, they need
            # to be cast back into integers.
            z = int(z)
            # Loop over each shell for species 'z'
            for l, occ in enumerate(occs):
                # And assign the associated occupancy where appropriate
                occupancies[(zs == z) & (ls == l)] = occ

        # Divide the occupancy by the number of shells
        return occupancies / (2 * ls + 1)

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "`SkfOccupationFeed` instances should be invoked via their "
            "`.forward` method now.")
        return self.forward(*args, **kwargs)

    @classmethod
    def from_database(cls, path: str, species: List[int], **kwargs
                      ) -> SkfOccupationFeed:
        """Instantiate an `SkfOccupationFeed` instance from an HDF5 database.

        Arguments:
            path: path to the HDF5 file in which the skf file data is stored.
            species: species for which occupancies are to be loaded.

        Keyword Arguments:
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            requires_grad: boolean indicating if gradient tracking should be
                enabled for the occupancies. If enabled, the relevant
                dictionaries and tensors will be converted into `ParameterDict`
                and `Parameter` instances respectively. [DEFAULT=False]

        Returns:
            occupancy_feed: An `SkfOccupationFeed` instance containing the
                requested occupancy information.

        Examples:
            >>> import torch
            >>> from tbmalt import OrbitalInfo
            >>> from tbmalt.physics.dftb.feeds import SkfOccupationFeed
            >>> from tbmalt.tools.downloaders import download_dftb_parameter_set
            >>> torch.set_default_dtype(torch.float64)

            # Download the auorg-1-1 parameter set
            >>> url = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
            >>> path = "auorg.h5"
            >>> download_dftb_parameter_set(url, path)

            # Definition of feeds
            >>> o_feed = SkfOccupationFeed.from_database(path, [1, 6])
            >>> shell_dict = {1: [0], 6: [0, 1]}

            # Occupancy information of an example system
            >>> o_feed.forward(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1]), shell_dict))
            tensor([2.0000, 0.6667, 0.6667, 0.6667,
                    1.0000, 1.0000, 1.0000, 1.0000])

        """
        requires_grad = kwargs.pop("requires_grad", False)

        def get_occupations(x: int):
            # Read occupancy values for the target species from the associated
            # Slater-Koster parameter file.
            occupations = Skf.read(path, (x, x), **kwargs).occupations

            # `Tensor` -> `Parameter` cast done manually now to give control
            # over the `requires_grad` option.
            return Parameter(occupations, requires_grad=requires_grad)

        return cls(ParameterDict({str(i): get_occupations(i) for i in species}))


class HubbardFeed(Feed):
    """Hubbard U feed entity that derives its data from a skf file.

    This provides a feed based method by which traditional DFTB Hubbard-U
    values can be accessed.

    Arguments:
        hubbard_us: A PyTorch parameter dictionary specifying the angular-
            -momenta resolved Hubbard-Us, keyed by atomic numbers (as strings)
            and valued by parameters. Dictionary keys must be strings
            as required by the PyTorch `ParameterDict` structure.

    Examples:
        >>> from tbmalt.physics.dftb.feeds import HubbardFeed
        >>> l_resolved = HubbardFeed(ParameterDict({"1": torch.tensor([0.5])}))

    Notes:
        Note that this method discriminates between orbitals based only on
        the azimuthal number of the orbital & the species to which it belongs.

    Todo:
        Add a test that throws an error if a shell resolved orbs is provided but
        `hubbard_u` is found to only be atom resolved; and vise versa. The skf
        database should also instruct the loader whether it is shell-resolved.
    """
    def __init__(self, hubbard_us: ParameterDict[str, Parameter]):
        super().__init__()

        self.hubbard_us = hubbard_us

        # Hubbard-U values must be supplied via a PyTorch `ParameterDict`.
        if not isinstance(hubbard_us, ParameterDict):
            raise TypeError(
                "Hubbard-Us must be stored within a `torch.nn.ParameterDict` "
                "entity. This allows PyTorch to a automatically detect valid "
                "optimisation targets as and when necessary."
            )

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by the `HubbardFeed` object."""
        return list(self.hubbard_us.values())[0].dtype

    @property
    def device(self) -> torch.device:
        """The device on which the `HubbardFeed` object resides."""
        return list(self.hubbard_us.values())[0].device

    def to(self, device: torch.device) -> HubbardFeed:
        """Return a copy of the `HubbardFeed` on the specified device.

        Arguments:
            device: Device to which all associated tensor should be moved.

        Returns:
            hubbard_u_feed: A copy of the `HubbardFeed` instance placed
                on the specified device.

        """
        return self.__class__(ParameterDict({
            k: v.to(device=device) for k, v in self.hubbard_us.items()}))

    def forward(self, orbs: OrbitalInfo) -> Tensor:
        """Hubbard U values.

        This returns the Hubbard U values for the atom.

        Arguments:
            orbs: orbs objects for the target systems.

        Returns:
            hubbard_us: Hubbard U values, either shell or atom resolved
                depending on status of `orbs.shell_resolved`.
        """

        # Construct a pair of arrays, 'zs' & `ls`, that can be used to look up
        # the species and shell number for each orbital.
        z_list, ls = orbs.atomic_numbers, orbs.shell_ls

        if orbs.shell_resolved:
            zs = prepeat_interleave(z_list, orbs.n_shells_on_species(z_list), -1)

            # Tensor into which the results will be placed
            hubbard_us = torch.zeros_like(zs, dtype=self.dtype)

            # Loop over all available Hubbard-U information
            for z, us in self.hubbard_us.items():
                z = int(z)
                # Loop over each shell for species 'z'
                for l, u in enumerate(us):
                    # And assign the associated Hubbard-Us where appropriate
                    hubbard_us[(zs == z) & (ls == l)] = u
        else:
            hubbard_us = torch.zeros_like(z_list, dtype=self.dtype)
            for z, us in self.hubbard_us.items():
                hubbard_us[z_list == int(z)] = us[0]

        return hubbard_us

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "`HubbardFeed` instances should be invoked via their "
            "`.forward` method now.")
        return self.forward(*args, **kwargs)

    @classmethod
    def from_database(cls, path: str, species: List[int], **kwargs
                      ) -> HubbardFeed:
        """Instantiate an `HubbardFeed` instance from an HDF5 database.

        Arguments:
            path: path to the HDF5 file in which the skf file data is stored.
            species: species for which Hubbard-U values are to be loaded.

        Keyword Arguments:
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            requires_grad: boolean indicating if gradient tracking should be
                enabled for the Hubbard-Us. If enabled, the relevant
                dictionaries and tensors will be converted into `ParameterDict`
                and `Parameter` instances respectively. [DEFAULT=False]

        Returns:
            hubbard_u_feed: A `HubbardFeed` instance containing the
                Hubbard-U values for the requested species.

        Examples:
            >>> import torch
            >>> from tbmalt import OrbitalInfo
            >>> from tbmalt.physics.dftb.feeds import HubbardFeed
            from tbmalt.tools.downloaders import download_dftb_parameter_set
            >>> torch.set_default_dtype(torch.float64)

            # Download the auorg-1-1 parameter set
            >>> url = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
            >>> path = "auorg.h5"
            >>> download_dftb_parameter_set(url, path)

            # Definition of feeds
            >>> u_feed = HubbardFeed.from_database(path, [1, 6])
            >>> shell_dict = {1: [0], 6: [0, 1]}

            # Hubbard U values of an example system
            >>> u_feed.forward(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1]), shell_dict))
            tensor([0.3647, 0.4196, 0.4196, 0.4196, 0.4196])

        """
        requires_grad = kwargs.pop("requires_grad", False)

        def get_hubbard_us(x: int):
            # Read Hubbard-U values for the target species from the associated
            # Slater-Koster parameter file.
            hubbard_us = Skf.read(path, (x, x), **kwargs).hubbard_us

            # `Tensor` -> `Parameter` cast done manually now to give control
            # over the `requires_grad` option.
            return Parameter(hubbard_us, requires_grad=requires_grad)

        return cls(ParameterDict({str(i): get_hubbard_us(i) for i in species}))


class RepulsiveSplineFeed(Feed):
    r"""Repulsive Feed using splines for DFTB calculations.

    Data is derived from a skf file. This feed uses splines to calculate the
    repulsive energy of a Geometry in the way it is defined for DFTB.

    Arguments:
        spline_data: Dictionary containing the tuples of atomic number pairs
            as keys and the corresponding spline data as values.
    """

    def __init__(self, spline_data: Dict[Tuple, Skf.RSpline]):
        super().__init__()

        warnings.warn(
            "The `RepulsiveSplineFeed` class is now deprecated and will be"
            "removed. Please use the `PairwiseRepulsiveEnergyFeed` and "
            "`DftbpRepulsiveSpline` classes instead.",
            category=DeprecationWarning)

        self.spline_data = {
            frozenset(interaction_pairs): data
            for interaction_pairs, data in spline_data.items()}

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by `RepulsiveSplineFeed` object."""
        return list(self.spline_data.values())[0].grid.dtype

    @property
    def device(self) -> torch.device:
        """The device on which the `RepulsiveSplineFeed` object resides."""
        return list(self.spline_data.values())[0].grid.device

    def __call__(self, geo: Geometry) -> Tensor:
        r"""Calculate the repulsive energy of a Geometry.

        Arguments:
            geo: `Geometry` object representing the system, or batch thereof,
                for which the repulsive energy should be calculated.

        Returns:
            Erep: The repulsive energy of the Geometry object(s).
        """
        batch_size, indxs, indx_pairs, normed_distance_vectors = self._calculation_prep(geo)

        Erep = torch.zeros((batch_size), device=self.device, dtype=self.dtype)

        for indx_pair in indx_pairs:
            atomnum1 = geo.atomic_numbers[..., indx_pair[0]].reshape((batch_size, ))
            atomnum2 = geo.atomic_numbers[..., indx_pair[1]].reshape((batch_size, ))

            distance = geo.distances[..., indx_pair[0], indx_pair[1]].reshape((batch_size, ))

            for batch_indx in range(batch_size):
                if atomnum1[batch_indx] == 0 or atomnum2[batch_indx] == 0:
                    continue
                add_Erep = self._repulsive_calc(distance[batch_indx], atomnum1[batch_indx], atomnum2[batch_indx])
                Erep[batch_indx] += add_Erep

        return Erep

    def gradient(self, geo: Geometry) -> Tensor:
        """Calculate the gradient of the repulsive energy.

        Arguments:
            geo: `Geometry` object representing the system, or batch thereof,
                for which the gradient of the repulsive energy should be
                calculated.

        returns:
            dErep: The gradient of the repulsive energy.
        """
        batch_size, indxs, indx_pairs, normed_distance_vectors = self._calculation_prep(geo)

        dErep = torch.zeros((batch_size, geo.atomic_numbers.size(dim=-1), 3), device=self.device, dtype=self.dtype)

        for indx_pair in indx_pairs:
            atomnum1 = geo.atomic_numbers[..., indx_pair[0]].reshape((batch_size, ))
            atomnum2 = geo.atomic_numbers[..., indx_pair[1]].reshape((batch_size, ))

            distance = geo.distances[..., indx_pair[0], indx_pair[1]].reshape((batch_size, ))

            for batch_indx in range(batch_size):
                if atomnum1[batch_indx] == 0 or atomnum2[batch_indx] == 0:
                    continue
                add_dErep = self._repulsive_calc(distance[batch_indx], atomnum1[batch_indx], atomnum2[batch_indx], grad=True)
                #TODO: Not yet batched
                dErep[batch_indx, indx_pair[0]] += add_dErep*normed_distance_vectors[batch_indx, indx_pair[0], indx_pair[1]]
                dErep[batch_indx, indx_pair[1]] += add_dErep*normed_distance_vectors[batch_indx,indx_pair[1], indx_pair[0]]
        if batch_size == 1:
            dErep = dErep.squeeze(0)
        return dErep

    def _calculation_prep(self, geo: Geometry
                          ) -> Tuple[int, Tensor, Tensor, Tensor]:
        """Preliminaries for repulsive energy & gradient calculation.

        Arguments:
            geo: `Geometry` object representing the system, or batch thereof,
                for which the calculation preparation steps are to be performed.

        returns:
            batch_size: The number of geometries in the batch.
            indxs: The indices of the atoms.
            indx_pairs: The indices of the interacting atom pairs as tuples.
            normed_distance_vectors: The normalized distance vectors between the atoms
        """
        if geo.atomic_numbers.dim() == 1: # this means it is not a batch
            batch_size = 1
        else:
            batch_size = geo.atomic_numbers.size(dim=0)

        indxs = torch.tensor(range(geo.atomic_numbers.size(dim=-1)), device=self.device)
        indx_pairs = torch.combinations(indxs)

        normed_distance_vectors = geo.distance_vectors / geo.distances.unsqueeze(-1)
        normed_distance_vectors[normed_distance_vectors.isnan()] = 0
        normed_distance_vectors = torch.reshape(
            normed_distance_vectors, (
                batch_size, normed_distance_vectors.shape[-3],
                normed_distance_vectors.shape[-2],
                normed_distance_vectors.shape[-1]))

        return batch_size, indxs, indx_pairs, normed_distance_vectors

    def _repulsive_calc(
            self, distance: Tensor, atomnum1: Union[Tensor, int],
            atomnum2: Union[Tensor, int], grad: bool = False
        ) -> Tensor:
        """Calculate the repulsive energy contribution between two atoms.

        Arguments:
            distance: The distance between the two atoms.
            atomnum1: The atomic number of the first atom.
            atomnum2: The atomic number of the second atom.

        returns:
            Erep: The repulsive energy contribution between the two atoms.
        """
        spline = self.spline_data[frozenset((int(atomnum1), int(atomnum2)))]
        tail_start = spline.grid[-1]
        exp_head_cutoff = spline.grid[0]

        if distance < spline.cutoff:
            if distance > tail_start:
                return self._tail(distance, tail_start, spline.tail_coef, grad=grad)
            elif distance > exp_head_cutoff:
                for ind in range(len(spline.grid)):
                    if distance < spline.grid[ind]:
                        return self._spline(distance, spline.grid[ind-1], spline.spline_coef[ind-1], grad=grad)
            else:
                return self._exponential_head(distance, spline.exp_coef, grad=grad)
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @classmethod
    def _exponential_head(cls, distance: Tensor, coeffs: Tensor, grad: bool = False) -> Tensor:
        r"""Exponential head calculation of the repulsive spline.

        Arguments:
            distance: The distance between the two atoms.
            coeffs: The coefficients of the exponential head.

        Returns:
            energy: The energy value of the exponential head.
                The energy is calculated as :math:`\exp(-coeffs[0] \cdot r + coeffs[1]) + coeffs[2]`.
        """
        a1 = coeffs[0]
        a2 = coeffs[1]
        a3 = coeffs[2]
        if not grad:
            return torch.exp(-a1*distance + a2) + a3
        else:
            return -a1*torch.exp(-a1*distance + a2)

    @classmethod
    def _spline(cls, distance: Tensor, start: Tensor, coeffs: Tensor, grad: bool = False) -> Tensor:
        r"""3rd order polynomial Spline calculation of the repulsive spline.

        Arguments:
            distance: The distance between the two atoms.
            start: The start of the spline segment.
            coeffs: The coefficients of the polynomial.

        Returns:
            energy: The energy value of the spline segment.
                The energy is calculated as :math:`coeffs[0] + coeffs[1]*(distance - start) + coeffs[2]*(distance - start)^2 + coeffs[3]*(distance - start)^3`.
        """
        rDiff = distance - start
        if not grad:
            energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3
            return energy
        else:
            denergy = coeffs[1] + 2*coeffs[2]*rDiff + 3*coeffs[3]*rDiff**2
            return denergy

    @classmethod
    def _tail(cls, distance: Tensor, start: Tensor, coeffs: Tensor, grad: bool = False) -> Tensor:
        r"""5th order polynomial trailing tail calculation of the repulsive spline.

        Arguments:
            distance: The distance between the two atoms.
            start: The start of the trailing tail segment.
            coeffs: The coefficients of the polynomial.

        Returns:
            energy: The energy value of the tail.
                The energy is calculated as :math:`coeffs[0] + coeffs[1]*(distance - start) + coeffs[2]*(distance - start)^2 + coeffs[3]*(distance - start)^3 + coeffs[4]*(distance - start)^4 + coeffs[5]*(distance - start)^5`.
        """
        rDiff = distance - start
        if not grad:
            energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3 + coeffs[4]*rDiff**4 + coeffs[5]*rDiff**5
            return energy
        else:
            denergy = coeffs[1] + 2*coeffs[2]*rDiff + 3*coeffs[3]*rDiff**2 + 4*coeffs[4]*rDiff**3 + 5*coeffs[5]*rDiff**4
            return denergy

    @classmethod
    def from_database(
            cls, path: str, species: List[int],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None
            ) -> 'RepulsiveSplineFeed':
        r"""Instantiate instance from a HDF5 database of Slater-Koster files.

        Instantiate a `RepulsiveSplineFeed` instance from a HDF5 database for the specified elements.

        Arguments:
            path: Path to the HDF5 file from which the repulsive interaction data should be taken.
            species: List of atomic numbers for which the repulsive spline data should be read.
            device: Device on which the feed object and its contents resides.

        Returns:
            repulsive_feed: A `RepulsiveSplineFeed` instance.

        Notes:
            This method will not instantiate `SkFeed` instances directly
            from human readable skf files, or a directory thereof. Thus, any
            such files must first be converted into their binary equivalent.
            This reduces overhead & file format error instabilities. The code
            block provide below shows how this can be done:

            >>> from tbmalt.io.skf import Skf
            >>> Zs = ['H', 'C', 'Au', 'S']
            >>> for file in [f'{i}-{j}.skf' for i in Zs for j in Zs]:
            >>>     Skf.read(file).write('my_skf.hdf5')
        """
        interaction_pairs = combinations_with_replacement(species, r=2)
        return cls({
            interaction_pair: Skf.read(
                path, interaction_pair, device=device, dtype=dtype
            ).r_spline for interaction_pair in interaction_pairs})


class DftbpRepulsiveSpline(Feed):
    """Repulsive spline representation for the repulsive DFTB interaction.

    The repulsive spline implementation matches that used by the DFTB+ package.

    The repulsive potential is partitioned into three regimes and is evaluated
    only within the specified cutoff radius, being zero beyond this distance.

    1. **Short-range exponential head** when distances ≤ first grid point:
       .. math::
           e^{-a_{1} r + a_{2}} + a_{3}

    2. **Intermediate cubic spline body** defined on each interval [r_i, r_{i+1}]:
       .. math::
           c_{0} + c_{1}(r - r_{0}) + c_{2}(r - r_{0})^{2} + c_{3}(r - r_{0})^{3}

    3. **Long-range polynomial tail** between the last spline point and cutoff:
       .. math::
           c_{0} + c_{1}(r - r_{n}) + c_{2}(r - r_{n})^{2}
           + c_{3}(r - r_{n})^{3} + c_{4}(r - r_{n})^{4} + c_{5}(r - r_{n})^{5}

    Arguments:
        grid: Distances for the primary spline segments including the start &
            end of the first & last segments respectively. As such there should
            be n+1 grid points, where n is the number of standard spline
            segments.
        cutoff: Cutoff radius for the spline's tail beyond which interactions
            are assumed to be zero.
        spline_coefficients: An n×4 tensor storing the coefficients for each of
            the primary spline segments.
        exponential_coefficients: A tensor storing the three coefficients of
            the short range exponential region.
        tail_coefficients: A tensor storing the six coefficients of the long -
            range tail region.
    """

    def __init__(
            self, grid: Tensor, cutoff: Tensor, spline_coefficients: Parameter,
            exponential_coefficients: Parameter, tail_coefficients: Parameter):

        super().__init__()
        self.grid = grid
        self.cutoff = cutoff
        self.spline_coefficients = spline_coefficients
        self.exponential_coefficients = exponential_coefficients
        self.tail_coefficients = tail_coefficients

        # Ensure parameters are correctly typed
        if isinstance(grid, Parameter) or grid.requires_grad:
            raise warnings.warn(
                "Setting the grid points as a freely tunable parameter is "
                "strongly advised against as it may result in unexpected "
                "behaviour. Please ensure that the `grid` argument is a "
                "standard `torch.Tensor` type rather than `torch.nn.Parameter` "
                "and that its \"requires_grad\" attribute is set to `False` "
                "unless you are sure of what you are doing.")

        if isinstance(cutoff, Parameter) or grid.requires_grad:
            raise TypeError(
                "The cutoff is not freely tunable parameter. Please ensure "
                "that the `cutoff` argument is a standard `torch.Tensor` type "
                "rather than `torch.nn.Parameter` & that its \"requires_grad\""
                "attribute is set to `False`.")

        if not isinstance(spline_coefficients, Parameter):
            raise TypeError("The spline coefficients must be a "
                            "`torch.nn.Parameter` instance.")

        if not isinstance(exponential_coefficients, Parameter):
            raise TypeError("The exponential coefficients must be a "
                            "`torch.nn.Parameter` instance.")

        if not isinstance(tail_coefficients, Parameter):
            raise TypeError("The tail coefficients must be a "
                            "`torch.nn.Parameter` instance.")

        # Ensure that the tensors are of the correct shape
        if spline_coefficients.ndim != 2 or spline_coefficients.shape[1] != 4:
            raise ValueError("Argument `spline_coefficients` should be an n×4 "
                             "tensor.")

        if spline_coefficients.shape[0] != grid.shape[0] - 1:
            raise ValueError(
                f"{grid.shape[0]} grid values were provided suggesting the "
                f"presence of {grid.shape[0] - 1} standard spline segments. "
                f"However, coefficients for {spline_coefficients.shape[0]} "
                f"segments were provided in `spline_coefficients`.")

        if exponential_coefficients.shape != torch.Size([3]):
            raise ValueError(
                f"Expected `exponential_coefficients` argument to be of shape "
                f"torch.Size([3]) but encountered "
                f"{exponential_coefficients.shape}.")

        if tail_coefficients.shape != torch.Size([6]):
            raise ValueError(
                f"Expected `tail_coefficients` argument to be of shape "
                f"torch.Size([6]) but encountered {tail_coefficients.shape}.")

    @property
    def spline_cutoff(self):
        """Cutoff distance of the last primary spline segment."""
        return self.grid[-1]

    @property
    def exponential_cutoff(self):
        """Cutoff distance of the short range exponential region."""
        return self.grid[0]

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction at the specified distance(s).

        Arguments:
            distances: Distance(s) at which the repulsive term is to be
                evaluated.

        Returns:
            repulsive: Repulsive interaction energy as evaluated at the
                specified distances.
        """
        results = torch.zeros_like(distances)

        # Mask for distances < cutoff
        under_cutoff = distances < self.cutoff

        # Within that subset, distinguish three further conditions:
        # 1) distances > spline_cutoff
        mask_1 = under_cutoff & (distances > self.spline_cutoff)
        # 2) distances > exponential_cutoff and <= spline_cutoff
        mask_2 = under_cutoff & (distances > self.exponential_cutoff
                                 ) & (distances <= self.spline_cutoff)
        # 3) distances <= exp_cutoff
        mask_3 = under_cutoff & (distances <= self.exponential_cutoff)

        # Evaluate the distances in each of the three main distance regimes
        # accordingly.
        results[mask_1] = self._tail(distances[mask_1])
        results[mask_2] = self._spline(distances[mask_2])
        results[mask_3] = self._exponential(distances[mask_3])

        return results

    def _exponential(self, distance: Tensor) -> Tensor:
        """Evaluate the exponential head of the repulsive interaction.

        The short-range exponential part of the repulsive interaction is
        applied when the atoms are closer than the starting distance of the
        first standard spline segment. The repulsive interaction within this
        region is described by the following exponential term:

        .. math::

            e^{-a_{1} r + a_{2}} + a_{3}

        Where "r" (`distances`) is the distance between the atoms and
        :math:`a_{i}` are the exponential coefficients.

        Arguments:
            distance: Distance(s) at which the exponential term is to be
                evaluated.

        Returns:
            repulsive: Exponential repulsive interaction as evaluated at the
                specified distances.
        """
        c = self.exponential_coefficients
        return torch.exp(-c[0] * distance + c[1]) + c[2]

    def _spline(self, distance: Tensor) -> Tensor:
        """Evaluate main spline body of the repulsive interaction.

        Distances between the exponential head and fifth order polynomial
        spline tail are evaluated using a third order polynomial spline of
        the form:

        .. math::

            c_{0}+c_{1}(r-r_{0})+c_{2}(r-r_{0})^{2}+c_{3}(r-r_{0})^{3}

        Where "r" (`distances`) is the distance between the atoms,
        :math:`r_{0}` is the start of the spline segment, and :math:`c_{i}` are
        the spline segment's coefficients.

        Arguments:
            distance: Distance(s) at which the primary spline term is to be
                evaluated.

        Returns:
            repulsive: Primary spline body repulsive interaction as evaluated
                at the specified distances.
        """
        indices = torch.searchsorted(self.grid, distance) - 1
        c = self.spline_coefficients[indices]
        r = distance - self.grid[indices]
        return c[:, 0] + c[:, 1] * r + c[:, 2] * r ** 2 + c[:, 3] * r ** 3

    def _tail(self, distance: Tensor) -> Tensor:
        """Evaluate the polynomial tail part of the repulsive interaction.

        Distance between the last standard spline segment's endpoint and the
        cutoff are represented by a tail spline of the form:

        .. math::

            c_{0}+c_{1}(r-r_{0})+c_{2}(r-r_{0})^{2}+c_{3}(r-r_{0})^{3}
                +c_{4}(r-r_{0})^{4}+c_{5}(r-r_{0})^{5}

        Where "r" (`distances`) is the distance between the atoms,
        :math:`r_{0}` is the start of the tail region, and :math:`c_{i}` are
        the tail spline's coefficients.

        Arguments:
            distance: Distance(s) at which the long-range spline tail term is
                to be evaluated.

        Returns:
            repulsive: Spline tail repulsive interaction as evaluated at the
                specified distances.
        """
        c = self.tail_coefficients
        r = distance - self.spline_cutoff
        r_poly = r.unsqueeze(-1).repeat(1, 5).cumprod(-1)
        return c[0] + (c[1:] * r_poly).sum(-1)

    @classmethod
    def from_skf(cls, skf: Skf, requires_grad: bool = False) -> DftbpRepulsiveSpline:
        """Instantiate a `DftbpRepulsiveSpline` instance from a `Skf` object.

        This method will read the repulsive spline data from an `Skf` instance
        representing an sfk formatted file and construct a repulsive spline
        of the form used by the DFTB+ package.

        Arguments:
            skf: An `Skf` instance representing an skf file from which the
                data parameterising the repulsive spline can be read.
            requires_grad: A boolean indicating if the gradient tracking should
                be enabled for the spline's coefficients. [DEFAULT=False]

        Returns:
            repulsive_feed: A `DftbpRepulsiveSpline` instance representing the
                repulsive interaction.

        Notes:
            This assumes the presence of repulsive spline feed in the skf file.
            However, this condition is not guaranteed as some skf files will
            provide a polynomial instead.
        """
        if skf.r_spline is None:
            raise AttributeError(
                f"Skf file {skf} does not define a repulsive spline.")

        return cls.from_r_spline(skf.r_spline, requires_grad=requires_grad)

    @classmethod
    def from_r_spline(
            cls, r_spline: Skf.RSpline, requires_grad: bool = False
    ) -> DftbpRepulsiveSpline:
        """Instantiate a `DftbpRepulsiveSpline` instance from a `Skf.RSpline` object.

        This method will use an `Skf.RSpline` data class to construct a
        repulsive spline of the form used by the DFTB+ package.

        Arguments:
            skf: An `Skf.RSpline` instance parameterising the repulsive spline.
            requires_grad: A boolean indicating if the gradient tracking should
                be enabled for the spline's coefficients. [DEFAULT=False]

        Returns:
            repulsive_feed: A `DftbpRepulsiveSpline` instance representing the
                repulsive interaction.
        """
        return cls(
            r_spline.grid, r_spline.cutoff,
            Parameter(r_spline.spline_coef, requires_grad=requires_grad),
            Parameter(r_spline.exp_coef, requires_grad=requires_grad),
            Parameter(r_spline.tail_coef, requires_grad=requires_grad))


class PairwiseRepulsiveEnergyFeed(Feed):
    """Sort range repulsive interaction feed for DFTB calculations.

    This feed uses distance dependent interpolator feeds to evaluate the
    total repulsive pair-wise interaction energy for a given system.

    Arguments:
        repulsive_feeds: A torch `ModuleDict` of pair-wise distance dependent
            repulsive feeds, such as `DftbpRepulsiveSpline`, keyed by strings
            representing tuples of the form `"(z₁, z₂)"`, where `z₁` & `z₂` are
            the atomic numbers of the associated element pair (with `z₁ ≤ z₂`).
            Keys must exactly match the string obtained from converting a tuple
            to a string, including the parentheses & spaces; for example,
            `"(1, 6)"`.

    Notes:
        In principle any feed may be placed within the ``repulsive_feeds``
        dictionary to represent repulsive interactions so long as its `forward`
        method takes a distances values and returns repulsive energy values.
    """

    def __init__(self, repulsive_feeds: ModuleDict[str, Feed]):
        super().__init__()

        self.repulsive_feeds = repulsive_feeds

        # Ensure that the repulsive feeds are stored within a ModuleDict
        # instance, rather than something else like a standard Python
        # dictionary.
        if not isinstance(repulsive_feeds, ModuleDict):
            raise TypeError(
                f"An instance of `torch.nn.ModuleDict` was expected for the "
                f"attribute `repulsive_feeds`, but a `{type(repulsive_feeds)}` "
                f"was encountered.")

        # Ensure the keys used in the repulsive feed module dictionary
        # match the expected form.
        for key in repulsive_feeds.keys():
            self.__validate_key(key)

    def forward(self, geometry: Geometry) -> Tensor:
        """Repulsive energy.

        Compute the pair-wise repulsive energy of the specified system using
        the element pair specific pair-wise repulsive sub-feeds.

        Arguments:
            geometry: System, or batch thereof, for which the repulsive
                interaction energy is to be computed.

        Returns:
            repulsive: Repulsive interaction energy as evaluated for the
                specified system(s).
        """

        # Tensor to hold the resulting repulsive energy.
        e_rep = torch.zeros(
            geometry.atomic_numbers.shape[0:geometry.atomic_numbers.ndim - 1],
            device=geometry.device)

        # For each species pair, atom indices and distances are yielded with a
        # forced batch dimension, facilitating batch-agnostic scatter addition.
        for species_pair, atomic_indices, distances in atomic_pair_distances(
                geometry, True, True):
            # Identify the repulsive feed associated with the current pair
            feed = self.repulsive_feeds[
                str((species_pair[0].item(), species_pair[1].item()))]

            # evaluate it at the relevant distances
            e_pairs = feed.forward(distances)

            # add the resulting energies to the current total(s)
            e_rep.scatter_add_(0, atomic_indices[0], e_pairs)

        return e_rep

    @classmethod
    def from_database(
            cls, path: str, species: List[int],
            requires_grad: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None):
        """Instantiate instance from an HDF5 database of Slater-Koster files.

        Instantiate a `PairwiseRepulsiveEnergyFeed` instance for the specified
        elements using repulsive data contained within a Slater-Koster HDF5
        file. More specifically, the repulsive spline coefficients will be
        parsed and used to construct a series of `DftbpRepulsiveSpline`
        instances.

        Arguments:
            path: Path to the HDF5 file from which repulsive splines data
                should be sourced.
            species: Species for which repulsive interactions should be
                loaded.
            requires_grad: When set to `True` gradient tracking will be enabled
                for all coefficients within the repulsive spline feeds.
                [DEFAULT=False]
            device: Device on which the feed object and its contents resides.
            dtype: dtype used by feed object.

        Returns:
            repulsive_energy_feed: An `PairwiseRepulsiveEnergyFeed` instance
                with the required repulsive interactions represented using
                `DftbpRepulsiveSpline` instances.
        """

        repulsive_feeds = ModuleDict()

        for pair in combinations_with_replacement(sorted(species), r=2):
            skf = Skf.read(path, pair, device=device, dtype=dtype)
            repulsive_feeds[str(pair)] = DftbpRepulsiveSpline.from_skf(
                skf, requires_grad=requires_grad)

        return cls(repulsive_feeds)

    @staticmethod
    def __validate_key(key_string):
        """Validate format of an element pair module-dictionary key.

        Validates that the provided string matches the expected format of a tuple
        converted to a string, i.e., "(z₁, z₂)".

        The string must:
        - Start with a left parenthesis '('
        - End with a right parenthesis ')'
        - Contain exactly two comma-separated integers
        - Have exactly one space after each comma
        - Ensure that z₁ ≤ z₂

        Arguments:
            key_string: The string to validate.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        errors = []

        # Check if the string starts with '(' and ends with ')'
        if not (key_string.startswith('(') and key_string.endswith(')')):
            errors.append("The string must start with '(' and end with ')'.")
        else:
            # Remove the parentheses
            content = key_string[1:-1]

            # Define the expected pattern
            pattern = r'^(\d+), (\d+)$'
            match = re.match(pattern, content)
            if not match:
                errors.append(
                    "The string must contain exactly two comma-separated integers, "
                    "with exactly one space after each comma.")
            else:
                z1, z2 = map(int, match.groups())

                # Check if z₁ ≤ z₂
                if z1 > z2:
                    errors.append(
                        "The first atomic number z₁ must be less than or equal to "
                        "the second atomic number z₂.")

        if errors:
            errors.insert(
                0,
                f"The string \"{key_string}\" used as a key in the"
                f"`repulsive_feeds` dictionary does not match the expected "
                f"format. It should represent a tuple converted to a string, "
                f"like '(1, 6)'.")

            raise ValueError('\n'.join(errors))
