# -*- coding: utf-8 -*-
"""Slater-Koster integral feed objects.

This contains all Slater-Koster integral feed objects. These objects are
responsible for generating the Slater-Koster integrals and for constructing
the associated Hamiltonian and overlap matrices.
"""
from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations_with_replacement
from typing import List, Literal, Optional, Dict, Tuple, Union
from scipy.interpolate import CubicSpline
import torch
from torch import Tensor
from torch.nn import Parameter, ParameterDict

from tbmalt import Geometry, OrbitalInfo, Periodicity
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt.io.skf import Skf, VCRSkf
from tbmalt.physics.dftb.slaterkoster import sub_block_rot
from tbmalt.data.elements import chemical_symbols
from tbmalt.ml import Feed
from tbmalt.common.batch import pack, prepeat_interleave, bT, bT2
from tbmalt.common.maths.interpolation import PolyInterpU, BicubInterp
from tbmalt.common.maths.interpolation import CubicSpline as CSpline
from tbmalt.common import unique

# Todo:
#   - Need to determine why this is so slow for periodic systems.

Tensor = torch.Tensor
Array = np.ndarray
interp_dict = {'polynomial': PolyInterpU, 'spline': CSpline,
               'bicubic': BicubInterp}


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

    def __init__(self, on_sites: Dict[int, Tensor],
                 off_sites: Dict[Tuple[int, int, int, int], CubicSpline],
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

            >>> from tbmalt.io.skf import Skf
            >>> Zs = ['H', 'C', 'Au', 'S']
            >>> for file in [f'{i}-{j}.skf' for i in Zs for j in Zs]:
            >>>     Skf.read(file).write('my_skf.hdf5')

        Examples:
            >>> from tbmalt import OrbitalInfo, Geometry
            >>> from tbmalt.physics.dftb.feeds import ScipySkFeed
            >>> from tbmalt.io.skf import Skf
            >>> from ase.build import molecule
            >>> import urllib
            >>> import tarfile
            >>> from os.path import join
            >>> import torch
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
                    Skf.read(skf_file).write(path := join(tmpdir,
                                                          'auorg.hdf5'))

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
                off_sites[pair + key] = CubicSpline(
                    *clip(skf.grid, value), extrapolate=False)

            # The X-Y.skf file may not contain all information. Thus some info
            # must be loaded from its Y-X counterpart.
            if pair[0] != pair[1]:
                skf_2 = Skf.read(path, tuple(reversed(pair)), device=device)
                for key, value in skf_2.__getattribute__(target).items():
                    if key[0] < key[1]:
                        off_sites[pair + (*reversed(key),)] = CubicSpline(
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

    This feed uses polynomial/cubic spline/bicubic interpolation & Slater-Koster
    transformations to construct Hamiltonian and overlap matrices via the
    traditional DFTB method.

    Arguments:
        on_sites: On-site integrals presented as a dictionary keyed by atomic
            numbers & valued by a tensor specifying all of associated the on-
            site integrals; i.e. one for each orbital.
        off_sites: Off-site integrals; dictionary keyed by tuples of the form
            (z₁, z₂, s₁, s₂), where zᵢ & sᵢ are the atomic & shell numbers of
            the interactions, & valued by Scipy `CubicSpline` entities. Note
            that z₁ must be less than or equal to z₂, see the notes section
            for further information.
        interpolation: interpolation type.
        device: Device on which the feed object and its contents resides.
        dtype: dtype used by feed object.
        vcr: Compression radii in DFTB orbs.
        is_local_onsite: `is_local_onsite` allows for constructing chemical
            environment dependent on-site energies.

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
        `CubicSpline` instances should not attempt to extrapolate, but rather
        return NaNs, i.e. 'extrapolate=False'. When interpolating `SkFeed`
        instances will identify and set all NaNs to zero.

    """

    def __init__(self, on_sites: Dict[int, Tensor],
                 off_sites: Dict[Tuple[int, int, int, int], CubicSpline],
                 interpolation: Literal[CSpline, PolyInterpU, BicubInterp],
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
        self.interpolation = interpolation

        self._vcr = None
        self.is_local_onsite = False

    def _off_site_blocks(self, atomic_idx_1: Array, atomic_idx_2: Array,
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

        # `BicubInterp` interpolation works for VCR
        if self.interpolation is BicubInterp:
            cr = torch.stack([self.vcr[*bT2(atomic_idx_1)], self.vcr[*bT2(atomic_idx_2)]]).T

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
                if self.interpolation is BicubInterp:
                    inte = self.off_sites[(z_1, z_2, i, j)](cr, dist)
                else:
                    inte = self.off_sites[(z_1, z_2, i, j)](dist)

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
              periodic: Periodic object containing distance matrix and position
                  vectors for periodic images.

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
                inte = self.off_sites[(z_1, z_2, i, j)](dist)
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
        mask_shell = torch.zeros_like(self.on_sites[int(z_1)]).bool()
        mask_shell[:(torch.arange(len(orbs.shell_dict[int(z_1)]))
                     * 2 + 1).sum()] = True

        # Construct the on-site blocks (if any are present)
        if any(on_site):
            if not self.is_local_onsite:
                blks[on_site] = torch.diag(self.on_sites[int(z_1)][mask_shell])
            elif self.is_local_onsite:
                blks[on_site] = torch.diag_embed(
                    self.on_sites[int(z_1)][mask_shell], dim1=-2, dim2=-1)

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
            interpolation: str = 'polynomial',
            requires_grad: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> 'SkFeed':
        r"""Instantiate instance from an HDF5 database of Slater-Koster files.

        Instantiate a `SkFeed` instance for the specified elements using
        integral tables contained within a Slater-Koster HDF5 database.

        Arguments:
            path: Path to the HDF5 file from which integrals should be taken.
            species: Integrals will only be loaded for the requested species.
            target: Specifies which integrals should be loaded options are
                "hamiltonian" and "overlap".
            interpolation: Define interpolation type.
            device: Device on which the feed object and its contents resides.
            dtype: dtype used by feed object.

        Returns:
            sk_feed: A `SkFeed` instance with the requested integrals.

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
                    Skf.read(skf_file).write(path := join(tmpdir,
                                                          'auorg.hdf5'))

            # Preparation of system to calculate
            >>> geo = Geometry.from_ase_atoms(molecule('H2'))
            >>> orbs = OrbitalInfo(geo.atomic_numbers,
                                     shell_dict={1: [0]})

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

        def clip(x, y):
            # Removes leading zeros from the sk data which may cause errors
            # when fitting the CubicSpline.
            start = torch.nonzero(y.sum(0), as_tuple=True)[0][0]
            return x[start:], y[:, start:].transpose(0, 1)

        # Ensure a valid target is selected
        if target not in ['hamiltonian', 'overlap']:
            ValueError('Invalid target selected; '
                       'options are "hamiltonian" or "overlap"')

        on_sites, off_sites = {}, {}
        interpolation = interp_dict[interpolation]
        params = {'extrapolate': False} if interpolation is CubicSpline else {}

        # The species list must be sorted to ensure that the lowest atomic
        # number comes first in the species pair.
        for pair in combinations_with_replacement(sorted(species), 2):

            skf = Skf.read(path, pair, device=device) if interpolation is not BicubInterp else\
                VCRSkf.read(path, pair, device=device)

            # Loop over the off-site interactions & construct the splines.
            for key, value in skf.__getattribute__(target).items():

                if interpolation is BicubInterp:
                    off_sites[pair + key] = interpolation(
                        skf.compression_radii.to(device),
                        value.transpose(0, 1).to(device),
                        skf.grid.to(device), **params)
                else:
                    off_sites[pair + key] = interpolation(
                        *clip(skf.grid.to(device), value.to(device)), **params)

                    # Add variables for spline training
                    if interpolation is CSpline and requires_grad:
                        off_sites[pair + key].abcd.requires_grad_(True)

            # The X-Y.skf file may not contain all information. Thus some info
            # must be loaded from its Y-X counterpart.
            if pair[0] != pair[1]:
                skf_2 = Skf.read(path, tuple(reversed(pair)))
                for key, value in skf_2.__getattribute__(target).items():
                    if key[0] < key[1]:
                        off_sites[pair + (*reversed(key),)] = interpolation(
                            *clip(skf_2.grid.to(device), value.to(device)), **params)

                        # Add variables for spline training
                        if interpolation is CSpline and requires_grad:
                            off_sites[pair + (*reversed(key),)
                                      ].abcd.requires_grad_(True)

                # Add variables for spline training
                if interpolation is CSpline and requires_grad:
                    off_sites[pair + key].abcd.requires_grad_(True)

            else:  # Construct the onsite interactions
                # Repeated so there's 1 value per orbital not just per shell.
                on_sites_vals = skf.on_sites.repeat_interleave(
                    torch.arange(len(skf.on_sites), device=device) * 2 + 1)

                if target == 'overlap':  # use an identify matrix for S
                    on_sites_vals = torch.ones_like(on_sites_vals, dtype=dtype, device=device)

                on_sites[pair[0]] = on_sites_vals

        return cls(on_sites, off_sites, interpolation, dtype, device)

    @property
    def vcr(self):
        """The various compression radii."""
        return self._vcr

    @vcr.setter
    def vcr(self, value):
        self._vcr = value

    def local_onsite(self, value: Tensor):
        """Only when is_local_onsite is True local_onsite will return Tensor."""
        return value if self.is_local_onsite else None

    def __str__(self):
        elements = ', '.join([
            chemical_symbols[i]for i in sorted(self.on_sites.keys())])
        return f'{self.__class__.__name__}({elements})'

    def __repr__(self):
        return str(self)


class SkfOccupationFeed(Feed):
    """Occupations feed entity that derives its data from a skf file.

    Arguments:
        occupancies: A dictionary specifying the angular-momenta resolved
            occupancies, keyed by atomic numbers (as strings) and valued by
            tensors or parameters. When using occupancies as standard inputs,
            provide a `Dict[str, Tensor]`. When using them as optimisation
            targets, they should be specified as `ParameterDict[str, Parameter]`,
            which enables PyTorch to automatically detect and optimise these
            parameters. The dictionary keys must be strings to ensure
            compatibility with PyTorch's `ParameterDict` structure.

    Examples:
        >>> from tbmalt.physics.dftb.feeds import SkfOccupationFeed
        >>> #                                                   fs, fp, fd
        >>> l_resolved = SkfOccupationFeed({"79": torch.tensor([1., 0., 10.])})

    Notes:
        Note that this method discriminates between orbitals based only on
        the azimuthal number of the orbital & the species to which it belongs.
    """

    # Developer's Notes:
    # This class will be abstracted and extended to allow for specification
    # via shell number which will avoid the current limits which only allow
    # for minimal orbs sets.

    def __init__(self,
                 occupancies: Union[
                     Dict[str, Tensor],
                     ParameterDict[str, Parameter]]
                 ):
        super().__init__()

        self.occupancies = occupancies

        # Ensure that all dictionary keys are strings
        if not all(isinstance(key, str) for key in occupancies):
            raise TypeError(
                "Occupancy dictionary keys must be strings. This is required "
                "to maintain consistency with the `torch.nn.ParameterDict` "
                "type which enforces this behaviour."
            )

        # When the occupancies have autograd enabled then they are considered
        # to be optimisation targets. In such a case they should be parameter
        # instances stored in a parameter dictionary.
        if (isinstance(occupancies, dict)
                and any(value.requires_grad for value in occupancies.values())):
            warnings.warn(
                "One or more of the supplied occupancy values has `requires_grad` "
                "set to `True`. In such cases one should supply the occupancies "
                "as `torch.nn.Parameter` instances stored within a "
                "`torch.nn.ParameterDict` entity. This allows PyTorch to "
                "automatically detect valid optimisation targets. The current "
                "type structure will compute gradients for the selected "
                "occupancies but will not attempt to optimise them.",
                Warning
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
        return self.__class__(self.occupancies.__class__(
            {k: v.to(device=device) for k, v in self.occupancies.items()}
        ))

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
            >>> from tbmalt import OrbitalInfo
            >>> from tbmalt.physics.dftb.feeds import SkfOccupationFeed
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
                    Skf.read(skf_file).write(path := join(tmpdir,
                                                          'auorg.hdf5'))

            # Definition of feeds
            >>> o_feed = SkfOccupationFeed.from_database(path, [1, 6])
            >>> shell_dict = {1: [0], 6: [0, 1]}

            # Occupancy information of an example system
            >>> o_feed(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1]), shell_dict))
            tensor([2.0000, 0.6667, 0.6667, 0.6667,
                    1.0000, 1.0000, 1.0000, 1.0000])

        """
        # If the "requires_grad" keyword argument is set to "True" then the
        # occupancies are considered to be optimisable targets; and thus should
        # be `Parameter` types stored in a `ParameterDict` rather than `Tensor`
        # types stored in a `Dict`.
        struct = ParameterDict if kwargs.pop("requires_grad", False) else dict

        return cls(struct(
            {str(i): Skf.read(path, (i, i), **kwargs).occupations
             for i in species}))


class HubbardFeed(Feed):
    """Hubbard U feed entity that derives its data from a skf file.

    This provides a feed based method by which traditional DFTB Hubbard-U
    values can be accessed.

    Arguments:
        hubbard_us: A dictionary specifying the angular-momenta resolved
            Hubbard-Us, keyed by atomic numbers (as strings) and valued by
            tensors or parameters. When using Hubbard-Us as standard inputs,
            provide a `Dict[str, Tensor]`. When using them as optimisation
            targets, they should be specified as `ParameterDict[str, Parameter]`,
            which enables PyTorch to automatically detect and optimise these
            parameters. The dictionary keys must be strings to ensure
            compatibility with PyTorch's `ParameterDict` structure.

    Examples:
        >>> from tbmalt.physics.dftb.feeds import HubbardFeed
        >>> l_resolved = HubbardFeed({"1": torch.tensor([0.5])})

    Notes:
        Note that this method discriminates between orbitals based only on
        the azimuthal number of the orbital & the species to which it belongs.

    Todo:
        Add a test that throws an error if a shell resolved orbs is provided but
        `hubbard_u` is found to only be atom resolved; and vise versa. The skf
        database should also instruct the loader whether it is shell-resolved.
    """
    def __init__(self,
                 hubbard_us: Union[
                     Dict[str, Tensor],
                     ParameterDict[str, Parameter]]
                 ):
        super().__init__()

        self.hubbard_us = hubbard_us

        # Ensure that all dictionary keys are strings
        if not all(isinstance(key, str) for key in hubbard_us):
            raise TypeError(
                "Hubbard-U dictionary keys must be strings. This is required "
                "to maintain consistency with the `torch.nn.ParameterDict` "
                "type which enforces this behaviour."
            )

        # When the Hubbard-Us have autograd enabled then they are considered
        # to be optimisation targets. In such a case they should be parameter
        # instances stored in a parameter dictionary.
        if (isinstance(hubbard_us, dict)
                and any(value.requires_grad for value in hubbard_us.values())):
            warnings.warn(
                "One or more of the supplied Hubbard-U values has `requires_grad` "
                "set to `True`. In such cases one should supply the Hubbard-Us "
                "as `torch.nn.Parameter` instances stored within a "
                "`torch.nn.ParameterDict` entity. This allows PyTorch to "
                "automatically detect valid optimisation targets. The current "
                "type structure will compute gradients for the selected "
                "Hubbard-Us but will not attempt to optimise them.",
                Warning
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
        return self.__class__(self.hubbard_us.__class__(
            {k: v.to(device=device) for k, v in self.hubbard_us.items()}
        ))

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
            >>> from tbmalt import OrbitalInfo
            >>> from tbmalt.physics.dftb.feeds import HubbardFeed
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
                    Skf.read(skf_file).write(path := join(tmpdir,
                                                          'auorg.hdf5'))

            # Definition of feeds
            >>> u_feed = HubbardFeed.from_database(path, [1, 6])
            >>> shell_dict = {1: [0], 6: [0, 1]}

            # Hubbard U values of an example system
            >>> u_feed.forward(OrbitalInfo(torch.tensor([6, 1, 1, 1, 1]), shell_dict))
            tensor([0.3647, 0.4196, 0.4196, 0.4196, 0.4196])

        """
        # If the "requires_grad" keyword argument is set to "True" then the
        # Hubbard-Us are considered to be optimisable targets; and thus should
        # be `Parameter` types stored in a `ParameterDict` rather than `Tensor`
        # types stored in a `Dict`.
        struct = ParameterDict if kwargs.pop("requires_grad", False) else dict

        return cls(struct(
            {str(i): Skf.read(path, (i, i), **kwargs).hubbard_us
             for i in species}))


class RepulsiveSplineFeed(Feed):
    r"""Repulsive Feed using splines for DFTB calculations. Data is derived from a skf file.

    This feed uses splines to calculate the repulsive energy of a Geometry in the way it is defined for DFTB.

    Arguments:
        spline_data: Dictionary containing the the tuples of atomic number pairs as keys and the corresponding spline data as values.
    """

    def __init__(self, spline_data: Dict[Tuple, Tensor]):
        super().__init__()
        self.spline_data = {frozenset(interaction_pairs):data for interaction_pairs,data in spline_data.items()}

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by `RepulsiveSplineFeed` object."""
        return list(self.spline_data.values())[0].grid.dtype

    @property
    def device(self) -> torch.device:
        """The device on which the `RepulsiveSplineFeed` object resides."""
        return list(self.spline_data.values())[0].grid.device

    def __call__(self, geo: Union[Geometry, Tensor]) -> Tensor:
        r"""Calculate the repulsive energy of a Geometry.

        Arguments:
            geo: Geometry object(s) for which the repulsive energy should be calculated. Either a single Geometry object or a batch of Geometry objects.

        Returns:
            Erep: The repulsive energy of the Geometry object(s).
        """
        if geo.atomic_numbers.dim() == 1: #this means it is not a batch
            batch_size = 1
        else:
            batch_size = geo.atomic_numbers.size(dim=0)

        indxs = torch.tensor(range(geo.atomic_numbers.size(dim=-1)), device=self.device)
        indx_pairs = torch.combinations(indxs)
        
        Erep = torch.zeros((batch_size), device=self.device, dtype=self.dtype)
        for indx_pair in indx_pairs:
            atomnum1 = geo.atomic_numbers[..., indx_pair[0]].reshape((batch_size, ))
            atomnum2 = geo.atomic_numbers[..., indx_pair[1]].reshape((batch_size, ))

            distance = geo.distances[..., indx_pair[0], indx_pair[1]].reshape((batch_size, ))

            for batch_indx in range(batch_size):
                if atomnum1[batch_indx] == 0 or atomnum2[batch_indx] == 0:
                    continue
                Erep[batch_indx] += self._repulsive_calc(distance[batch_indx], atomnum1[batch_indx], atomnum2[batch_indx])

        return Erep

    def _repulsive_calc(self, distance: Tensor, atomnum1: Union[Tensor, int], atomnum2: Union[Tensor, int]) -> Tensor:
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
                return self._tail(distance, tail_start, spline.tail_coef)
            elif distance > exp_head_cutoff:
                for ind in range(len(spline.grid)):
                    if distance < spline.grid[ind]:
                        return self._spline(distance, spline.grid[ind-1], spline.spline_coef[ind-1])
            else:
                return self._exponential_head(distance, spline.exp_coef)
        return 0
   
    @classmethod
    def _exponential_head(cls, distance: Tensor, coeffs: Tensor) -> Tensor:
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

        return torch.exp(-a1*distance + a2) + a3

    @classmethod 
    def _spline(cls, distance: Tensor, start: Tensor, coeffs: Tensor) -> Tensor:
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
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3
        return energy

    @classmethod 
    def _tail(cls, distance: Tensor, start: Tensor, coeffs: Tensor) -> Tensor:
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
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3 + coeffs[4]*rDiff**4 + coeffs[5]*rDiff**5
        return energy



    @classmethod
    def from_database(cls, path: str, species: List[int], dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> 'RepulsiveSplineFeed':
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
        return cls({interaction_pair: Skf.read(path, interaction_pair, device=device, dtype=dtype).r_spline for interaction_pair in interaction_pairs})



