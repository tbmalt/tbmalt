# -*- coding: utf-8 -*-
"""Slater-Koster integral feed objects.

This contains all Slater-Koster integral feed objects. These objects are
responsible for generating the Slater-Koster integrals and for constructing
the associated Hamiltonian and overlap matrices.
"""
import numpy as np
from itertools import combinations_with_replacement
from typing import List, Literal, Optional, Dict, Tuple
from scipy.interpolate import CubicSpline
import torch

from tbmalt import Geometry, Basis
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt.io.skf import Skf
from tbmalt.physics.dftb.slaterkoster import sub_block_rot
from tbmalt.data.elements import chemical_symbols

Tensor = torch.Tensor
Array = np.ndarray


def _enforce_numpy(v):
    """Helper function to ensure entity is a numpy array."""
    if isinstance(v, np.ndarray):
        return v
    elif isinstance(v, Tensor):
        return v.detach().cpu().numpy()
    else:
        return np.array(v)


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
        error is the second atomic number is greater than the first; e.g. the
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

    def _off_site_blocks(self, atomic_idx_1: Array, atomic_idx_2: Array,
                         geometry: Geometry, basis: Basis) -> Tensor:
        """Compute atomic interaction blocks (off-site only).

        Constructs the off-site atomic blocks using Slater-Koster integral
        tables.

        Arguments:
              atomic_idx_1: Indices of the 1'st atom associated with each
                desired interaction block.
              atomic_idx_2: Indices of the 2'nd atom associated with each
                  desired interaction block.
              geometry: The systems to which the atomic indices relate.
              basis: Orbital information associated with said systems.

          Returns:
              blocks: Requested atomic interaction sub-blocks.
        """

        # Identify atomic numbers associated with the interaction
        z_1 = int(geometry.atomic_numbers[atomic_idx_1[0:1].T])
        z_2 = int(geometry.atomic_numbers[atomic_idx_2[0:1].T])

        # Get the species' shell lists (basically a list of azimuthal numbers)
        shells_1, shells_2 = basis.shell_dict[z_1], basis.shell_dict[z_2]

        # Inter-atomic distance and distance vector calculator.
        dist_vec = (geometry.positions[atomic_idx_2.T]
                    - geometry.positions[atomic_idx_1.T])
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

    def blocks(self, atomic_idx_1: Array, atomic_idx_2: Array,
               geometry: Geometry, basis: Basis, **kwargs) -> Tensor:
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
              basis: Orbital information associated with said systems.

          Returns:
              blocks: Requested atomic interaction sub-blocks.

          Warnings:
              This is not backpropagatable.

        """
        # Atomic index arrays must be numpy arrays
        atomic_idx_1 = _enforce_numpy(atomic_idx_1)
        atomic_idx_2 = _enforce_numpy(atomic_idx_2)

        # Get the atomic numbers of the atoms
        zs_1 = (zs := geometry.atomic_numbers)[atomic_idx_1.T]
        zs_2 = zs[atomic_idx_2.T]

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
        n_rows, n_cols = basis.n_orbs_on_species(torch.stack((z_1, z_2)))
        blks = torch.empty(len(atomic_idx_1), n_rows, n_cols, dtype=self.dtype,
                           device=self.device)

        # Identify which are on-site blocks and which are off-site
        on_site = self._partition_blocks(atomic_idx_1, atomic_idx_2)

        if any(on_site):  # Construct the on-site blocks (if any are present)
            blks[on_site] = torch.diag(self.on_sites[int(z_1)])

        if any(~on_site):  # Then the off-site blocks
            blks[~on_site] = self._off_site_blocks(
                atomic_idx_1[~on_site], atomic_idx_2[~on_site], geometry, basis)

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
            from human readable skf files, or a directory thereof. Thus, any
            such files must first be converted into their binary equivalent.
            This reduces overhead & file format error instabilities. The code
            block provide below shows how this can be done:

            >>> from tbmalt.io.skf import Skf
            >>> Zs = ['H', 'C', 'Au', 'S']
            >>> for file in [f'{i}-{j}.skf' for i in Zs for j in Zs]:
            >>>     Skf.read(file).write('my_skf.hdf5')


        """
        # As C-H & C-H interactions are the same only one needs to be loaded.
        # Thus only off-site interactions where z₁≤z₂ are generated. However,
        # integrals are split over the two Slater-Koster tables, thus both must
        # be loaded.

        def clip(x, y):
            # Removes leading zeros from the sk data which may cause errors
            # when fitting the CubicSpline.
            start = torch.nonzero(y.sum(0), as_tuple=True)[0][0]
            return x[start:], y[:, start:].T  # Transpose here to save effort

        # Ensure a valid target is selected
        if target not in ['hamiltonian', 'overlap']:
            ValueError('Invalid target selected; '
                       'options are "hamiltonian" or "overlap"')

        on_sites, off_sites = {}, {}

        # The species list must be sorted to ensure that the lowest atomic
        # number comes first in the species pair.
        for pair in combinations_with_replacement(sorted(species), 2):
            skf = Skf.read(path, pair)
            # Loop over the off-site interactions & construct the splines.
            for key, value in skf.__getattribute__(target).items():
                off_sites[pair + key] = CubicSpline(
                    *clip(skf.grid, value), extrapolate=False)

            # The X-Y.skf file may not contain all information. Thus some info
            # must be loaded from its Y-X counterpart.
            if pair[0] != pair[1]:
                skf_2 = Skf.read(path, tuple(reversed(pair)))
                for key, value in skf_2.__getattribute__(target).items():
                    if key[0] < key[1]:
                        off_sites[pair + (*reversed(key),)] = CubicSpline(
                            *clip(skf_2.grid, value), extrapolate=False)

            else:  # Construct the onsite interactions
                # Repeated so theres 1 value per orbital not just per shell.
                on_sites_vals = skf.on_sites.repeat_interleave(
                    torch.arange(len(skf.on_sites)) * 2 + 1).to(device)

                if target == 'overlap':  # use an identify matrix for S
                    on_sites_vals = torch.ones_like(on_sites_vals)

                on_sites[pair[0]] = on_sites_vals

        return cls(on_sites, off_sites, dtype, device)

    def __str__(self):
        elements = ', '.join([
            chemical_symbols[i]for i in sorted(self.on_sites.keys())])
        return f'{self.__class__.__name__}({elements})'

    def __repr__(self):
        return str(self)
