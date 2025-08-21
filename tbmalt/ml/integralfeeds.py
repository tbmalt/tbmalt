# -*- coding: utf-8 -*-
"""Electronic integral feed objects.

This contains all the electronic integral feed objects. These objects are
responsible for generating the Hamiltonian and overlap matrices. The on-site
and off-site blocks are constructed by the `on_site_blocks` & `off_site_blocks`
class methods respectively.

Warning:
     This is development stage code and is subject to significant change.
"""
from abc import ABC
from typing import Union

import numpy as np
import torch
from h5py import Group
from numpy import ndarray as Array
from tbmalt import Geometry, OrbitalInfo
from tbmalt.structures.geometry import atomic_pair_indices
from tbmalt.ml import Feed
from tbmalt.ml.calculator import Calculator
from tbmalt.common.batch import bT
from torch import Tensor


def indices(dims, dtype=None, device=None):
    """Pytorch implementation of numpy.indices

    Note this is a temporary function which will be abstracted.
    """
    dtype = torch.long if dtype is None else dtype
    dims = tuple(dims)
    res = torch.empty((n_dims := len(dims),) + dims, dtype=dtype)
    sp = (1,) * n_dims
    for i, d in enumerate(dims):
        idx = torch.arange(d, dtype=dtype).view(
            sp[:i] + (d,) + sp[i + 1:]
        )
        res[i] = idx
    return res.to(device=device)


class IntegralFeed(Feed, ABC):
    r"""ABC for Hamiltonian and overlap matrix constructors.

    Subclasses of this abstract base class are responsible for constructing
    the Hamiltonian and overlap matrices.

    Arguments:
        device: Device on which the feed object and its contents resides.
        dtype: Floating point dtype used by feed object.

    """

    def __init__(self, dtype, device):
        super().__init__()
        # These variables must NEVER be modified outside the .to method!
        self.__device = device
        self.__dtype = dtype

    def __init_subclass__(cls, **kwargs):
        """Initialises subclass structures.

        When a new subclass is initialised this will decorate various methods
        with a selection of wrapper functions. These wrappers will automate
        some of the more repetitive error handling operations needed.
        """
        pass

    @property
    def device(self) -> torch.device:
        """The device on which the feed object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        # Instruct users to use the ".to" method if wanting to change device.
        name = self.__class__.__name__
        raise AttributeError(f'{name} object\'s dtype can only be modified '
                             'via the ".to" method.')

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by feed object."""
        return self.__dtype

    def __enforce_orbital_consistency(self):
        """Wrapper function to ensure orbital consistency.

        This ensures that all the requested sub-blocks are of the same shape
        during calls to the ``off_site_blocks`` & ``on_site_blocks`` methods.
        Issues will be encountered if different sized sub-blocks are requested
        during a single call; this wrapper just ensures a helpful exception is
        provided to the user.
        """
        pass

    @staticmethod
    def partition_blocks(atomic_idx_1: Tensor, atomic_idx_2: Tensor
                         ) -> Tensor:
        """Mask identifying on-site interaction pairs.

        This helper function constructs a mask which is True wherever the
        associated atom index pair corresponds to an on-site interaction.

        Arguments:
            atomic_idx_1: Atomic indices of the 1'st atom associated with each
                desired interaction block.
            atomic_idx_2: Atomic indices of the 2'nd atom associated with each
                desired interaction block.

        Returns:
            on_site_mask: A tensor that is truthy wherever the atom index
                pair corresponds to an on-site interaction.
        """
        on_site_mask = torch.eq(atomic_idx_1, atomic_idx_2)
        if on_site_mask.ndim != 1:
            on_site_mask = on_site_mask.all(dim=-1)
        return on_site_mask

    def blocks(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
               geometry: Geometry, orbs: OrbitalInfo, **kwargs) -> Tensor:
        r"""Compute atomic interaction blocks.

        Returns the atomic blocks associated with the atoms in ``atomic_idx_1``
        interacting with those in ``atomic_idx_2``. The № of interaction blocks
        returned will be equal to the length of the two index lists; i.e. *not*
        one for every combination.

        Arguments:
            atomic_idx_1: Atomic indices of the 1'st atom associated with each
                desired interaction block.
            atomic_idx_2: Atomic indices of the 2'nd atom associated with each
                desired interaction block.
            geometry: The systems to which the atomic indices relate.
            orbs: Orbital information associated with said systems.

        Returns:
            blocks: Requested atomic interaction sub-blocks.

        Warnings:
            All requested blocks must have the same shape & size.

        Notes:
            If both indices of an index pair refer to the same atom then the
            block will represent an on-site interaction, otherwise it will be
            an off-site interaction.

            The index tensor has been split in two; while this increases
            verbosity it decreases the complexity needed to ensure batch
            agnosticism.

            If operating on a batch of systems; index tensors will be 2xN where
            the 1'st & 2'nd rows specify the batch & atom indices respectively.
            Otherwise, they will be one dimensional tensors of length N where N
            is the number of interactions.

        """
        # Developers Notes
        # This method is mostly called from ``matrix``; which loops over all
        # species combinations, H-C, H-C, C-C, ..., etc. & request all of the
        # interaction blocks associated with that species pair.
        #
        # The index arrays can be used to retrieve information, such as
        # positions, like so ``geometry.positions[atomic_idx_1.T]``.
        #
        # The ``partition_blocks`` helper method makes splitting interactions
        # into off- and on-site blocks easier when necessary.
        #
        raise NotImplementedError()

    def matrix(self, geometry: Geometry, orbs: OrbitalInfo, **kwargs) -> Tensor:
        """Construct the hermitian matrix associated with this feed.

        Arguments:
            geometry: Systems whose matrices are to be constructed.
            orbs: Orbital information associated with said systems.

        Keyword Arguments:
            kwargs: Any keyword arguments provided are passed during calls to
                the `blocks` method.

        Returns:
            matrix: The resulting matrices.

        Warnings:
            This method is in rapid development and is thus subject to change
            and or abstraction.

        """
        # Developers Notes
        # There is still quite a bit left to do in this function; even when one
        # discounts optimisation, cleaning, and improving batch agnosticism.

        # Construct the matrix into which the results will be placed
        mat = torch.zeros(orbs.orbital_matrix_shape,
                          dtype=self.dtype, device=self.device)

        # Loop over pairwise interactions by species type
        for pair, a_idx in atomic_pair_indices(
                geometry, ignore_periodicity=True):
            a_idx = a_idx.T

            # Reshape atom index list to be more amenable to advanced indexing.
            a_idx_l = a_idx[:, :-1].squeeze(1)
            b_idx_l = a_idx[:, 3 - a_idx.shape[-1]::2].squeeze(1)

            # Get the matrix indices associated with the target blocks.
            blk_idx = self.atomic_block_indices(a_idx_l, b_idx_l, orbs)

            # Construct the blocks for these interactions
            blks = self.blocks(a_idx_l, b_idx_l, geometry, orbs, **kwargs)

            # Assign data to the matrix. As on-site blocks are not masked out
            # during the transpose assignment the transpose assignment must
            # be done first (this is only matters for complex diagonal blocks).
            mat.transpose(-1, -2)[*blk_idx] = blks.conj()
            mat[*blk_idx] = blks

        return mat

    def matrix_from_calculator(self, calculator: Calculator, **kwargs):
        """Construct the hermitian matrix associated with this feed.

        This method extracts the required arguments from the supplied
        ``calculator`` argument & redirects to the `matrix` method. This allows
        for a `Calculator` to call a feeds `matrix` method in a call signature
        agnostic manner.

        Args:
            calculator: a calculator object from which the necessary data can
                be extracted.

        Keyword Arguments:
            kwargs: Any keyword arguments provided are passed during calls to
                the `blocks` method.

        Returns:
            matrix: The resulting matrices.

        Notes:
                Each `IntegralFeed` object's `matrix` method may require any
                number of arbitrary arguments. Consequently, it is unreasonable
                to expect a `Calculator` instance to know how to correctly
                invoke the `matrix` method for any given feed, particularly
                without resorting to introspection and reflection, which would
                introduce an unnecessary degree of complexity. Therefore, this
                pseudo-overload method is established, enabling `Calculator`s
                to simply pass themselves in as the sole argument. This
                approach allows the feed to extract the necessary data
                independently and make the call to `matrix` itself. While this
                is potentially more prone to errors, the errors that may arise
                are expected to be clearer and easier to resolve, thus
                enhancing the overall maintainability and debuggability.
        """

        return self.matrix(calculator.geometry, calculator.orbs, **kwargs)

    def atomic_block_indices(self, atomic_idx_1: Tensor, atomic_idx_2: Tensor,
                             orbs: OrbitalInfo) -> Tensor:
        """Returns the indices of the specified blocks.

        This method identifies the blocks associated with the specified atom
        pairs and returns the indexing arrays that can be used to retrieve
        said blocks.

        Arguments:
            atomic_idx_1: Atomic indices of the 1'st atom associated with each
                desired interaction block.
            atomic_idx_2: Atomic indices of the 2'nd atom associated with each
                desired interaction block.
            orbs: Orbital information associated with said systems.

        Returns:
            block_indices: the indices of the specified blocks.
        """

        is_batch = atomic_idx_1.ndim == 2

        if isinstance(atomic_idx_1, Array):
            import warnings
            warnings.warn(
                "Indices must be provided as torch tensors not numpy arrays")
            atomic_idx_1 = torch.tensor(atomic_idx_1)
            atomic_idx_2 = torch.tensor(atomic_idx_2)

        # Block indices are mostly used in their transposed form
        idx_i, idx_j = bT(atomic_idx_1), bT(atomic_idx_2)

        # Non-batch tensors must be inflated to permit batch agnosticism
        idx_i, idx_j = torch.atleast_2d(idx_i), torch.atleast_2d(idx_j)

        # Find the index at which each atomic block starts.
        blk_starts = (opa := orbs.orbs_per_atom).cumsum(-1) - opa

        # Row/column offset template; used to build the indices specifying
        # the location of a block's elements in the target matrix.
        # The internal list comprehension just gets the number of orbitals
        # present on the first and second atoms.
        rt, ct = indices((opa[*i[..., 0:1]].item() for i in [idx_i, idx_j]),
                         device=self.device)

        # Get the block's row & column indices.
        blk_idx = torch.stack((rt + blk_starts[*idx_i].view(-1, 1, 1),
                               ct + blk_starts[*idx_j].view(-1, 1, 1)))

        if is_batch:  # Add bach indices, if appropriate.
            b_idx = idx_i[0]
            blk_idx = torch.cat((bT(b_idx.expand(*rt.T.shape, -1))[None, :], blk_idx))

        return blk_idx

    def to(self, device: torch.device) -> 'IntegralFeed':
        """Returns a copy of the `SkFeed` instance on the specified device.

        This method creates and returns a new copy of the `SkFeed` instance
        on the specified device "``device``".

        Arguments:
            device: Device on which the clone should be placed.

        Returns:
            sk_feed: A copy of the `SkFeed` instance placed on the specified
                device.

        Notes:
            If the `SkFeed` instance is already on the desired device then
            `self` will be returned.
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, source: Union[str, Group]) -> 'IntegralFeed':
        """Load a stored integral feed object.

        This is only for loading preexisting ``IntegralFeed`` objects, from
        HDF5 databases, not instantiating new ones.

        Arguments:
            source: Name of a file to load the integral feed from or an HDF5
                group from which it can be extracted.

        Returns:
            ski_feed: A ``IntegralFeed`` object.

        """
        raise NotImplementedError()

    def save(self, target: Union[str, Group]):
        """Save the integral feed to an HDF5 database.

        Arguments:
            target: Name of a file to save the integral feed to or an HDF5
                group in which it can be saved.

        Notes:
            If `target` is a string then a new HDF5 database will be created
            at the path specified by the string. If an HDF5 entity was given
            then a new HDF5 group will be created and added to it.

            Under no circumstances should this just pickle an object. Doing so
            is unstable, unsafe and inflexible.

            It is good practice to save the name of the class so that the code
            automatically knows how to unpack it.
        """
        if isinstance(target, str):
            # Create a HDF5 database and save the feed to it
            raise NotImplementedError()
        elif isinstance(target, Group):
            # Create a new group, save the feed in it and add it to the Group
            raise NotImplementedError()
