# -*- coding: utf-8 -*-
"""Electronic integral feed objects.

This contains all the electronic integral feed objects. These objects are
responsible for generating the Hamiltonian and overlap matrices. The on-site
and off-site blocks are constructed by the `on_site_blocks` & `off_site_blocks`
class methods respectively.

Warning this is development stage code and is subject to significant change.
"""
from abc import ABC
from typing import Union, Tuple

import numpy as np
import torch
from h5py import Group
from numpy import ndarray as Array
from tbmalt import Geometry, Basis
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


class IntegralFeed(ABC):
    """ABC for Hamiltonian and overlap matrix constructors.

    Subclasses of this abstract base class are responsible for constructing
    the Hamiltonian and overlap matrices.

    Arguments:
        device: Device on which the feed object and its contents resides.
        dtype: Floating point dtype used by feed object.
    """

    def __init__(self, dtype, device):
        # These variables must NEVER be modified outside of the .to method!
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

    def __enforce_array(self):
        """Wrapper function to ensure atomic indices are supplied as arrays.

        Issues arise when using torch.Tensors for advanced indexing operations
        in bach agnostic code. Thus numpy.ndarrays are used instead. If a user
        inadvertently supplies a torch.Tensor then the results can be highly
        unpredictable. Thus this wrapper checks for and auto-converts torch
        tensors into numpy arrays as needed before warning the user.
        """
        pass

    @staticmethod
    def _partition_blocks(atomic_idx_1: Array, atomic_idx_2: Array
                          ) -> Array:
        """Mask identifying on-site interaction pairs.

        This helper function constructs a mask which is True wherever the
        associated atom index pair corresponds to an on-site interaction.

        Arguments:
            atomic_idx_1: Atomic indices of the 1'st atom associated with each
                desired interaction block.
            atomic_idx_2: Atomic indices of the 2'nd atom associated with each
                desired interaction block.

        Returns:
            on_site_mask: A numpy array that is truthy wherever the atom index
                pair corresponds to an on-site interaction.
        """
        return np.array(list(map(np.all, atomic_idx_1 == atomic_idx_2)))

    # @abstractmethod
    def blocks(self, atomic_idx_1: Array, atomic_idx_2: Array,
               geometry: Geometry, basis: Basis) -> Tensor:
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
            basis: Orbital information associated with said systems.

        Returns:
            blocks: Requested atomic interaction sub-blocks.

        Warnings:
            All requested blocks must have the same shape & size.

        Notes:
            If both indices of an index pair refer to the same atom then the
            block will represent an on-site interaction, otherwise it will be
            an off-site interaction.

            Indices are passed as numpy arrays as advanced indexing is not
            always triggered when indexing with tensors and as no clean
            alternative is readily apparent. Furthermore, the index array has
            been split in two; while this increases verbosity it decreases the
            complexity needed to ensure batch agnosticism.

            If operating on a batch of systems; index arrays will be 2xN where
            the 1'st & 2'nd rows specify the batch & atom indices respectively.
            Otherwise, they will be one dimensional arrays of length N where N
            is the number of interactions.

        """
        # Developers Notes
        # Indexing operations can get messy here due to inconsistencies in how
        # advanced indexing is implemented in pytorch. Normally, this would not
        # be a significant issue; however, batch agnostic code is hard to write
        # when advanced indexing works for some systems but not others. Thus
        # a pair of numpy arrays are used instead. Do *NOT* use torch tensors
        # as they will result in highly inconsistent behaviour. Later, a wrapper
        # will decorate this function to ensure spurious tensors are converted
        # to arrays. Any alternative solutions are very much welcome here!
        #
        # This method is mostly called from ``matrix``; which loops over all
        # species combinations, H-C, H-C, C-C, ..., etc. & request all of the
        # interaction blocks associated with that species pair.
        #
        # The index arrays can be used to retrieve information, such as
        # positions, like so ``geometry.positions[atomic_idx_1.T]``.
        #
        # The ``_partition_blocks`` helper method make splitting interactions
        # into off- and on-site blocks easier when necessary.
        #
        raise NotImplementedError()

    def matrix(self, geometry: Geometry, basis: Basis) -> Tensor:
        """Construct the hermitian matrix associated with this feed.

        Arguments:
            geometry: Systems whose matrices are to be constructed.
            basis: Orbital information associated with said systems.

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
        mat = torch.zeros(basis.orbital_matrix_shape,
                          dtype=self.dtype, device=self.device)

        # Identify all unique species combinations.
        unique_interactions = torch.combinations(
            geometry.unique_atomic_numbers(), with_replacement=True)

        # Find the index at which each atomic block starts.
        block_starts = (opa := basis.orbs_per_atom).cumsum(-1) - opa

        # Construct an element-element pair matrix
        an_mat_a = basis.atomic_number_matrix('atomic')

        # Loop over the unique interaction pairs
        for interaction in unique_interactions:
            # Create a tensor specifying the indices for each atom in each
            # relevant interaction; but only look for interactions within
            # systems not between different systems (obviously). For a batch
            # of systems this will take the form:
            #   [[batch_idx, atom_1_idx, atom_2_idx]... for each interaction]
            # and for a single system:
            #   [[atom_1_idx, atom_2_idx]... for each interaction]
            a_idx = torch.nonzero((an_mat_a == interaction).all(-1))

            # If no relevant interactions are encountered then skip this loop.
            # This can be removed once unique_atomic_numbers has been updated.
            if a_idx.nelement() == 0:
                continue

            # Cull the lower triangle of homo-atomic interactions to avoid
            # double computation.
            if interaction[0] == interaction[1]:
                a_idx = a_idx[torch.where(a_idx[..., -2].le(a_idx[..., -1]))]

            # Reshape atom index list to be more amenable to advanced indexing.
            # This approach is a little messy but reduces memory on the cpu.
            a_idx_l = a_idx[:, :-1].squeeze(1).cpu().numpy()
            b_idx_l = a_idx[:, 3 - a_idx.shape[-1]::2].squeeze(1).cpu().numpy()

            # Construct the blocks for these interactions
            blocks = self.blocks(a_idx_l, b_idx_l, geometry, basis)

            # During assigment, data is normally assigned row-by-row. While
            # this isn't an issue for single row blocks it will cause mangling
            # of multi-row data to; e.g, when attempting to assign two 3x3
            # blocks [a-i & j-r] to a tensor, the desired outcome would be
            # tensor A), however, a more likely outcome is the tensor B).
            # A)┌                           ┐ B)┌                           ┐
            #   │ .  .  .  .  .  .  .  .  . │   │ .  .  .  .  .  .  .  .  . │
            #   │ a  b  c  .  .  .  j  k  l │   │ a  b  c  .  .  .  d  e  f │
            #   │ d  e  f  .  .  .  m  n  o │   │ g  h  i  .  .  .  j  k  l │
            #   │ g  h  i  .  .  .  p  q  r │   │ m  n  o  .  .  .  p  q  r │
            #   │ .  .  .  .  .  .  .  .  . │   │ .  .  .  .  .  .  .  .  . │
            #   └                           ┘   └                           ┘
            # Thus an indexing tensor is required that can map the flattened
            # block data into the results tensor without mangling. The code
            # for this operation have been intentionally left verbose to make
            # modification easier.

            m, n = blocks.shape[-2:]  # Block shape

            # Row and column offset reference tensors; this is effectively an
            # index mask that is able to rebuild a flattened block.
            rt, ct = indices((m, n), device=self.device)

            # Row & column indicating where each block starts in the final
            # matrix; this should be the top left corner.
            block_start_row = block_starts[a_idx_l.T]
            block_start_column = block_starts[b_idx_l.T]

            # Finally build the row, column, & batch index maps
            c_i = (ct.view(-1, 1) + block_start_column).T.flatten()
            r_i = (rt.view(-1, 1) + block_start_row).T.flatten()
            b_i = a_idx.T[0].repeat_interleave(n*m) if mat.ndim == 3 else ...

            # Assign data to the results matrix. Note that on-site blocks are
            # not masked out during the transpose assignment. Thus, transpose
            # assignment must be done first.
            mat[b_i, c_i, r_i] = blocks.flatten().conj()
            mat[b_i, r_i, c_i] = blocks.flatten()

        return mat

    #@abstractmethod
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


class TestFeed(IntegralFeed):
    def __init__(self):
        super().__init__(torch.long, None)
        self.c = 0

    def _off_site_blocks(self, atomic_idx_1: Array, atomic_idx_2: Array,
                         geometry: Geometry, basis: Basis) -> Tensor:
        # The off-site blocks returned by this implementation are complete
        # gibberish; but are a little more useful during initial debugging
        # than using random numbers.
        opa = basis.orbs_per_atom
        no1, no2 = opa[atomic_idx_1.T][0], opa[atomic_idx_2.T][0]

        # Generate reproducible gibberish
        t1 = torch.arange(no1 * no2, dtype=torch.long).view(
            1, no1, no2).repeat(len(atomic_idx_1), 1, 1).clone()
        block = t1 + torch.arange(self.c, self.c + len(atomic_idx_1)).view(-1, 1, 1)
        self.c += len(atomic_idx_1)
        return block

    def _on_site_blocks(self, atomic_idx: Array, geometry: Geometry,
                       basis: Basis) -> Tensor:
        # Again this will return gibberish
        no = basis.orbs_per_atom[atomic_idx.T][0]
        return torch.eye(no, dtype=self.dtype).repeat((len(atomic_idx), 1, 1))

    def blocks(self, atomic_idx_1: Array, atomic_idx_2: Array,
               geometry: Geometry, basis: Basis) -> Tensor:

        opa = basis.orbs_per_atom
        no1, no2 = opa[atomic_idx_1.T][0], opa[atomic_idx_2.T][0]

        results = torch.empty(len(atomic_idx_1), no1, no2, dtype=self.dtype)

        is_on_site = self._partition_blocks(atomic_idx_1, atomic_idx_2)

        if any(is_on_site):
            results[is_on_site] = self._on_site_blocks(
                atomic_idx_1[is_on_site], geometry, basis)

        if any(~is_on_site):
            results[~is_on_site] = self._off_site_blocks(
                atomic_idx_1[~is_on_site], atomic_idx_2[~is_on_site], geometry, basis)

        return results


if __name__ == '__main__':
    from ase.build import molecule
    torch.set_printoptions(linewidth=2000)
    # Construct and look at a pair of systems a bach & an isolated system.
    for geometry in [Geometry.from_ase_atoms([molecule('CH4'), molecule('C2H4')]),
                     Geometry.from_ase_atoms(molecule('CH4'))]:
        # Build the basis object
        basis = Basis(geometry.atomic_numbers, {1: [0], 6: [0, 1]}, True)
        # Construct a simple test feed
        feed = TestFeed()
        # Get the matrix
        matrix = feed.matrix(geometry, basis)
        # Print it out and check it is symmetric.
        print(matrix)
        assert torch.allclose(matrix, matrix.transpose(-1,-2))