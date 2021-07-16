# -*- coding: utf-8 -*-
"""A container to hold data associated with a chemical system's bases.

This module provides the `Basis` data structure class and its associated code.
The `Basis` class is intended to hold data needed to describe the number and
identities of a chemical systems bases.
"""
from typing import Dict, List, Union, Literal, Optional, Any
import numpy as np
from h5py import Group
import torch
from torch import Tensor, Size, arange
from tbmalt.common.batch import pack
from tbmalt.common import split_by_size
from tbmalt.common.constants import MAX_ATOMIC_NUMBER


Form = Literal['full', 'shell', 'atomic']


class Basis:
    """Data container for all information relating to a system's basis set.

    This class collects all information about a system's basis set and its
    orbitals into one place. This also permits calculations to

    Arguments:
        atomic_numbers: Atomic numbers of the atoms present in the system.
        shell_dict: A dictionary that yields the angular momenta of each shell
            for a given atomic number e.g. `{1: [0], 2: [0, 0, 0] 6: [0, 1]}`
            designates H as having 1s shell, He as having 3s shells and C as
            having 1s & 1p shell. Values must be lists of integers, not torch
            tensors.
        shell_resolved: If True, calculations will be shell-resolved, if False
            they will be atom-resolved. This is used to automatically
            return the correct resolution information. [DEFAULT=False]

    Attributes:
        n_orbitals (Tensor): orbital count.
        n_shells (Tensor): shell count.
        n_atoms (Tensor): atoms count.
        orbital_matrix_shape (Size): Anticipated orbital matrix shape, e.g.
            Hamiltonian or overlap matrices.
        shell_matrix_shape (Size): Similar to ``orbital_matrix_shape`` but
            with only one matrix-element per shell.
        atomic_matrix_shape (Size): Similar to ``shell_matrix_shape`` but
            with only one matrix-element per atom.
        shell_ls (Tensor): Azimuthal number associated with each shell.
        shell_ns (Tensor): The number of each shell, as defined by the order
            in which they were specified for relevant species.

    Warnings:
        The order that each element's shell is specified in ``shell_dict`` is
        taken as the order in which integrals are yielded by `SkFeed` objects;
        i.e. `SkFeed.off_site(..., [atom_1_shell, atom_2_shell], ...)` will
        return the integral associated with the atom_1_shell'th shell on atom
        1 and the atom_2_shell'th shell on atom 2.
    """
    # Developers Notes:
    # Basis instances only ever yield tensors used for masking or indexing
    # operations. As such the device on which they return results is somewhat
    # irrelevant. However, results are placed on the anticipated device anyway
    # to maintain expected behaviour.

    def __init__(self, atomic_numbers: Union[List[Tensor], Tensor],
                 shell_dict: Dict[int, List[int]],
                 shell_resolved: bool = False):

        # Ensure that shell_dict yields lists with integers in & nothing else
        if not all([isinstance(j, int) and isinstance(i, list)
                    for i in shell_dict.values() for j in i]):
            raise ValueError('"shell_dict" must yield lists of integers.')

        # __DATA ATTRIBUTES__
        self.shell_resolved = shell_resolved
        self.shell_dict = shell_dict
        self.atomic_numbers = pack(atomic_numbers)

        # __META ATTRIBUTES__
        self.__device = self.atomic_numbers.device  # <- Don't change directly
        # pylint: disable=C0103
        batch = self.atomic_numbers.ndim == 2  # <─┬ Used only during init
        kwargs = {'device': self.__device}  # <─────┘

        # __HELPER ATTRIBUTES__

        def map_to_atom_number(val):
            """Helps to build _shells/orbitals_per_species"""
            return (torch.zeros(MAX_ATOMIC_NUMBER, dtype=torch.long)
                .scatter(0, torch.tensor(list(shell_dict.keys())),
                torch.tensor(val)).to(**kwargs))

        # These allow for the Nº of shells/orbitals on a given species to be
        # looked-up in a batch-wise manor without the need for dictionaries.
        self._shells_per_species = map_to_atom_number(
            [len(v) for v in shell_dict.values()])
        self._orbitals_per_species = map_to_atom_number(
            [sum([l * 2 + 1 for l in v]) for v in shell_dict.values()])

        self.shell_ns, self.shell_ls = torch.tensor(
            [(i, l) for n in self.atomic_numbers.view(-1) if n != 0
             for i, l in enumerate(shell_dict[int(n)])], **kwargs).T

        if batch:
            def batch_reshape(t):
                """Reshape "shell_ns" & "shell_ls" if batched"""
                return pack(split_by_size(t,
                    self._shells_per_species[self.atomic_numbers].sum(-1)),
                    value=-1)
            self.shell_ns = batch_reshape(self.shell_ns)
            self.shell_ls = batch_reshape(self.shell_ls)

        # __COUNTABLE ATTRIBUTES__
        self.n_atoms: Tensor = self.atomic_numbers.count_nonzero(-1)
        self.n_shells: Tensor = (self.shell_ls != -1).sum(-1)
        self.n_orbitals: Tensor = self.orbs_per_atom.sum(-1)

        # __SHAPE ATTRIBUTES__
        m1 = self.n_atoms.max()
        m2 = self.n_shells.max()
        m3 = self.n_orbitals.max()
        # _n adds an extra dimension when in batch mode
        n = Size([len(self.atomic_numbers)]) if batch else Size()
        self.atomic_matrix_shape: Size = n + Size([m1, m1])
        self.shell_matrix_shape: Size = n + Size([m2, m2])
        self.orbital_matrix_shape: Size = n + Size([m3, m3])

    @property
    def device(self) -> torch.device:
        """The device on which the `Basis` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        # Instruct users to use the ".to" method if wanting to change device.
        raise AttributeError('Basis object\'s dtype can only be modified '
                             'via the ".to" method.')

    @property
    def orbs_per_atom(self) -> Tensor:
        """Number of orbitals associated with each atom."""
        return self._orbitals_per_species[self.atomic_numbers.view(-1)
                                   ].view_as(self.atomic_numbers)

    @property
    def orbs_per_shell(self) -> Tensor:
        """Number of orbitals associated with each shell."""
        return torch.where(self.shell_ls != -1, self.shell_ls * 2 + 1, 0)

    @property
    def orbs_per_res(self):
        """Returns `orbs_per_atom` if atom resolved & `orbs_per_shell` if not."""
        # This property allows for resolution agnostic programming.
        return (self.orbs_per_shell if self.shell_resolved
                else self.orbs_per_atom)

    @property
    def on_atoms(self) -> Tensor:
        """Identifies which atom each orbital belongs to."""
        if (opa := self.orbs_per_atom).ndim == 1:
            return _repeat_range(opa, device=self.__device)
        else:
            return pack([_repeat_range(o, device=self.__device) for o in opa],
                        value=-1)

    @property
    def on_shells(self) -> Tensor:
        """Identifies which shell each orbital belongs to."""
        if (ops := self.orbs_per_shell).ndim == 1:
            return _repeat_range(ops, device=self.__device)
        else:
            return pack([_repeat_range(o, device=self.__device) for o in ops],
                        value=-1)

    @property
    def on_res(self) -> Tensor:
        """Returns ``on_atoms`` if atom resolved & ``on_shells`` if not. """
        # This property allows for resolution agnostic programming.
        return self.on_shells if self.shell_resolved else self.on_atoms

    def to(self, device: device) -> 'Basis':
        """Returns a copy of the `Basis` instance on the specified device.

        This method creates and returns a new copy of the `Basis` instance
        on the specified device "``device``".

        Arguments:
            device: Device to which all associated tensors should be moved.

        Returns:
            basis: Copy of the instance placed on the specified device.

        Notes:
            If the `Basis` instance is already on the desired device then
            `self` will be returned.

        """
        return self.__class__(self.atomic_numbers.to(device=device),
                              self.shell_dict, self.shell_resolved)

    @classmethod
    def from_hdf5(cls, source: Group, device: Optional[torch.device] = None
                  ) -> 'Basis':
        """Instantiate a `Basis` instances from an HDF5 group.

        Arguments:
            source: An HDF5 group(s) containing `Basis` instance.
            device: Device on which to place tensors. [DEFAULT=None]

        Returns:
            basis: The resulting `Basis` object.
        """
        shell_dict = {int(k): v[()].tolist() for k, v in
                      source['shell_dict'].items()}
        shell_resolved = bool(source['shell_resolved'][()])
        atomic_numbers = torch.tensor(source['atomic_numbers'][()],
                                      device=device)
        return cls(atomic_numbers, shell_dict, shell_resolved)

    def to_hdf5(self, target: Group):
        """Saves `Basis` instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the `Basis` should be saved.

        Notes:
            This function does not create its own group as it expects that
            ``target`` is the group into which data should be writen.
        """
        # Store the shell dictionary, atomic numbers & resolution
        shell_dict = target.create_group('shell_dict')
        for k, v in self.shell_dict.items():
            shell_dict.create_dataset(str(k), data=np.array(v))

        target.create_dataset('atomic_numbers',
                              data=self.atomic_numbers.detach().cpu().numpy())

        target.create_dataset('shell_resolved', data=self.shell_resolved)

    def matrix_shape(self, form: Form) -> Size:
        """Returns the expected shape of a matrix based on ``form``.

        This returns either `orbital_matrix_shape`, `shell_matrix_shape` or
        `atomic_matrix_shape` based on the value provided for ``form``.

        Arguments:
            form: Specifies the form of the atomic number matrix:

                - "full": One matrix-element for each orbital-orbital pair.
                - "shell": One matrix-element for each shell-shell pair.
                - "atomic": One matrix-element for each atom-atom pair.

        Returns:
            shape: A `torch.Size` object specifying the associated shape.
        """
        # This is mostly a convenience function and to avoid code reception
        try:
            shape = {'full': self.orbital_matrix_shape,
                     'shell': self.shell_matrix_shape,
                     'atomic': self.atomic_matrix_shape}[form]
            return shape

        except KeyError as e:  # Warn if an unknown "form" option is used
            raise ValueError(f'"{form}" is not a valid option for "form".'
                             f'valid values are {Form.__args__}')

    def azimuthal_matrix(self, form: Form = 'full', sort: bool = False,
                         mask_on_site: bool = False, mask_diag: bool = False,
                         mask_lower: bool = False) -> Tensor:
        r"""Azimuthal quantum numbers for each basis-basis interaction.

        Tensor defining the azimuthal quantum numbers (ℓ) associated with each
        orbital-orbital interaction element.  Alternately, a shell form of
        the azimuthal matrix can be returned which defines only one element
        per shell-shell interaction block. Segments of the matrix can be
        masked out with -1 values using the `mask_*` arguments.

        Arguments:
            form: Specifies the form of the azimuthal matrix:

                - "full": one matrix-element per orbital-orbital interaction.
                - "shell": Causes the ℓ-matrix to be returned in shell form
                  where each shell-shell interaction block is represented by a
                  single element.

                See the notes section for more information. [DEFAULT="full"]
            sort: Sort along the last dimension so the lowest ℓ value in each
                ℓ-pair comes first. [DEFAULT=False]
            mask_on_site: Masks on-site blocks, but leaves diagonals
                unaffected.[DEFAULT=False]
            mask_diag: Masks diagonal elements in full matrix mode but has no
                effect on `shell` form matrices. [DEFAULT=False]
            mask_lower: Masks lower triangle of the ℓ-matrix. [DEFAULT=False]

        Returns:
            azimuthal_matrix: Tensor defining the azimuthal quantum numbers
                associated with the orbital-orbital interactions.

        Notes:
            For an N-orbital system a NxNx2 matrix will be returned where the
            i'th, j'th vector lists the ℓ values of the i'th & j'th orbitals;
            these being the orbitals associated with the i'th, j'th element of
            the Hamiltonian & overlap matrices.

            The ``sort`` option can be used to enable all interaction of a
            given type to be gathered with a single call, e.g. [0, 1] rather
            than two separate calls [0, 1] & [1, 0].

            For an H4C2Au2 molecule, the full azimuthal matrix with diagonal,
            on-site & lower-triangle masking will take the form:

            .. image:: ../images/Azimuthal_matrix.svg
                :width: 400
                :align: center
                :alt: Azimuthal matrix

            Where all elements in the ss blocks equal [0, 0], sp: [0, 1],
            ps: [1, 0], etc. Note that if ``sort`` is True then the ps
            elements will converted from [1, 0] to [0, 1] during sorting.
            Black & grey indicate areas of the matrix that have been masked
            out by setting their values to -1. A masking value of -1 is
            chosen as it is a invalid azimuthal quantum number, & is thus
            less likely to result in collision. If ``shell`` is True then
            the matrix will be reduced to:

            .. image:: ../images/Azimuthal_matrix_block.svg
                :width: 186
                :align: center
                :alt: Azimuthal matrix (shell form)

            As the diagonal elements are contained within the on-site blocks
            it is not possible to selectively mask & unmask them. Note that
            masking of matrix diagonals are affected only by the ``mask_diag``
            option and **not** the ``mask_on_site`` option.

            A dense matrix implementation is used by this function as the
            sparse matrix code in pytorch, as of version-1.7, is not stable
            or mature enough to support the types of operations that the
            returned tensor is intended to perform.

            Note that an atomic reduction of the azimuthal matrix is not
            possible. Hence there is no "atomic" option like that found in
            ``atomic_number_matrix``.

        Raises:
            ValueError: If an invalid ``form`` option is passed.

        Warnings:
            Azimuthal sorting (``sort``) should be used with care as it will
            introduce inconsistencies with the other matrices. Only use this
            option if you know what you are doing.
        """

        if form == 'atomic':
            raise ValueError('Azimuthal matrix can not be in "atomic" form')

        shape = self.matrix_shape(form)  # <- shape of the output matrix
        batch = self.atomic_numbers.dim() == 2

        # Code is full/shell mode agnostic to reduce code duplication. however
        # some properties must be set at the start to make this work.
        if form == 'full':
            counts = self.shell_ls * 2 + 1
            counts[counts == -1] = 0
            basis_list = self.shell_ls.repeat_interleave(counts.view(-1))
            if batch:
                basis_list = pack(split_by_size(basis_list, self.n_orbitals)
                                  , value=-1)
        else:
            basis_list = self.shell_ls

        # Repeat and expand the vectors to get the final NxNx2 matrix tensor.
        l_mat = _rows_to_NxNx2(basis_list, shape, -1)

        # If masking out parts of the matrix
        if mask_on_site | mask_lower | mask_diag:
            # Create a blank mask.
            mask = torch.full_like(l_mat, False, dtype=torch.bool)
            if mask_on_site:  # Mask out on-site blocks if required.
                idx_mat = self.index_matrix(form)
                mask[idx_mat[..., 0] == idx_mat[..., 1]] = True

            if form == 'full': # Mask/unmask the diagonals as instructed
                mask.diagonal(dim1=-3, dim2=-2)[:] = mask_diag

            # Add lower triangle of the matrix to the mask, if told to do so
            if mask_lower:
                if not batch:
                    mask[tuple(torch.tril_indices(*shape, -1))] = True
                else:
                    a, b = tuple(torch.tril_indices(*shape[1:], -1))
                    mask[:, a, b] = True  # ↑ Indexing hack ↑

            # Apply the mask and set all the masked values to -1
            l_mat[mask] = -1

        # Sort angular momenta terms if requested & return the matrix
        return l_mat if not sort else l_mat.sort(-1)[0]

    def shell_number_matrix(self, form: Form = 'full') -> Tensor:
        r"""Shell numbers associated with each orbital-orbital pair.

        This is analogous to the ``azimuthal_matrix`` tensor but with shell
        numbers rather than azimuthal quantum numbers. The shell number for
        each species are taken from the order in which they are defined in
        the `Basis.shell_dict`.

        Arguments:
            form: Specifies the form of the shell number matrix:

                - "full": 1 matrix-element per orbital-orbital interaction.
                - "shell": 1 matrix-element per shell-shell interaction block.

                A more in-depth description of the this argument's effects is
                given in :meth:`.azimuthal_matrix`. [DEFAULT="full"]

        Returns:
            shell_number_matrix: An NxNx2 tensor specifying the shell numbers
                associated with each interaction. N can be the number orbitals
                , shells or atoms depending on ``form``.

        Raises:
            ValueError: If an invalid ``form`` option is passed

        """
        if form == 'atomic':
            raise ValueError('Shell matrix can not be given in "atomic" form')

        # 1) Make the first row of the NxN matrix
        s_mat = self.shell_ns.clone()

        if form == 'full':  # If the full matrix is required: expand the s_mat
            s_mat = s_mat.repeat_interleave(torch.where(
                self.shell_ls == -1, 0, self.shell_ls * 2 + 1).view(-1))
            if self.atomic_numbers.dim() == 2:  # "if batch"
                s_mat = pack(split_by_size(s_mat, self.n_orbitals), value=-1)

        # Convert the rows into a full NxNx2 tensor and return it
        return _rows_to_NxNx2(s_mat, self.matrix_shape(form), -1)

    def atomic_number_matrix(self, form: Form = 'full') -> Tensor:
        r"""Atomic numbers associated with each orbital-orbital pair.

        This is analogous to the ``azimuthal_matrix`` tensor but with atomic
        numbers rather than azimuthal quantum numbers. Shell and atomic forms
        can also be returned.

        Arguments:
            form: Specifies the form of the atomic number matrix:

                - "full": 1 matrix-element per orbital-orbital interaction.
                - "shell": 1 matrix-element per shell-shell interaction block.
                - "atomic": 1 matrix-element per atom-atom interaction block.

                A more in-depth description of the this argument's effects is
                given in :meth:`.azimuthal_matrix`. [DEFAULT="full"]

        Returns:
            atomic_number_matrix: An NxNx2 tensor specifying the atomic
                numbers associated with each interaction. N can be the number
                of orbitals, shells or atoms depending on ``form``.

        Raises:
            ValueError: If an invalid ``form`` option is passed.

        """
        batch = self.atomic_numbers.dim() == 2

        # Make the first row of the NxN matrix
        if form == 'full':
            indices = self.on_atoms
            if batch:  # Workaround for issue 55143
                indices[torch.where(indices == -1)] = self.n_atoms.max() - 1
            an_mat = self.atomic_numbers.gather(-1, indices)

        elif form == 'shell':
            an = self.atomic_numbers.view(-1)
            an_mat = an.repeat_interleave(self._shells_per_species[an])
            if batch:
                an_mat = pack(split_by_size(an_mat, self.n_shells), value=0)

        else:  # 'form' must be "atomic" then
            an_mat = self.atomic_numbers

        # Convert the rows into a full NxNx2 tensor and return it
        return _rows_to_NxNx2(an_mat, self.matrix_shape(form), 0)

    def index_matrix(self, form: Form = 'full') -> Tensor:
        """Indices of the atoms associated with each orbital pair.

        Produces a tensor specifying the indices of the atoms associate with
        each orbital-orbital pair. This is functionality identical to the
        ``atomic_number_matrix`` operation; differing only in that it returns
        atom indices rather than atomic numbers. See ``atomic_number_matrix``
        documentation for more information.

        Arguments:
            form: Specifies the form of the index matrix:

                - "full": 1 matrix-element per orbital-orbital interaction.
                - "shell": 1 matrix-element per shell-shell interaction block.
                - "atomic": 1 matrix-element per atom-atom interaction block.

                A more in-depth description of the this argument's effects is
                given in :meth:`.azimuthal_matrix`. [DEFAULT="full"]

        Returns:
            index_matrix : A NxNx2 tensor specifying the indices of the atoms
                associated with each interaction. N can be the number of
                orbitals, shells or atoms depending on ``form``.

        Raises:
            ValueError: If an invalid ``form`` option is passed.
        """
        shape = self.matrix_shape(form)  # <- shape of the output matrix
        ans = self.atomic_numbers

        # Make the first row of the NxN matrix
        if form == 'full':
            i_mat = self.on_atoms

        elif form == 'shell':
            i_mat = arange(
                self.n_atoms.max(), device=self.__device).expand_as(
                ans).repeat_interleave(self._shells_per_species[ans.view(-1)])

            if self.atomic_numbers.dim() == 2:  # "if batch"
                i_mat = pack(split_by_size(i_mat, self.n_shells), value=-1)

        else:  # Must be atomic form otherwise
            i_mat = arange(self.n_atoms.max(), device=self.__device).expand(
                shape[:-1]).clone()
            i_mat[ans == 0] = -1  # Pre-mask to make masking work later on

        # Convert the rows into a full NxNx2 tensor and return it
        return _rows_to_NxNx2(i_mat, shape, -1)


def _rows_to_NxNx2(vec: Tensor, shape: Size, pad: Optional[Any] = None
    ) -> Tensor:
    """Takes a row & converts it into its final NxNx2 shape.

    Arguments:
        vec: The vector that is to be expanded into a tensor.
        shape: The final, batch agnostic, shape of the tensor.
        pad: Values which are to be padded after expansion.

    """
    # Expand rows into first NxN slice of the matrix and mask as needed
    mat = vec.unsqueeze(-2).expand(shape).clone()
    mat[mat[..., 0, :] == pad] = pad  # Mask down columns
    # Form the NxN slice into the full NxNx2 tensor and return it.
    return torch.stack((mat.transpose(-1, -2), mat), -1)


def _repeat_range(tensor, **kwargs):
    """Combines `torch.arange` and `torch.repeat_interleave` methods."""
    # Convenience function to abstract repeated verbose calls
    return arange(len(tensor), **kwargs).repeat_interleave(tensor)
