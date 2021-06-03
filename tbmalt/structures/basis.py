# -*- coding: utf-8 -*-
"""A container to hold data associated with a chemical system's bases.

This module provides the `Basis` data structure class and its associated code.
The `Basis` class is intended to hold data needed to describe the number and
identities of a chemical systems bases.
"""
from typing import Dict, List, Union, Literal
import torch
from torch import Tensor, Size
from tbmalt.common.batch import pack


class Basis:
    """Data container for all information relating to a system's basis set.

    This class collects all information about a system's basis set and its
    orbitals into one place. This also permits calculations to

    Arguments:
        atomic_numbers: Atomic numbers of the atoms present in the system.
        max_ls: A dictionary specifying the maximum permitted angular momentum
            associated with a each atomic number. keys must be integers not
            torch tensors.
        shell_resolved: If True, calculations will be shell-resolved, if False
            they will be atom-resolved. This is used to automatically
            return the correct resolution information. [DEFAULT=False]

    Attributes:
        n_orbitals (Tensor): Number of orbitals present.
        n_subshells (Tensor): Number of subshells present.
        n_atoms (Tensor): Number of atoms present.
        orbital_matrix_shape (Size): Associated orbital matrix shape, e.g.
            Hamiltonian or overlap matrices.
        subshell_matrix_shape (Size): Similar to ``orbital_matrix_shape`` but
            with only one matrix-element per subshell block.
        atomic_matrix_shape (Size): Similar to ``subshell_matrix_shape`` but
            with only one matrix-element per atom.

    """
    # Developers Notes:
    # This class is used wherever information is needed about a system's basis
    # set or its orbitals. It is designed to permit code to be written which is
    # agnostic to the orbital-resolved/atom-resolved nature of the calculation.
    #
    # Class instances have two primary cached properties named ``_basis_list``
    # & ``_basis_blocks``; which are derived from the two class properties
    # ``_look_up`` & ``_blocks`` respectively. These help speed up calls to
    # the ``azimuthal_matrix`` function without needing to store the underling
    # result. which would be memory intensive. The ``_look_up`` list stores a
    # vector for each angular momenta:
    #     _look_up = [
    #         [0],  # <- for an S-subshell
    #         [1, 1, 1],  # <- for a P-subshell
    #         [2, 2, 2, 2, 2],  # <- for a D-subshell
    #         [3, 3, 3, 3, 3, 3, 3],  # <- for an F-subshell
    #         ...  # So on and so forth
    #     ]
    # This is used to generate atomic azimuthal identity blocks. For example a
    # carbon atom has s & p subshells, thus it combines the s & p orbitals:
    #     Carbon_atom = [0, 1, 1, 1,]
    #     or
    #     Carbon_atom = torch.cat([_look_up[0], _look_up[1]])
    # These are then used to build up the azimuthal identity for the whole
    # system ``_basis_list``. To save memory, ``_basis_list`` stores vectors
    # rather than concatenating them into a single list. The ``_blocks``
    # property is a list of truthy boolean tensors of size NxN where
    #   N = 2ℓ + 1
    # ℓ being the azimuthal quantum number. These are used in constructing the
    # mask which blanks out the on site terms from the ``azimuthal_matrix``.
    # By storing both ``_basis_list`` and ``_basis_blocks`` as composites of
    # ``_look_up`` & ``_blocks`` the amount of memory per system can be
    # reduced. The two secondary cached properties: ``_sub_basis_list`` &
    # ``_sub_basis_blocks`` are the orbital-resolved variates.

    # Variable Setup
    _max_l = 5  # <- Sets the maximum permitted angular momentum.
    _ls = torch.arange(_max_l)

    # Cache List Construction
    # pylint: disable=E741,W0106
    _sub_look_up = [torch.arange(l + 1) for l in _ls]
    _look_up = [l.repeat_interleave(2 * l + 1) for l in _sub_look_up]
    _sub_blocks = [torch.full((l + 1, l + 1), True) for l in _ls]
    _blocks = [torch.full((l, l), True) for l in (_ls + 1) ** 2]
    _orbs_per_shell = [2 * l + 1 for l in _sub_look_up]

    # Cache List Extension: Add dummy ℓ=-1 value for padded batches
    _orbs_per_shell.append(torch.tensor([0]))

    def __init__(self, atomic_numbers: Union[Tensor, List[Tensor]],
                 max_ls: Dict[int, int], shell_resolved: bool = False):

        # Ensure that the keys of `max_ls` are integers and Not tensors
        key = list(max_ls.keys())[0]
        if not isinstance(key, int):
            raise ValueError(f'Keys of "max_ls" must be ints not {type(key)}')
        # Clone the max_ls dict and modify it to accept padding values
        max_ls = max_ls.copy()
        max_ls[0] = -1

        # pylint: disable=C0103
        _batch = isinstance(atomic_numbers, list) or atomic_numbers.ndim == 2

        # __DATA ATTRIBUTES__
        self.shell_resolved = shell_resolved
        self.atomic_numbers = pack(atomic_numbers)

        self.max_l_on_atom: Tensor = torch.tensor(
            [max_ls[int(i)] if i.ndim == 0  # If single system
             else [max_ls[int(j)] for j in i]  # If multiple systems
             for i in self.atomic_numbers])

        # __COUNTABLE ATTRIBUTES__
        self.n_orbitals: Tensor = self.orbs_per_atom.sum(-1)
        self.n_subshells: Tensor = (self.max_l_on_atom + 1).sum(-1)
        self.n_atoms: Tensor = self.atomic_numbers.count_nonzero(-1)

        # __SHAPE ATTRIBUTES__
        # pylint: disable=C0103
        _m1, _m2 = self.n_atoms.max(), self.n_subshells.max()
        _m3 = self.n_orbitals.max()
        # _n adds an extra dimension when in batch mode
        _n = Size([len(atomic_numbers)]) if _batch else Size()
        self.atomic_matrix_shape: Size = _n + Size([_m1, _m1])
        self.subshell_matrix_shape: Size = _n + Size([_m2, _m2])
        self.orbital_matrix_shape: Size = _n + Size([_m3, _m3])

        # __CACHE ATTRIBUTES__
        def apply_loop(look_up):  # Convenience function
            if self.max_l_on_atom.dim() == 1:
                return [look_up[o] for o in self.max_l_on_atom]
            else:
                return [[look_up[o] for o in s if o != -1]
                        for s in self.max_l_on_atom]

        # Construct cache properties
        self._basis_list = apply_loop(self._look_up)
        self._sub_basis_list = apply_loop(self._sub_look_up)
        self._basis_blocks = apply_loop(self._blocks)
        self._sub_basis_blocks = apply_loop(self._sub_blocks)

        if _batch:
            # Masks for clearing padding values when in batch mode & to get
            # around PyTorch issue 55143.
            def make_mask(e):  # Convenience function
                return pack([torch.full([i], True)for i in e], value=False)

            self._masks = {'orbital': make_mask(self.n_orbitals),
                           'shell': make_mask(self.n_subshells),
                           'atomic': make_mask(self.n_atoms)}

            # Add "full" and "block" aliases for convenience
            self._masks.update({'full': self._masks['orbital'],
                                'block': self._masks['shell']})

    @property
    def orbs_per_atom(self) -> Tensor:
        """Number of orbitals associated with each atom."""
        return (self.max_l_on_atom + 1) ** 2

    @property
    def orbs_per_shell(self) -> Tensor:
        """Number of orbitals associated with each shell."""
        # Get number of orbitals per-shell, per-system, concatenate, & pack.
        if self.atomic_numbers.dim() == 1:
            return torch.cat([self._orbs_per_shell[l]
                              for l in self.max_l_on_atom])
        else:
            return pack([torch.cat([self._orbs_per_shell[l] for l in i])
                         for i in self.max_l_on_atom])

    @property
    def orbs_per_res(self):
        """Returns ``orbs_per_atom`` if atom resolved & ``orbs_per_shell`` if not."""
        # This property allows for resolution agnostic programming.
        return (self.orbs_per_shell if self.shell_resolved
                else self.orbs_per_atom)

    @property
    def on_atoms(self) -> Tensor:
        """Identifies which atom each orbital belongs to."""
        if self.atomic_numbers.dim() == 1:
            return torch.arange(len(self.orbs_per_atom)
                                ).repeat_interleave(self.orbs_per_atom)
        else:
            return pack([torch.arange(len(o)).repeat_interleave(o)
                         for o in self.orbs_per_atom], value=-1)

    @property
    def on_shells(self) -> Tensor:
        """Identifies which shell each orbital belongs to."""
        if self.atomic_numbers.dim() == 1:
            return torch.arange(len(self.orbs_per_shell)
                                ).repeat_interleave(self.orbs_per_shell)
        else:
            return pack([torch.arange(len(o)).repeat_interleave(o)
                         for o in self.orbs_per_shell], value=-1)

    @property
    def on_res(self) -> Tensor:
        """Returns ``on_atoms`` if atom resolved & ``on_shells`` if not. """
        # This property allows for resolution agnostic programming.
        return self.on_shells if self.shell_resolved else self.on_atoms

    def azimuthal_matrix(self, form: Literal['full', 'block'] = 'full',
                         sort: bool = False, mask_on_site: bool = True,
                         mask_lower: bool = True, mask_diag: bool = True
                         ) -> Tensor:
        r"""Azimuthal quantum numbers for each basis-basis interaction.

        Tensor defining the azimuthal quantum numbers (ℓ) associated with each
        orbital-orbital interaction element.  Alternately, a block form of
        the azimuthal matrix can be returned which defines only one element
        per subshell-subshell interaction block. Segments of the matrix can be
        masked out with -1 values using the `mask_*` arguments.

        Arguments:
            form: Specifies the form of the azimuthal matrix:

                - "full": One matrix-element for each orbital-orbital
                  interaction.
                - "block": Causes the ℓ-matrix to be returned in block form
                  where each subshell-subshell interaction block is
                  represented by only a single element.

                See the notes section for more information. [DEFAULT="full"]
            sort: Sort along the last dimension so the lowest ℓ value in each
                ℓ-pair comes first. [DEFAULT=False]
            mask_on_site : `bool`, optional
                Masks on-site blocks. Note that the diagonals are not masked
                when in full matrix mode. [DEFAULT=True]
            mask_lower : `bool`, optional
                Masks lower triangle of the ℓ-matrix. [DEFAULT=True]
            mask_diag : `bool`, optional
                Masks diagonal elements in full matrix mode. [DEFAULT=True]

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
            less likely to result in collision. If ``block`` is True then
            the matrix will be reduced to:

            .. image:: ../images/Azimuthal_matrix_block.svg
                :width: 186
                :align: center
                :alt: Azimuthal matrix (block form)

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
            KeyError: If an invalid ``form`` option is passed.
        """
        if form not in ['full', 'block']:
            raise KeyError(f'"{form}" is not a valid option for `form`')

        batch = self.atomic_numbers.dim() == 2

        # Code is full/block mode agnostic to reduce code duplication. however
        # some properties must be set at the start to make this work.
        if form != 'block':  # If returning the full matrix
            shape = self.orbital_matrix_shape  # <- shape of the output matrix
            basis_list = self._basis_list  # <- see class docstring
            basis_blocks = self._basis_blocks  # <- see class docstring
        else:  # If returning the reduced block matrix
            shape = self.subshell_matrix_shape
            basis_list = self._sub_basis_list
            basis_blocks = self._sub_basis_blocks

        # Repeat basis list to get ℓ-values for the 1'st orbital in each
        # interaction. Expand function is used as it is faster than repeat.
        if not batch:
            l_mat = torch.cat(basis_list).expand(shape)
        else:
            l_mat = pack([torch.cat(i) for i in basis_list],
                         value=-1).unsqueeze(-2).expand(shape).clone()
            # Propagate the padding values to the second dimension
            l_mat[(l_mat == -1).transpose(-1, -2)] = -1

        # Convert from an NxNx1 matrix into the NxNx2 azimuthal matrix
        l_mat = torch.stack((l_mat.transpose(-1, -2), l_mat), -1)

        # If masking out parts of the matrix
        if mask_on_site | mask_lower | mask_diag:
            # Create a blank mask.
            mask = torch.full_like(l_mat, False, dtype=torch.bool)
            if mask_on_site:  # Mask out on-site blocks if required.
                if not batch:
                    mask[torch.block_diag(*basis_blocks)] = True
                else:
                    mask[pack([torch.block_diag(*i) for i in basis_blocks],
                              value=True)] = True

            # Add lower triangle of the matrix to the mask, if told to do so
            if mask_lower:
                if not batch:
                    mask[tuple(torch.tril_indices(*shape, -1))] = True
                else:
                    a, b = tuple(torch.tril_indices(*shape[1:], -1))
                    mask[:, a, b] = True  # ↑ Indexing hack ↑

            # If not in block mode mask/unmask the diagonals as instructed
            # Mask/Unmask the the
            if form != 'block':  # <- Only valid for non block matrices
                # If mask_diag True; the diagonal will be masked, if False it
                # will be unmasked.
                mask.diagonal(dim1=-3, dim2=-2)[:] = mask_diag

            # Apply the mask and set all the masked values to -1
            l_mat[mask] = -1

        # Sort the angular momenta terms, if requested, and return the
        # azimuthal_matrix.
        return l_mat if not sort else l_mat.sort(-1)[0]

    def atomic_number_matrix(self,
                             form: Literal['full', 'block', 'atomic'] = 'full'
                             ) -> Tensor:
        r"""Atomic numbers associated with each orbital-orbital pair.

        This is analogous to the ``azimuthal_matrix`` tensor but with atomic
        numbers rather than azimuthal quantum numbers. Block and atomic forms
        can also be returned.

        Arguments:
            form: Specifies the form of the atomic number matrix:

                - "full": One matrix-element for each orbital-orbital
                  interaction.
                - "block": One matrix-element for each subshell-subshell
                  interaction block.
                - "atomic": One matrix-element for each atom-atom interaction
                  block.

                A more in-depth description of the this argument's effects is
                given in :meth:`.azimuthal_matrix`. [DEFAULT="full"]

        Returns:
            atomic_number_matrix: An NxNx2 tensor specifying the atomic
                numbers associated with each interaction. N can be the number
                of orbitals, subshells or atoms depending on ``form``.

        Raises:
            ValueError: If an invalid ``form`` option is passed.

        """
        try:  # Identify the final desired shape
            shape = {'full': self.orbital_matrix_shape,
                     'block': self.subshell_matrix_shape,
                     'atomic': self.atomic_matrix_shape}[form]
        except KeyError as e:  # Warn if using an unknown "form" option
            raise type(e)(f'"{form}" is not a valid option for `form`')

        batch = self.atomic_numbers.dim() == 2

        # 1) Make the first row of the NxN matrix
        if form == 'full':
            indices = self.on_atoms
            if batch:  # Workaround for issue 55143
                indices[~self._masks['orbital']] = self.n_atoms.max() - 1
            an_mat = self.atomic_numbers.gather(-1, indices)

        elif form == 'block':
            if not batch:
                an_mat = self.atomic_numbers.repeat_interleave(
                    self.max_l_on_atom + 1)
            else:  # There is no easy alternative to the for-loop method
                an_mat = pack([an.repeat_interleave(ml + 1) for an, ml in
                               zip(self.atomic_numbers, self.max_l_on_atom)])

        else:  # 'form' must be "atomic" then
            an_mat = self.atomic_numbers

        # 2) Expand the row into the first NxN slice of the matrix
        if not batch:
            an_mat = an_mat.expand(shape).clone()
        else:
            an_mat[~self._masks[form]] = -0  # Mask along rows
            an_mat = an_mat.unsqueeze(-2).expand(shape).clone()
            an_mat[~self._masks[form]] = -0  # Mask down columns

        # 3) Form the NxN slice into the full NxNx2 tensor and return it.
        return torch.stack((an_mat.transpose(-1, -2), an_mat), -1)

    def index_matrix(self, form: Literal['full', 'block', 'atomic'] = 'full'
                     ) -> Tensor:
        """Indices of the atoms associated with each orbital pair.

        Produces a tensor specifying the indices of the atoms associate with
        each orbital-orbital pair. This is functionality identical to the
        ``atomic_number_matrix`` operation; differing only in that it returns
        atom indices rather than atomic numbers. See ``atomic_number_matrix``
        documentation for more information.

        Arguments:
            form: Specifies the form of the index matrix:

                - "full": One matrix-element for each orbital-orbital
                  interaction.
                - "block": One matrix-element for each subshell-subshell
                  interaction block.
                - "atomic": One matrix-element for each atom-atom interaction
                  block.

                A more in-depth description of the this argument's effects is
                given in :meth:`.azimuthal_matrix`. [DEFAULT="full"]

        Returns:
            index_matrix : A NxNx2 tensor specifying the indices of the atoms
                associated with each interaction. N can be the number of
                orbitals, sub-shells or atoms depending on ``form``.

        Raises:
            ValueError: If an invalid ``form`` option is passed.
        """

        try:  # Identify the final desired shape
            shape = {'full': self.orbital_matrix_shape,
                     'block': self.subshell_matrix_shape,
                     'atomic': self.atomic_matrix_shape}[form]
        except KeyError as e:  # Warn if using an unknown "form" option
            raise type(e)(f'"{form}" is not a valid option for `form`')

        batch = self.atomic_numbers.dim() == 2

        # 1) Make the first row of the NxN matrix
        if form == 'full':
            i_mat = self.on_atoms

        elif form == 'block':
            if not batch:
                i_mat = torch.arange(
                    len(self.max_l_on_atom)
                ).repeat_interleave(self.max_l_on_atom + 1)

            else:
                i_mat = pack([torch.arange(len(ml)).repeat_interleave(ml + 1)
                              for ml in self.max_l_on_atom])

        else:  # Must be atomic form otherwise
            i_mat = torch.arange(self.n_atoms.max()
                                 ).expand(shape[:-1]).clone()

        # 2) Expand the row into the first NxN slice of the matrix
        if not batch:
            i_mat = i_mat.expand(shape).clone()
        else:
            i_mat[~self._masks[form]] = -1  # Mask along rows
            i_mat = i_mat.unsqueeze(-2).expand(shape).clone()
            i_mat[~self._masks[form]] = -1  # Mask down columns

        # 3) Form the NxN slice into the full NxNx2 tensor and return it.
        return torch.stack((i_mat.transpose(-1, -2), i_mat), -1)
