# -*- coding: utf-8 -*-
"""A container to hold data associated with periodic boundary conditions.

This module provides the `Cell` data structure class and its associated
code. The `Cell` class is intended to hold information of periodic
boundary conditions. Along with `Geometry` module, a periodic
chemical system can be described.
"""
from abc import ABC
from typing import Union, List, Optional, Type
import torch
import numpy as np
from ase.lattice.bravais import Lattice
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units

Tensor = torch.Tensor


class Cell(ABC):
    """ABC for objects responsible for supplying geometric information
    specially on periodic systems.

    Subclasses of the abstract base class provides lattice vectors to describe
    the periodic boundary condition. Properties of the lattice vectors, e.g.
    lengths and angles, can be returned. Both single and batch system(s) can
    be represented in this class.

    Arguments:
        cell: Lattice vectors of the periodic systems.
        frac: Whether using fractional coordinates to describe periodic
            systems. [DEFAULT: None]

    """

    def __init__(self, cell: Union[Tensor, List[Tensor], Type[Lattice]],
                 frac: Optional[bool] = None, **kwargs):

        # Create an interface for importing cells using an object
        if not isinstance(cell, (Tensor, list)):
            if hasattr(cell, 'cells'):  # -> A custom cell object
                cell = cell.cells
            elif isinstance(cell, Lattice):  # -> An ase Lattice object
                cell = torch.tensor(cell.cell[:], dtype=kwargs.get('dtype'),
                                    device=kwargs.get('device', None))
            else:
                raise ValueError('Custom cell object should include attribute '
                                 'cells or be an ase.Lattice object.')

        self.cell: Tensor = pack(cell)

        # Check the dimension of lattice vectors
        if self.cell.ndim < 2 or self.cell.ndim > 3:
            raise ValueError('Dimension of cell should be 2 or 3.')

        # Make sure frac is bool
        self._frac: bool = False if frac is None else frac

        # Number of batches if in batch mode
        self._n_batch: Optional[int] = (None if self.cell.ndim == 2
                                        else len(cell))

        self._device = kwargs.get('device', self.cell.device)
        self._dtype = kwargs.get('dtype', self.cell.dtype)


class Pbc(Cell):
    """Return cell information for normal pbc.

    Arguments:
        cell: Lattice vectors of the periodic systems.
        frac: Whether using fractional coordinates to describe periodic
            systems. [DEFAULT: None]

    Attributes:
        cell: Lattice vectors of the periodic systems.
        pbc: The type of periodic boundary condition.

    Examples:
        >>> from tbmalt.structures.cell import Pbc
        >>> import torch
        >>> cells = [torch.tensor([[4., 0., 0.], [0., 5., 0.], [0., 0., 6.]]),
                      torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]])]
        >>> frac = False
        >>> cell = Pbc(cells, frac)
        >>> print(cell.cell)
        tensor([[[4., 0., 0.], [0., 5.., 0.], [0., 0., 6.]],
                [[5., 0., 0.], [0., 5., 0.], [0., 0.,5.]]])
        >>> print(cell.pbc)
        tensor([True, True, True])

        """

    def __init__(self, cell: Union[Tensor, List[Tensor], Type[Lattice]],
                 frac: Optional[bool] = None, **kwargs):
        super().__init__(cell, frac, **kwargs)
        self.pbc: Tensor = self._check()

    def _check(self):
        """Check the format of lattice vectors and other cell information."""
        # Check the format of lattice vectors
        if self.cell.size(dim=-2) != 3:
            raise ValueError('Input cell should be defined by three '
                             'lattice vectors.')

        # Masks of periodic systems
        _is_periodic: Tensor = self.cell.ne(0).any(-1).any(-1)

        # Check whether cluster using fraction coordinates
        if not _is_periodic.all() and self._frac:
            raise ValueError('Cluster should not be defined by fraction '
                             'coordinates.')

        # Check the dimensions of periodic boundary condition
        _sum_dim = self.cell.ne(0).any(-1).sum(dim=-1)
        _dim_pe = self.cell.ne(0).any(-1)

        if not self._n_batch:  # -> single
            # Type of pbc
            pbc = _dim_pe
        else:  # -> batch
            # Mixing of different pbc
            if not torch.all(torch.tensor([isd == _sum_dim[0]
                                           for isd in _sum_dim])):
                raise NotImplementedError(
                    'Mixing of different types of pbcs is not supported.')
            else:
                # Check directions of periodic boundary condition
                if not torch.all(torch.tensor([torch.all(idp == _dim_pe[0])
                                               for idp in _dim_pe])):
                    # Different periodic directions but same type of pbc
                    pbc = _dim_pe
                else:
                    # Same periodic directions
                    pbc = _dim_pe[0]

        return pbc

    @staticmethod
    def frac_to_cartesian(cell: Tensor, positions: Tensor) -> Tensor:
        """Transfer fractional coordinates to cartesian coordinates."""
        # Whether fraction coordinates in the range [0, 1)
        if torch.any(positions >= 1) or torch.any(positions < 0):

            # Operate on a copy so that the original is not modified.
            positions = positions.clone()

            positions = torch.abs(positions) - torch.floor(
                torch.abs(positions))

        # Transfer from fractional to cartesian coordinates
        positions: Tensor = torch.matmul(positions, cell)

        return positions

    @staticmethod
    def cell_unit_transfer(cell: Tensor, units: str) -> Tensor:
        """Transfer the unit of lattice vectors to bohr."""
        return cell * length_units[units]

    @staticmethod
    def get_cell_lengths(cell: Tensor) -> Tensor:
        """Get the length of each lattice vector."""
        return torch.linalg.norm(cell, dim=-1)

    @staticmethod
    def get_cell_angles(cell: Tensor) -> Tensor:
        """Get the angles' alpha, beta and gamma of lattice vectors."""
        _cos = torch.nn.CosineSimilarity(dim=-1)
        cosine = torch.stack([_cos(cell[..., 1], cell[..., 2]),
                              _cos(cell[..., 0], cell[..., 2]),
                              _cos(cell[..., 0], cell[..., 1])], -1)

        return torch.acos(cosine) * 180 / np.pi


class Pbc_helical(Cell):
    """Return cell information for helical pbc."""

    def __init__(self, cell: Union[Tensor, List[Tensor]],
                 frac=None, pbc: Optional[str] = None, **kwargs):
        super().__init__(cell, frac, **kwargs)
        if pbc == 'helical':
            raise NotImplementedError('Helical pbc is not implemented.')
