# -*- coding: utf-8 -*-
"""Code associated with performing cell translation.

This module implement cell translation for 1D & 2D & 3D periodic
boundary conditions. Distance matrix and position vectors for pbc
will be constructed.
"""
from typing import Union, Tuple, Optional
import torch
import numpy as np
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units

Tensor = torch.Tensor


class Periodic:
    """Calculate the translation vectors for cells for 1D & 2D & 3D
    periodic boundary condition.

    Arguments:
        positions: Coordinates of the atoms in the central cell.
        n_atoms: Number of atoms in the central cell.
        latvec: Lattice vector, with Bohr as default unit.
        cutoff: Interaction cutoff distance for reading SK table, with Bohr
            as default unit.

    Keyword Arguments:
        distance_extension: Extension of cutoff in SK tables to smooth tails.
        positive_extension: Extension for the positive lattice vectors.
        negative_extension: Extension for the negative lattice vectors.

    Attributes:
        positions: Coordinates of the atoms in the central cell.
        n_atoms: Number of atoms in the central cell.
        latvec: Lattice vector.
        cutoff: Global cutoff for the diatomic interactions.
        cellvec: Cell translation vectors in relative coordinates.
        rcellvec: Cell translation vectors in absolute units.
        n_cells: Number of lattice cells.
        positions_vec: Position vector matrix between atoms in different cells.
        periodic_distances: Distance matrix between atoms in different cells.

    Notes:
        When representing batch systems, the 'cellvec' and 'periodic_distances'
        will be padded with large numbers. Mixing of periodic & non-periodic
        systems and mixing of different types of pbc is forbidden.

    Examples:
        >>> from tbmalt import Periodic
        >>> H2 = Periodic(torch.tensor([[0.00, 0.00, 0.00],
                                       [0.00, 0.00, 0.79]]),
                          torch.tensor(1), torch.eye(3) * 3, 2.0)
        >>> print(H2.positions_vec.shape)
        torch.Size([125, 2, 2, 3])

    """

    def __init__(self, positions: Tensor, n_atoms: Tensor, latvec: Tensor,
                 cutoff: Union[Tensor, float], **kwargs):
        self.positions: Tensor = positions
        self.n_atoms: Tensor = n_atoms
        self.latvec, self.cutoff = self._check(latvec, cutoff, **kwargs)

        self._device = self.latvec.device
        self._dtype = self.latvec.dtype

        dist_ext = kwargs.get('distance_extension', 1.0)

        # Global cutoff for the diatomic interactions
        self.cutoff: Tensor = self.cutoff + dist_ext

        # Inverse lattice vectors
        self._invlatvec, self._mask_zero = self._inverse_lattice()

        # Cell translation
        self.cellvec, self.rcellvec, self.n_cells = (
            self.get_cell_translations(**kwargs))

        # Position vectors and distance matrix
        self.positions_vec, self.periodic_distances = (
            self._get_periodic_distance())

    def _check(self, latvec, cutoff, **kwargs) -> Tuple[Tensor, Tensor]:
        """Check dimension, type of lattice vector and cutoff."""
        # Default lattice vector is from geometry, thus default unit is bohr
        unit = kwargs.get('unit', 'bohr')

        # Lattice vectors can still be updated
        if isinstance(latvec, list):
            latvec = pack(latvec)

        # Check the shape of lattice vectors
        if latvec.ndim < 2 or latvec.ndim > 3:
            raise ValueError('lattice vector dimension should be 2 or 3')

        # Number of batches if in batch mode
        self._n_batch: Optional[int] = (None if latvec.ndim == 2
                                        else len(latvec))

        # Check the format of cutoff
        if type(cutoff) is float:
            cutoff = torch.tensor([cutoff])
            if cutoff.ndim == 0:
                cutoff = cutoff.unsqueeze(0)
            elif cutoff.ndim >= 2:
                raise ValueError(
                    'cutoff should be 0, 1 dimension tensor or float')
        elif type(cutoff) is Tensor:
            if cutoff.ndim == 0:
                cutoff = cutoff.unsqueeze(0)
            elif cutoff.ndim >= 2:
                raise ValueError(
                    'cutoff should be 0, 1 dimension tensor or float')
        elif type(cutoff) is not float:
            raise TypeError('cutoff should be tensor or float')

        if unit != 'bohr':
            latvec = latvec * length_units[unit]
            cutoff = cutoff * length_units[unit]

        return latvec, cutoff

    def get_cell_translations(self, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Implement cell translation."""

        dd = {'dtype': self._dtype, 'device': self._device}

        # Extensions for lattice vectors
        pos_ext = kwargs.get('positive_extension', 1)
        neg_ext = kwargs.get('negative_extension', 1)

        # Ranges of cell translation on three dimensions
        _tmp = torch.floor(self.cutoff * torch.norm(
            self._invlatvec, dim=-1).T).T
        ranges = torch.stack([-(neg_ext + _tmp), pos_ext + _tmp])

        # For 1D/2D cell translation, non-periodic direction will be zero
        ranges[torch.stack([self._mask_zero, self._mask_zero])] = 0

        # length of the first, second and third column in ranges
        leng = ranges[1, :].long() - ranges[0, :].long() + 1

        # number of cells
        n_cells = leng[..., 0] * leng[..., 1] * leng[..., 2]

        # Cell translation vectors in relative coordinates
        if not self._n_batch:  # -> single
            cellvec = torch.stack([
                torch.linspace(ranges[0, 0], ranges[1, 0],
                               leng[0], **dd).repeat_interleave(leng[2] * leng[1]),
                torch.linspace(ranges[0, 1], ranges[1, 1],
                               leng[1], **dd).repeat(leng[0]).repeat_interleave(leng[2]),
                torch.linspace(ranges[0, 2], ranges[1, 2],
                               leng[2], **dd).repeat(leng[0] * leng[1])])

        else:  # -> batch
            # Large values are padded at the end of short cell vectors.
            cellvec = pack([torch.stack([
                torch.linspace(iran[0, 0], iran[1, 0],
                               ile[0], **dd).repeat_interleave(ile[2] * ile[1]),
                torch.linspace(iran[0, 1], iran[1, 1],
                               ile[1], **dd).repeat(ile[0]).repeat_interleave(ile[2]),
                torch.linspace(iran[0, 2], iran[1, 2],
                               ile[2], **dd).repeat(ile[0] * ile[1])])
                            for ile, iran in zip(
                                    leng, ranges.transpose(-2, -3))], value=1e3)

        # Cell translation vectors in absolute units
        rcellvec = (torch.matmul(self.latvec.transpose(-1, -2),
                                 cellvec)).transpose(-1, -2)

        return cellvec.transpose(-1, -2), rcellvec, n_cells

    def _get_periodic_distance(self) -> Tuple[Tensor, Tensor]:
        """Get positions and distances between central and neighbouring cells."""
        # Number of atoms in central cell
        n_atoms = self.n_atoms

        # Positions of atoms in all images
        positions = (self.rcellvec.unsqueeze(-2) +
                     self.positions.unsqueeze(-3))

        # Position vectors
        positions_vec = (positions.unsqueeze(-3) -
                         self.positions.unsqueeze(-3).unsqueeze(-2))

        # Distance matrix, large values will be padded for batch systems
        if not self._n_batch:  # -> single
            distance = torch.sqrt(
                ((positions.repeat(1, n_atoms, 1) - torch.repeat_interleave(
                    self.positions, n_atoms, 0)) ** 2).sum(-1).reshape(
                        -1, n_atoms, n_atoms))

        else:  # -> batch
            distance = pack([torch.sqrt(((ipos[:, :inat].repeat(1, inat, 1) -
                                          icp[:inat].repeat_interleave(inat, 0)
                                          ) ** 2).sum(-1)).reshape(-1, inat, inat)
                             for ipos, icp, inat in zip(
                                     positions, self.positions, n_atoms)
                             ], value=1e3)

        return positions_vec, distance

    def _inverse_lattice(self) -> Tuple[Tensor, Tensor]:
        """Get inverse lattice vectors."""
        # Build a mask for zero vectors in 1D/2D lattice vectors
        mask_zero = self.latvec.eq(0).all(-1)
        _latvec = self.latvec + torch.diag_embed(mask_zero.type(
            self.latvec.dtype))

        eye = torch.eye(_latvec.shape[-1],
                        device=self._device, dtype=self._dtype)

        if self._n_batch is not None:
            eye = eye.repeat(self._n_batch, 1, 1)

        # Inverse lattice vectors
        invlat = torch.linalg.solve(_latvec, eye).transpose(-1, -2)

        invlat[mask_zero] = 0

        return invlat, mask_zero

    @property
    def reciprocal_lattice(self) -> Tensor:
        """Get reciprocal lattice vectors."""
        return 2 * np.pi * self._invlatvec

    @property
    def get_reciprocal_volume(self) -> Tensor:
        """Get reciprocal lattice unit cell volume."""
        return abs(torch.det(2 * np.pi * (self._invlatvec.transpose(0, 1))))

    @property
    def cellvol(self) -> Tensor:
        """Get unit cell volume."""
        return abs(torch.det(self.latvec))

    @property
    def neighbour(self) -> Tensor:
        """A mask to choose atoms of images inside the cutoff distance."""
        if len(self.cutoff) == 1:
            return self.periodic_distances.le(self.cutoff)
        else:
            return torch.stack([ipd.le(ico) for ipd, ico in zip(
                self.periodic_distances, self.cutoff)])

    @property
    def neighbour_vector(self) -> Tensor:
        """Get positions for neighbour list within cutoff distance."""
        # Mask for images containing atoms within cutoff distance
        _mask = self.neighbour.any(-1).any(-1)

        if not self._n_batch:  # -> single
            neighbour_vec = self.positions_vec[_mask]
        else:  # -> batch
            neighbour_vec = pack([self.positions_vec[ibatch][_mask[ibatch]]
                                  for ibatch in range(self._n_batch)], value=1e3)

        return neighbour_vec

    @property
    def neighbour_distance(self) -> Tensor:
        """Get distance matrix for neighbour list within cutoff distance."""
        # Mask for images containing atoms within cutoff distance
        _mask = self.neighbour.any(-1).any(-1)

        if not self._n_batch:  # -> single
            neighbour_dist = self.periodic_distances[_mask]
        else:  # -> batch
            neighbour_dist = pack([self.periodic_distances[ibatch][_mask[ibatch]]
                                   for ibatch in range(self._n_batch)], value=1e3)

        return neighbour_dist
