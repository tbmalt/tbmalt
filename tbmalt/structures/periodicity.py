# -*- coding: utf-8 -*-
"""Code associated with performing cell translation.

This module implement cell translation for 1D & 2D & 3D periodicity
boundary conditions. Distance matrix and position vectors for pbc
will be constructed.
"""
from typing import Union, Tuple, Optional
from abc import ABC, abstractmethod
import torch
import numpy as np
from tbmalt.common.batch import pack, merge, bT, bT2
from tbmalt.common import cached_property
from tbmalt.data.units import length_units

Tensor = torch.Tensor

# Todo (and points to address):
#   - In order for the caches to be stable and reliable a recursive dependency
#     is created between the `Geometry` & `Periodicity` classes. While this is
#     necessary in the short-term, one might consider extending the `Geometry`
#     `Geometry` class so that it, or a child class, incorporates the required
#     functionality.
#   - Does having different extensions for both the positive and negative
#     provide any real advantage, or can we get away with a single value.
#   - Why can't the cell search be modified to account for the issues that are
#     addressed but the positive and negative extensions?
#   - The cutoff value is modified within the init method as soon as it is
#     provided to the class by the user. This can create unexpected behaviour.
#     The user and external code should be responsible for specifying the exact
#     cutoff value.


class Periodicity(ABC):
    """Properties and functionality associated with periodic systems.

    Derived classes can be used to calculate various properties of 1D, 2D, & 3D
    periodic systems, such as the translation vectors. This is not intended to
    store much in the way of data itself but rather provides usefull methods.
    The periodicity entity will instead source infromation from the `Geometry`
    object it is provided.

    Arguments:
        geometry: Geometry object from which informaiton like atomic positions
            and lattice parameters are to be sourced.
        cutoff: Interaction cutoff distance for reading SK table, with Bohr
            as default unit.

    Keyword Arguments:
        distance_extension: Extension of cutoff in SK tables to smooth tails.
        positive_extension: Extension for the positive lattice vectors.
        negative_extension: Extension for the negative lattice vectors.

    Attributes:
        cutoff: Global cutoff for the diatomic interactions.
        cellvec: Cell translation vectors in relative coordinates.
        rcellvec: Cell translation vectors in absolute units.
        n_cells: Number of lattice cells.
        positions_vec: Position vector matrix between atoms in different cells.
        periodic_distances: Distance matrix between atoms in different cells.

    Notes:
        This class make extensive use of cached properties and thus may use
        more memory than one might expect.

        When representing batch systems, the 'cellvec' and 'periodic_distances'
        will be padded with large numbers. Mixing of periodicity & non-periodicity
        systems and mixing of different types of pbc is forbidden.

    Examples:
        >>> from tbmalt import Periodicity, Geometry
        >>> geometry = Geometry(torch.tensor([1, 1]), torch.rand(2,3))
        >>> H2 = Periodicity(
        >>>     geometry, torch.tensor(
        >>>         [[0.00, 0.00, 0.00], [0.00, 0.00, 0.79]]),
        >>>     torch.tensor(2), torch.eye(3) * 3, 2.0)
        >>> print(H2.positions_vec.shape)
        torch.Size([125, 2, 2, 3])

    """
    def __init__(
            self, geometry: "Geometry", cutoff: Union[Tensor, float],
            **kwargs):


        self.geometry = geometry

        geometry.lattice, self._cutoff = self._check(geometry.lattice, cutoff, **kwargs)

        # Intrinsic state attributes
        self._device = self.lattice.device
        self._dtype = self.lattice.dtype

        # Extensions for lattice vectors
        self._positive_extension = kwargs.get('positive_extension', 1)
        self._negative_extension = kwargs.get('negative_extension', 1)

        dist_ext = kwargs.get('distance_extension', 1.0)

        # Global cutoff for the diatomic interactions
        self._cutoff: Tensor = self._cutoff + dist_ext

        # Note that this is set within the `cellvec` property method.
        self._n_cells = None

    @property
    def cutoff(self):
        """Global cutoff for the diatomic interactions."""
        return self._cutoff

    @property
    def positions(self):
        return self.geometry.positions

    @positions.setter
    def positions(self, value):
        self.geometry = value

    @property
    def lattice(self):
        return self.geometry.lattice

    @lattice.setter
    def lattice(self, value):
        self.geometry.lattice = value

    @property
    def n_atoms(self):
        return self.geometry.n_atoms

    @cached_property('cellvec')
    def n_cells(self):
        """Number of periodicity cell images within the cutoff distance."""

        # The `cellvec` property has been marked as a dependency even though
        # the number of cells (`n_cells`) is not technically its dependency.
        # This is done as it is easiest to just set the value as `cellvec` is
        # being constructed.
        return self._n_cells

    @cached_property("positions", "rcellvec")
    def positions_vec(self):
        """Distance vectors between atoms in the central and neighbouring cells.
        """

        # Positions of atoms in all images
        positions = (self.rcellvec.unsqueeze(-2) +
                     self.positions.unsqueeze(-3))

        # distance vectors
        distance_vec = (positions.unsqueeze(-3) -
                        self.positions.unsqueeze(-3).unsqueeze(-2))

        return distance_vec


    @staticmethod
    def get_periodic_distances(
            cell_translation_vectors: Tensor, positions: Tensor, n_atoms=None,
            **kwargs):
        """Distances between atoms in the central and neighbouring cells."""

        # Positions of atoms in all images
        positions_expanded = (cell_translation_vectors.unsqueeze(-2) +
                     positions.unsqueeze(-3))

        # Distance matrix, large values will be padded for batch systems
        if positions.ndim == 2:  # -> single
            distance = torch.sqrt(
                ((positions_expanded.repeat(1, n_atoms, 1) - torch.repeat_interleave(
                    positions, n_atoms, 0)) ** 2).sum(-1).reshape(
                    -1, n_atoms, n_atoms))

        else:  # -> batch
            if n_atoms is None:
                raise ValueError(
                    "Number of atoms per system must be specified when working"
                    " with batches as it cannot be inferred easily.")
            distance = pack(
                [torch.sqrt(((ipos[:, :inat].repeat(1, inat, 1) -
                              icp[:inat].repeat_interleave(inat, 0)
                              ) ** 2).sum(-1)).reshape(-1, inat, inat)
                 for ipos, icp, inat in zip(
                    positions_expanded, positions, n_atoms)
                 ], value=1e3)

        return distance

    @cached_property("positions", "rcellvec")
    def periodic_distances(self):
        """Distances between atoms in the central and neighbouring cells."""
        return self.get_periodic_distances(
            self.rcellvec, self.positions, n_atoms=self.n_atoms)


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
        return abs(torch.det(self.lattice))


    @staticmethod
    def get_neighbours(periodic_distances, cutoff) -> Tensor:
        """A mask to choose atoms of images inside the cutoff distance."""
        if len(cutoff) == 1:
            return periodic_distances.le(cutoff)
        else:
            return torch.stack([ipd.le(ico) for ipd, ico in zip(
                periodic_distances, cutoff)])

    @property
    def neighbour(self) -> Tensor:
        """A mask to choose atoms of images inside the cutoff distance."""
        return self.get_neighbours(self.periodic_distances, self._cutoff)


    @cached_property("positions_vec")
    def neighbour_vector(self) -> Tensor:
        """Get positions for neighbour list within cutoff distance."""
        # Mask for images containing atoms within cutoff distance
        _mask = self.neighbour.any(-1).any(-1)

        # Pre-fetched to prevent having to go through the cache each time
        # within the periodic system's list comprehension.
        positions_vector = self.positions_vec

        if not self._n_batch:  # -> single
            neighbour_vec = positions_vector[_mask]
        else:  # -> batch
            neighbour_vec = pack([positions_vector[ibatch][_mask[ibatch]]
                                  for ibatch in range(self._n_batch)], value=1e3)

        return neighbour_vec

    @cached_property("periodic_distances")
    def neighbour_distance(self) -> Tensor:
        """Get distance matrix for neighbour list within cutoff distance."""
        # Mask for images containing atoms within cutoff distance
        _mask = self.neighbour.any(-1).any(-1)

        # Pre-fetched to prevent having to go through the cache each time
        # within the periodic system's list comprehension.
        periodic_distances = self.periodic_distances

        if not self._n_batch:  # -> single
            neighbour_dist = periodic_distances[_mask]
        else:  # -> batch
            neighbour_dist = pack(
                [periodic_distances[ibatch][_mask[ibatch]]
                 for ibatch in range(self._n_batch)], value=1e3)

        return neighbour_dist

    @property
    def _n_batch(self) -> Optional[int]: return (
        None if self.lattice.ndim == 2 else len(self.lattice))

    @property
    def _mask_zero(self): return self.lattice.eq(0).all(-1)

    @staticmethod
    def inverse_lattice_vector(lattice_vector, **kwargs):
        """inverse lattice vectors."""

        mask_zero = kwargs.get(
            'mask_zero', lattice_vector.eq(0).all(-1))

        n_batch = kwargs.get(
            "n_batch",
            None if lattice_vector.ndim == 2 else lattice_vector.shape[0])

        _latvec = lattice_vector + torch.diag_embed(
            mask_zero.type(lattice_vector.dtype))

        eye = torch.eye(
            _latvec.shape[-1], device=lattice_vector.device,
            dtype=lattice_vector.dtype)

        if n_batch is not None:
            eye = eye.repeat(n_batch, 1, 1)

        # Inverse lattice vectors
        invlat = torch.linalg.solve(_latvec, eye).transpose(-1, -2)

        invlat[mask_zero] = 0

        return invlat

    @cached_property("lattice")
    def _invlatvec(self):
        """inverse lattice vectors."""

        return self.inverse_lattice_vector(
            self.lattice, n_batch=self._n_batch, mask_zero=self._mask_zero)

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

    @staticmethod
    def get_cell_translation_vectors(
            lattice_vector: Tensor, cell_translation_vector_indices: Tensor):
        """Cell translation vectors in absolute units."""
        return (
            torch.matmul(lattice_vector.transpose(-1, -2),
                         cell_translation_vector_indices.transpose(-1, -2))
        ).transpose(-1, -2)

    @cached_property('lattice', 'cellvec')
    def rcellvec(self):
        """Cell translation vectors in absolute units."""
        return self.get_cell_translation_vectors(
            self.lattice, self.cellvec
        )

    @staticmethod
    def get_cell_translation_vector_indices(
            inverse_lattice_vector: Tensor, cutoff,
            negative_extension: float = 1, positive_extension: float = 1,
            **kwargs):
        """Calculate cell translation vector indices."""

        # This is exposed as a static method for the sake of the coulomb module

        # Issues, will change with the cutoff value thus this value should be
        # static.

        dd = {'dtype': inverse_lattice_vector.dtype,
              'device': inverse_lattice_vector.device}

        # Ranges of cell translation on three dimensions
        _tmp = bT(torch.floor(cutoff * bT(torch.norm(inverse_lattice_vector, dim=-1))))

        ranges = torch.stack([-(negative_extension + _tmp),
                                positive_extension + _tmp])

        # For 1D/2D cell translation, non-periodicity direction will be zero
        mask_zero = kwargs.get(
            'mask_zero', inverse_lattice_vector.eq(0).all(-1))
        ranges[torch.stack([mask_zero, mask_zero])] = 0

        # length of the first, second and third column in ranges
        leng = ranges[1, :].long() - ranges[0, :].long() + 1

        # number of cells
        n_cells = leng[..., 0] * leng[..., 1] * leng[..., 2]

        # Cell translation vectors in relative coordinates
        if inverse_lattice_vector.ndim == 2:  # -> single
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
                    leng, ranges.transpose(-2, -3))], value=1e4)

        return cellvec.transpose(-1, -2), n_cells

    @cached_property('_invlatvec')
    def cellvec(self) -> Tensor:
        """Implement cell translation."""
        # This just redirects to the static method.

        # It is worth noting that the `_n_cells` value will also be set here
        # for performance and stability reasons.
        cell_vectors, n_cells = self.get_cell_translation_vector_indices(
            self._invlatvec, self._cutoff, self._positive_extension,
            self._negative_extension, mask_zero=self._mask_zero)

        self._n_cells = n_cells

        return cell_vectors

    @staticmethod
    def frac_to_cartesian(cell: Tensor, positions: Tensor) -> Tensor:
        """Transfer fractional coordinates to cartesian coordinates."""
        pass

    @staticmethod
    def get_cell_lengths(cell: Tensor) -> Tensor:
        """Get the length of each lattice vector."""
        pass

    @staticmethod
    def get_cell_angles(cell: Tensor) -> Tensor:
        """Get the angles' alpha, beta and gamma of lattice vectors."""
        pass


class Triclinic(Periodicity):
    """Represents PBC systems that can be described by three basis vectors.

    This extends the functional of the `Periodicity` abstract base class to
    perform functionality associated with triclinic systems. Note that the
    term "triclinic" is use here to represent any and all boxed periodic
    systems that can be described using three basis vectors.

    Arguments:
        positions: Coordinates of the atoms in the central cell.
        lattice: Lattice vector, with Bohr as default unit.
        cutoff: Interaction cutoff distance for reading SK table, with Bohr
            as default unit.

    Keyword Arguments:
        distance_extension: Extension of cutoff in SK tables to smooth tails.
        positive_extension: Extension for the positive lattice vectors.
        negative_extension: Extension for the negative lattice vectors.

    Attributes:
        positions: Coordinates of the atoms in the central cell.
        lattice: Lattice vector.
        cutoff: Global cutoff for the diatomic interactions.
        cellvec: Cell translation vectors in relative coordinates.
        rcellvec: Cell translation vectors in absolute units.
        n_cells: Number of lattice cells.
        positions_vec: Position vector matrix between atoms in different cells.
        periodic_distances: Distance matrix between atoms in different cells.

    """

    def __init__(self, geometry,
                 cutoff: Union[Tensor, float], **kwargs):
        super().__init__(geometry, cutoff, **kwargs)

    @property
    def pbc(self):
        """Check the format of lattice vectors and other cell information."""

        n_batch: Optional[int] = (None if self.lattice.ndim == 2
                                  else len(self.lattice))

        # Check the format of lattice vectors
        if self.lattice.size(dim=-2) != 3:
            raise ValueError('Input cell should be defined by three '
                             'lattice vectors.')

        # Masks of periodicity systems
        _is_periodic: Tensor = self.lattice.ne(0).any(-1).any(-1)

        # Check the dimensions of periodicity boundary condition
        _sum_dim = self.lattice.ne(0).any(-1).sum(dim=-1)
        _dim_pe = self.lattice.ne(0).any(-1)

        if not n_batch:  # -> single
            # Type of pbc
            pbc = _dim_pe
        else:  # -> batch
            # Mixing of different pbc
            if not torch.all(torch.tensor([isd == _sum_dim[0]
                                           for isd in _sum_dim])):
                raise NotImplementedError(
                    'Mixing of different types of pbcs is not supported.')
            else:
                # Check directions of periodicity boundary condition
                if not torch.all(torch.tensor([torch.all(idp == _dim_pe[0])
                                               for idp in _dim_pe])):
                    # Different periodicity directions but same type of pbc
                    pbc = _dim_pe
                else:
                    # Same periodicity directions
                    pbc = _dim_pe[0]

        return pbc

    @staticmethod
    def frac_to_cartesian(lattice: Tensor, positions: Tensor) -> Tensor:
        """Transfer fractional coordinates to cartesian coordinates.

        Arguments:
            lattice: Lattice vectors.
            positions: Atomic positions.
        """
        # Whether fraction coordinates in the range [0, 1)
        if torch.any(positions >= 1) or torch.any(positions < 0):
            # Operate on a copy so that the original is not modified.
            positions = positions.clone()

            positions = torch.abs(positions) - torch.floor(
                torch.abs(positions))

        # Transfer from fractional to cartesian coordinates
        positions: Tensor = torch.matmul(positions, lattice)

        return positions

    def get_cell_lengths(self) -> Tensor:
        """Get the length of each lattice vector."""
        return torch.linalg.norm(self.lattice, dim=-1)

    def get_cell_angles(self) -> Tensor:
        """Get the angles' alpha, beta and gamma of lattice vectors."""
        _cos = torch.nn.CosineSimilarity(dim=-1)
        cosine = torch.stack([_cos(self.lattice[..., 1], self.lattice[..., 2]),
                              _cos(self.lattice[..., 0], self.lattice[..., 2]),
                              _cos(self.lattice[..., 0], self.lattice[..., 1])], -1)

        return torch.acos(cosine) * 180 / np.pi


