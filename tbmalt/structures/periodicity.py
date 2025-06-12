# -*- coding: utf-8 -*-
"""Code associated with performing cell translation.

This module implement cell translation for 1D & 2D & 3D periodic
boundary conditions. Distance matrix and position vectors for pbc
will be constructed.
"""
from typing import Union, Optional
from abc import ABC, abstractmethod
import torch
from torch import Tensor
import numpy as np
from tbmalt.common.batch import pack, bT
from tbmalt.common import cached_property


class Periodicity(ABC):
    """Properties and functionality associated with periodic systems.

    Derived classes can be used to calculate various properties of 1D, 2D, & 3D
    periodic systems, such as the translation vectors. This is not intended to
    store much in the way of data itself but rather provides useful methods.
    The periodicity entity will instead source information from the `Geometry`
    object it is provided.

    Arguments:
        geometry: Geometry object from which information like atomic positions
            and lattice parameters are to be sourced.
        cutoff: Interaction cutoff distance for interactions in units of Bohr.

    Attributes:
        geometry: the associated `Geometry` instance.
        cutoff: Interaction cutoff distance for interactions in units of Bohr.

    Notes:
        It is important to note that `Periodicity` entities are recursively
        linked to their corresponding `Geometry` instances. As such, the manual
        creation of `Periodicity` type instances, such as `Triclinic`, is not
        supported. Instead, one should provide the `lattice_vector` & `cutoff`
        arguments to the `Geometry` class which will create a `Periodicity`
        instance assigned to the `Geometry.periodicity` attribute. The
        recursive nature of the `Periodicity` and `Geometry` classes is
        something that will be addressed in future updates.
        
        This class make extensive use of cached properties and thus may use
        more memory than one might expect.

        When representing batch systems, the 'cellvec' and 'periodic_distances'
        will be padded with large numbers. Mixing of periodic & non-periodic
        systems and mixing of different types of pbc is forbidden.

    Examples:

        >>> import torch
        >>> from tbmalt import Geometry
        >>> geometry = Geometry(torch.tensor([1, 1]), torch.rand(2, 3),
        ...                     lattice_vector=torch.eye(3) * 3.0, cutoff=2.0)
        >>> pbc = geometry.periodicity
        >>> print(pbc.positions_vec.shape)
        torch.Size([27, 2, 2, 3])

    """
    def __init__(
            self, geometry: "Geometry", cutoff: Union[Tensor, float]):

        cutoff = self._check(geometry.lattice, cutoff)

        self.geometry = geometry

        # Intrinsic state attributes
        self._device = self.lattice.device
        self._dtype = self.lattice.dtype

        self.cutoff = cutoff

        # Note that this is set within the `cellvec` property method.
        self._n_cells = None

    @property
    def positions(self):
        return self.geometry.positions

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
        """Number of periodic cell images within the cutoff distance."""

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
        return self.get_neighbours(self.periodic_distances, self.cutoff)


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

    def _check(self, latvec: Tensor, cutoff: Union[Tensor, float]) -> Tensor:
        """Check dimension, type of lattice vector and cutoff."""

        # Lattice vector packing is not necessary here for the same reasons
        # outlined above at the end of this function.

        # # Lattice vectors can still be updated
        # if isinstance(latvec, list):
        #     latvec = pack(latvec)

        # Check the shape of lattice vectors
        if latvec.ndim < 2 or latvec.ndim > 3:
            raise ValueError('lattice vector dimension should be 2 or 3')

        # Check the format of cutoff
        if type(cutoff) is float:
            cutoff = torch.tensor([cutoff])
        elif type(cutoff) is Tensor:
            if cutoff.ndim == 0:
                cutoff = cutoff.unsqueeze(0)
            elif cutoff.ndim >= 2:
                raise ValueError(
                    'cutoff should be 0, 1 dimension tensor or float')
        elif type(cutoff) is not float:
            raise TypeError('cutoff should be tensor or float')

        # Unit conversion within `Periodicity` entities disabled for now.
        # Currently, instances are only created from within the `Geometry`
        # class which already performs the unit conversion. Furthermore,
        # it would be best not to modify the `Geometry` instance when
        # creating a new `Periodicity` object. This will get cleaned up
        # later when the two classes are decoupled.

        # # Default lattice vector is from geometry, thus default unit is bohr
        # unit = kwargs.get('unit', 'bohr')
        # if unit != 'bohr':
        #     latvec = latvec * length_units[unit]
        #     cutoff = cutoff * length_units[unit]

        return cutoff

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
            **kwargs):
        """Calculate cell translation vector indices."""

        # This is exposed as a static method for the sake of the coulomb module

        # Issues, will change with the cutoff value thus this value should be
        # static.

        dd = {'dtype': inverse_lattice_vector.dtype,
              'device': inverse_lattice_vector.device}

        # Ranges of cell translation on three dimensions
        n_images = bT(torch.ceil(cutoff * bT(torch.norm(
            inverse_lattice_vector, dim=-1))))

        ranges = torch.stack([-n_images, n_images])

        # For 1D/2D cell translation, non-periodic direction will be zero
        mask_zero = kwargs.get(
            'mask_zero', inverse_lattice_vector.eq(0).all(-1))
        ranges[torch.stack([mask_zero, mask_zero])] = 0

        # length of the first, second and third column in ranges
        leng = ranges[1, :].long() - ranges[0, :].long() + 1

        # number of cells
        n_cells = leng[..., 0] * leng[..., 1] * leng[..., 2]

        # Cell translation vectors in relative coordinates
        if inverse_lattice_vector.ndim == 2:  # -> single
            # Would be worth replacing this with `torch.cartesian_prod` at
            # some point.
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
            self._invlatvec, self.cutoff, mask_zero=self._mask_zero)

        self._n_cells = n_cells

        return cell_vectors

    @staticmethod
    def frac_to_cartesian(lattice: Tensor, positions_frac: Tensor) -> Tensor:
        """Convert fractional coordinates to cartesian coordinates.

        Arguments:
            lattice: Lattice vectors.
            positions_frac: Positions of the atoms in fractional coordinates.

        Returns:
            positions_cart: Positions of the atoms in Cartesian coordinates.
        """
        pass

    @staticmethod
    def get_cell_lengths(cell: Tensor) -> Tensor:
        """Get the length of each lattice vector."""
        pass

    @staticmethod
    def get_cell_angles(cell: Tensor) -> Tensor:
        """Get the angles' alpha, beta and gamma of lattice vectors."""
        pass

    @property
    @abstractmethod
    def pbc(self) -> Tensor:
        """Directions along which the system is deemed to be periodic."""
        pass


class Triclinic(Periodicity):
    """Represents PBC systems that can be described by three basis vectors.

    This extends the functional of the `Periodicity` abstract base class to
    perform functionality associated with triclinic systems. Note that the
    term "triclinic" is use here to represent any and all boxed periodic
    systems that can be described using three basis vectors.

    Arguments:
        geometry: Geometry object from which information like atomic positions
            and lattice parameters are to be sourced.
        cutoff: Interaction cutoff distance for interactions in units of Bohr.

    Attributes:
        geometry: the associated `Geometry` instance.
        cutoff: Interaction cutoff distance for interactions in units of Bohr.

    Examples:

        >>> import torch
        >>> from tbmalt import Geometry
        >>> geometry = Geometry(torch.tensor([1, 1]), torch.rand(2, 3),
        ...                     lattice_vector=torch.eye(3) * 3.0, cutoff=2.0)
        >>> pbc = geometry.periodicity
        >>> print(pbc.positions_vec.shape)
        torch.Size([27, 2, 2, 3])

    """

    def __init__(self, geometry: "Geometry", cutoff: Union[Tensor, float]):
        super().__init__(geometry, cutoff)

    @property
    def pbc(self) -> Tensor:
        """Directions along which the system is deemed to be periodic."""

        # This will also check the format of lattice vectors and other cell
        # information

        n_batch: Optional[int] = (None if self.lattice.ndim == 2
                                  else len(self.lattice))

        # Check the format of lattice vectors
        if self.lattice.size(dim=-2) != 3:
            raise ValueError('Input cell should be defined by three '
                             'lattice vectors.')

        # Masks of periodic systems
        _is_periodic: Tensor = self.lattice.ne(0).any(-1).any(-1)

        # Check the dimensions of periodic boundary condition
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
    def frac_to_cartesian(lattice: Tensor, positions_frac: Tensor) -> Tensor:
        """Convert fractional coordinates to cartesian coordinates.

        Arguments:
            lattice: Lattice vectors.
            positions_frac: Positions of the atoms in fractional coordinates.

        Returns:
            positions_cart: Positions of the atoms in Cartesian coordinates.
        """
        # Wrap the fractional coordinates to the domain [0, 1) if necessary.
        if not (positions_frac.ge(0).all() and positions_frac.lt(1).all()):
            positions_frac = positions_frac.remainder(1.0)

        # Covert from fractional to cartesian coordinates & return the result
        return torch.matmul(positions_frac, lattice)

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


