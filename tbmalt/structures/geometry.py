# -*- coding: utf-8 -*-
"""A container to hold data associated with a chemical system's structure.

This module provides the `Geometry` data structure class and its associated
code. The `Geometry` class is intended to hold any & all data needed to fully
describe a chemical system's structure.
"""
from typing import Union, List, Optional, Type, Tuple
import warnings
from operator import itemgetter
import torch
from torch import Tensor
import numpy as np
from h5py import Group
from ase import Atoms
from ase.lattice.bravais import Lattice
from tbmalt.structures.periodicity import Triclinic, Periodicity
from tbmalt.common.batch import pack, merge, deflate
from tbmalt.data.units import length_units
from tbmalt.data import chemical_symbols

# Todo:
#   - Currently periodic systems are hard-coded to use the `Triclinic` periodic
#     helper class. Note that triclinic is used in reference to the general
#     geometric shape and not the specific crystal structure. This is done
#     because only one type of periodic boundary condition is supported at this
#     time. Later this should be generalised. Hardcoding is present in both the
#     `__init__` and `__preprocess` methods.
#   - The `distance_vectors` & `distances` properties should be converted into
#     cached functions & updated to respect periodic boundary conditions.


class Geometry:
    """Data structure for storing geometric information on molecular systems.

    The `Geometry` class stores any information that is needed to describe a
    chemical system; atomic numbers, positions, etc. This class also permits
    batch system representation. However, mixing of PBC & non-PBC systems is
    strictly forbidden.

    Arguments:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Atomic coordinates are specified via an "Nx3" tensor for
            single system instances, where "N" is the nubmer of atoms. For
            batch instanes either i) a single zero-padded "BxMxN" tensor may
            be specified, where "B" is the number of batches and "M" is the
            number of atoms in the largest system, or ii) a list of "Nx3"
            tensors may be provided.
        lattice_vector: Lattice vectors of the periodicity systems. This is
            argument, commonly a 3x3 tensor, is only relevant for periodic
            systems. This is used by underlying `Periodicity` instances to
            construct periodic dependant properties. [DEFAULT: None]
        frac: Whether using fractional coordinates to describe periodicity
            systems. [DEFAULT: False]
        cutoff: Global cutoff for the diatomic interactions in periodicity
            systems. [DEFAULT: 9.98].
        units: Unit in which ``positions``, ``lattice_vector``, & ``cutoff``
            were specified. For a list of available units see :mod:`.units`
            [DEFAULT='bohr'].

    Attributes:
        atomic_numbers: Atomic numbers of the atoms.
        positions: Coordinates of the atoms.
        n_atoms: Number of atoms in the system.
        periodicity: Periodicity object that offers helper methods for
            periodic systems. Geometric properties like distances should
            be accessed via this entity when working with periodic systems.
        lattice: the lattice vectors.

    Notes:
        When representing multiple systems, the `atomic_numbers` & `positions`
        tensors will be padded with zeros. Tensors generated from ase atoms
        objects or HDF5 database entities will not share memory with their
        associated numpy arrays, nor will they inherit their dtype.

    Warnings:
        At this time, 1D & 2D & 3D periodicity boundary conditions are supported,
        but mixing of different types of periodicity boundary conditions is
        forbidden. The mixing of fractional and cartesian coordinates is
        also forbidden. While Helical periodicity boundary condition is not
        supported.

    Examples:
        Geometry instances may be created by directly passing in the atomic
        numbers & atom positions

        >>> import torch
        >>> from tbmalt import Geometry
        >>> H2 = Geometry(torch.tensor([1, 1]),
        >>>               torch.tensor([[0.00, 0.00, 0.00],
        >>>                             [0.00, 0.00, 0.79]]))
        >>> print(H2)
        Geometry(H2)

        Or from an ase.Atoms object

        >>> from tbmalt import Geometry
        >>> from ase.build import molecule
        >>> CH4_atoms = molecule('CH4')
        >>> print(CH4_atoms)
        Atoms(symbols='CH4', pbc=False)
        >>> CH4 = Geometry.from_ase_atoms(CH4_atoms)
        >>> print(CH4)
        Geometry(CH4)

        Multiple systems can be represented by a single ``Geometry`` instance.
        To do this, simply pass in lists or packed tensors where appropriate.

    """

    __slots__ = ['atomic_numbers', '_positions', 'n_atoms', 'periodicity',
                 'lattice', '_n_batch',
                 '__dtype', '__device']

    # Developers notes technically the `cutoff` argument has the default value
    # of `None`. However, this is assigned to 9.98 bohr internally after the
    # fact. This is done to prevent the default value from undergoing a unit
    # conversion if the user provides the atomic positions and lattice vectors
    # in units other than atomic units.

    def __init__(
            self, atomic_numbers: Union[Tensor, List[Tensor]],
            positions: Union[Tensor, List[Tensor]],
            lattice_vector: Optional[
                Union[Tensor, List[Tensor], Type[Lattice]]] = None,
            frac: bool = False,
            cutoff: Optional[Union[Tensor, float]] = None,
            units: str = 'bohr', **kwargs):

        # Perform general preprocessing and sanitisation of the inputs
        atomic_numbers, positions, lattice_vector, cutoff = self.__preprocess(
            atomic_numbers, positions, lattice_vector, cutoff, frac, units)

        self.atomic_numbers: Tensor = atomic_numbers
        self._positions: Tensor = positions

        # These are static, private variables and must NEVER be modified!
        self.__device = self._positions.device
        self.__dtype = self._positions.dtype

        self.n_atoms: Tensor = self.atomic_numbers.count_nonzero(-1)

        # Number of batches if in batch mode (for internal use only)
        self._n_batch: Optional[int] = (None if self.atomic_numbers.dim() == 1
                                        else len(atomic_numbers))

        if lattice_vector is not None:
            # Cutoff distance for the diatomic interactions in periodicity systems
            cutoff = torch.tensor(
                [9.98], device=self.__device, dtype=self.__dtype
            ) if cutoff is None else cutoff

            self.lattice = lattice_vector

            # Fetch any `Periodicity` specific keyword arguments.
            pbc_keys = ["positive_extension", "negative_extension", "tail_distance"]
            pbc_kwargs = {k: v for k, v in kwargs.items() if k in pbc_keys}

            # Currently periodicity if hardcoded to be triclinic
            self.periodicity: Periodicity = Triclinic(self, cutoff, **pbc_kwargs)

            if units != "bohr" and len(pbc_kwargs) != 0:
                raise NotImplementedError(
                    "Periodicity related keyword arguments are not compatible "
                    "with automatic unit conversion.")

        else:
            self.periodicity = None
            self.lattice = None

    @staticmethod
    def __preprocess(
            atomic_numbers: Union[Tensor, List[Tensor]],
            positions: Union[Tensor, List[Tensor]],
            lattice_vector: Optional[Union[Tensor, List[Tensor], Type[Lattice]]],
            cutoff: Optional[Union[Tensor, float]],
            frac: bool, units: str):
        """Perform general safety checks and conversion operations.

        This method just abstracts a lot of overly verbose and messy safety
        checks and conversions that are performed on the inputs. This is done
        to make the `__init__` method a little cleaner.
        """

        # Preprocessing:
        # 1) Ensure that `atomic_numbers` and `positions` are packed as needed,
        #    `pack` only effects lists of tensors, and deflate is used to
        #    remove any unnecessary padding.
        if isinstance(atomic_numbers, list):
            atomic_numbers = deflate(pack(atomic_numbers))

        if isinstance(positions, list):
            # Warn when batching gradient-tracked leaf-nodes.
            if any([i.requires_grad and i.is_leaf for i in positions]):
                warnings.warn(
                    "Care must be taken, gradient-tracked leaf-node detected "
                    "in auto-batch request. Atomic positions have been "
                    "supplied as a list of tensors that are to be pack "
                    "into a single batch tensor. One or more of the supplied "
                    "tensors are gradient-tracked leaf-nodes, however these "
                    "will not be stored within the  `Geometry` class, only the "
                    "packed tensor that results from them. As such auto-grad "
                    "related operations, such as `Geometry.positions.grad`, may "
                    "not behave as expected.")
            positions = pack(positions)[..., :atomic_numbers.shape[-1], :]

        device = positions.device

        # 2) If the lattice vectors are supplied as ase.lattice.bravais.Lattice
        #    instances then the lattice vector arrays must be extracted from
        #    them. Then make sure that the lattice vectors are pytorch arrays
        #    and are packed if required. Note that care must be taken here not
        #    to accidentally cause the lattice vectors to be deflated. A row
        #    of all zeros is valid in a lattice vector and thus should not be
        #    pruned.
        if isinstance(lattice_vector, Lattice):
            lattice_vector = lattice_vector.cell.array

        # Cast from a list ase `Lattice` instance to list of PyTorch arrays
        elif (isinstance(lattice_vector, list)
              and isinstance(lattice_vector[0], Lattice)):
            lattice_vector = [torch.tensor(
                i.cell.array, device=device,
                dtype=positions.dtype) for i in lattice_vector]

        if isinstance(lattice_vector, list):
            lattice_vector = pack(lattice_vector).to(device)

        elif isinstance(lattice_vector, np.ndarray):
            lattice_vector = torch.tensor(lattice_vector).to(device)

        # 3) Ensure tensors are on the same device (only two present currently)
        if device != atomic_numbers.device:
            raise RuntimeError('All tensors must be on the same device!')

        # 4) Lattice vectors many not be zero dimensional
        if lattice_vector is not None and (
                ~lattice_vector.ne(0).any(-1).any(-1)).any():
            raise ValueError('Lattice vectors may not be zero dimensional!')

        # 5) Convert fractional positions to their Cartesian values
        if frac and lattice_vector is not None:
            positions = Triclinic.frac_to_cartesian(lattice_vector, positions)

        # 6) a non-periodic system cannot be given in fractional coordinates
        if frac and lattice_vector is None:
            raise ValueError(
                'Fractional coordinates cannot be used for clusters!')

        # 7) Perform unit conversions as and when required. TBMaLT uses atomic
        #    units internally; i.e. Bohr for distance.
        if units != 'bohr':
            conversion_factor = length_units[units]
            positions = positions * conversion_factor
            if lattice_vector is not None:
                lattice_vector = lattice_vector * conversion_factor
            if cutoff is not None:
                cutoff = cutoff * conversion_factor

        # 8) Check for shape discrepancies in `positions`, `atomic_numbers`, &
        #   `cells`. A position/atomic-number mismatch would have caused an
        #   error in the first step so it technically does not need to be
        #   explicitly checked for.
        check_1 = positions.shape[:-1] == atomic_numbers.shape
        assert check_1, '`positions` & `atomic_numbers` shape mismatch found'
        if lattice_vector is not None and atomic_numbers.ndim != 1:
            check_2 = positions.shape[0] == lattice_vector.shape[0]
            assert check_2, '`positions` & `lattice_vector` shape mismatch found'

        return atomic_numbers, positions, lattice_vector, cutoff

    def _update(self):
        pass

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value
        if self.periodicity is not None:
            self.periodicity._positions = value

    @property
    def device(self) -> torch.device:
        """The device on which the `Geometry` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        # Instruct users to use the ".to" method if wanting to change device.
        raise AttributeError('Geometry object\'s dtype can only be modified '
                             'via the ".to" method.')

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by geometry object."""
        return self.__dtype

    @property
    def is_periodic(self) -> bool:
        """If there is any periodicity boundary conditions."""
        return False if self.periodicity is None else True

    @property
    def distances(self) -> Tensor:
        """Distance matrix between atoms in the system."""

        # Ensure padding area is zeroed out
        # But don't modify in place
        dist = torch.cdist(self.positions, self.positions, p=2).clone()
        dist[self._mask_dist] = 0

        # cdist bug, sometimes distances diagonal is not zero
        idx = torch.arange(dist.shape[-1])
        if not (dist[..., idx, idx].eq(0)).all():
            dist[..., idx, idx] = 0.0

        return dist

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        dist_vec = self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3)
        dist_vec[self._mask_dist] = 0
        return dist_vec

    @property
    def chemical_symbols(self) -> list:
        """Chemical symbols of the atoms present."""
        return batch_chemical_symbols(self.atomic_numbers)

    @property
    def pbc(self) -> Union[bool, Tensor]:
        """Directions along which the system is deemed to be periodic."""
        return self.periodicity.pbc if self.periodicity is not None else False

    @property
    def _mask_dist(self):
        """Mask for clearing padding values in the distance matrix."""

        # Identify which atoms are not padding atoms. It is assumed that the
        # value "0" is used for padding that atomic number tensor.
        mask = self.atomic_numbers != 0

        # If no padding atoms are present then no masking is needed, thus the
        # value `False` is returned; i.e. mask out no parts of a tensor.
        if (mask != 0).all():
            return False
        # If padding values are found then expand the mask into a 2D slice for
        # each system present in the current `Geometry` instance.
        else:
            return ~(mask.unsqueeze(-2) * mask.unsqueeze(-1))

    def unique_atomic_numbers(self) -> Tensor:
        """Identifies and returns a tensor of unique atomic numbers.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_atomic_numbers: A tensor specifying the unique atomic
                numbers present.
        """
        return torch.unique(self.atomic_numbers[self.atomic_numbers.ne(0)])

    @classmethod
    def from_ase_atoms(cls, atoms: Union[Atoms, List[Atoms]],
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None,
                       units: str = 'angstrom') -> 'Geometry':
        """Instantiates a Geometry instance from an `ase.Atoms` object.

        Multiple atoms objects can be passed in to generate a batched Geometry
        instance which represents multiple systems.

        Arguments:
            atoms: Atoms object(s) to instantiate a Geometry instance from.
            device: Device on which to create any new tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            units: Length unit used by `Atoms` object. [DEFAULT='angstrom']

        Returns:
            geometry: The resulting ``Geometry`` object.

        Raises:
            NotImplementedError: If there is a mixing of `ase.Atoms` objects
            that have both periodicity boundary conditions and non-periodicity
            boundary conditions.
        """
        # If not specified by the user; ensure that the default dtype is used,
        # rather than inheriting from numpy. Failing to do this will case some
        # *very* hard to diagnose errors.
        dtype = torch.get_default_dtype() if dtype is None else dtype

        # Check periodicity systems
        def check_not_periodic(atom_instance):
            """Check whether a system is periodicity."""
            return True if atom_instance.pbc.any() else False

        if not isinstance(atoms, list):  # If a single system
            # Check PBC systems
            _pbc = check_not_periodic(atoms)
            if not _pbc:
                return cls(  # Create a Geometry instance and return it
                    torch.tensor(atoms.get_atomic_numbers(), device=device),
                    torch.tensor(atoms.positions, device=device, dtype=dtype),
                    units=units)
            else:
                return cls(  # Create a Geometry instance and return it
                    torch.tensor(atoms.get_atomic_numbers(), device=device),
                    torch.tensor(atoms.positions, device=device, dtype=dtype),
                    torch.tensor(atoms.cell[:], device=device, dtype=dtype),
                    units=units)

        else:  # If a batch of systems
            # Check PBC systems
            _pbc = [check_not_periodic(a) for a in atoms]
            if not torch.any(torch.tensor(_pbc)):  # -> A batch of non-PBC
                return cls(  # Create a batched Geometry instance and return it
                    [torch.tensor(a.get_atomic_numbers(), device=device)
                     for a in atoms],
                    [torch.tensor(a.positions, device=device, dtype=dtype)
                     for a in atoms],
                    units=units)
            elif torch.all(torch.tensor(_pbc)):  # -> A batch of PBC
                return cls(  # Create a batched Geometry instance and return it
                    [torch.tensor(a.get_atomic_numbers(), device=device)
                     for a in atoms],
                    [torch.tensor(a.positions, device=device, dtype=dtype)
                     for a in atoms],
                    [torch.tensor(a.cell[:], device=device, dtype=dtype)
                     for a in atoms],
                    units=units)
            else:
                raise NotImplementedError(
                    'Mixing of PBC and non-PBC is not supported.')

    @classmethod
    def from_hdf5(cls, source: Union[Group, List[Group]],
                  device: Optional[torch.device] = None,
                  dtype: Optional[torch.dtype] = None,
                  units: str = 'bohr') -> 'Geometry':
        """Instantiate a `Geometry` instances from an HDF5 group.

        Construct a `Geometry` entity using data from an HDF5 group. Passing
        multiple groups, or a single group representing multiple systems, will
        return a batched `Geometry` instance.

        Arguments:
            source: An HDF5 group(s) containing geometry data.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            units: Unit of length used by the data. [DEFAULT='bohr']

        Returns:
            geometry: The resulting ``Geometry`` object.

        """
        # If not specified by the user; ensure that the default dtype is used,
        # rather than inheriting from numpy. Failing to do this will case some
        # *very* hard to diagnose errors.
        dtype = torch.get_default_dtype() if dtype is None else dtype

        # If a single system or a batch system
        if not isinstance(source, list):
            # Read & parse a datasets from the database into a System instance
            # & return the result.
            if 'lattice_vector' not in source.keys():  # Check PBC systems
                return cls(torch.tensor(source['atomic_numbers'][()],
                                        device=device),
                           torch.tensor(source['positions'][()],
                                        dtype=dtype, device=device),
                           units=units)
            else:
                return cls(torch.tensor(source['atomic_numbers'][()],
                                        device=device),
                           torch.tensor(source['positions'][()],
                                        dtype=dtype, device=device),
                           torch.tensor(source['lattice_vector'][()],
                                        dtype=dtype, device=device),
                           units=units)

        else:
            # Check PBC systems
            _pbc = ['lattice_vector' in s.keys() for s in source]
            if not torch.any(torch.tensor(_pbc)):  # -> A batch of non-PBC
                return cls(  # Create a batched Geometry instance and return it
                    [torch.tensor(s['atomic_numbers'][()], device=device)
                     for s in source],
                    [torch.tensor(s['positions'][()], device=device, dtype=dtype)
                     for s in source],
                    units=units)
            elif torch.all(torch.tensor(_pbc)):  # -> A batch of PBC:
                return cls(  # Create a batched Geometry instance and return it
                    [torch.tensor(s['atomic_numbers'][()], device=device)
                     for s in source],
                    [torch.tensor(s['positions'][()], device=device, dtype=dtype)
                     for s in source],
                    [torch.tensor(s['lattice_vector'][()], device=device, dtype=dtype)
                     for s in source],
                    units=units)
            else:
                raise NotImplementedError(
                    'Mixing of PBC and non-PBC is not supported.')

    def to_hdf5(self, target: Group):
        """Saves Geometry instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the system's data should be saved.

        Notes:
            This function does not create its own group as it expects that
            ``target`` is the group into which data should be writen.
        """
        # Short had for dataset creation
        add_data = target.create_dataset

        # Add datasets for atomic_numbers, positions, lattice, and pbc
        add_data('atomic_numbers', data=self.atomic_numbers.cpu().numpy())
        pos = add_data('positions', data=self.positions.cpu().numpy())
        if self.periodicity is not None:
            add_data(
                'lattice_vector', data=self.periodicity.lattice.cpu().numpy())

        # Add units meta-data to the atomic positions
        pos.attrs['unit'] = 'bohr'

    def to(self, device: torch.device) -> 'Geometry':
        """Returns a copy of the `Geometry` instance on the specified device.

        This method creates and returns a new copy of the `Geometry` instance
        on the specified device "``device``".

        Arguments:
            device: Device to which all associated tensors should be moved.

        Returns:
            geometry: A copy of the `Geometry` instance placed on the
                specified device.

        Notes:
            If the `Geometry` instance is already on the desired device then
            `self` will be returned.
        """
        # Developers Notes: It is imperative that this function gets updated
        # whenever new attributes are added to the `Geometry` class. Otherwise
        # this will return an incomplete `Geometry` object.
        if self.atomic_numbers.device == device:
            return self
        else:
            if self.periodicity is None:
                return self.__class__(self.atomic_numbers.to(device=device),
                                      self.positions.to(device=device))
            else:
                pbc = self.periodicity
                return self.__class__(
                    self.atomic_numbers.to(device=device),
                    self.positions.to(device=device),
                    lattice_vector=self.lattice.to(device=device),
                    cutoff=pbc.base_cutoff.to(device=device),
                    positive_extension=pbc.positive_extension,
                    negative_extension=pbc.negative_extension,
                    tail_distance=pbc.tail_distance)

    def __getitem__(self, selector) -> 'Geometry':
        """Permits batched Geometry instances to be sliced as needed."""
        # Block this if the instance has only a single system
        if self.atomic_numbers.ndim != 2:
            raise IndexError(
                'Geometry slicing is only applicable to batches of systems.')

        # Select the desired atomic numbers and positions.
        new_zs = deflate(self.atomic_numbers[selector, ...])
        new_pos = self.positions[selector, ...][..., :new_zs.shape[-1], :]

        if self.is_periodic:
            pbc = self.periodicity
            return self.__class__(
                new_zs, new_pos, lattice_vector=self.lattice[selector, ...],
                cutoff=pbc.base_cutoff.max(),
                positive_extension=pbc.positive_extension,
                negative_extension=pbc.negative_extension,
                tail_distance=pbc.tail_distance)

        else:
            return self.__class__(new_zs, new_pos)


    def __add__(self, other: 'Geometry') -> 'Geometry':
        """Combine two `Geometry` objects together."""
        if self.__class__ != other.__class__:
            raise TypeError(
                'Addition can only take place between two Geometry objects.')

        # Catch for situations where one or both systems are not batched.
        s_batch = self.atomic_numbers.ndim == 2
        o_batch = other.atomic_numbers.ndim == 2

        an_1 = torch.atleast_2d(self.atomic_numbers)
        an_2 = torch.atleast_2d(other.atomic_numbers)

        pos_1 = self.positions
        pos_2 = other.positions

        cell_1 = self.lattice
        cell_2 = other.lattice

        pos_1 = pos_1 if s_batch else pos_1.unsqueeze(0)
        pos_2 = pos_2 if o_batch else pos_2.unsqueeze(0)

        if cell_1 is not None and not s_batch:
            cell_1 = cell_1.unsqueeze(0)

        if cell_2 is not None and not o_batch:
            cell_2 = cell_2.unsqueeze(0)

        if cell_1 is None and cell_2 is None:  # -> Two non-PBC objects
            return self.__class__(merge([an_1, an_2]), merge([pos_1, pos_2]))
        elif (cell_1 is not None and
              cell_2 is not None):  # -> Two PBC objects

            pbc_1, pbc_2 = self.periodicity, other.periodicity

            base_cutoff = torch.max(
                pbc_1.base_cutoff.max(), pbc_2.base_cutoff.max())

            positive_extension = max(
                pbc_1.positive_extension, pbc_2.positive_extension)

            negative_extension = max(
                pbc_1.negative_extension, pbc_2.negative_extension)

            tail_distance = max(
                pbc_1.tail_distance, pbc_2.tail_distance)

            return self.__class__(
                merge([an_1, an_2]),
                merge([pos_1, pos_2]),
                merge([cell_1, cell_2]),
                cutoff=base_cutoff,
                positive_extension=positive_extension,
                negative_extension=negative_extension,
                tail_distance=tail_distance
                )

        else:  # -> One PBC object and one non-PBC object
            raise TypeError(
                'Addition can not take place between a PBC object and '
                'a non-PBC object.')

    def __eq__(self, other: 'Geometry') -> bool:
        """Check if two `Geometry` objects are equivalent."""
        # Note that batches with identical systems but a different order will
        # return False, not True.

        if self.__class__ != other.__class__:
            raise TypeError(f'"{self.__class__}" ==  "{other.__class__}" '
                            f'evaluation not implemented.')

        def shape_and_value(a, b):
            if a is None and b is None:
                return True
            elif a is not None and b is not None:
                return a.shape == b.shape and torch.allclose(a, b)
            else:
                return False

        return all([
            shape_and_value(self.atomic_numbers, other.atomic_numbers),
            shape_and_value(self.positions, other.positions),
            shape_and_value(self.lattice, other.lattice)
        ])

    def __repr__(self) -> str:
        """Creates a string representation of the Geometry object."""

        # Return Geometry(CH4) for a single system & Geometry(CH4, H2O, ...)
        # for multiple systems. Only the first & last two systems get shown if
        # there are more than four systems (this prevents endless spam).

        def get_formula(atomic_numbers: Tensor) -> str:
            """Helper function to get reduced formula."""
            # If n atoms > 30; then use the reduced formula
            if len(atomic_numbers) > 30:
                return ''.join([f'{chemical_symbols[z]}{n}' if n != 1 else
                                f'{chemical_symbols[z]}' for z, n in
                                zip(*atomic_numbers.unique(return_counts=True))
                                if z != 0])  # <- Ignore zeros (padding)

            # Otherwise list the elements in the order they were specified
            else:
                return ''.join(
                    [f'{chemical_symbols[int(z)]}{int(n)}' if n != 1 else
                     f'{chemical_symbols[z]}' for z, n in
                     zip(*torch.unique_consecutive(atomic_numbers,
                                                   return_counts=True))
                     if z != 0])

        if self.atomic_numbers.dim() == 1:  # If a single system
            formula = get_formula(self.atomic_numbers)
        else:  # If multiple systems
            if self.atomic_numbers.shape[0] <= 4:  # If n<=4 systems; show all
                formulas = [get_formula(an) for an in self.atomic_numbers]
                formula = ', '.join(formulas)
            else:  # If n>4; show only the first and last two systems
                formulas = [get_formula(an) for an in
                            self.atomic_numbers[[0, 1, -2, -1]]]
                formula = '{}, {}, ..., {}, {}'.format(*formulas)

        if self.periodicity is None:
            # Wrap the formula(s) in the class name and return
            return f'{self.__class__.__name__}({formula})'
        else:  # Add PBC information
            if self.pbc.ndim == 1:  # If same periodicity direction
                formula_pbc = 'pbc=' + str(self.pbc.tolist())
            else:  # If multiple directions
                if self.pbc.shape[0] <= 4:  # Show all
                    formula_pbc = 'pbc=' + str(self.pbc.tolist())
                else:  # Show only the first and last two systems
                    formulas_pbc = [str(pd) for pd in
                                    self.pbc[[0, 1, -2, -1]].tolist()]
                    formula_pbc = 'pbc={}, {}, ..., {}, {}'.format(
                        *formulas_pbc)
            return f'{self.__class__.__name__}({formula}, {formula_pbc})'

    def __str__(self) -> str:
        """Creates a printable representation of the System."""
        # Just redirect to the `__repr__` method
        return repr(self)

    def clone(self) -> 'Geometry':
        """Returns a copy of the `Geometry` instance.
        This method creates and returns a new copy of the `Geometry` instance.
        Returns:
            geometry: A copy of the `Geometry` instance.
        """
        if self.periodicity is None:
            return self.__class__(self.atomic_numbers.clone(),
                                  self.positions.clone())
        else:
            pbc = self.periodicity
            return self.__class__(
                self.atomic_numbers.clone(),
                self.positions.clone(),
                lattice_vector=self.lattice.clone(),
                cutoff=pbc.base_cutoff.clone(),
                positive_extension=pbc.positive_extension,
                negative_extension=pbc.negative_extension,
                tail_distance=pbc.tail_distance)

    def detach(self) -> 'Geometry':
        """Returns a copy of the `Geometry` instance.
        This method creates and returns a new copy of the `Geometry` instance.
        Returns:
            geometry: A copy of the `Geometry` instance.
        """
        if self.periodicity is None:
            return self.__class__(self.atomic_numbers.detach(),
                                  self.positions.detach())
        else:
            pbc = self.periodicity
            return self.__class__(
                self.atomic_numbers.detach(),
                self.positions.detach(),
                lattice_vector=self.lattice.detach(),
                cutoff=pbc.base_cutoff.detach(),
                positive_extension=pbc.positive_extension,
                negative_extension=pbc.negative_extension,
                tail_distance=pbc.tail_distance)


####################
# Helper Functions #
####################
def batch_chemical_symbols(atomic_numbers: Union[Tensor, List[Tensor]]
                           ) -> list:
    """Converts atomic numbers to their chemical symbols.

    This function allows for en-mass conversion of atomic numbers to chemical
    symbols.

    Arguments:
        atomic_numbers: Atomic numbers of the elements.

    Returns:
        symbols: The corresponding chemical symbols.

    Notes:
        Padding vales, i.e. zeros, will be ignored.

    """
    a_nums = atomic_numbers

    # Catch for list tensors (still faster doing it this way)
    if isinstance(a_nums, list) and isinstance(a_nums[0], Tensor):
        a_nums = pack(a_nums, value=0)

    # Convert from atomic numbers to chemical symbols via a itemgetter
    symbols = np.array(  # numpy must be used as torch cant handle strings
        itemgetter(*a_nums.flatten())(chemical_symbols)
    ).reshape(a_nums.shape)
    # Mask out element "X", aka padding values
    mask = symbols != 'X'
    if symbols.ndim == 1:
        return symbols[mask].tolist()
    else:
        return [s[m].tolist() for s, m in zip(symbols, mask)]


def unique_atom_pairs(
        geometry: Geometry, return_ordered_pairs: bool = False) -> Tensor:
    """Returns a tensor specifying all unique atom pairs.

    This takes `Geometry` instance and identifies all atom pairs. This use
    useful for identifying all possible two body interactions possible within
    a given system.

    Arguments:
         geometry: `Geometry` instance representing the target system.
         return_ordered_pairs: `bool` indicating whether to consider pairs as
            ordered. If set to `False`, pairs like (1, 2) and (2, 1) are
            treated as equivalent and only the former is returned. If set to
            `True`, both (2, 1) and (1, 2) will be returned as they will be
            considered to be distinct. [DEFAULT: False]

    Returns:
        unique_atom_pairs: A tensor specifying all unique atom pairs.
    """

    unique_atomic_numbers = geometry.unique_atomic_numbers()

    if return_ordered_pairs:
        n_global = len(unique_atomic_numbers)
        return torch.stack([
            unique_atomic_numbers.repeat_interleave(n_global),
            unique_atomic_numbers.repeat(n_global)]).T
    else:
        return torch.combinations(unique_atomic_numbers,
                                  with_replacement=True)


def atomic_pair_distances(
        geometry: Geometry, ignore_self: bool = False,
        force_batch_index: bool = False) -> Tuple[Tensor, Tensor]:
    """Atomic pair distance generator.

    Generates all unique element-pair types and, for each such pair, yields the
    indices of the matching atom pairs along with the distance between each
    pair. Useful for enumerating all pairwise interactions present in a system
    or batch thereof, including possible consideration of periodic boundary
    conditions.

    Arguments:
        geometry: The system, or batch thereof, for which pair-wise
            interaction indices are to be generated.
        ignore_self: Boolean indicating whether self-interaction pairs should
            be ignored. If enabled, any pair where both indices refer to the
            same atom (i.e. an atom interacting with itself) is excluded.
            This only pertains to self-interactions and not interactions
            with copies in neighbouring images.[DEFAULT: False]
        force_batch_index: If ``True``, forces the returned ``pair_indices``
            to include a leading batch index dimension even for single systems.
            By default (``False``), a batch dimension is only included if
            ``geometry`` actually represents a batch. [DEFAULT: False]

    Yields:
        pair: A tensor of shape `(2,)` specifying the atomic numbers of the two
            species that define the pair. For example, `(6, 8)` for a carbon -
            oxygen pair.
        pair_indices: A tensor of integer indices that identify each occurrence
            of that ``pair`` in the system(s). The exact shape and meaning of
            each row depends on three factors:
            (1) whether the system is single or batched,
            (2) whether the system is periodic, and
            (3) whether ``force_batch_index`` is set:

            - **Single, non-periodic**:
              - `force_batch_index=False` ⇒ shape `(2, P)`
              - `force_batch_index=True`  ⇒ shape `(3, P)`
            - **Single, periodic**:
              - `force_batch_index=False` ⇒ shape `(3, P)`
              - `force_batch_index=True`  ⇒ shape `(4, P)`
            - **Batched, non-periodic** ⇒ shape `(3, P)`
            - **Batched, periodic** ⇒ shape `(4, P)`

            In all cases, the final two rows (or columns, after transpose)
            indicate the atomic indices of the interacting atoms. If there
            is a batch dimension, it occupies the first row. If the system
            is periodic, there is an additional row for the periodic cell
            index (or indices, depending on implementation). The remaining
            rows, if present, are the atomic indices within that batch and/or
            cell context.
        distances: A tensor of distances for each indexed pair, with the same
            “pair-count” dimension `P`. For periodic systems this will be the
            periodic distance; otherwise the direct distance.
    """

    # Compute distances between all atom pairs
    distances = geometry.periodicity.periodic_distances \
        if geometry.is_periodic else geometry.distances

    # Enforcing the presence of a batch dimension if requested
    if force_batch_index and geometry.atomic_numbers.ndim == 1:
        distances = distances.view(
            -1, *distances.shape[-3 if geometry.is_periodic else -2:])

    # Wrap the `atomic_pair_indices` generator to extend its returns to
    # include distances.
    for pair, idx in atomic_pair_indices(
            geometry, ignore_self, force_batch_index):
        yield pair, idx, distances[*idx]


def atomic_pair_indices(
        geometry: Geometry, ignore_self: bool = False,
        force_batch_index: bool = False) -> Tuple[Tensor, Tensor]:
    """Atomic pair index generator

    Generates all unique element-pair types & yields the corresponding indices
    of atoms pairs that match those types.

    Generates all unique element-pair types and, for each such pair, yields
    the indices of the atoms (and possibly cells and/or batches) that
    match that element-pair. The result is useful for enumerating all
    pairwise interactions present in a single system or a batch of systems,
    including possible consideration of periodic boundary conditions.

    Arguments:
        geometry: The system, or batch thereof, for which pair-wise
            interaction indices are to be generated.
        ignore_self: Boolean indicating whether self-interaction pairs should
            be ignored. If enabled, any pair where both indices refer to the
            same atom (i.e. an atom interacting with itself) is excluded.
            This only pertains to self-interactions and not interactions
            with copies in neighbouring images.[DEFAULT: False]
        force_batch_index: If ``True``, forces the returned ``pair_indices``
            to include a leading batch index dimension even for single systems.
            By default (``False``), a batch dimension is only included if
            ``geometry`` actually represents a batch. [DEFAULT: False]

    Yields:
        pair: A tensor of shape `(2,)` specifying the atomic numbers of the two
            species that define the pair. For example, `(6, 8)` for a carbon -
            oxygen pair.
        pair_indices: A tensor of integer indices that identify each occurrence
            of that ``pair`` in the system(s). The exact shape and meaning of
            each row depends on three factors:
            (1) whether the system is single or batched,
            (2) whether the system is periodic, and
            (3) whether ``force_batch_index`` is set:

            - **Single, non-periodic**:
              - `force_batch_index=False` ⇒ shape `(2, P)`
              - `force_batch_index=True`  ⇒ shape `(3, P)`
            - **Single, periodic**:
              - `force_batch_index=False` ⇒ shape `(3, P)`
              - `force_batch_index=True`  ⇒ shape `(4, P)`
            - **Batched, non-periodic** ⇒ shape `(3, P)`
            - **Batched, periodic** ⇒ shape `(4, P)`

            In all cases, the final two rows (or columns, after transpose)
            indicate the atomic indices of the interacting atoms. If there
            is a batch dimension, it occupies the first row. If the system
            is periodic, there is an additional row for the periodic cell
            index (or indices, depending on implementation). The remaining
            rows, if present, are the atomic indices within that batch and/or
            cell context.
    """
    if geometry.is_periodic:
        return _atomic_pair_indices_periodic(geometry, ignore_self, force_batch_index)
    else:
        return _atomic_pair_indices(geometry, ignore_self, force_batch_index)


def _atomic_pair_indices(
        geometry: Geometry, ignore_self: bool = False,
        force_batch_index: bool = False) -> Tuple[Tensor, Tensor]:
    """Atomic pair index generator for clusters.

    Generates all unique element-pair types & yields the corresponding indices
    of atoms pairs that match those types.

    Arguments:
        geometry: The system, or batch thereof, for which pair-wise
            interaction indices are to be generated.
        ignore_self: Boolean indicating whether self-interaction pairs should
            be ignored. If enabled, pairs involving an atom interacting with
            itself will be filtered out. Caution is advised when using this
            flag in conjunction with periodic systems. [DEFAULT: False]
        force_batch_index: `bool` indicating whether to force the presence of
            batch indices in the returned ``pair_indices`` tensor. By default,
            batch indices are only present when the system in question is a
            batch. However, it is sometimes useful to include a batch index
            for single systems to aid in writing batch agnostic code.
            [DEFAULT: False]

    Yields:
        pair: Atomic-number pair tensor of shape (2, ) specifying which
            species pair the indices correspond to.
        pair_indices: Atomic index array identifying the atoms associated with
            each interaction of the corresponding ``pair`` type. For a single
            system this will be of shape 2xP, which, for each pair "P",
            provides the atomic indices of the two atoms. For batches, this
            will be 3xP where the first dimension specifies which system in
            the batch the atoms belong to.
    """

    # Construct the atom resolved atomic number matrix
    shape = geometry.atomic_numbers.shape + geometry.atomic_numbers.shape[-1:]

    if force_batch_index and len(shape) == 2:
        shape = torch.Size([1]) + shape

    atomic_number_matrix = torch.stack((
        geometry.atomic_numbers.unsqueeze(-1).expand(shape),
        geometry.atomic_numbers.unsqueeze(-2).expand(shape)), dim=-1)

    # Loop over the set of unique atomic number pairs
    for pair in unique_atom_pairs(geometry):

        # Get the atomic indices of all such atom pairs.
        pair_indices = torch.nonzero((atomic_number_matrix == pair).all(-1))

        # Skip the iteration if no such atom pairs exist.
        if pair_indices.nelement() == 0:
            continue

        # The means by which pair indices are generated results in homo-atomic
        # interaction pairs indices being duplicated. Thus, a filtering step
        # is required for homo-atomic pairs.
        if pair[0] == pair[1]:

            # Also purge self-interactions if instructed to do so.
            if ignore_self:
                pair_indices = pair_indices[torch.where(
                    pair_indices[..., -2].lt(pair_indices[..., -1]))]

            else:
                pair_indices = pair_indices[torch.where(
                    pair_indices[..., -2].le(pair_indices[..., -1]))]

        yield pair, pair_indices.T


def _atomic_pair_indices_periodic(
        geometry: Geometry, ignore_self: bool = False,
        force_batch_index: bool = False) -> Tuple[Tensor, Tensor]:
    """Atomic pair index generator for periodic systems.

    Generates all unique element-pair types & yields the corresponding indices
    of atoms pairs that match those types.

    Arguments:
        geometry: The system, or batch thereof, for which pair-wise
            interaction indices are to be generated.
        ignore_self: Boolean indicating whether self-interaction pairs should
            be ignored. If enabled, pairs involving an atom interacting with
            itself will be filtered out. Caution is advised when using this
            flag in conjunction with periodic systems. [DEFAULT: False]
        force_batch_index: `bool` indicating whether to force the presence of
            batch indices in the returned ``pair_indices`` tensor. By default,
            batch indices are only present when the system in question is a
            batch. However, it is sometimes useful to include a batch index
            for single systems to aid in writing batch agnostic code.
            [DEFAULT: False]

    Yields:
        pair: Atomic-number pair tensor of shape (2, ) specifying which
            species pair the indices correspond to.
        pair_indices: Atomic index array identifying the atoms associated with
            each interaction of the corresponding ``pair`` type. For a single
            system this will be of shape 3xP, which, for each pair "P",
            provides the cell index and the atomic indices of the two atoms.
            For batches, this will be 4xP where the first dimension specifies
            which system in the batch the atoms belong to.
    """
    # Identify index of the origin cell(s). This is used later on to mask out
    # on-site interactions if needed
    origin_cell_idx = (geometry.periodicity.n_cells - 1) // 2

    # Non-periodic interaction pairs are duplicated so that there is one for
    # every possible periodic image. Although the number of images differs from
    # system to system, the largest of the batch is taken. This will create
    # many ghost interactions involving non-existent images for most systems
    # within a batch. However, these invalid interactions get filtered out by
    # the neighbour list used just before the yield.
    n_images = geometry.periodicity.n_cells.max()

    # Image index array used when duplicating interactions and adding in image
    # index.
    cell_idxs = torch.arange(n_images, device=geometry.device)

    # Is this a batched system, and should it be treated as a batch?
    is_batch = geometry.atomic_numbers.ndim == 2
    should_treat_as_batch = is_batch or force_batch_index

    # Neighbour list to identify which interactions are in range.
    neighbours = geometry.periodicity.neighbour

    # Expand the neighbours tensor if forcing the present of batch indices
    if force_batch_index:
        neighbours = neighbours.view(-1, *neighbours.shape[-3:])

    # Loop over the non-periodic indices and expand them to account for
    # periodicity. The `ignore_self` flag is not passed on otherwise indices
    # for atoms interacting with copies of themselves would be missing.
    for pair, idx in _atomic_pair_indices(geometry, False, force_batch_index):
        # Expand the pair indices to include interactions with all periodic cells
        idx_p = torch.vstack([
            cell_idxs.repeat(idx.shape[-1]),
            torch.repeat_interleave(idx, n_images, -1)])

        # Adjust the indices so the batch index comes at the start as expected
        if should_treat_as_batch:
            idx_p = idx_p[[1, 0, 2, 3]]

        # Remove on-site interactions if requested
        if ignore_self and pair[0] == pair[1]:
            # Expand origin cells indices for batches of systems
            origin = origin_cell_idx[idx_p[0]] if is_batch else origin_cell_idx

            # Filter out interactions within the origin cell between an atom
            # and itself.
            mask = idx_p[-3].ne(origin).logical_or(idx_p[-1].ne(idx_p[-2]))
            idx_p = idx_p.T[*torch.where(mask)].T

        # Filter the interaction list so that only those in range remain
        idx_p = idx_p.T[neighbours[*idx_p]].T

        yield pair, idx_p
