# -*- coding: utf-8 -*-
"""A container to hold data associated with a chemical system's structure.

This module provides the `Geometry` data structure class and its associated
code. The `Geometry` class is intended to hold any & all data needed to fully
describe a chemical system's structure.
"""
from typing import Union, List, Optional, Type
from operator import itemgetter
import torch
import numpy as np
from h5py import Group
from ase import Atoms
from ase.lattice.bravais import Lattice
from tbmalt.structures.periodicity import Triclinic
from tbmalt.common.batch import pack, merge, deflate
from tbmalt.data.units import length_units
from tbmalt.data import chemical_symbols
Tensor = torch.Tensor


# Todo:
#   - The `clone` and `detach` methods do not support periodic systems.
#   - Currently the cutoff value is being stored in two places, the `Geometry`
#     class and the `Periodicity` class. Furthermore, it is being modified
#     within the latter. There should be a single centralised place in which
#     this value is stored.
#   - Currently periodic systems are hard-coded to use the `Triclinic` periodic
#     helper class. Note that triclinic is used in reference to the general
#     geometric shape and not the specific crystal structure. This is done
#     because only one type of periodic boundary condition is supported at this
#     time. Later this should be generalised.



class Geometry:
    """Data structure for storing geometric information on molecular systems.

    The `Geometry` class stores any information that is needed to describe a
    chemical system; atomic numbers, positions, etc. This class also permits
    batch system representation. However, mixing of PBC & non-PBC systems is
    strictly forbidden.

    Arguments:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Coordinates of the atoms.
        lattice_vector: Lattice vectors of the periodicity systems. [DEFAULT: None]
        frac: Whether using fractional coordinates to describe periodicity
            systems. [DEFAULT: False]
        cutoff: Global cutoff for the diatomic interactions in periodicity
            systems. [DEFAULT: 9.98].
        units: Unit in which ``positions``, ``cells``, and ``cutoff`` were
            specified. For a list of available units see :mod:`.units`
            [DEFAULT='bohr'].

    Attributes:
        atomic_numbers: Atomic numbers of the atoms.
        positions: Coordinates of the atoms.
        n_atoms: Number of atoms in the system.
        periodicity: Periodicity object that offer helper methods for periodic
            systems.
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

        >>> from tbmalt import Geometry
        >>> H2 = Geometry(torch.tensor([1, 1]),
        >>>               torch.tensor([[0.00, 0.00, 0.00],
        >>>                             [0.00, 0.00, 0.79]]))
        >>> print(H2)
        Geometry(H2)

        Or from an ase.Atoms object

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
                 'lattice', '_cutoff', '_n_batch',
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
            units: str = 'bohr'):

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
            # Todo: Abstract cutoff out of the Geometry class
            # Cutoff distance for the diatomic interactions in periodicity systems
            self._cutoff = torch.tensor(
                [9.98], device=self.__device, dtype=self.__dtype
            ) if cutoff is None else cutoff

            self.lattice = lattice_vector
            self.periodicity = Triclinic(self, self._cutoff)

        else:
            # Todo: Abstract cutoff out of the Geometry class
            self._cutoff = None
            self.periodicity = None
            self.lattice = None
        # Periodicity system specific components

    @staticmethod
    def __preprocess(
            atomic_numbers: Union[Tensor, List[Tensor]],
            positions: Union[Tensor, List[Tensor]],
            lattice_vector: Optional[Union[Tensor, List[Tensor], Type[Lattice]]],
            cutoff: Optional[Union[Tensor, float]],
            frac: bool, units: str):
        """
        This method just abstracts a lot of overly verbose and messy safety
        checks and conversions that are performed on the inputs. This is done
        to make the `__init__` method a little cleaner.
        """

        # Preprocessing:
        # 1) Ensure that `atomic_numbers` and `positions` are packed as needed,
        #    `pack` only effects lists of tensors, and deflate is used to
        #    remove any unnecessary padding.
        atomic_numbers = deflate(pack(atomic_numbers))
        positions = pack(positions)[..., :atomic_numbers.shape[-1], :]

        # 2) If the lattice vectors are supplied as ase.lattice.bravais.Lattice
        #    instances then the lattice vector arrays must be extracted from
        #    them. Then make sure that the lattice vectors are pytorch arrays
        #    and are packed if required. Note that care must be taken here not
        #    to accidentally cause the lattice vectors to be deflated. A row
        #    of all zeros is valid in a lattice vector and thus should not be
        #    pruned.
        if isinstance(lattice_vector, Lattice):
            lattice_vector = lattice_vector.cell.array

        elif (isinstance(lattice_vector, list)
              and isinstance(lattice_vector[0], Lattice)):
            lattice_vector = [i.cell.array for i in lattice_vector]

        if isinstance(lattice_vector, list):
            lattice_vector = pack(lattice_vector).to(positions.device)
        elif isinstance(lattice_vector, np.ndarray):
            lattice_vector = torch.tensor(lattice_vector).to(positions.device)

        # 3) Ensure tensors are on the same device (only two present currently)
        if positions.device != atomic_numbers.device:
            raise RuntimeError('All tensors must be on the same device!')

        # 4) Lattice vectors many not be zero dimensional
        if lattice_vector is not None and (
                ~lattice_vector.ne(0).any(-1).any(-1)).any():
            raise ValueError('Lattice vectors may not be zero dimensional!')

        # 5) Convert fractional positions to their Cartesian values
        if frac and lattice_vector is not None:
            # Todo: Remove hard-coding to specific periodicity
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
        # Todo: Modify to account for PBC
        
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
        # Todo: Modify to account for PBC
        dist_vec = self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3)
        dist_vec[self._mask_dist] = 0
        return dist_vec

    @property
    def chemical_symbols(self) -> list:
        """Chemical symbols of the atoms present."""
        return batch_chemical_symbols(self.atomic_numbers)

    @property
    def pbc(self) -> Union[bool, Tensor]:
        """Directions along which the system is deemed to be periodicity."""
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
                return self.__class__(self.atomic_numbers.to(device=device),
                                      self.positions.to(device=device),
                                      self.lattice.to(device=device))

    def __getitem__(self, selector) -> 'Geometry':
        """Permits batched Geometry instances to be sliced as needed."""
        # Block this if the instance has only a single system
        if self.atomic_numbers.ndim != 2:
            raise IndexError(
                'Geometry slicing is only applicable to batches of systems.')

        # Select the desired atomic numbers, positions and cells. Making sure
        # to remove any unnecessary padding.
        new_zs = deflate(self.atomic_numbers[selector, ...])
        new_pos = self.positions[selector, ...][..., :new_zs.shape[-1], :]
        new_lattice = self.lattice[selector, ...] \
            if self.lattice is not None else self.lattice

        return self.__class__(
            new_zs, new_pos, new_lattice, cutoff=self._cutoff)

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

        cutoff_1 = self._cutoff
        cutoff_2 = other._cutoff

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

            return self.__class__(
                merge([an_1, an_2]), merge([pos_1, pos_2]),
                merge([cell_1, cell_2]), cutoff=torch.max(cutoff_1, cutoff_2))
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
            raise NotImplementedError(
                "This operation does not support periodic systems")

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
            raise NotImplementedError(
                "This operation does not support periodic systems")


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


def unique_atom_pairs(geometry: Geometry) -> Tensor:
    """Returns a tensor specifying all unique atom pairs.

    This takes `Geometry` instance and identifies all atom pairs. This use
    useful for identifying all possible two body interactions possible within
    a given system.

    Arguments:
         geometry: `Geometry` instance representing the target system.

    Returns:
        unique_atom_pairs: A tensor specifying all unique atom pairs.
    """
    uan = geometry.unique_atomic_numbers()
    n_global = len(uan)
    return torch.stack([uan.repeat(n_global),
                        uan.repeat_interleave(n_global)]).T

