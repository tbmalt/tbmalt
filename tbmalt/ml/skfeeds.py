# -*- coding: utf-8 -*-
"""Slater-Koster integral feed objects.

This contains all Slater-Koster integral feed objects. These objects are
responsible for generating the Slater-Koster integrals used in constructing
Hamiltonian and overlap matrices. The on-site and off-site terms are yielded
by the `on_site` and `off_site` class methods respectively.
"""

from typing import Union, Tuple
from abc import ABC, abstractmethod
from inspect import getfullargspec
from warnings import warn
from h5py import Group
import torch
from torch import Tensor


class _SkFeed(ABC):
    """ABC for objects responsible for supplying Slater-Koster integrals.

    Subclasses of the this abstract base class are responsible for supplying
    the Slater-Koster integrals needed to construct the Hamiltonian & overlap
    matrices.

    Arguments:
        device: Device on which the `SkFeed` object and its contents resides.
        dtype: Floating point dtype used by `SkFeed` object.

    Developers Notes:
        This class provides a common fabric upon which all Slater-Koster
        integral feed objects are built. As the `_SkFeed` class is in its
        infancy it is subject to change; e.g. the addition of an `update`
        method which allows relevant model variables to be updated via a
        single call during backpropagation.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        # These are static, private variables and must NEVER be modified!
        self.__device = device
        self.__dtype = dtype

    def __init_subclass__(cls, check_sig: bool = True):
        """Check the signature of subclasses' methods.

        Issues non-fatal warnings if invalid signatures are detected in
        subclasses' `off_site` or `on_site` methods. Both methods must accept
        an arbitrary number of keyword arguments, i.e. `**kwargs`. The
        `off_site` & `on_site` method must take the keyword arguments
        (atom_pair, shell_pair, distances) and (atomic_numbers) respectively.

        This behaviour is enforced to maintain consistency between the various
        subclasses of `_SkFeed`'; which is necessary as the various subclasses
        will likely differ significantly from one another & may become quite
        complex.

        Arguments:
            check_sig: Signature check not performed if ``check_sig = False``.
                This offers a way to override these warnings if needed.
        """

        def check(func, has_args):
            sig = getfullargspec(func)
            name = func.__qualname__
            if check_sig:  # This check can be skipped
                missing = ', '.join(has_args - set(sig.args))
                if len(missing) != 0:
                    warn(f'Signature Warning: keyword argument(s) "{missing}"'
                         f' missing from method "{name}"',
                         stacklevel=4)

            if sig.varkw is None:  # This check cannot be skipped
                warn(f'Signature Warning: method "{name}" must accept an '
                     f'arbitrary keyword arguments, i.e. **kwargs.',
                     stacklevel=4)

        check(cls.off_site, {'atom_pair', 'shell_pair', 'distances'})
        check(cls.on_site, {'atomic_numbers'})

    @property
    def device(self) -> torch.device:
        """The device on which the geometry object resides."""
        return self.__device

    @device.setter
    def device(self, value):
        # Instruct users to use the ".to" method if wanting to change device.
        name = self.__class__.__name__
        raise AttributeError(f'{name} object\'s dtype can only be modified '
                             'via the ".to" method.')

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by geometry object."""
        return self.__dtype

    @abstractmethod
    def off_site(self, atom_pair: Tensor, shell_pair: Tensor,
                 distances: Tensor, **kwargs) -> Tensor:
        """Evaluate the selected off-site Slater-Koster integrals.

        This evaluates & returns the off-site Slater-Koster integrals between
        orbitals `l_pair` on atoms `atom_pair` at the distances specified by
        `distances`. Note that only one `atom_pair` & `shell_pair` can be
        evaluated at at time. The dimensionality of the the returned tensor
        depends on the number of distances evaluated & the number of bonding
        integrals associated with the interaction.

        Arguments:
            atom_pair: Atomic numbers of the associated atoms.
            shell_pair: Shell numbers associated with the interaction.
            distances: Distances between the atoms pairs.

        Keyword Arguments:
            atom_indices: Tensor: The indices of the atoms associated with the
                 ``distances`` specified. This is automatically passed in by
                 the Slater-Koster transformation code.

        Return:
            integrals: Off-site integrals between orbitals ``shell_pair`` on
                atoms ``atom_pair`` at the specified distances.

        Developers Notes:
            The Slater-Koster transformation passes "atom_pair", "shell_pair",
            & "distances" as keyword arguments. This avoids having to change
            the Slater-Koster transformation code every time a new feed is
            created. These four arguments were made default as they will be
            required by most Slater-Koster feed implementations. A warning
            will be issued if a `_SkFeed` subclass is found to be missing any
            of these arguments. However, this behaviour can be suppressed by
            adding the class argument `check_sig=False`.

            It is imperative that this method accepts an arbitrary number of
            keyword arguments, i.e. has a `**kwarg` argument. This allows for
            additional data to be passed in. By default the Slater-Koster
            transformation code will add the keyword argument "atom_indices".
            This specifies the indices of the atoms involved, which is useful
            if the feed takes into account environmental dependency.

            Any number of additional arguments can be added to this method.
            However, to get the Slater-Koster transform code to pass this
            information through one must pass the requisite data as keyword
            arguments to the Slater-Koster transform function itself. As it
            will pass through any keyword arguments it encounters.

        """
        pass

    @abstractmethod
    def on_site(self, atomic_numbers: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        """Returns the specified on-site terms.

        Arguments:
            atomic_numbers: Atomic numbers for which on-site terms should be
                returned.

        Keyword Arguments:
            atom_indices: Tensor: The indices of the atoms associated with the
                 ``distances`` specified. This is automatically passed in by
                 the Slater-Koster transformation code.

        Returns:
            on_sites: Tuple of on-site term tensors, one for each atom in
                ``atomic_numbers``.

        Developers Notes:
            See the documentation for the _SkFeed.off_site method for
            more information.

        """
        pass

    @abstractmethod
    def to(self, device: torch.device) -> 'SkFeed':
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
        pass

    @classmethod
    def load(cls, source: Union[str, Group]) -> 'SkFeed':
        """Load a stored Slater Koster integral feed object.

        This is only for loading preexisting Slater-Koster feed objects, from
        HDF5 databases, not instantiating new ones.

        Arguments:
            source: Name of a file to load the integral feed from or an HDF5
                group from which it can be extracted.

        Returns:
            ski_feed: A Slater Koster integral feed object.

        """
        raise NotImplementedError()

    def save(self, target: Union[str, Group]):
        """Save the Slater Koster integral feed to an HDF5 database.

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


# Type alias to improve PEP484 readability
SkFeed = _SkFeed
