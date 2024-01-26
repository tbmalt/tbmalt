
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from functools import wraps
import torch

from tbmalt import Geometry, OrbitalInfo




class Calculator(ABC):

    def __init__(self, dtype: torch.dtype, device: torch.device,
                 mass: Optional[Dict[int, float]] = None):

        self.__dtype = dtype
        self.__device = device

        self._geometry = None
        self._orbs = None
        self._ml_params = None

    @property
    def device(self) -> torch.device:
        """The device on which the feed object resides."""
        return self.__device

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by feed object."""
        return self.__dtype

    @abstractmethod
    def forward(self, cache: Optional[Dict[str, Any]] = None):
        pass

    def __call__(self, geometry: Geometry, orbs: OrbitalInfo,
                 cache: Optional[Dict[str, Any]] = None, **kwargs):
        """Run the calculator instance.

        Arguments:
            geometry: System(s) upon which the calculation is to be run.
            orbs: Orbital information associated with said system(s).
            cache: A cache entity that may be used to bootstrap the calculation.

        Returns:
            total_energy: total energy of the target system(s). If repulsive
                interactions are not considered, i.e. the repulsion feed is
                omitted, then this will be the band structure energy. If finite
                temperature is active then this will technically be the Mermin
                energy.

        """
        # If a new geometry & orbs object have been provided then it's assumed
        # that the user wants to use the calculator on a new system. Otherwise
        # assume the user wishes to rerun the calculator on the current system.
        # The latter is useful if rerunning a calculation after updating one or
        # more of the feed objects.

        g_spec, b_spec = geometry is not None, orbs is not None

        # Either geometry and orbs are passed or neither are. It is not logical
        # to provide only one.
        if all([g_spec, b_spec]) != any([g_spec, b_spec]):
            raise ValueError(
                '"geometry" & "orbs" are mutually inclusive arguments; either '
                'both must be provided or neither.')

        # If no geometry/orbs objects were provide and none are currently set
        # then no calculation can be run.
        if not g_spec and self.geometry is None:
            raise AttributeError(
                '"geometry" & "orbs" objects must be specified before a '
                'calculation can run')

        # Reset the calculator, update the `geometry` & `orbs` attributes, if
        # required, then call the `forward` method.
        # reinitialise, and then call the `forward` method.
        if not g_spec:
            self.reset()
        else:
            self.reset()
            self._geometry, self._orbs = geometry, orbs

        return self.forward(cache=cache, **kwargs)

    @property
    def is_batch(self):
        """True of operating on a batch of systems."""
        return self.geometry.positions.dim() == 3

    @property
    def is_periodic(self):
        """If there is any periodicity boundary conditions."""
        return self.geometry.is_periodic

    @property
    def geometry(self):
        return self._geometry

    @property
    def orbs(self):
        return self._orbs

    @abstractmethod
    def reset(self):
        pass

    def cache(self) -> Dict[str, Any]:
        """
        This returns a cache that may be fed into the next call to bootstrap the
        startup. This is only meaningful when restarting a calculation on a
        system after the calculator has been reset.
        """
        pass
