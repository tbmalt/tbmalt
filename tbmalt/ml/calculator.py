
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
from torch import Tensor
from torch.nn import Module

from tbmalt import Geometry, OrbitalInfo


class Calculator(Module, ABC):

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.__dtype = dtype
        self.__device = device

        self._geometry = None
        self._orbs = None

    @property
    def device(self) -> torch.device:
        """The device on which the feed object resides."""
        return self.__device

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by feed object."""
        return self.__dtype

    @abstractmethod
    def forward(
            self, geometry: Geometry, orbs: OrbitalInfo,
            cache: Optional[Dict[str, Any]] = None) -> Tensor:
        raise NotImplementedError("Abstract method has not been implemented")

    @property
    def is_batch(self):
        """True of operating on a batch of systems."""
        return self.geometry.positions.dim() == 3

    @property
    def is_periodic(self):
        """If there are any periodic boundary conditions."""
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
