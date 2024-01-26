"""
Integral feeds for xTB.

This module implements integral feeds derived from `tbmalt`s `IntegralFeed`
class and the corresponding `matrix` method. The matrices are, however, not
calculated as in `tbmalt` but are just taken from `dxtb`. Hence, periodicity
boundary conditions are _not_ yet available for the xTB Hamiltonian.

The `dxtb` library is included as an external dependency in form of a (currently
private) git submodule. The submodule can be initialized with
 - `git submodule init`
 - `git submodule update`
"""

from typing import Optional, Union

import torch

from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml import Feed
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt.ml.module import require_args

try:
    from external.dxtb.origin.dxtb.ncoord import get_coordination_number, exp_count
    from external.dxtb.origin.dxtb.basis import IndexHelper
    from external.dxtb.origin.dxtb.integral import Overlap
    from external.dxtb.origin.dxtb.param import Param, get_elem_angular
    from external.dxtb.origin.dxtb.xtb import Hamiltonian
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        (
            "The module 'dxtb' is not yet publicly available. "
            "Access can be granted upon reasonable request."
        )
    ) from e


class XtbOccupationFeed(Feed):
    """
    Feed for reference occupation.

    A separate feed is required because the existing ones can only handle
    a minimal orbs. The extended tight-binding model, however, employs only
    a *mostly* minimal orbs.
    """

    def __init__(
        self,
        par: Param,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.par = par
        self.__device = device
        self.__dtype = dtype

    @property
    def device(self) -> Union[torch.device, None]:
        """The device on which the feed object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        # Instruct users to use the ".to" method if wanting to change device.
        name = self.__class__.__name__
        raise AttributeError(
            f"{name} object's device can only be modified " 'via the ".to" method.'
        )

    def to(self, device: torch.device) -> "XtbOccupationFeed":
        """
        Returns a copy of the `XtbOccupationFeed` instance on the specified
        device. Here, this would be equivalent to a setter as there are no
        tensors stored in the class.

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        XtbOccupationFeed
            A copy of the `XtbOccupationFeed` instance placed on the specified device.

        Notes
        -----
        If the `XtbOccupationFeed` instance is already on the desired device,
        `self` will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__(self.par, self.dtype, device)

    @property
    def dtype(self) -> Union[torch.dtype, None]:
        """Floating point dtype used by feed object."""
        return self.__dtype

    def __call__(self, orbs: OrbitalInfo) -> torch.Tensor:
        numbers = orbs.atomic_numbers

        # helper for mapping atomic, orbital and shell indices
        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(self.par.element))

        # second argument (positions) is only required for settings dtype/device
        return Hamiltonian(
            numbers,
            self.par,
            ihelp,
            device=self.device,
            dtype=self.dtype,
        ).get_occupation()


class Gfn1OccupationFeed(XtbOccupationFeed):
    """
    Occupation feed for the GFN1-xTB reference occupation.
    """

    pass


class Gfn2OccupationFeed(XtbOccupationFeed):
    """
    Occupation feed for the GFN2-xTB reference occupation.
    """

    def __init__(self, *args):
        raise NotImplementedError("GFN2 not yet implemented.")


class XtbOverlapFeed(IntegralFeed):
    """
    Integral feed for the xTB overlap.
    """

    def __init__(
        self,
        par: Param,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dtype, device)
        self.par = par

    @require_args("geometry")
    def matrix(self, geometry: Geometry):
        """
        Construct the overlap matrix.
        The orbs is constructed within the dxtb's `Hamiltonian` class.

        Parameters
        ----------
        geometry : Geometry
            Systems whose matrices are to be constructed.

        Returns
        -------
        torch.Tensor
            The overlap matrices.
        """

        numbers = geometry.atomic_numbers
        positions = geometry.positions

        # helper for mapping atomic, orbital and shell indices
        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(self.par.element))

        # constructing overlap
        s = Overlap(
            numbers,
            self.par,
            ihelp,
            device=self.device,
            dtype=self.dtype,
        )
        return s.build(positions)


class XtbHamiltonianFeed(IntegralFeed):
    """
    Integral feed for the xTB Hamiltonian.
    """

    def __init__(
        self,
        par: Param,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dtype, device)
        self.par = par

    @require_args("geometry", "overlap")
    def matrix(self, geometry: Geometry, overlap: torch.Tensor) -> torch.Tensor:
        """
        Construct the Hamiltonian matrix.
        The orbs is constructed within the dxtb's `Hamiltonian` class.

        Parameters
        ----------
        geometry : Geometry
            Systems whose matrices are to be constructed.

        Returns
        -------
        torch.Tensor
            The Hamiltonian matrices.
        """
        numbers = geometry.atomic_numbers
        positions = geometry.positions

        # calculate coordintation number
        cn = get_coordination_number(numbers, positions, exp_count)

        # helper for mapping atomic, orbital and shell indices
        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(self.par.element))

        # constructing hamiltonian and overlap
        h0 = Hamiltonian(
            numbers,
            self.par,
            ihelp,
            device=self.device,
            dtype=self.dtype,
        )
        return h0.build(positions, overlap, cn=cn)


class Gfn1OverlapFeed(XtbOverlapFeed):
    """
    Integral feed for the GFN1-xTB overlap.
    """

    pass


class Gfn2OverlapFeed(XtbOverlapFeed):
    """
    Integral feed for the GFN2-xTB overlap.
    """

    def __init__(self, *args):
        raise NotImplementedError("GFN2 not yet implemented.")


class Gfn1HamiltonianFeed(XtbHamiltonianFeed):
    """
    Integral feed for the GFN1-xTB Hamiltonian.
    """

    pass


class Gfn2HamiltonianFeed(XtbHamiltonianFeed):
    """
    Integral feed for the GFN2-xTB Hamiltonian.
    """

    def __init__(self, *args):
        raise NotImplementedError("GFN2 not yet implemented.")
