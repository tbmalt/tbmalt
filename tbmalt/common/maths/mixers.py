# -*- coding: utf-8 -*-
"""A selection of mixing algorithms for aiding and accelerating convergence.

This module contains a selection of different mixing algorithms. These methods
can be used to accelerate and/or stabilise the convergence of fixed-point
iterations. Within the context of TBMaLT, these are intended to help converge
the self-consistent-field/charge (SCF/SCC) cycles.

In the majority of cases it is fundamentally impossible to identify from the
input alone whether the user is attempting to mix a single system or a batch
of such systems. This an ambiguity renders the shape-agnostic & batch-agnostic
programming paradigms incompatible with one another. As such one paradigm must
be abandoned in favor of the other. Thus, all mixing instances require the
user to explicitly state whether they are operating on a single system or on a
batch of systems.
"""
from abc import ABC, abstractmethod
from numbers import Real
from typing import Union, Optional
import warnings
import torch
from torch import Tensor


class _Mixer(ABC):
    """This is the abstract base class upon which all mixers are to be based.

    This abstract base class provides the template on which all mixers are to
    be built.

    Arguments:
        is_batch: Set to true when mixing a batch of systems, False when not.
        mix_param: Mixing parameter, ∈(0, 1), controls the extent of mixing.
            Larger values result in more aggressive mixing. [DEFAULT=0.05]
        tolerance: Max permitted difference in between successive iterations
            for a system to be considered "converged". [DEFAULT=1E-6]

    """
    def __init__(self, is_batch: bool, mix_param: Real = 0.05, tolerance: Real = 1E-6):

        self.mix_param = mix_param

        self.tolerance = tolerance

        self._is_batch = is_batch

        # Integer tracking the number of mixing iterations performed thus far.
        self._step_number: int = 0

        # Difference between the current & previous systems
        self._delta: Optional[Tensor] = None

    def _setup_hook(self, dtype: torch.dtype):
        """Conducts any required initialisation operations.

        Any setup code that needs only to be once, during the first call to
        "`__call__`" is placed here. This abstraction helps to clean up the
        `__call__` function by removing any code that is not needed for the
        mixing operation. By default this code will ensure that the requested
        tolerance limit is within the precision of the dtype being used. If
        the tolerance is too tight, a warning is issued and the tolerance is
        downgraded.

        Arguments:
            dtype: The dtype being used.

        Notes:
            It is expected that new mixer classes will locally override this
            function. *HOWEVER*, all new mixers will need to execute the code
            present in this function. To prevent needless code repetition all
            local ``_setup_hook`` functions should make a call to the parent
            version via `super()._setup_hook(dtype)'.

        """
        # Check the specified tolerance can be achieved
        if torch.finfo(dtype).resolution > self.tolerance:
            warnings.warn(  # If not issue a warning
                f'{self.__class__.__name__}: Tolerance value {self.tolerance}'
                f' cannot be achieved by a {dtype}. DOWNGRADING!',
                UserWarning, stacklevel=2)
            # And downgrade to a tolerance just below the precision limit
            self.tolerance = torch.finfo(dtype).resolution * 5.0

    @abstractmethod
    def __call__(self, x_new: Tensor, x_old: Optional[Tensor] = None
                 ) -> Tensor:
        """Performs the mixing operation & returns the newly mixed system.

        Notes:
            This should contain only the code required to carry out the mixing
            operation.

        """
        pass

    def __repr__(self) -> str:
        """Returns representative string."""
        return f'{self.__class__.__name__}({self._step_number})'

    @property
    def step_number(self) -> int:
        """Integer tracking the number of mixing iterations performed."""
        return self._step_number

    @property
    def converged(self) -> Union[Tensor, bool]:
        """Tensor of bools indicating convergence status of the system(s).

        A system is considered to have converged if the maximum absolute
        difference between the current and previous systems is less than
        the ``tolerance`` value.

        """
        # The return type is set as Union[Tensor, bool] to aid type-checking.
        # Check that mixing has been conducted
        assert self._delta is not None, 'Nothing has been mixed'

        if not self._is_batch:  # If not in batch mode
            return self._delta.abs().max() < self.tolerance
        else:  # If operating in batch mode
            if self._delta.dim() == 1:
                # Catch needed when operating on a batch of scalars.
                return self._delta.abs() < self.tolerance
            else:
                return self._delta.flatten(-self._delta.dim() + 1
                                           ).abs().max(-1)[0] < self.tolerance

    @property
    def delta(self) -> Tensor:
        """Difference between the current & previous systems"""
        # This may need to be locally overridden if "_delta" needs to be
        # reshaped prior to it being returned.
        return self._delta

    @abstractmethod
    def reset(self):
        """Resets the mixer to its initial state."""
        # This should reset all attributes like, _step_count, _delta, etc.
        # Effectively placing the mixer instance into the state it was in
        # prior to the first mixing operation.
        pass

    @abstractmethod
    def cull(self, cull_list: Tensor):
        """Purge select systems form the mixer.

        This is useful when a subset of systems have converged during mixing.

        Arguments:
            cull_list: Tensor with booleans indicating which systems should be
                culled (True) and which should remain (False).

        """
        # This code should carry out all the operations necessary to remove a
        # system from the mixer. All mixing operations after this point should
        # mix only the remaining systems. This function should check that
        # "is_batch" is True first; as it makes no sense to attempt a cull
        # when not in batch mode.
        pass


class Simple(_Mixer):
    r"""Simple linear mixer mixing algorithm.

    Iteratively mixes pairs of systems via a simple linear combination:

    .. math::

        (1-f)x_n + fx_{n-1}

    Where :math:`x_n`/:math:`x_{n-1}` are the new/old systems respectively &
    :math:`f` is the mixing parameter, i.e. the fraction of :math:`x_n` that
    is to be retained. Given a suitable tolerance, and a small enough mixing
    parameter, the `Simple` mixer is guaranteed to converge, however, it also
    tends to be significantly slower than other, more advanced, methods.

    Arguments:
        is_batch: Set to true when mixing a batch of systems, False when not.
        mix_param: Mixing parameter, ∈(0, 1), controls the extent of mixing.
            Larger values result in more aggressive mixing. [DEFAULT=0.05]
        tolerance: Maximum permitted difference in charge between successive
            iterations for a system to be considered "converged".

    Notes:
        Mixer instances require the user to explicitly specify during
        initialisation whether it is a single system or a batch of systems
        that are to be mixed.

    Examples:
        More in-depth examples of mixing algorithms can be found in the
        tutorial section of the documentation.

        The attractive fixed point of the function:

        >>> from torch import tensor, sqrt
        >>> def func(x):
        >>>     return tensor([0.5 * sqrt(x[0] + x[1]),
        >>>                    1.5 * x[0] + 0.5 * x[1]])

        can be idenfied using the ``Simple`` mixer as follows:

        >>> from tbmalt.common.maths.mixers import Simple
        >>> x = tensor([2., 2.])  # Initial guess
        >>> mixer = Simple(False, tolerance=1E-4, mix_param=0.8)
        >>> for i in range(100):
        >>>     x = mixer(func(x), x)
        >>> print(x)
        tensor([1., 3.])
    """

    def __init__(self, is_batch: bool, mix_param: Real = 0.05, tolerance: Real = 1E-6):
        # Pass required inputs to parent class.
        super().__init__(is_batch, mix_param, tolerance)

        # Holds the system from the previous iteration.
        self._x_old: Union[Tensor, None] = None

    def __call__(self, x_new: Tensor, x_old: Optional[Tensor] = None
                 ) -> Tensor:
        """Performs the simple mixing operation & returns the mixed system.

        Iteratively mix pairs of systems via a simple linear combination:

        .. math::

            (1-f)x_n + fx_{n-1}

        Arguments:
            x_new: New system(s) that is to be mixed.
            x_old: Previous system(s) that is to be mixed. Only required for
                the first mix operation.

        Returns:
            x_mix: Newly mixed system(s).

        Notes:
            The ``x_old`` argument is normally identical to the ``x_mix``
            value returned from the previous iteration, which is stored by the
            class internally. As such, the ``x_old`` argument can be omitted
            from all but the first step if desired.

        """
        # If this is the first mixing step, make a call to the setup function
        if self._step_number == 0:
            self._setup_hook(x_new.dtype)

        # Increment the step number variable
        self._step_number += 1

        # Use the previous x_old value if none was specified
        x_old = self._x_old if x_old is None else x_old
        x_new = x_new

        # Check all tensor dimensions match
        assert x_old.shape == x_new.shape,\
            'new & old systems must have matching shapes.'

        # Perform the mixing operation to create the new mixed x value
        x_mix = x_old + (x_new - x_old) * self.mix_param

        # Update the _x_old attribute
        self._x_old = x_mix

        # Update the delta
        self._delta = (x_mix - x_old)

        # Return the newly mixed system
        return x_mix

    def reset(self):
        """Resets the mixer to its initial state.

        Calling this function will rest the class & its internal attributes.
        However, any properties set during the initialisation process, e.g.
        ``mix_param``, ``threshold``, etc. will be retained.

        """
        # Reset the step_number, delta, and x_old parameters
        self._step_number = 0
        self._delta = self._x_old = None

    def cull(self, cull_list: Tensor):
        """Purge select systems form the mixer.

        This is useful when a subset of systems have converged during mixing.

        Arguments:
            cull_list: Tensor with booleans indicating which systems should be
                culled (True) and which should remain (False).

        """
        assert self._is_batch, 'Cull only valid for batch mixing'
        # Invert cull_list, gather & reassign x_old and _delta so only those
        # marked False remain.
        cull = ~cull_list
        self._x_old, self._delta = self._x_old[cull], self._delta[cull]
