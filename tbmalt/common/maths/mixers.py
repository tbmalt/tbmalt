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
from typing import Union, Optional
from numbers import Real
from abc import ABC, abstractmethod
from functools import wraps
import warnings
import torch
from torch import Tensor


class _Mixer(ABC):
    """This is the abstract base class upon which all mixers are to be based.

    This abstract base class provides the template on which all mixers are to
    be built.

    Arguments:
        is_batch: Set to true when mixing a batch of systems, False when not.
        tolerance: Max permitted difference in between successive iterations
            for a system to be considered "converged". [DEFAULT=1E-6]

    """
    def __init__(self, is_batch: bool, tolerance: Real = 1E-6):
        self.tolerance = tolerance

        self._is_batch = is_batch

        # Integer tracking the number of mixing iterations performed thus far.
        self._step_number: int = 0

        # Difference between the current & previous systems
        self._delta: Optional[Tensor] = None

    def __init_subclass__(cls):
        def _tolerance_threshold_check(func):
            """Wrapper to check the validity of the current tolerance value.

            This wraps `__call__` to ensure that a tolerance threshold check
            gets carried out on the first call, i.e. when `_step_number` is
            zero. If the specified tolerance cannot be achieved then a warning
            will be issued and the tolerance will be downgraded to just below
            the precision limit.
            """
            @wraps(func)
            def wrapper(*args, **kwargs):
                self = args[0]
                # pylint: disable=W0212
                if self._step_number == 0:
                    dtype = args[1].dtype
                    if torch.finfo(dtype).resolution > self.tolerance:
                        warnings.warn(
                            f'{cls.__name__}: Tolerance value '
                            f"{self.tolerance} can't be achieved by a {dtype}"
                            '. DOWNGRADING!', UserWarning, stacklevel=2)
                        self.tolerance = torch.finfo(dtype).resolution * 5.0
                return func(*args, **kwargs)
            return wrapper

        cls.__call__ = _tolerance_threshold_check(cls.__call__)

    @abstractmethod
    def _setup_hook(self):
        """Conducts any required initialisation operations.

        Any setup code that needs only to be once, during the first call to
        "`__call__`" is placed here. This abstraction helps to clean up the
        `__call__` function by removing any code that is not needed for the
        mixing operation.

        Notes:
            It is expected that new mixer classes will locally override this
            function.
        """
        pass

    @abstractmethod
    def __call__(self, x_new: Tensor, x_old: Optional[Tensor] = None
                 ) -> Tensor:
        """Performs the mixing operation & returns the newly mixed system.

        This should contain only the code required to carry out the mixing
        operation.

        Notes:
            In all implementations the `x_new` argument **MUST** be the first
            non-self argument, x_old should be an optional keyword argument.
            The first call to this function, i.e. when `self._step_number` is
            zero, should make a call to `_setup_hook`, if applicable.
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

        # >>> from torch import tensor, sqrt
        # >>> def func(x):
        # >>>     return tensor([0.5 * sqrt(x[0] + x[1]),
        # >>>                    1.5 * x[0] + 0.5 * x[1]])
        #
        # can be idenfied using the ``Simple`` mixer as follows:
        #
        # >>> from tbmalt.common.maths.mixers import Simple
        # >>> x = tensor([2., 2.])  # Initial guess
        # >>> mixer = Simple(False, tolerance=1E-4, mix_param=0.8)
        # >>> for i in range(100):
        # >>>     x = mixer(func(x), x)
        # >>> print(x)
        tensor([1., 3.])
    """

    def __init__(self, is_batch: bool, mix_param: Real = 0.05,
                 tolerance: Real = 1E-6):
        # Pass required inputs to parent class.
        super().__init__(is_batch, tolerance)

        self.mix_param = mix_param

        # Holds the system from the previous iteration.
        self._x_old: Optional[Tensor] = None

    def _setup_hook(self):
        """NullOp to satisfy abstractmethod requirement so parent class.

        The simple mixer is unique in that it does not require any setup. Thus
        an empty function has been created to satisfy the requirements of the
        parent class.
        """
        pass

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
        # Increment the step number variable
        self._step_number += 1

        # Use the previous x_old value if none was specified
        x_old = self._x_old if x_old is None else x_old

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
        self._x_old = self._x_old[cull]
        self._delta = self._delta[cull]


class Anderson(_Mixer):
    """Accelerated Anderson mixing algorithm.

    Anderson acceleration, also known as Pulay mixing & DIIS, is a method for
    accelerating convergence. Upon instantiation a callable instance will be
    returned. Calls to this instance will take, as its arguments, two input
    systems and will return a single mixed system.

    Arguments:
        is_batch: Set to true when mixing a batch of systems, False when not.
        mix_param: Mixing parameter, ∈(0, 1), controls the extent of mixing.
            Larger values result in more aggressive mixing. [DEFAULT=0.05]
        tolerance: Maximum permitted difference in charge between successive
            iterations for a system to be considered "converged".
        generations: Number of generations to use during mixing. [DEFAULT=4]
        diagonal_offset: Offset added to the equation system's diagonal's
            to prevent a linear dependence during the mixing process. If set
            to `None` then rescaling will be disabled. [DEFAULT=0.01]
        init_mix_param: Mixing parameter to use during the initial simple
            mixing steps. [DEFAULT=0.01]

    Notes:
        Note that simple mixing will be used for the first ``generations``
        number of steps

        The Anderson mixing functions primarily follow the equations set out
        by Eyert [Eyert]_. However, this code borrows heavily from the DFTB+
        implementation [DFTB]_. This deviates from the DFTB+ implementation
        in that it does not compute or use the theta zero values, as they
        cause stability issues in this implementation. For more information on
        Anderson mixing see See "Anderson Acceleration, Mixing and
        Extrapolation" [Anderson]_.

    Warnings:
        Setting ``generations`` too high can lead to a linearly dependent set
        of equations. However, this effect can be mitigated through the use of
        the ``diagonal_offset`` parameter.

    References:
        .. [Eyert] Eyert, V. (1996). A Comparative Study on Methods for
           Convergence Acceleration of Iterative Vector Sequences. Journal of
           Computational Physics, 124(2), 271–285.
        .. [DFTB] Hourahine, B., Aradi, B., Blum, V., Frauenheim, T. et al.,
           (2020). DFTB+, a software package for efficient approximate density
           functional theory based atomistic simulations. The Journal of
           Chemical Physics, 152(12), 124101.
        .. [Anderson] Anderson, D. M. (2018). Comments on “Anderson
           Acceleration, Mixing and Extrapolation.” Numerical Algorithms,
           80(1), 135–234.

    """
    def __init__(self, is_batch: bool, mix_param: Real = 0.05,
                 generations: int = 6, diagonal_offset=0.01,
                 init_mix_param: Real = 0.01, tolerance: Real = 1E-6):

        super().__init__(is_batch, tolerance)

        self.mix_param = mix_param
        self.generations = generations
        self.init_mix_param = init_mix_param
        self.diagonal_offset = diagonal_offset

        # Holds "x" history and "x" delta history
        self._x_hist: Optional[Tensor] = None
        self._f: Optional[Tensor] = None

        # Systems are flatted during input to permit shape agnostic
        # programming. The shape information is stored here.
        self._shape_in: Optional[list] = None
        self._shape_out: Optional[list] = None

    def _setup_hook(self, x_new: Tensor):
        """Perform post instantiation initialisation operation.

        This instantiates internal variables.

        Arguments:
            x_new: New system(s) that is to be mixed.
        """
        dtype, device = x_new.dtype, x_new.device

        # Tensors are converted to _shape_in when passed in and back to their
        # original shape _shape_out when returned to the user.
        self._shape_out = list(x_new.shape)
        if self._is_batch:
            self._shape_in = list(x_new.reshape(x_new.shape[0], -1).shape)
        else:
            self._shape_in = list(torch.flatten(x_new).shape)

        # Instantiate the x history (x_hist) and the delta history 'd_hist'
        size = (self.generations + 1, *self._shape_in)
        self._x_hist = torch.zeros(size, dtype=dtype, device=device)
        self._f = torch.zeros(size, dtype=dtype, device=device)

    @property
    def delta(self) -> Tensor:
        """Difference between the current & previous systems"""
        return self._delta.reshape(self._shape_out)

    def __call__(self, x_new: Tensor, x_old: Optional[Tensor] = None
                 ) -> Tensor:
        """Perform Anderson mixing operation.

        This takes a new system ``x_new`` & uses its past history to generate
        a new "mixed" system via the anderson method.

        Arguments:
            x_new: New system(s) that is to be mixed.
            x_old: Previous system(s) that is to be mixed. Only required for
                the first mix operation.

        Returns:
            x_mix: Newly mixed system(s).

        Notes:
            Simple mixing will be used for the first n steps, where n is the
            number of previous steps to be use in the mixing process.

        """
        if self._step_number == 0:  # Call setup hook if this is the 1st cycle
            self._setup_hook(x_new)

        self._step_number += 1  # Increment step_number

        # Following Eyert's notation, "f" refers to the delta:
        #   F = x_new - x_old
        # However, for clarity "x_hist" is used in place of Eyert's "x".

        # Inputs must be reshaped to ensure they a vector (or batch thereof)
        x_new = x_new.reshape(self._shape_in)

        # If x_old specified; overwrite last entry in self._x_hist.
        x_old = (self._x_hist[0] if x_old is None
                 else x_old.reshape(self._shape_in))

        # Calculate x_new - x_old delta & assign to the delta history _f
        self._f[0] = x_new - x_old

        # If a sufficient history has been built up then use Anderson mixing
        if self._step_number > self.generations:
            # Setup and solve the linear equation system, as described in
            # equation 4.3 (Eyert), to get the coefficients "thetas":
            #   a(i,j) =  <F(l) - F(l-i)|F(l) - F(l-j)>
            #   b(i)   =  <F(l) - F(l-i)|F(l)>
            # here dF = <F(l) - F(l-i)|
            df = self._f[0] - self._f[1:]
            a = torch.einsum('i...v,j...v->...ij', df, df)
            b = torch.einsum('h...v,...v->...h', df, self._f[0])

            # Rescale diagonals to prevent linear dependence on the residual
            # vectors by adding 1 + offset^2 to the diagonals of "a", see
            # equation 8.2 (Eyert)
            if self.diagonal_offset is not None:
                a += (torch.eye(a.shape[-1], device=x_new.device)
                      * (self.diagonal_offset ** 2))

            # Solve for the coefficients. As torch.solve cannot solve for 1D
            # tensors a blank dimension must be added
            thetas = torch.squeeze(torch.solve(torch.unsqueeze(b, -1), a)[0])

            # Construct the 2'nd terms of eq 4.1 & 4.2 (Eyert). These are
            # the "averaged" histories of x and F respectively:
            #   x_bar = sum(j=1 -> m) ϑ_j(l) * (|x(l-j)> - |x(l)>)
            #   f_bar = sum(j=1 -> m) ϑ_j(l) * (|F(l-j)> - |F(l)>)
            # These are not the x_bar & F_var values of eq. 4.1 & 4.2 (Eyert)
            # yet as they are still missing the 1st terms.
            x_bar = torch.einsum('...h,h...v->...v', thetas,
                                 (self._x_hist[1:] - self._x_hist[0]))
            f_bar = torch.einsum('...h,h...v->...v', thetas, -df)

            # The first terms of equations 4.1 & 4.2 (Eyert):
            #   4.1: |x(l)> and & 4.2: |F(l)>
            # Have been replaced by:
            #   ϑ_0(l) * |x(j)> and ϑ_0(l) * |x(j)>
            # respectively, where "ϑ_0(l)" is the coefficient for the current
            # step and is defined as (Anderson):
            #   ϑ_0(l) = 1 - sum(j=1 -> m) ϑ_j(l)
            # Code deviates from DFTB+ here to prevent "stability issues"
            # theta_0 = 1 - torch.sum(thetas)
            # x_bar += theta_0 * self._x_hist[0]  # <- DFTB+
            # f_bar += theta_0 * self._f[0]  # <- DFTB+
            x_bar += self._x_hist[0]
            f_bar += self._f[0]

            # Calculate the new mixed dQ following equation 4.4 (Eyert):
            #   |x(l+1)> = |x_bar(l)> + beta(l)|f_bar(l)>
            # where "beta" is the mixing parameter
            x_mix = x_bar + (self.mix_param * f_bar)

        # If there is insufficient history for Anderson; use simple mixing
        else:
            x_mix = self._x_hist[0] + (self._f[0] * self.init_mix_param)
            return x_mix.reshape(self._shape_out)

        # Shift f & x_hist over; a roll follow by a reassignment is
        # necessary to avoid an inplace error. (gradients remain intact)
        self._f = torch.roll(self._f, 1, 0)
        self._x_hist = torch.roll(self._x_hist, 1, 0)

        # Assign the mixed x to the x_hist history array. The last x_mix value
        # is saved on the assumption that it will be used in the next step.
        self._x_hist[0] = x_mix
        self._delta = self._f[1]  # Save the last difference to _delta

        # Reshape the mixed system back into the expected shape and return it
        return x_mix.reshape(self._shape_out)

    def cull(self, cull_list: bool):
        """Purge select systems form the mixer.

        This is useful when a subset of systems have converged during mixing.

        Arguments:
            cull_list: Tensor with booleans indicating which systems should be
                culled (True) and which should remain (False).

        """
        assert self._is_batch, 'Cull only valid for batch mixing'
        # Invert the cull_list, gather & reassign self._delta self._x_hist &
        # self._f so only those marked False remain.
        cull = ~cull_list
        self._delta = self._delta[cull]
        self._f = self._f[:, cull]
        self._x_hist = self._x_hist[:, cull]

        # Adjust the the shapes accordingly
        n = list(cull_list).count(True)
        self._shape_in[0] -= n
        self._shape_out[0] -= n

    def reset(self):
        """Reset mixer to its initial state."""
        # Reset all internal attributes.
        self._step_number = 0
        self._x_hist = self._f = self._delta = None
        self._shape_in = self._shape_out = None
