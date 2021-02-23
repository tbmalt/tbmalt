# -*- coding: utf-8 -*-
"""A collection of common mathematical functions.

This module contains a collection of batch-operable, back-propagatable
mathematical functions.
"""
from typing import Tuple, Union, Literal, Optional
import torch
import numpy as np
Tensor = torch.Tensor


def gaussian(x: Union[Tensor, float], mean: Union[Tensor, float],
             std: Union[Tensor, float]) -> Tensor:
    r"""Gaussian distribution function.

    A one dimensional Gaussian function representing the probability density
    function of a normal distribution. This Gaussian takes on the form:

    .. math::

        g(x) = \frac{1}{\sigma\sqrt{2\pi}}exp
            \left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\right)

    Where σ (`std`) is the standard deviation, μ (`mean`) is the mean & x is
    the point at which the distribution is to be evaluated. Multiple values
    can be passed for batch operation, see the `Notes` section for further
    information.

    Arguments:
        x: Value(s) at which to evaluate the gaussian function.
        mean: Expectation value(s), i.e the mean.
        std: Standard deviation.

    Returns:
        g: The gaussian function(s) evaluated at the specified `x`, `mean` &
            `std` value(s).

    Raises:
        TypeError: Raised if neither ``x`` or ``mu`` are of type torch.Tensor.

    Notes:
        Multiple `x`, `mean` & `std` values can be specified for batch-wise
        evaluation. Note that at least one argument must be a torch.Tensor
        entity, specifically `x` or `mean`, else this function will error out.
        However, zero dimensional tensors are acceptable.

    Examples:
        Evaluating multiple points within a single distribution:

        >>> import tbmalt.common.maths as tb_maths
        >>> import matplotlib.pyplot as plt
        >>> x_vals = torch.linspace(0, 1, 100)
        >>> y_vals = tb_maths.gaussian(x_vals, 0.5, 0.5)
        >>> plt.plot(x_vals, y_vals, '-k')
        >>> plt.show()

        Evaluating points on a pair of distributions with differing means:

        >>> x_vals = torch.linspace(0, 1, 100)
        >>> y1, y2 = tb_maths.gaussian(x_vals, torch.tensor([0.25, 0.75]), 0.5)
        >>> plt.plot(x_vals, y1, '-r')
        >>> plt.plot(x_vals, y2, '-b')
        >>> plt.show()

    """
    # Evaluate the gaussian at the specified value(s) and return the result
    return (torch.exp(-0.5 * torch.pow((x - mean) / std, 2))
            / (std * np.sqrt(2 * np.pi)))


def hellinger(p: Tensor, q: Tensor) -> Tensor:
    r"""Calculate the Hellinger distance between pairs of 1D distributions.

    The Hellinger distance can be used to quantify the similarity between a
    pair of discrete probability distributions which have been evaluated at
    the same sample points.

    Arguments:
        p: Values observed in the first distribution.
        q: Values observed in the second distribution.

    Returns:
        distance: Hellinger distance between each `p`, `q` distribution pair.

    Notes:
        The Hellinger distance is computed as:

        .. math::

             H(p,q)= \frac{1}{\sqrt{2}}\sqrt{\sum_{i=1}^{k}
                \left( \sqrt{p_i} - \sqrt{q_i}  \right)^2}

        Multiple pairs of distributions can be evaluated simultaneously by
        passing in a 2D torch.Tensor in place of a 1D one.

    Raises:
        ValueError: When elements in `p` or `q` are found to be negative.

    Warnings:
        As `p` and `q` ar probability distributions they must be positive. If
        not, a terminal error will be encountered during backpropagation.

    """
    # Raise a ValueError if negative values are encountered. Negative values
    # will throw an error during backpropagation of the sqrt function.
    if torch.sum(p.detach() < 0) != 0 or torch.sum(q.detach() < 0) != 0:
        raise ValueError('All elements in p & q must be positive.')

    # Calculate & return the Hellinger distance between distribution pair(s)
    # Note that despite what the pytorch documentation states torch.sum does
    # in-fact take the "axis" argument.
    return torch.sqrt(
        torch.sum(
            torch.pow(torch.sqrt(p) - torch.sqrt(q), 2),
            -1)
    ) / np.sqrt(2)


class _SymEigB(torch.autograd.Function):
    # State that this can solve for multiple systems and that the first
    # dimension should iterate over instance of the batch.
    r"""Solves standard eigenvalue problems for real symmetric matrices.

    This solves standard eigenvalue problems for real symmetric matrices, and
    can apply conditional or Lorentzian broadening to the eigenvalues during
    the backwards pass to increase gradient stability.

    Notes:
        Results from backward passes through eigen-decomposition operations
        tend to suffer from numerical stability [*]_  issues when operating
        on systems with degenerate eigenvalues. Fortunately,  the stability
        of such operations can be increased through the application of eigen
        value broadening. However, such methods will induce small errors in
        the returned gradients as they effectively mutate  the eigen-values
        in the backwards pass. Thus, it is important to be aware that while
        increasing the extent of  broadening will help to improve stability
        it will also increase the error in the gradients.

        Two different broadening methods have been  implemented within this
        class. Conditional broadening as described by Seeger [MS2019]_, and
        Lorentzian as detailed by Liao [LH2019]_. During the forward pass the
        `torch.symeig` function is used to calculate both the eigenvalues &
        the eigenvectors (U & :math:`\lambda` respectively). The gradient
        is then calculated following:

        .. math:: \bar{A} = U (\bar{\Lambda} + sym(F \circ (U^t \bar{U}))) U^T

        Where bar indicates a value's gradient passed in from  the previous
        layer, :math:`\Lambda` is the diagonal matrix associated with the
        :math:`\bar{\lambda}` values,  :math:`\circ`  is the so  called
        Hadamard product, sym is the symmetrisation operator and F is:

        .. math:: F_{i, j} = \frac{I_{i \ne j}}{h(\lambda_i - \lambda_j)}

        Where, for conditional broadening, h is:

        .. math:: h(t) = max(|t|, \epsilon)sgn(t)

        and for, Lorentzian broadening:

        .. math:: h(t) = \frac{t^2 + \epsilon}{t}

        The advantage of conditional broadening is that is is only applied
        when it is needed, thus the errors induced in the gradients will be
        restricted to systems whose gradients would be nan's otherwise.
        The Lorentzian method, on the other hand, will apply broadening to
        all systems, irrespective of whether or not it is necessary. Note
        that if the h function is a unity operator then this is identical
        to a standard backwards pass through an eigen-solver.


        .. [*] Where stability is defined as the propensity of a function to
               return nan values or some raise an error.

    References:
        .. [MS2019] Seeger, M., Hetzel, A., Dai, Z., & Meissner, E. Auto-
                    Differentiating Linear Algebra. ArXiv:1710.08717 [Cs,
                    Stat], Aug. 2019. arXiv.org,
                    http://arxiv.org/abs/1710.08717.
        .. [LH2019] Liao, H.-J., Liu, J.-G., Wang, L., & Xiang, T. (2019).
                    Differentiable Programming Tensor Networks. Physical
                    Review X, 9(3).
        .. [Lapack] www.netlib.org/lapack/lug/node54.html (Accessed 10/08/2020)

        """

    # Note that 'none' is included only for testing purposes
    KNOWN_METHODS = ['cond', 'lorn', 'none']

    @staticmethod
    def forward(ctx, a: Tensor, method: str = 'cond', factor: float = 1E-12
                ) -> Tuple[Tensor, Tensor]:
        """Calculate the eigenvalues and eigenvectors of a symmetric matrix.

        Finds the eigenvalues and eigenvectors of a real symmetric
        matrix using the torch.symeig function.

        Arguments:
            a: A real symmetric matrix whose eigenvalues & eigenvectors will
                be computed.
            method: Broadening method to used, available options are:

                    - "cond" for conditional broadening.
                    - "lorn" for Lorentzian broadening.

                See class doc-string for more info on these methods.
                [DEFAULT='cond']
            factor: Degree of broadening (broadening factor). [Default=1E-12]

        Returns:
            w: The eigenvalues, in ascending order.
            v: The eigenvectors.

        Notes:
            The ctx argument is auto-parsed by PyTorch & is used to pass data
            from the .forward() method to the .backward() method. This is not
            normally described in the docstring but has been done here to act
            as an example.

        Warnings:
            Under no circumstances should `factor` be a torch.tensor entity.
            The `method` and `factor` parameters MUST be passed as positional
            arguments and NOT keyword arguments.

        """
        # Check that the method is of a known type
        if method not in _SymEigB.KNOWN_METHODS:
            raise ValueError('Unknown broadening method selected.')

        # Compute eigen-values & vectors using torch.symeig.
        w, v = torch.symeig(a, eigenvectors=True)

        # Save tensors that will be needed in the backward pass
        ctx.save_for_backward(w, v)

        # Save the broadening factor and the selected broadening method.
        ctx.bf, ctx.bm = factor, method

        # Store dtype/device to prevent dtype/device mixing
        ctx.dtype, ctx.device = a.dtype, a.device

        # Return the eigenvalues and eigenvectors
        return w, v

    @staticmethod
    def backward(ctx, w_bar: Tensor, v_bar: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates gradients of the eigen decomposition operation.

        Evaluates gradients of the matrix from which the eigenvalues
        and eigenvectors were taken.

        Arguments:
            w_bar: Gradients associated with the the eigenvalues.
            v_bar: Gradients associated with the eigenvectors.

        Returns:
            a_bar: Gradients associated with the `a` tensor.

        Notes:
            See class doc-string for a more detailed description of this
            method.

        """
        # Equation to variable legend
        #   w <- λ
        #   v <- U

        # __Preamble__
        # Retrieve eigenvalues (w) and eigenvectors (v) from ctx
        w, v = ctx.saved_tensors

        # Retrieve, the broadening factor and convert to a tensor entity
        if not isinstance(ctx.bf, Tensor):
            bf = torch.tensor(ctx.bf, dtype=ctx.dtype, device=ctx.device)
        else:
            bf = ctx.bf

        # Retrieve the broadening method
        bm = ctx.bm

        # Form the eigenvalue gradients into diagonal matrix
        lambda_bar = w_bar.diag_embed()

        # Identify the indices of the upper triangle of the F matrix
        tri_u = torch.triu_indices(*v.shape[-2:], 1)

        # Construct the deltas
        deltas = w[..., tri_u[1]] - w[..., tri_u[0]]

        # Apply broadening
        if bm == 'cond':  # <- Conditional broadening
            deltas = 1 / torch.where(torch.abs(deltas) > bf,
                                     deltas, bf) * torch.sign(deltas)
        elif bm == 'lorn':  # <- Lorentzian broadening
            deltas = deltas / (deltas**2 + bf)
        elif bm == 'none':  # <- Debugging only
            deltas = 1 / deltas
        else:  # <- Should be impossible to get here
            raise ValueError(f'Unknown broadening method {bm}')

        # Construct F matrix where F_ij = v_bar_j - v_bar_i; construction is
        # done in this manner to avoid 1/0 which can cause intermittent and
        # hard-to-diagnose issues.
        F = torch.zeros(*w.shape, w.shape[-1], dtype=ctx.dtype,
                        device=w_bar.device)
        # Upper then lower triangle
        F[..., tri_u[0], tri_u[1]] = deltas
        F[..., tri_u[1], tri_u[0]] -= F[..., tri_u[0], tri_u[1]]

        # Construct the gradient following the equation in the doc-string.
        a_bar = v @ (lambda_bar
                     + sym(F * (v.transpose(-2, -1) @ v_bar))
                     ) @ v.transpose(-2, -1)

        # Return the gradient. PyTorch expects a gradient for each parameter
        # (method, bf) hence two extra Nones are returned
        return a_bar, None, None


def _eig_sort_out(w: Tensor, v: Tensor, ghost: bool = True
                  ) -> Tuple[Tensor, Tensor]:
    """Move ghost eigen values/vectors to the end of the array.

    Discuss the difference between ghosts (w=0) and auxiliaries (w=1)

    Performing and eigen-decomposition operation on a zero-padded packed
    tensor results in the emergence of ghost eigen-values/vectors. This can
    cause issues downstream, thus they are moved to the end here which means
    they can be easily clipped off should the user wish to do so.

    Arguments:
        w: The eigen-values.
        v: The eigen-vectors.
        ghost: Ghost-eigen-vlaues are assumed to be 0 if True, else assumed to
            be 1. If zero padded then this should be True, if zero padding is
            turned into identity padding then False should be used. This will
            also change the ghost eigenvalues from 1 to zero when appropriate.
            [DEFAULT=True]

    Returns:
        w: The eigen-values, with ghosts moved to the end.
        v: The eigen-vectors, with ghosts moved to the end.

    """
    val = 0 if ghost else 1

    # Create a mask that is True when an eigen value is zero/one
    mask = torch.eq(w, val)
    # and its associated eigen vector is a column of a identity matrix:
    # i.e. all values are 1 or 0 and there is only a single 1. This will
    # just all zeros if columns are not one-hot.
    is_one = torch.eq(v, 1)  # <- precompute
    mask &= torch.all(torch.eq(v, 0) | is_one, dim=1)
    mask &= torch.sum(is_one, dim=1) <= 1  # <- Only a single "1" at most.

    # Convert any auxiliary eigenvalues into ghosts
    if not ghost:
        w = w - mask.type(w.dtype)

    # Pull out the indices of the true & ghost entries and cat them together
    # so that the ghost entries are at the end.
    # noinspection PyTypeChecker
    indices = torch.cat((
        torch.stack(torch.where(~mask)),
        torch.stack(torch.where(mask))),
        dim=-1,
    )

    # argsort fixes the batch order and stops eigen-values accidentally being
    # mixed between different systems. As PyTorch's argsort is not stable, i.e.
    # it dose not respect any order already present in the data, numpy's argsort
    # must be used for now.
    sorter = np.argsort(indices[0].cpu(), kind='stable')

    # Apply sorter to indices; use a tuple to make 1D & 2D cases compatible
    sorted_indices = tuple(indices[..., sorter])

    # Fix the order of the eigen values and eigen vectors.
    w = w[sorted_indices].reshape(w.shape)
    # Reshaping is needed to allow sorted_indices to be used for 2D & 3D
    v = v.transpose(-1, -2)[sorted_indices].reshape(v.shape).transpose(-1, -2)

    # Return the eigenvalues and eigenvectors
    return w, v


def eighb(a: Tensor,
          b: Tensor = None,
          scheme: Literal['chol', 'lowd'] = 'chol',
          broadening_method: Optional[Literal['cond', 'lorn']] = 'cond',
          factor: float = 1E-12,
          sort_out: bool = True,
          aux: bool = True,
          **kwargs) -> Tuple[Tensor, Tensor]:
    r"""Solves general & standard eigen-problems, with optional broadening.

    Solves standard and generalised eigenvalue problems of the from Az = λBz
    for a real symmetric matrix ``a`` and can apply conditional or Lorentzian
    broadening to the eigenvalues during the backwards pass to increase
    gradient stability. Multiple  matrices may be passed in batch major form,
    i.e. the first axis iterates over entries.

    Arguments:
        a: Real symmetric matrix whose eigen-values/vectors will be computed.
        b: Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem.
        scheme: Scheme to covert generalised eigenvalue problems to standard
            ones:

                - "chol": Cholesky factorisation. [DEFAULT='chol']
                - "lowd": Löwdin orthogonalisation.

            Has no effect on solving standard problems.

        broadening_method: Broadening method to used:

                - "cond": conditional broadening. [DEFAULT='cond']
                - "lorn": Lorentzian broadening.
                - None: no broadening (uses torch.symeig).

        factor: The degree of broadening (broadening factor). [Default=1E-12]
        sort_out: If True; eigen-vector/value tensors are reordered so that
            any "ghost" entries are moved to the end. "Ghost" are values which
            emerge as a result of zero-padding. [DEFAULT=True]
        aux: Converts zero-padding to identity-padding. This this can improve
            the stability of backwards propagation. [DEFAULT=True]

    Keyword Args:
        direct_inv (bool): If True then the matrix inversion will be computed
            directly rather than via a call to torch.solve. Only relevant to
            the cholesky scheme. [DEFAULT=False]

    Returns:
        w: The eigenvalues, in ascending order.
        v: The eigenvectors.

    Notes:
        Results from backward passes through eigen-decomposition operations
        tend to suffer from numerical stability [*]_  issues when operating
        on systems with degenerate eigenvalues. Fortunately,  the stability
        of such operations can be increased through the application of eigen
        value broadening. However, such methods will induce small errors in
        the returned gradients as they effectively mutate  the eigen-values
        in the backwards pass. Thus, it is important to be aware that while
        increasing the extent of  broadening will help to improve stability
        it will also increase the error in the gradients.

        Two different broadening methods have been  implemented within this
        class. Conditional broadening as described by Seeger [MS2019]_, and
        Lorentzian as detailed by Liao [LH2019]_. During the forward pass the
        `torch.symeig` function is used to calculate both the eigenvalues &
        the eigenvectors (U & :math:`\lambda` respectively). The gradient
        is then calculated following:

        .. math:: \bar{A} = U (\bar{\Lambda} + sym(F \circ (U^t \bar{U}))) U^T

        Where bar indicates a value's gradient, passed in from the previous
        layer, :math:`\Lambda` is the diagonal matrix associated with the
        :math:`\bar{\lambda}` values, :math:`\circ`  is the so called Hadamard
        product, :math:`sym` is the symmetrisation operator and F is:

        .. math:: F_{i, j} = \frac{I_{i \ne j}}{h(\lambda_i - \lambda_j)}

        Where, for conditional broadening, h is:

        .. math:: h(t) = max(|t|, \epsilon)sgn(t)

        and for, Lorentzian broadening:

        .. math:: h(t) = \frac{t^2 + \epsilon}{t}

        The advantage of conditional broadening is that is is only  applied
        when it is needed, thus the errors induced in the gradients will be
        restricted to systems whose gradients would be nan's otherwise. The
        Lorentzian method, on the other hand, will apply broadening to all
        systems, irrespective of whether or not it is necessary. Note that if
        the h function is a unity operator then this is identical to a
        standard backwards pass through an eigen-solver.

        Mathematical discussions regarding the Cholesky decomposition are
        made with reference to the  "Generalized Symmetric Definite
        Eigenproblems" chapter of Lapack. [Lapack]_

        When operating in batch mode the zero valued padding columns and rows
        will result in the generation of "ghost" eigen-values/vectors. These
        are mostly harmless, but make it more difficult to extract the actual
        eigen-values/vectors. This function will move the "ghost" entities to
        the ends of their respective lists, making it easy to clip them out.

        .. [*] Where stability is defined as the propensity of a function to
               return nan values or some raise an error.

    Warnings:
        If operating upon zero-padded packed tensors then degenerate and  zero
        valued eigen values will be encountered. This will **always** cause an
        error during the backwards pass unless broadening is enacted.

        As ``torch.symeig`` sorts its results prior to returning them, it is
        likely that any "ghost" eigen-values/vectors, which result from zero-
        padded packing, will be located in the middle of the returned arrays.
        This makes down-stream processing more challenging. Thus, the sort_out
        option is enabled by default. This results in the "ghost" values being
        moved to the end. **However**, this method identifies any entry with a
        zero-valued eigenvalue and an eigenvector which can be interpreted as
        a column of an identity matrix as a ghost.

    References:
        .. [MS2019] Seeger, M., Hetzel, A., Dai, Z., & Meissner, E. Auto-
                    Differentiating Linear Algebra. ArXiv:1710.08717 [Cs,
                    Stat], Aug. 2019. arXiv.org,
                    http://arxiv.org/abs/1710.08717.
        .. [LH2019] Liao, H.-J., Liu, J.-G., Wang, L., & Xiang, T. (2019).
                    Differentiable Programming Tensor Networks. Physical
                    Review X, 9(3).
        .. [Lapack] www.netlib.org/lapack/lug/node54.html (Accessed 10/08/2020)

        """

    # Initial setup to make function calls easier to deal with
    # If smearing use _SymEigB otherwise use the internal torch.syeig function
    func = _SymEigB.apply if broadening_method else torch.symeig
    # Set up for the arguments
    args = (broadening_method, factor) if broadening_method else (True,)

    if aux:
        # Convert from zero-padding to identity padding
        is_zero = torch.eq(a, 0)
        mask = torch.all(is_zero, dim=-1) & torch.all(is_zero, dim=-2)
        a = a + torch.diag_embed(mask.type(a.dtype))

    if b is None:  # For standard eigenvalue problem
        w, v = func(a, *args)  # Call the required eigen-solver

    else:  # Otherwise it will be a general eigenvalue problem

        # Cholesky decomposition can only act on positive definite matrices;
        # which is problematic for zero-padded tensors. Similar issues are
        # encountered in the Löwdin scheme. To ensure positive definiteness
        # the diagonals of padding columns/rows are therefore set to 1.

        # Create a mask which is True wherever a column/row pair is 0-valued
        is_zero = torch.eq(b, 0)
        mask = torch.all(is_zero, dim=-1) & torch.all(is_zero, dim=-2)

        # Set the diagonals at these locations to 1
        b = b + torch.diag_embed(mask.type(a.dtype))

        # For Cholesky decomposition scheme
        if scheme == 'chol':

            # Perform Cholesky factorization (A = LL^{T}) of B to attain L
            l = torch.cholesky(b)

            # Compute the inverse of L:
            if kwargs.get('direct_inv', False):
                # Via the direct method if specifically requested
                l_inv = torch.inverse(l)
            else:
                # Otherwise compute via an indirect method (default)
                l_inv = torch.solve(torch.eye(a.shape[-1], dtype=a.dtype,
                                              device=b.device), l)[0]
                #  RuntimeError: Expected b and A to be on the same device, but found b on cpu and A on cuda:0 instead.
            # Transpose of l_inv: improves speed in batch mode
            l_inv_t = torch.transpose(l_inv, -1, -2)

            # To obtain C, perform the reduction operation C = L^{-1}AL^{-T}
            c = l_inv @ a @ l_inv_t

            # The eigenvalues of Az = λBz are the same as Cy = λy; hence:
            w, v_ = func(c, *args)

            # Eigenvectors, however, are not, so they must be recovered:
            #   z = L^{-T}y
            v = l_inv_t @ v_

        elif scheme == 'lowd':  # For Löwdin Orthogonalisation scheme

            # Perform the BV = WV eigen decomposition.
            w, v = func(b, *args)

            # Embed w to construct "small b"; inverse power is also done here
            # to avoid inf values later on.
            b_small = torch.diag_embed(w ** -0.5)

            # Construct symmetric orthogonalisation matrix via:
            #   B^{-1/2} = V b^{-1/2} V^{T}
            b_so = v @ b_small @ v.transpose(-1, -2)

            # A' (a_prime) can then be constructed as: A' = B^{-1/2} A B^{-1/2}
            a_prime = b_so @ a @ b_so

            # Decompose the now orthogonalised A' matrix
            w, v_prime = func(a_prime, *args)

            # the correct eigenvector is then recovered via
            #   V = B^{-1/2} V'
            v = b_so @ v_prime

        else:  # If an unknown scheme was specified
            raise ValueError('Unknown scheme selected.')

    # If sort_out is enabled, then move ghosts to the end.
    if sort_out:
        w, v = _eig_sort_out(w, v, not aux)

    # Return the eigenvalues and eigenvectors
    return w, v


def sym(x: Tensor, dim0: int = -1, dim1: int = -2) -> Tensor:
    """Symmetries the specified tensor.

    Arguments:
        x: The tensor to be symmetrised.
        dim0: First dimension to be transposed. [DEFAULT=-1]
        dim1: Second dimension to be transposed [DEFAULT=-2]

    Returns:
        x_sym: The symmetrised tensor.

    """
    return (x + x.transpose(dim0, dim1)) / 2
