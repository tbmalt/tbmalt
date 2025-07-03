# -*- coding: utf-8 -*-
"""Interpolation for general purpose."""
from typing import Tuple, Union, Optional
from numbers import Real
import warnings
import torch
from torch.nn import Parameter
from tbmalt.common.batch import pack, bT
from tbmalt.ml import Feed
Tensor = torch.Tensor


class BicubInterpSpl(Feed):
    """Special purpose Bicubic like interpolation method.

    This bicubic like interpolation method is designed to interpolate Slater-
    Koster integrals over both compression radii and distances.

    Arguments:
        compr: Grid points for interpolation, 1D Tensor.
        zmesh: 2D, 3D or 4D Tensor, 2D is for single integral with various
            compression radii, 3D is for multi integrals.
        hs_grid: Distances at which the ``hamiltonian`` & ``overlap``
            elements were evaluated.

    Notes:
        This feed does not contain any optimisable parameters as it is not
        intended to be optimised directly. Instead, it is the compression
        radii of the atoms that are to be optimised.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import torch
        >>> x = torch.arange(0., 5., 0.25)
        >>> y = torch.arange(0., 5., 0.25)
        >>> xx, yy = torch.meshgrid(x, y)
        >>> z = torch.sin(xx) + torch.cos(yy)
        >>> bi_interp = BicubInterpSpl(x, z)
        >>> xnew = torch.arange(0., 5., 1e-2)
        >>> ynew = torch.arange(0., 5., 1e-2)
        >>> znew = bi_interp(torch.stack([xnew, ynew]).T)
        >>> plt.plot(x, z.diagonal(), 'ro-', xnew, znew, 'b-')
        >>> plt.show()

    References:
        .. [wiki] https://en.wikipedia.org/wiki/Bicubic_interpolation

    """

    def __init__(self, compr: Tensor, zmesh: Tensor, hs_grid=None):
        super().__init__()

        assert zmesh.shape[-2] == zmesh.shape[-2], \
            'Size of last two dimensions of zmesh must be the same.'
        if zmesh.dim() < 2 or zmesh.dim() > 4:
            raise ValueError(f'zmesh should be 2, 3, or 4D, get {zmesh.dim()}')

        if hs_grid is not None:
            assert zmesh.dim() == 4, 'zemsh dimension error'

        self.compr = compr
        self.zmesh = zmesh
        self.hs_grid = hs_grid

        # Internal attributes used during interpolation
        self._nx0, self._nx1, self._nx2 = None, None, None
        self._nind, self._nx_1 = None, None

        # Device type of the tensor in this class
        self._device = compr.device

    def __call__(self, rr: Tensor, distances=None):
        """Calculate bicubic interpolation.

        If distances is not None, the polynomial interpolation will be
        used to interpolate distances from ``hamiltonian`` & ``overlap``.
        Then the bicubic interpolation will be used to interpolate rr
        from the interpolated ``hamiltonian`` & ``overlap`` values.

        Arguments:
            rr: The points to be interpolated for the first dimension and
                second dimension.
            distances: Distances between atoms.

        Returns:
            znew: Interpolation values with given rr, or rr and distances.

        """
        if self.hs_grid is not None:
            assert distances is not None, 'distances should not be None'

        xi = rr if rr.dim() == 2 else rr.unsqueeze(0)
        if not xi.is_contiguous():
            xi = xi.contiguous()

        self.batch = xi.shape[0]
        self.arange_batch = torch.arange(self.batch)

        if self.hs_grid is not None:  # with DFTB+ distance interpolation
            assert distances is not None, 'if hs_grid is not None, '+ \
                'distances is expected'

            ski = PolyInterpU(self.hs_grid, Parameter(self.zmesh, requires_grad=False))
            zmesh = ski.forward(distances).permute(0, -2, -1, 1)
        elif self.zmesh.dim() == 2 or self.zmesh.dim() == 3:
            zmesh = self.zmesh.repeat(rr.shape[0], 1, 1)
        else:
            raise ValueError("Incompatible z-mesh dimension")

        coeff = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                              [-3., 3., -2., -1.], [2., -2., 1., 1.]],
                              device=self._device)
        coeff_ = torch.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                               [0., 1., -2., 1.], [0., 0., -1., 1.]],
                               device=self._device)

        # get the nearest grid points, 1st and second neighbour indices of xi
        self._get_indices(xi)

        # this is to transfer x to fraction and its square, cube
        x_fra = (xi - self.compr[self._nx0]) / (
                self.compr[self._nx1] - self.compr[self._nx0])
        xmat = torch.stack([x_fra ** 0, x_fra ** 1, x_fra ** 2, x_fra ** 3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self._fmat0th(zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = \
            self._fmat1th(zmesh, f00, f10, f01, f11)
        fmat = torch.stack([torch.stack([f00, f01, f02, f03]),
                            torch.stack([f10, f11, f12, f13]),
                            torch.stack([f20, f21, f22, f23]),
                            torch.stack([f30, f31, f32, f33])])

        pdim = [2, 0, 1] if fmat.dim() == 3 else [2, 3, 0, 1]
        a_mat = torch.matmul(torch.matmul(coeff, fmat.permute(pdim)), coeff_)

        znew = torch.stack([torch.matmul(torch.matmul(
            xmat[:, i, 0], a_mat[i]), xmat[:, i, 1]) for i in range(self.batch)])
        return znew

    def forward(self, rr: Tensor, distances=None):
        """Perform the bicubic-like interpolation.

        If distances is not None, the polynomial interpolation will be
        used to interpolate distances from ``hamiltonian`` & ``overlap``.
        Then the bicubic interpolation will be used to interpolate rr
        from the interpolated ``hamiltonian`` & ``overlap`` values.

        Arguments:
            rr: The points to be interpolated for the first dimension and
                second dimension.
            distances: Distances between atoms.

        Returns:
            znew: Interpolation values with given rr, or rr and distances.

        """
        return self(rr, distances=distances)

    def _get_indices(self, xi):
        """Get indices and repeat indices."""
        # Note that this function assigns to _nx0, _nind, _nx1, _nx_1, & _nx2
        self._nx0 = torch.searchsorted(self.compr, xi.detach()) - 1

        # get all surrounding 4 grid points indices and repeat indices
        self._nind = torch.tensor([ii for ii in range(self.batch)])
        self._nx1 = torch.clamp(torch.stack([ii + 1 for ii in self._nx0]), 0,
                                len(self.compr) - 1)
        self._nx_1 = torch.clamp(torch.stack([ii - 1 for ii in self._nx0]), 0)
        self._nx2 = torch.clamp(torch.stack([ii + 2 for ii in self._nx0]), 0,
                                len(self.compr) - 1)

    def _fmat0th(self, zmesh: Tensor):
        """Construct f(0/1, 0/1) in fmat."""
        f00 = zmesh[self.arange_batch, self._nx0[..., 0], self._nx0[..., 1]]
        f10 = zmesh[self.arange_batch, self._nx1[..., 0], self._nx0[..., 1]]
        f01 = zmesh[self.arange_batch, self._nx0[..., 0], self._nx1[..., 1]]
        f11 = zmesh[self.arange_batch, self._nx1[..., 0], self._nx1[..., 1]]
        return f00, f10, f01, f11

    def _fmat1th(self, zmesh: Tensor, f00: Tensor, f10: Tensor, f01: Tensor,
                 f11: Tensor):
        """Get the 1st derivative of four grid points over x, y and xy."""

        # Helper functions to avoid code repetition
        def f1(i, j):
            return zmesh[self.arange_batch, i[..., 0], j[..., 1]]

        def f2(f1, f2, n1, n2, i, j):
            return bT(bT(f1 - f2) / (n1[..., i] - n2[..., j]))

        # Bring class instance attributes into the local name space to avoid
        # repeatedly having to call `self`.
        nx_1, nx0, nx1, nx2 = self._nx_1, self._nx0, self._nx1, self._nx2

        f_10 = f1(nx_1, nx0)
        f_11 = f1(nx_1, nx1)
        f0_1 = f1(nx0, nx_1)
        f02 = f1(nx0, nx2)
        f1_1 = f1(nx1, nx_1)
        f12 = f1(nx1, nx2)
        f20 = f1(nx2, nx0)
        f21 = f1(nx2, nx1)

        # calculate the derivative: (F(1) - F(-1) / (2 * grid)
        fy00 = f2(f01, f0_1, nx1, nx_1, 1, 1)
        fy01 = f2(f02, f00, nx2, nx0, 1, 1)
        fy10 = f2(f11, f1_1, nx1, nx_1, 1, 1)
        fy11 = f2(f12, f10, nx2, nx0, 1, 1)
        fx00 = f2(f10, f_10, nx1, nx_1, 0, 0)
        fx01 = f2(f20, f00, nx2, nx0, 0, 0)
        fx10 = f2(f11, f_11, nx1, nx_1, 0, 0)
        fx11 = f2(f21, f01, nx2, nx0, 0, 0)

        fxy00, fxy11 = fy00 * fx00, fx11 * fy11
        fxy01, fxy10 = fx01 * fy01, fx10 * fy10

        return (fy00, fy01, fy10, fy11, fx00, fx01, fx10, fx11, fxy00,
                fxy01, fxy10, fxy11)


class PolyInterpU(Feed):
    """Polynomial interpolation method with uniform grid points.

    The boundary condition will use `poly_to_zero` function, which make the
    polynomial values smoothly converge to zero at the boundary.

    Arguments:
        x: Grid points for interpolation, 1D Tensor.
        y: Values to be interpolated at each grid point. Note that this must
            be a `torch.nn.Parameter` rather than a standard torch `Tensor`.
        tail: Distance to smooth the tail.
        delta_r: Delta distance for 1st, 2nd derivative.
        n_interp: Number of total interpolation grid points.
        n_interp_r: Number of right side interpolation grid points.

    Attributes:
        delta_r: Delta distance for 1st, 2nd derivative.
        tail: Distance to smooth the tail.
        n_interp: Number of total interpolation grid points.
        n_interp_r: Number of right side interpolation grid points.

    Notes:
        The `PolyInterpU` class, which is taken from the DFTB+, assumes a
        uniform grid. Here, the y and x arguments are the values to be
        interpolated and their associated grid points respectively. The tail
        end of the spline is smoothed to zero, meaning that extrapolated
        points will rapidly, but smoothly, decay to zero.

    Examples:
        >>> import torch
        >>> from torch.nn import Parameter
        >>> import matplotlib.pyplot as plt
        >>> x = torch.linspace(0, 2. * torch.pi, 100)
        >>> y = Parameter(torch.sin(x), requires_grad=False)
        >>> poly = PolyInterpU(x, y, n_interp=8, n_interp_r=4)
        >>> new_x = torch.rand(10) * 2. * torch.pi
        >>> new_y = poly.forward(new_x)
        >>> plt.plot(x, y, 'k-')
        >>> plt.plot(new_x, new_y, 'rx')
        >>> plt.show()

    """

    # TODO:
    #   - Resolve bug causing incorrect value to be returned when
    #     interpolating at the last grid point.
    #   - Either implement cache based updating to detect when changes have
    #     been made to tensor contents or make it clear through documentation
    #     what is not permitted to be modified.

    def __init__(
            self, x: Tensor, y: Parameter, tail: Real = 1.0,
            delta_r: Real = 1E-5, n_interp: int = 8, n_interp_r: int = 4):

        super().__init__()

        if not torch.allclose(x.diff(), x[1] - x[0]):
            raise ValueError("Spacing of grid points must be uniform.")

        # Ensure that the y values are a parameter instance
        if not isinstance(y, Parameter):
            warnings.warn(
                "An instance of `torch.nn.Parameter` was expected for the "
                "attribute `y`, but a `torch.Tensor` was received. The tensor "
                "will be automatically cast to a parameter.",
                UserWarning)
            y = Parameter(y, requires_grad=y.requires_grad)

        self._x = x
        self._y = y
        self.delta_r = delta_r

        self._tail = tail
        self.n_interp = n_interp
        self.n_interp_r = n_interp_r

        # Device type of the tensor in this class
        self._device = x.device

        # Get grid points with external tail for index operation
        self._x_ext = self._build_external_tail()

        # Check xx is uniform & that len(xx) > n_interp
        if len(x) < n_interp:
            raise ValueError(f'`n_interp` ({n_interp}) exceeds the number of'
                             f'data points `xx` ({len(x)}).')

    @property
    def y(self) -> Tensor:
        """Interpolation values"""
        return self._y

    @y.setter
    def y(self, y: Tensor):
        if not isinstance(y, Parameter):
            warnings.warn(
                "An instance of `torch.nn.Parameter` was expected for the "
                "attribute `y`, but a `torch.Tensor` was received. The tensor "
                "will be automatically cast to a parameter.",
                UserWarning)
            y = Parameter(y, requires_grad=y.requires_grad)
        self._y = y

    @property
    def x(self) -> Tensor:
        """Interpolation grid points"""
        return self._x

    @x.setter
    def x(self, x: Tensor):
        # If a new array of x values have been provided then ensure that it is
        # the right length.
        if not torch.allclose(x.diff(), x[1] - x[0]):
            raise ValueError("Spacing of grid points must be uniform.")

        self._x = x
        self._x_ext = self._build_external_tail()

    @property
    def tail(self) -> Real:
        """Distance over which to smooth the tail to zero."""
        return self._tail

    @tail.setter
    def tail(self, tail):
        self._tail = tail
        self._x_ext = self._build_external_tail()

    @property
    def device(self) -> torch.device:
        """Device on which data is located."""
        return self._device

    def forward(self, rr: Tensor) -> Tensor:
        """Get interpolation according to given rr.

        Arguments:
            rr: interpolation points for single and batch.

        Returns:
            result: Interpolation values with given rr.

        """
        n_grid_points = len(self._x)  # -> number of grid points

        grid_step = self.x[1] - self.x[0]
        r_max = self._x[-1] + self._tail

        interpolate = torch.logical_and(self._x[0] <= rr, rr <= self._x[-1])
        extrapolate = torch.logical_and(self._x[-1] < rr, rr <= r_max)

        result = (
            torch.zeros(rr.shape, device=self._device)
            if self._y.dim() == 1
            else torch.zeros(rr.shape[0], *self._y.shape[1:], device=self._device)
        )

        if interpolate.any():

            ind = torch.floor((rr[interpolate] - self.x[0]) / grid_step).long()

            ind_last = torch.clamp(ind + self.n_interp_r, min=self.n_interp_r, max=n_grid_points - 1)

            idxs = (ind_last - self.n_interp + 1)[:, None] + torch.arange(self.n_interp, device=self._device)

            result[interpolate] = poly_interp(
                self._x[idxs], self._y[idxs], rr[interpolate])

        # Beyond the grid => extrapolation with polynomial of 5th order
        if extrapolate.any():

            dr = rr[extrapolate] - r_max

            dr = dr.unsqueeze(-1) if self._y.dim() == 2 else dr

            xa = self.x[-self.n_interp:]

            xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)

            yb = self._y[-self.n_interp:]

            yb = yb.unsqueeze(0).repeat_interleave(dr.shape[0], dim=0)

            y0 = poly_interp(xa, yb, xa[:, self.n_interp - 1] - self.delta_r)
            y2 = poly_interp(xa, yb, xa[:, self.n_interp - 1] + self.delta_r)

            y1 = self._y[-1]
            y1p = (y2 - y0) / (2.0 * self.delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (self.delta_r * self.delta_r)

            result[extrapolate] = poly_to_zero(
                dr, -1.0 * self._tail, -1.0 / self._tail, y1, y1p, y1pp)

        return result

    def _build_external_tail(self):
        """Compute and return the external tail.

        Returns:
            external_tail: The external tail.
        """
        xx = self._x
        tail = self._tail
        return torch.linspace(
            xx[0], xx[-1] + tail,
            len(xx) + int(tail / (xx[1] - xx[0])),
            device=self._device)

    def __call__(self, *args, **kwargs) -> Tensor:
        warnings.warn(
            "`PolyInterpU` instances should be invoked via their "
            "`.forward` method now.")
        return self.forward(*args, **kwargs)


def poly_to_zero(xx: Tensor, dx: Tensor, inv_x: Tensor,
                 y0: Tensor, y0p: Tensor, y0pp: Tensor) -> Tensor:
    """Get interpolation if beyond the grid range with 5th order polynomial.

    Arguments:
        xx: Grid points.
        dx: The grid point range for y0 and its derivative.
        inv_x: Reciprocal of dx
        y0: Values to be interpolated at each grid point.
        y0p: First derivative of y0.
        y0pp: Second derivative of y0.

    Returns:
        yy: The interpolation values with given xx points in the tail.

    Notes:
        The function `poly_to_zero` realize the interpolation of the points
        beyond the range of grid points, which make the polynomial values
        smoothly converge to zero at the boundary. The variable dx determines
        the point to be zero. This code is consistent with the function
        `poly5ToZero` in DFTB+.

    """
    dx1 = y0p * dx
    dx2 = y0pp * dx * dx
    dd = 10.0 * y0 - 4.0 * dx1 + 0.5 * dx2
    ee = -15.0 * y0 + 7.0 * dx1 - 1.0 * dx2
    ff = 6.0 * y0 - 3.0 * dx1 + 0.5 * dx2
    xr = xx * inv_x
    yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr

    return yy


def poly_interp(xp: Tensor, yp: Tensor, rr: Tensor) -> Tensor:
    """Interpolation with given uniform grid points.

    Arguments:
        xp: The grid points, 2D Tensor, first dimension is for different
            system and second is for the corresponding grids in each system.
        yp: The values at the gird points.
        rr: Points to be interpolated.

    Returns:
        yy: Interpolation values corresponding to input rr.

    Notes:
        The function `poly_interp` is designed for both single and multi
        system interpolation. Therefore, xp will be 2D Tensor.

    """
    assert xp.dim() == 2, 'xp is not 2D Tensor'
    device = xp.device
    nn0, nn1 = xp.shape[0], xp.shape[1]
    index_nn0 = torch.arange(nn0, device=device)
    icl = torch.zeros(nn0, device=device).long()
    cc, dd = yp.clone(), yp.clone()
    dxp = abs(rr - xp[index_nn0, icl])

    # find the most close point to rr (single atom pair or multi pairs)
    _mask, ii = torch.zeros(len(rr), device=device) == 0.0, 0.0
    _dx_new = abs(rr - xp[index_nn0, 0])
    while (_dx_new < dxp).any():
        ii += 1
        assert ii < nn1 - 1, 'index ii range from 0 to %s' % nn1 - 1
        _mask = _dx_new < dxp
        icl[_mask] = ii
        dxp[_mask] = abs(rr - xp[index_nn0, ii])[_mask]

    yy = yp[index_nn0, icl]

    for mm in range(nn1 - 1):
        for ii in range(nn1 - mm - 1):
            r_tmp0 = xp[index_nn0, ii] - xp[index_nn0, ii + mm + 1]

            # use transpose to realize div: (N, M, K) / (N)
            r_tmp1 = ((cc[index_nn0, ii + 1] - dd[index_nn0, ii]).transpose(
                0, -1) / r_tmp0).transpose(0, -1)
            cc[index_nn0, ii] = ((xp[index_nn0, ii] - rr) *
                                 r_tmp1.transpose(0, -1)).transpose(0, -1)
            dd[index_nn0, ii] = ((xp[index_nn0, ii + mm + 1] - rr) *
                                 r_tmp1.transpose(0, -1)).transpose(0, -1)
        if (2 * icl < nn1 - mm - 1).any():
            _mask = 2 * icl < nn1 - mm - 1
            yy[_mask] = (yy + cc[index_nn0, icl])[_mask]
        else:
            _mask = 2 * icl >= nn1 - mm - 1
            yy[_mask] = (yy + dd[index_nn0, icl - 1])[_mask]
            icl[_mask] = icl[_mask] - 1

    return yy


class CubicSpline(Feed):
    """Cupic spline interpolator.

    An entity for piecewise interpolation of data via a cupic polynomial
    spline which is twice continuously differentiable.

    Arguments:
        x: A one dimensional tensor specifying the interpolation grid points,
            i.e. the knot locations.
        y: Interpolation values, i.e. the knot values, associated with each
            grid point. For single series interpolation this should be an array
            of length "n", where "n" is the number of grid points present in
            ``x``. For batch interpolation this should be an "m" by "n"
            tensor. Note that this must be a `Parameter` rather than `Tensor`
            instance.
        tail: Distance over which to smooth the tail.

    Keyword Args:
        coefficients: 0th, 1st, 2nd and 3rd order parameters in cubic spline.

    References:
        .. [csi_wiki] https://en.wikipedia.org/wiki/Spline_(mathematics)

    Examples:
        >>> from tbmalt.common.maths.interpolation import CubicSpline
        >>> import torch
        >>> from torch.nn import Parameter
        >>> x = torch.linspace(1, 10, 10)
        >>> y = Parameter(torch.sin(x), requires_grad=False)
        >>> spline = CubicSpline(x, y)
        >>> spline.forward(torch.tensor([3.5]))
        #   tensor([-0.3526])
        >>> torch.sin(torch.tensor([3.5]))
        #   tensor([-0.3508])

    """

    def __init__(self, x: Tensor, y: Parameter, tail: Real = 1.0,
                 **kwargs):
        super().__init__()

        # X-knot values must be of an anticipated type
        if not isinstance(x, Tensor) or isinstance(x, Parameter):
            raise TypeError("The x-knot values must be a `torch.Tensor`"
                            " instance.")

        # Same for the y-knot values
        if not isinstance(y, Parameter):
            raise TypeError(
                "The y-knot values must be a `torch.nn.Parameter` instance.")

        assert y.dim() <= 2, '"CubicSpline" only support 1D or 2D interpolation'

        # Ensure that there is not a mismatch between the number of x and y
        # points supplied: one-dimensional case only.
        if y.ndim == 1 and not (len(y) == len(x)):
            raise ValueError(
                "Mismatch detected in the number of supplied `x` "
                f"({len(x)}) and `y` ({len(y)}) values."
            )

        # This is just a repeat of the above check but for the two-dimensional
        # case.
        elif y.ndim == 2 and y.shape[0] != len(x):
            raise ValueError(
                f"Array shape mismatch detected, anticipated a `y` array of "
                f"the shape ({len(x)}, n), encountered {tuple(y.shape)} instead."
            )

        # Prevent users from unintentionally optimizing the knot locations.
        if isinstance(x, Parameter) or x.requires_grad:
            warnings.warn(
                "Setting the knot positions 'x' as a freely tunable parameter"
                " is strongly advised against as it may lead to instability or"
                " incorrect behavior of the spline during optimisation."
                " Please ensure that the `x` argument is a `torch.tensor`"
                " type rather than a `torch.nn.Parameter` and that its"
                "\"requires_grad\" attribute set to `False`.",
                UserWarning, stacklevel=2)

        self.xp = x
        self._y = y

        # Coefficients will be build when the `coefficients` property is
        # first invoked. Unless the user has supplied the spline coefficients
        # manually.
        self._coefficients: Optional[Tensor] = None
        if "coefficients" in kwargs.keys():
            warnings.warn(
                "Manual specification of coefficients is deprecated.",
                DeprecationWarning, stacklevel=2)
            self._coefficients: Optional[Tensor] = kwargs.get("coefficients")

        self.grid_step = x[1] - x[0]
        self.tail = tail

        # Device type of the tensor in this class
        self._device = x.device

        # Store the version tracking information for the y-knot values so that
        # the coefficients can be updated as and when needed.
        self._y_version, self._y_id = None, None

    @property
    def y(self) -> Parameter:
        """Value of the spline at each grid point."""
        return self._y

    @y.setter
    def y(self, value: Parameter):
        # Y-knot values must be of an anticipated type
        if not isinstance(value, Parameter):
            raise TypeError(
                "y-knot values must be a `torch.nn.Parameter` instance.")

        # Finally, set new y-knot value tensor.
        self._y = value

    @property
    def coefficients(self):
        """The spline coefficients."""

        # If the spline's y-knot values were modified by something or someone
        # external to the `CubicSpline` class, then the coefficients will no
        # longer accurately reflect the current state of the spline. Thus, the
        # coefficients values must now be recalculated so that they are consistent
        # with the updated y-knot values.
        if self._knots_were_changed or self._coefficients is None:

            ypt = bT(self._y)

            # get the first dim of x
            nx = self.xp.shape[0]
            device = self.xp.device
            assert nx > 3  # the length of x variable must > 3

            # get the difference between grid points
            dxp = self.xp[1:] - self.xp[:-1]
            dyp = ypt[..., 1:] - ypt[..., :-1]

            # get b, c, d from reference website: step 3~9, first calculate c
            A = torch.zeros(nx, nx, device=device)
            A.diagonal()[1:-1] = 2 * (dxp[:-1] + dxp[1:])  # diag
            A[torch.arange(nx - 1), torch.arange(nx - 1) + 1] = dxp  # off-diag
            A[torch.arange(nx - 1) + 1, torch.arange(nx - 1)] = dxp
            A[0, 0], A[-1, -1] = 1.0, 1.0
            A[0, 1], A[1, 0] = 0.0, 0.0  # natural condition
            A[-1, -2], A[-2, -1] = 0.0, 0.0

            B = torch.zeros(*ypt.shape, device=device)
            B[..., 1:-1] = 3 * (dyp[..., 1:] / dxp[1:] - dyp[..., :-1] / dxp[:-1])
            B = B.permute(1, 0) if B.dim() == 2 else B

            cc = torch.linalg.lstsq(A, B)[0]
            cc = cc.permute(1, 0) if cc.dim() == 2 else cc
            bb = dyp / dxp - dxp * (cc[..., 1:] + 2 * cc[..., :-1]) / 3
            dd = (cc[..., 1:] - cc[..., :-1]) / (3 * dxp)

            self._coefficients = pack([ypt, bb, cc, dd])

            # The version tracking data must now be updated. The y-knot version
            # is updated otherwise the coefficients will keep being regenerated
            # over and over again. The coefficients version must also be updated
            # to prevent an infinite recursion.
            self._update_knot_version_tracking_data()

        # Finally return the coefficients
        return self._coefficients

    def forward(self, xnew: Tensor) -> Tensor:
        """Evaluate the polynomial linear or cubic spline.

        Arguments:
            xnew: Points to be interpolated.

        Returns:
            ynew: Interpolation values with given points.

        """
        # boundary condition of xnew
        assert xnew.ge(self.xp[0]).all(), \
            f'input should not be less than {self.xp[0]}'

        # Note that this deviates from the DFTB+ implementation in that
        # the there is no special treatment for the region between the
        # penultimate and final grid points. Furthermore, distances
        # less than the first grid point will trigger an error.

        # Note here that the tail actually ends one grid point earlier
        r_max = (self.xp[-1] + self.tail - self.grid_step)

        interpolate = torch.logical_and(self.xp[0] <= xnew, xnew <= self.xp[-1])
        extrapolate = torch.logical_and(self.xp[-1] < xnew, xnew <= r_max)
        ypt = bT(self.y)

        result = (
            torch.zeros(xnew.shape, device=self._device)
            if ypt.dim() == 1
            else torch.zeros(xnew.shape[0], ypt.shape[0],
                             device=self._device)
        )

        # interpolation of xx which not in the tail
        if interpolate.any():
            # get the nearest grid point index of distance in grid points
            ind = torch.floor(
                ((xnew - self.xp[0]) / self.grid_step)).detach().long()
            ind = ind[interpolate]
            dx = xnew[interpolate] - self.xp[ind]
            aa, bb, cc, dd = self.coefficients[..., ind]
            interp = aa + bb * dx + cc * dx**2 + dd * dx**3
            interp = interp.transpose(0, -1) if interp.ndim > 1 else interp
            result[interpolate] = interp

        if extrapolate.any():
            dr = xnew[extrapolate] - r_max
            dr = dr.unsqueeze(-1) if ypt.dim() == 2 else dr

            y0, y1, y2 = self._y[-3:]
            r1 = (y2 - y0) / (2.0 * self.grid_step)
            r2 = (y2 + y0 - 2.0 * y1) / self.grid_step ** 2

            dx1 = 1.0 / self.grid_step
            dd = (((y2 - y1) * dx1 - r1) * dx1 - 0.5 * r2) * dx1
            y1p = (3.0 * dd * self.grid_step + r2) * self.grid_step + r1
            y1pp = 6.0 * dd * self.grid_step + r2

            dx = self.grid_step - self.tail

            result[extrapolate] = poly_to_zero(
                dr, dx, 1.0 / dx, y2, y1p, y1pp)

        return result

    def __call__(self, *args, **kwargs) -> Tensor:
        warnings.warn(
            "`CubicSpline` instances should be invoked via their "
            "`.forward` method now.")
        return self.forward(*args, **kwargs)

    def _update_knot_version_tracking_data(self):
        self._y_version = self._y._version
        self._y_id = id(self._y)

    @property
    def _knots_were_changed(self):
        """Check if the y-knot values have updated."""
        ver_delta = self._y_version != self._y._version
        id_delta = self._y_id != id(self._y)
        return ver_delta or id_delta
