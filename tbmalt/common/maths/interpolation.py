# -*- coding: utf-8 -*-
"""Interpolation for general purpose."""
from typing import Tuple
from numbers import Real
import torch
from tbmalt.common.batch import pack, bT
Tensor = torch.Tensor


class BicubInterp:
    """Bicubic interpolation method.

    The bicubic interpolation is designed to interpolate the integrals with
    given compression radii or distances.

    Arguments:
        compr: Grid points for interpolation, 1D Tensor.
        zmesh: 2D, 3D or 4D Tensor, 2D is for single integral with various
            compression radii, 3D is for multi integrals.
        hs_grid: Distances at which the ``hamiltonian`` & ``overlap``
            elements were evaluated.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import torch
        >>> x = torch.arange(0., 5., 0.25)
        >>> y = torch.arange(0., 5., 0.25)
        >>> xx, yy = torch.meshgrid(x, y)
        >>> z = torch.sin(xx) + torch.cos(yy)
        >>> bi_interp = BicubInterp(x, z)
        >>> xnew = torch.arange(0., 5., 1e-2)
        >>> ynew = torch.arange(0., 5., 1e-2)
        >>> znew = bi_interp(torch.stack([xnew, ynew]).T)
        >>> plt.plot(x, z.diagonal(), 'ro-', xnew, znew, 'b-')
        >>> plt.show()

    References:
        .. [wiki] https://en.wikipedia.org/wiki/Bicubic_interpolation

    """

    def __init__(self, compr: Tensor, zmesh: Tensor, hs_grid=None):

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

            ski = PolyInterpU(self.hs_grid, self.zmesh)
            zmesh = ski(distances).permute(0, -2, -1, 1)
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


class PolyInterpU:
    """Polynomial interpolation method with uniform grid points.

    The boundary condition will use `poly_to_zero` function, which make the
    polynomial values smoothly converge to zero at the boundary.

    Arguments:
        xx: Grid points for interpolation, 1D Tensor.
        yy: Values to be interpolated at each grid point.
        tail: Distance to smooth the tail.
        delta_r: Delta distance for 1st, 2nd derivative.
        n_interp: Number of total interpolation grid points.
        n_interp_r: Number of right side interpolation grid points.

    Attributes:
        xx: Grid points for interpolation, 1D Tensor.
        yy: Values to be interpolated at each grid point.
        delta_r: Delta distance for 1st, 2nd derivative.
        tail: Distance to smooth the tail.
        n_interp: Number of total interpolation grid points.
        n_interp_r: Number of right side interpolation grid points.
        grid_step: Distance between each gird points.

    Notes:
        The `PolyInterpU` class, which is taken from the DFTB+, assumes a
        uniform grid. Here, the yy and xx arguments are the values to be
        interpolated and their associated grid points respectively. The tail
        end of the spline is smoothed to zero, meaning that extrapolated
        points will rapidly, but smoothly, decay to zero.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> x = torch.linspace(0, 2. * torch.pi, 100)
        >>> y = torch.sin(x)
        >>> poly = PolyInterpU(x, y, n_interp=8, n_interp_r=4)
        >>> new_x = torch.rand(10) * 2. * torch.pi
        >>> new_y = poly(new_x)
        >>> plt.plot(x, y, 'k-')
        >>> plt.plot(new_x, new_y, 'rx')
        >>> plt.show()

    """

    def __init__(self, xx: Tensor, yy: Tensor, tail: Real = 1.0,
                 delta_r: Real = 1E-5, n_interp: int = 8, n_interp_r: int = 4):
        self.xx = xx
        self.yy = yy
        self.delta_r = delta_r

        self.tail = tail
        self.n_interp = n_interp
        self.n_interp_r = n_interp_r
        self.grid_step = xx[1] - xx[0]

        # Device type of the tensor in this class
        self._device = xx.device

        # Get grid points with external tail for index operation
        self.n_tail = int(self.tail / self.grid_step)
        self.xx_ext = torch.linspace(
            self.xx[0], self.xx[-1] + self.tail,
            len(self.xx) + self.n_tail, device=self._device)

        # Check xx is uniform & that len(xx) > n_interp
        dxs = xx[1:] - xx[:-1]
        check_1 = torch.allclose(dxs, torch.full_like(dxs, self.grid_step))
        assert check_1, 'Grid points xx are not uniform'
        if len(xx) < n_interp:
            raise ValueError(f'`n_interp` ({n_interp}) exceeds the number of'
                             f'data points `xx` ({len(xx)}).')

    def __call__(self, rr: Tensor) -> Tensor:
        """Get interpolation according to given rr.

        Arguments:
            rr: interpolation points for single and batch.

        Returns:
            result: Interpolation values with given rr.

        """
        n_grid_point = len(self.xx)  # -> number of grid points

        ind = torch.searchsorted(self.xx_ext, rr).to(self._device)
        result = (
            torch.zeros(rr.shape, device=self._device)
            if self.yy.dim() == 1
            else torch.zeros(rr.shape[0], *self.yy.shape[1:], device=self._device)
        )

        # => polynomial fit
        if (ind <= n_grid_point).any():
            _mask = torch.logical_and(ind <= n_grid_point, ind != 0)

            # get the index of rr in grid points
            ind_last = (ind[_mask] + self.n_interp_r + 1).long()
            ind_last[ind_last > n_grid_point] = n_grid_point
            ind_last[ind_last < self.n_interp + 1] = self.n_interp + 1

            # gather xx and yy for both single and batch
            xa = self.xx[0] + (ind_last.unsqueeze(1) - self.n_interp - 1 +
                               torch.arange(self.n_interp, device=self._device)
                               ) * self.grid_step
            yb = torch.stack([self.yy[ii - self.n_interp - 1: ii - 1]
                              for ii in ind_last]).to(self._device)

            result[_mask] = poly_interp(xa, yb, rr[_mask])

        # Beyond the grid => extrapolation with polynomial of 5th order
        max_ind = n_grid_point - 1 + int(self.tail / self.grid_step)
        is_tail = ind.masked_fill(ind.ge(n_grid_point) * ind.le(max_ind), -1).eq(-1)
        if is_tail.any():
            r_max = self.xx[-2] + self.tail
            dr = rr[is_tail] - r_max
            dr = dr.unsqueeze(-1) if self.yy.dim() == 2 else dr
            ilast = n_grid_point

            # get grid points and grid point values
            xa = (ilast - self.n_interp + torch.arange(
                self.n_interp, device=self._device) - 1) * self.grid_step + self.xx[0]
            yb = self.yy[ilast - self.n_interp - 1: ilast - 1]
            xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)
            yb = yb.unsqueeze(0).repeat_interleave(dr.shape[0], dim=0)

            # get derivative
            y0 = poly_interp(xa, yb, xa[:, self.n_interp - 1] - self.delta_r)
            y2 = poly_interp(xa, yb, xa[:, self.n_interp - 1] + self.delta_r)
            y1 = self.yy[ilast - 2]
            y1p = (y2 - y0) / (2.0 * self.delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (self.delta_r * self.delta_r)

            result[is_tail] = poly_to_zero(
                dr, -1.0 * self.tail, -1.0 / self.tail, y1, y1p, y1pp)

        return result


def poly_to_zero(xx: Tensor, dx: Tensor, inv_dist: Tensor,
                 y0: Tensor, y0p: Tensor, y0pp: Tensor) -> Tensor:
    """Get interpolation if beyond the grid range with 5th order polynomial.

    Arguments:
        y0: Values to be interpolated at each grid point.
        y0p: First derivative of y0.
        y0pp: Second derivative of y0.
        xx: Grid points.
        dx: The grid point range for y0 and its derivative.

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
    xr = xx * inv_dist
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
        systems interpolation. Therefore xp will be 2D Tensor.

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


class CubicSpline(torch.nn.Module):
    """Polynomial natural cubic spline.

    Arguments:
        xx: Grid points for interpolation, 1D Tensor.
        yy: Values to be interpolated at each grid point.
        tail: Distance over which to smooth the tail.
        delta_r: Delta distance for 1st and 2nd derivative.
        n_interp: Number of total interpolation grid points for the tail.

    Keyword Args:
        abcd: 0th, 1st, 2nd and 3rd order parameters in cubic spline.

    References:
        .. [csi_wiki] https://en.wikipedia.org/wiki/Spline_(mathematics)

    Examples:
        >>> import tbmalt.common.maths.interpolation as interp
        >>> import torch
        >>> x = torch.linspace(1, 10, 10)
        >>> y = torch.sin(x)
        >>> fit = interp.Spline(x, y)
        >>> fit(torch.tensor([3.5]))
        >>> tensor([-0.3526])
        >>> torch.sin(torch.tensor([3.5]))
        >>> tensor([-0.3508])

    """

    def __init__(self, xx: Tensor, yy: Tensor, tail: Real = 1.0,
                 delta_r: Real = 1E-5, n_interp: int = 8, **kwargs):
        super(CubicSpline, self).__init__()

        assert yy.dim() <= 2, '"CubicSpline" only support 1D or 2D interpolation'

        self.xp = xx
        self.yp = yy.T if yy.dim() == 2 and yy.shape[0] == xx.shape[0] else yy

        self.grid_step = xx[1] - xx[0]
        self.delta_r = delta_r
        self.tail = tail
        self.n_interp = n_interp

        # Device type of the tensor in this class
        self._device = xx.device

        aa, bb, cc, dd = kwargs.get("abcd") if "abcd" in kwargs.keys()\
            else CubicSpline._get_abcd(self.xp, self.yp)

        self.abcd = pack([aa, bb, cc, dd])

    def forward(self, xnew: Tensor):
        """Evaluate the polynomial linear or cubic spline.

        Arguments:
            xnew: Points to be interpolated.

        Returns:
            ynew: Interpolation values with given points.

        """
        # boundary condition of xnew
        assert xnew.ge(self.xp[0]).all(),\
            f'input should not be less than {self.xp[0]}'
        n_grid_point = len(self.xp)

        result = (
            torch.zeros(xnew.shape, device=self._device)
            if self.yp.dim() == 1
            else torch.zeros(xnew.shape[0], self.yp.shape[0],
                             device=self._device)
        )

        # get the nearest grid point index of distance in grid points
        ind = self.__get_nearest_grid_point_indices(xnew)

        # interpolation of xx which not in the tail
        if (ind <= n_grid_point).any():
            _mask = torch.logical_and(ind <= n_grid_point, ind != 0)
            result[_mask] = self._cubic(xnew[_mask], ind[_mask] - 1)

        # Adjust the results to account for tail interpolation, if required
        r_max = self.xp[-2] + self.tail
        max_ind = n_grid_point - 1 + int(self.tail / self.grid_step)
        is_tail = ind.masked_fill(ind.ge(n_grid_point) * ind.le(max_ind), -1).eq(-1)

        if is_tail.any():
            self.__compute_tail_interpolation(xnew, is_tail, r_max, result)


        return result

    def __get_nearest_grid_point_indices(self, xnew: Tensor):
        n_tail = int((self.tail / self.grid_step).round())
        xx_ext = torch.linspace(
            self.xp[0], self.xp[-1] + self.tail,
            len(self.xp) + n_tail, device=self._device)
        return torch.searchsorted(xx_ext.detach(), xnew)

    def __compute_tail_interpolation(self, xnew: Tensor, is_tail, r_max, result):
        dr = xnew[is_tail] - r_max
        dr = dr.unsqueeze(-1) if self.yp.dim() == 2 else dr
        ilast = len(self.xp)

        # get grid points and grid point values
        xa = ((ilast
              - self.n_interp
              + torch.arange(self.n_interp, device=self._device) - 1)
              * self.grid_step + self.xp[0])

        yb = bT(self.yp[..., ilast - self.n_interp - 1: ilast - 1])
        xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)
        yb = yb.unsqueeze(0).repeat_interleave(dr.shape[0], dim=0)

        # get derivative
        y0 = poly_interp(xa, yb, xa[:, self.n_interp - 1] - self.delta_r)
        y2 = poly_interp(xa, yb, xa[:, self.n_interp - 1] + self.delta_r)
        y1 = self.yp[..., ilast - 2]
        y1p = (y2 - y0) / (2.0 * self.delta_r)
        y1pp = (y2 + y0 - 2.0 * y1) / (self.delta_r * self.delta_r)
        # dr = dr.repeat(self.yp.shape[0], 1).T if self.yp.dim() == 2 else dr

        result[is_tail] = poly_to_zero(
            dr, -1.0 * self.tail, -1.0 / self.tail, y1, y1p, y1pp)

    def _cubic(self, xnew: Tensor, ind: Tensor):
        """Calculate cubic spline interpolation."""
        dx = xnew - self.xp[ind]
        aa, bb, cc, dd = self.abcd
        interp = (
            aa[..., ind]
            + bb[..., ind] * dx
            + cc[..., ind] * dx ** 2
            + dd[..., ind] * dx ** 3
        )

        return interp.transpose(-1, 0) if interp.dim() > 1 else interp

    @staticmethod
    def _get_abcd(xp, yp) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get aa, bb, cc, dd parameters for cubic spline interpolation.

        Arguments:
            xp: Grid points for interpolation, 1D Tensor.
            yp: Values to be interpolated at each grid point.

        Returns:
            parameters: a tuple storing the aa, bb, cc, and dd parameters.

        """
        # get the first dim of x
        nx = xp.shape[0]
        device = xp.device
        assert nx > 3  # the length of x variable must > 3

        # get the difference between grid points
        dxp = xp[1:] - xp[:-1]
        dyp = yp[..., 1:] - yp[..., :-1]

        # get b, c, d from reference website: step 3~9, first calculate c
        A = torch.zeros(nx, nx, device=device)
        A.diagonal()[1:-1] = 2 * (dxp[:-1] + dxp[1:])  # diag
        A[torch.arange(nx - 1), torch.arange(nx - 1) + 1] = dxp  # off-diag
        A[torch.arange(nx - 1) + 1, torch.arange(nx - 1)] = dxp
        A[0, 0], A[-1, -1] = 1.0, 1.0
        A[0, 1], A[1, 0] = 0.0, 0.0  # natural condition
        A[-1, -2], A[-2, -1] = 0.0, 0.0

        B = torch.zeros(*yp.shape, device=device)
        B[..., 1:-1] = 3 * (dyp[..., 1:] / dxp[1:] - dyp[..., :-1] / dxp[:-1])
        B = B.permute(1, 0) if B.dim() == 2 else B

        cc = torch.linalg.lstsq(A, B)[0]
        cc = cc.permute(1, 0) if cc.dim() == 2 else cc
        bb = dyp / dxp - dxp * (cc[..., 1:] + 2 * cc[..., :-1]) / 3
        dd = (cc[..., 1:] - cc[..., :-1]) / (3 * dxp)

        return yp, bb, cc, dd
