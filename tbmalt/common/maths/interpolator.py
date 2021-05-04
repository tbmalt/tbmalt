"""Interpolation for general purpose."""
import torch
Tensor = torch.Tensor


class PolyInterpU:
    """Polynomial interpolation method with uniform grid points.

    Arguments:
        xx: Grid points of distances, 1D Tensor.
        yy: Slater-Koste Integral tables.

    Notes:
        The `PolyInterpU` class is originally from DFTB+ polynomial
        interpolation and assumes a uniform grid. The xx is grid points and
        yy is values of the polynomial. If used in Slater-Koster integrals
        interpolation, yy should be integrals corresponding to each grid
        points. For the grid points at the tail of grid points, a s
        mmoth-to-zero method is applied.

    """

    def __init__(self, xx: Tensor, yy: Tensor, **kwargs):
        self.xx, self.yy = self._check(xx, yy, **kwargs)

    def _check(self, xx: Tensor, yy: Tensor, **kwargs):
        """Check input parameters."""
        self.ninterp: int = kwargs.get('n_interpolation', 8)

        self.incr = xx[1] - xx[0]
        self.ngridpoint = len(xx)  # -> number of grid points

        # Check if xx is uniform
        _incr = xx[1:] - xx[:-1]
        assert torch.allclose(_incr, torch.ones(*_incr.shape) * self.incr)

        # Input size of SKF must larger than ninterp
        if self.ngridpoint < self.ninterp:
            raise ValueError("Not enough grid points for interpolation!")

        return xx, yy


    def __call__(self, rr: Tensor, ninterp=8, delta_r=1E-5, tail=1) -> Tensor:
        """Interpolation SKF according to distance from integral tables.

        Arguments:
            rr: interpolation points for batch.
            ninterp: Number of total interpolation grid points.
            delta_r: Delta distance for 1st, 2nd derivative.
            tail: Distance to smooth the tail, unit is bohr.

        Returns:
            result: Interpolation values with given rr.

        """
        rmax = (self.ngridpoint - 1) * self.incr + tail
        ind = torch.floor(rr / self.incr).int()
        result = torch.zeros(*rr.shape)

        # => polynomial fit
        if (ind <= self.ngridpoint).any():
            _mask = ind <= self.ngridpoint

            # get the index of rr in grid points
            ind_last = (ind[_mask] + ninterp / 2 + 1).int()
            ind_last[ind_last > self.ngridpoint] = self.ngridpoint
            ind_last[ind_last < ninterp + 1] = ninterp + 1

            # gather xx and yy for both single and batch
            xa = (ind_last.unsqueeze(1) - ninterp + torch.arange(ninterp)
                  ) * self.incr  # get the interpolation gird points
            yb = torch.stack([self.yy[ii - ninterp - 1: ii - 1]
                              for ii in ind_last])
            result[_mask] = poly_interp(xa, yb, rr[_mask])

        # Beyond the grid => extrapolation with polynomial of 5th order
        max_ind = self.ngridpoint - 1 + int(tail / self.incr)
        is_tail = ind.masked_fill(ind.ge(self.ngridpoint) * ind.le(max_ind), -1).eq(-1)
        if is_tail.any():
            dr = rr[is_tail] - rmax
            ilast = self.ngridpoint

            # get grid points and grid point values
            xa = (ilast - ninterp + torch.arange(ninterp)) * self.incr
            yb = self.yy[ilast - ninterp - 1: ilast - 1]
            xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)
            yb = yb.unsqueeze(0).repeat_interleave(dr.shape[0], dim=0)

            # get derivative
            y0 = poly_interp(xa, yb, xa[:, ninterp - 1] - delta_r)
            y2 = poly_interp(xa, yb, xa[:, ninterp - 1] + delta_r)
            y1 = self.yy[ilast - 2]
            y1p = (y2 - y0) / (2.0 * delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)

            result[is_tail] = poly_to_zero(dr, -1.0 * tail, y1, y1p, y1pp)

        return result


def poly_to_zero(xx: Tensor, dx: Tensor,
               y0: Tensor, y0p: Tensor, y0pp: Tensor) -> Tensor:
    """Get integrals if beyond the grid range with 5th order polynomial.

    Arguments:
        y0: Values of interpolation grid point values.
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
    xr = xx / dx
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
    assert xp.dim() == 2
    nn0, nn1 = xp.shape[0], xp.shape[1]
    index_nn0 = torch.arange(nn0)
    icl = torch.zeros(nn0).long()
    cc, dd = yp.clone(), yp.clone()
    dxp = abs(rr - xp[index_nn0, icl])

    # find the most close point to rr (single atom pair or multi pairs)
    _mask, ii = torch.zeros(len(rr)) == 0.0, 0.0
    _dx_new = abs(rr - xp[index_nn0, 0])
    while (_dx_new < dxp).any():
        ii += 1
        assert ii < nn1 - 1  # index ii range from 0 to nn1 - 1
        _mask = _dx_new < dxp
        icl[_mask] = ii
        dxp[_mask] = abs(rr - xp[index_nn0, ii])[_mask]

    yy = yp[index_nn0, icl]

    for mm in range(nn1 - 1):
        for ii in range(nn1 - mm - 1):
            rtmp0 = xp[index_nn0, ii] - xp[index_nn0, ii + mm + 1]

            # use transpose to realize div: (N, M, K) / (N)
            rtmp1 = ((cc[index_nn0, ii + 1] - dd[index_nn0, ii]).transpose(
                0, -1) / rtmp0).transpose(0, -1)
            cc[index_nn0, ii] = ((xp[index_nn0, ii] - rr) *
                                 rtmp1.transpose(0, -1)).transpose(0, -1)
            dd[index_nn0, ii] = ((xp[index_nn0, ii + mm + 1] - rr) *
                                 rtmp1.transpose(0, -1)).transpose(0, -1)
        if (2 * icl < nn1 - mm - 1).any():
            _mask = 2 * icl < nn1 - mm - 1
            yy[_mask] = (yy + cc[index_nn0, icl])[_mask]
        else:
            _mask = 2 * icl >= nn1 - mm - 1
            yy[_mask] = (yy + dd[index_nn0, icl - 1])[_mask]
            icl[_mask] = icl[_mask] - 1

    return yy
