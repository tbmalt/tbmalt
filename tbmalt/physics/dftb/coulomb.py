# -*- coding: utf-8 -*-
"""Code associated with coulombic interactions.

This module calculate the Ewald summation for periodic
boundary conditions.
"""
import torch
import numpy as np
from typing import Optional, Tuple
from scipy import special
from tbmalt import Geometry, Periodic
from tbmalt.structures.cell import Pbc
from tbmalt.common.batch import pack
from abc import ABC
Tensor = torch.Tensor
_euler = 0.5772156649


class Coulomb:
    """Class to assist the calculation of coulomb interaction by ABC 'Ewald'.

    The 'Coulomb' class checks the type of periodic boundary condition and
    decides which subclass of ewald summation to use.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary conditoin.

    Keyword Arguments:
        tol_ewald: EWald tolerance.
        method: Method to obtain parameters of alpha and cutoff.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.

    Examples:
        >>> from tbmalt import Geometry, Periodic, Coulomb
        >>> import torch
        >>> cell = torch.tensor([[2., 0., 0.], [0., 4., 0.], [0., 0., 2.]])
        >>> pos = torch.tensor([[0., 0., 0.], [0., 2., 0.]])
        >>> num = torch.tensor([1, 1])
        >>> cutoff = torch.tensor([9.98])
        >>> system = Geometry(num, pos, cell, units='a')
        >>> periodic = Periodic(system, system.cells, cutoff=cutoff)
        >>> coulomb = Coulomb(system, periodic, method='search')
        >>> print(coulomb.invrmat)
        tensor([[-0.4778, -0.2729],
                [-0.2729, -0.4778]])

    """

    def __init__(self, geometry: Geometry, periodic: Periodic, **kwargs):
        self.geometry: Geometry = geometry
        self.periodic: Periodic = periodic

        # Check the type of pbc and choose corresponding subclass
        if self.geometry.pbc.ndim == 1:  # -> Single
            _sum_dim = self.geometry.pbc.sum(dim=-1)
        else:  # -> Batch
            _sum_dim = self.geometry.pbc[0].sum(dim=-1)

        if _sum_dim == 1:  # -> 1D pbc
            coulomb = Ewald1d(self.geometry, self.periodic, **kwargs)
        elif _sum_dim == 2:  # -> 2D pbc
            coulomb = Ewald2d(self.geometry, self.periodic, **kwargs)
        elif _sum_dim == 3:  # -> 3D pbc
            coulomb = Ewald3d(self.geometry, self.periodic, **kwargs)

        self.invrmat: Tensor = coulomb.invrmat


class Ewald(ABC):
    """ABC for calculating the coulombic interaction in periodic geometry.

    'Ewald' class calculates the long range coulombic interaction using Ewald
    summation, which consists of three different parts, i.e. two rapidly
    converging terms (real space sum, reciprocal space sum) and a
    self-correction term. Formulas for different periodic boundary conditions
    are given in the coresponding references.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary conditoin.
        param: Parameter used for calculation. Cell volumn for 3D pbc,
            cell length for 1D & 2D pbc.

    Keyword Arguments:
        tol_ewald: Tolerance of ewald summation.
        method: Method to obtain parameters of alpha, maxr and maxg.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.
        alpha: Splitting parameter for the Ewald summation
        maxr: The longest real space vector that gives a bigger
            contribution to the EWald sum than tolerance.
        maxg: The longest reciprocal vector that gives a bigger
            contribution to the EWald sum than tolerance.

    Notes:
        There are two available methods to generate adjustable parameters for
        Ewald summation. The default method uses empirical formulas from
        experience to obtain alpha, maxr and maxg. While these parameters can
        also be obtained by searching, which give exactly the same results in
        DFTB+ for 3d pbc.

    Warning:
        The result of 1D ewald summation is sensitive to the selection of
        splitting parameter alpha. Using the default method rather than
        searching can achieve the convergence. Besides, mixing of different pbc
        is not supported.

    References:
        [1]: Journal of Computational Physics 285 (2015): 280-315.
        [2]: The Journal of chemical physics 136.16 (2012): 164111.
        [3]: Advances in Computational Mathematics 42.1 (2016): 227-248.
        [4]: Chemical physics letters 340.1-2 (2001): 157-164.

    """

    def __init__(self, geometry: Geometry, periodic: Periodic, param: Tensor,
                 **kwargs):

        # Read input geometry
        self.geometry: Geometry = geometry
        self.periodic: Periodic = periodic
        self.latvec: Tensor = self.geometry.cells
        self.natom: Tensor = self.geometry.n_atoms
        self.coord: Tensor = self.geometry.positions
        self.recvec: Tensor = self.periodic.reciprocal_lattice

        self._device = self.latvec.device
        self._dtype = self.latvec.dtype

        # Number of batches if in batch mode (for internal use only)
        self._n_batch: Optional[int] = (None if self.latvec.dim() == 2
                                        else len(self.latvec))

        # Parameter used for calculation
        self.param: Tensor = param if self._n_batch else param.unsqueeze(0)

        # Tolerance of ewald summation
        self.tol_ewald: Tensor = kwargs.get(
            'tol_ewald', torch.tensor(1e-9, device=self._device,
                                      dtype=self._dtype))

        # Method to obtain parameters for calculation
        self.method: str = kwargs.get('method', 'default')

        # Maximun number of iteration when searching alpha
        self.nsearchiter: int = kwargs.get('nsearchiter', 30)

        # Maximum number of atoms in geometry
        self._max_natoms: Tensor = torch.max(self.natom)

        # Default method to obtain parameters by empirical formulas
        if self.method == 'default':

            # Splitting parameter
            self.alpha: Tensor = self._default_alpha()

            ff = torch.sqrt(-torch.log(self.tol_ewald))

            # The longest real space vector
            self.maxr: Tensor = ff / self.alpha

            # The longest reciprocal vector
            self.maxg: Tensor = 2.0 * self.alpha * ff

        else:  # -> Obtain parameters by searching
            self.alpha: Tensor = self._get_alpha()
            self.maxr: Tensor = self._get_maxr()
            self.maxg: Tensor = self._get_maxg()

        # The updated lattice points
        self.rcellvec_ud, self.ncell_ud = self._update_latvec()

        # The updated neighbour lists
        self.distmat, self.neighbour = self._update_neighbour()

        # Real part of the Ewald summation
        self.ewald_r, self.mask_g = self._invr_periodic_real()

        # Reciprocal part of the Ewald summation
        self.ewald_g: Tensor = self._invr_periodic_reciprocal()

        # 1/R matrix for the periodic geometry
        self.invrmat: Tensor = self._invr_periodic()

    def _update_latvec(self) -> Tuple[Tensor, Tensor]:
        """Update the lattice points for reciprocal Ewald summation."""
        update = Periodic(self.geometry, self.recvec, cutoff=self.maxg,
                          distance_extention=0, positive_extention=0,
                          negative_extention=0)

        return update.rcellvec, update.ncell

    def _update_neighbour(self) -> Tuple[Tensor, Tensor]:
        """Update the neighbour lists for real Ewald summation."""
        update = Periodic(self.geometry, self.latvec, cutoff=self.maxr,
                          distance_extention=0)

        return update.periodic_distances, update.neighbour

    def _invr_periodic(self) -> Tensor:
        """Calculate the 1/R matrix for the periodic geometry."""
        # Extra contribution for self interaction
        if not self._n_batch:  # -> Single
            extra = torch.eye(self._max_natoms, device=self._device,
                              dtype=self._dtype) * 2.0 * self.alpha / np.sqrt(
                np.pi)
        else:  # -> Batch
            extra = torch.eye(self._max_natoms,
                              device=self._device, dtype=self._dtype
                              ).unsqueeze(0).repeat_interleave(
                self._n_batch, dim=0) * (2.0 * self.alpha / np.sqrt(np.pi)
                                         ).unsqueeze(-1).unsqueeze(-1)

        invr = self.ewald_r + self.ewald_g - extra
        invr[self.mask_g] = 0

        return invr

    def _invr_periodic_reciprocal(self) -> Tensor:
        """Calculate the reciprocal part of 1/R matrix."""
        # Lattice points for the reciprocal sum
        n_low = torch.ceil(torch.clone(self.ncell_ud / 2.0))

        # Single
        if not self._n_batch:
            gvec_tem = self.rcellvec_ud[int(n_low):]
            mask = torch.sum(torch.clone(gvec_tem) ** 2, -1) < self.maxg ** 2
            gvec = gvec_tem[mask]

            # Vectors for calculating the reciprocal Ewald sum
            rr = self.coord.repeat(self.natom, 1) -\
                self.coord.repeat(1, self.natom).view(-1, 3)

            # The reciprocal Ewald sum
            recsum = self._ewald_reciprocal_single(rr, gvec, self.alpha,
                                                   self.param)
            ewald_g = torch.reshape(
                recsum, (self._max_natoms, self._max_natoms))
            ewald_g[self.mask_g] = 0

        # Batch
        # Large values are padded in the end of short vectors
        else:
            gvec_tem = pack([torch.unsqueeze(self.rcellvec_ud[
                ibatch, int(n_low[ibatch]): int(2 * n_low[ibatch] - 1)], 0)
                for ibatch in range(self._n_batch)], value=1e3)
            dd2 = torch.sum(torch.clone(gvec_tem) ** 2, -1)
            mask = dd2 < self.maxg.unsqueeze(-1).unsqueeze(-1) ** 2
            gvec = pack([gvec_tem[ibatch, mask[ibatch]]
                         for ibatch in range(self._n_batch)], value=1e3)

            # Vectors for calculating the reciprocal Ewald sum
            rr = self.coord.repeat(1, self._max_natoms, 1) - self.coord.repeat(
                1, 1, self._max_natoms).view(self._n_batch, -1, 3)

            # The reciprocal Ewald sum
            recsum = self._ewald_reciprocal(rr, gvec, self.alpha, self.param)
            ewald_g = torch.reshape(recsum, (self._n_batch, self._max_natoms,
                                             self._max_natoms))
            ewald_g[self.mask_g] = 0

        return ewald_g

    def _invr_periodic_real(self) -> Tuple[Tensor, Tensor]:
        """Calculate the real part of 1/R matrix."""
        if not self._n_batch:  # -> Single
            ewaldr_tmp = self._ewald_real_single()
        else:  # -> Batch
            ewaldr_tmp = self._ewald_real()

        # Mask for summation
        mask = ewaldr_tmp < float('inf')
        mask_real = self.neighbour & mask
        ewaldr_tmp[~mask_real] = 0
        ewald_r = torch.sum(ewaldr_tmp, dim=-3)

        # Mask used for calculation of reciprocal part
        mask_g = ewald_r == 0

        return ewald_r, mask_g

    def _get_alpha(self) -> Tensor:
        """Get optimal alpha for the Ewald sum from searching."""
        # Mask for zero vector
        maskg = self.recvec.ne(0).any(-1)
        maskr = self.latvec.ne(0).any(-1)

        # Ewald parameter
        if not self._n_batch:  # -> Single
            alphainit = torch.tensor([1.0e-8], device=self._device,
                                     dtype=self._dtype)
            # Length of the shortest vector in reciprocal space
            min_g = torch.sqrt(torch.min(torch.sum(
                self.recvec[maskg] ** 2, -1), 0, keepdim=True).values)

            # Length of the shortest vector in real space
            min_r = torch.sqrt(torch.min(torch.sum(
                self.latvec[maskr] ** 2, -1), 0, keepdim=True).values)

        else:  # -> Batch
            alphainit = torch.tensor([1.0e-8],  device=self._device,
                                     dtype=self._dtype).repeat(self._n_batch)
            # Length of the shortest vector in reciprocal space
            min_g = torch.sqrt(torch.min(torch.sum(
                self.recvec[maskg].unsqueeze(
                    0).view(self._n_batch, -1, 3) ** 2, -1), 1).values)

            # Length of the shortest vector in real space
            min_r = torch.sqrt(torch.min(torch.sum(
                self.latvec[maskr].unsqueeze(
                    0).view(self._n_batch, -1, 3) ** 2, -1), 1).values)

        alpha = torch.clone(alphainit)

        # Difference between reciprocal and real parts of the decrease
        # of Ewald sum.
        diff = self._diff_rec_real(alpha, min_g, min_r, self.param)
        ierror = 0

        # Mask for batch calculation
        mask = diff < - self.tol_ewald

        # Loop to find the alpha
        while (alpha[mask] < float('inf')).all():
            alpha[mask] = 2.0 * alpha[mask]
            diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                             min_r[mask], self.param[mask])
            mask = diff < - self.tol_ewald
            if (~mask).all():
                break
        if torch.max(alpha >= float('inf')):
            ierror = 1
        elif torch.max(alpha == alphainit):
            ierror = 2

        if ierror == 0:
            alphaleft = 0.5 * alpha
            mask = diff < self.tol_ewald
            while (alpha[mask] < float('inf')).all():
                alpha[mask] = 2.0 * alpha[mask]
                diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                                 min_r[mask], self.param[mask])
                mask = diff < self.tol_ewald
                if (~mask).all():
                    break

        if torch.max(alpha >= float('inf')):
            ierror = 3

        if ierror == 0:
            alpharight = alpha
            alpha = (alphaleft + alpharight) / 2.0
            iiter = 0
            diff = self._diff_rec_real(alpha, min_g, min_r, self.param)
            mask = torch.abs(diff) > self.tol_ewald
            while iiter <= self.nsearchiter:
                mask_neg = diff < 0
                alphaleft[mask_neg] = alpha[mask_neg]
                alpharight[~mask_neg] = alpha[~mask_neg]
                alpha[mask] = (alphaleft[mask] + alpharight[mask]) / 2.0
                diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                                 min_r[mask], self.param[mask])
                mask = torch.abs(diff) > self.tol_ewald
                iiter += 1
                if (~mask).all():
                    break
            if iiter > self.nsearchiter:
                ierror = 4

        if ierror != 0:
            raise ValueError('Fail to get optimal alpha for Ewald sum.')

        return alpha

    def _get_maxg(self) -> Tensor:
        """Get the longest reciprocal vector that gives a bigger
        contribution to the EWald sum than tolerance."""
        ginit = torch.tensor([1.0e-8], device=self._device,
                             dtype=self._dtype).repeat(self.alpha.size(0))
        ierror = 0
        xx = torch.clone(ginit)
        yy = self._gterm(xx, self.alpha, self.param)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while (xx[mask] < float('inf')).all():
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self._gterm(
                xx[mask], self.alpha[mask], self.param[mask])
            mask = yy > self.tol_ewald
            if (~mask).all():
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == ginit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self._gterm(xleft, self.alpha, self.param)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while iiter <= self.nsearchiter:
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self._gterm(
                    xx[mask], self.alpha[mask], self.param[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all():
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')

        return xx

    def _get_maxr(self) -> Tensor:
        """Get the longest real space vector that gives a bigger
           contribution to the EWald sum than tolerance."""
        rinit = torch.tensor([1.0e-8], device=self._device,
                             dtype=self._dtype).repeat(self.alpha.size(0))
        ierror = 0
        xx = torch.clone(rinit)
        yy = self._rterm(xx, self.alpha)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while (xx[mask] < float('inf')).all():
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self._rterm(xx[mask], self.alpha[mask])
            mask = yy > self.tol_ewald
            if (~mask).all():
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == rinit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self._rterm(xleft, self.alpha)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while iiter <= self.nsearchiter:
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self._rterm(xx[mask], self.alpha[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all():
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')

        return xx

    def _ewald_real(self) -> Tensor:
        """Batch calculation of the Ewald sum in the real part for a certain
        vector length."""
        return torch.erfc(self.alpha.unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1) * self.distmat) / self.distmat

    def _ewald_real_single(self) -> Tensor:
        """Calculation of the Ewald sum in the real part for a certain
        vector length."""
        return torch.erfc(self.alpha * self.distmat) / self.distmat

    def _diff_rec_real(self, alpha: Tensor, min_g: Tensor,
                       min_r: Tensor, param: Tensor) -> Tensor:
        """Returns the difference between reciprocal and real parts of the
        decrease of Ewald sum."""
        return (self._gterm(4.0 * min_g, alpha, param) - self._gterm(
            5.0 * min_g, alpha, param)) - (self._rterm(2.0 * min_r, alpha) -
                                           self._rterm(3.0 * min_r, alpha))


class Ewald3d(Ewald):
    """Implement of ewald summation for 3d periodic boundary condition.

    Subclass of the 'Ewald' class, containing formulas for 3d pbc.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary conditoin.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.

    """

    def __init__(self, geometry: Geometry, periodic: Periodic, **kwargs):
        param = periodic.cellvol
        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self) -> Tensor:
        """Returns the default value of alpha."""
        return (self.natom / self.param ** 2) ** (1/6) * np.pi ** 0.5

    def _ewald_reciprocal(self, rr: Tensor, gvec: Tensor,
                          alpha: Tensor, vol: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))
        recsum = torch.sum((torch.exp(- g2 / (4.0 * alpha.unsqueeze(-1) ** 2))
                            / g2).unsqueeze(-2) * torch.cos(dot), -1)
        tem = 2.0 * recsum * 4.0 * np.pi / vol.unsqueeze(-1)

        return (tem - (np.pi / (self.param * self.alpha ** 2)
                       ).unsqueeze(-1)).unsqueeze(-2)

    def _ewald_reciprocal_single(self, rr: Tensor, gvec: Tensor,
                                 alpha: Tensor, vol: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))
        recsum = torch.sum((torch.exp(
            - g2 / (4.0 * alpha ** 2)) / g2) * torch.cos(dot), -1)

        return 2.0 * recsum * 4.0 * np.pi / vol - np.pi / (
            self.param * self.alpha ** 2)

    def _gterm(self, len_g: Tensor, alpha: Tensor, cellvol: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return 4.0 * np.pi * (torch.exp((-0.25 * len_g ** 2) / (alpha ** 2))
                              / (cellvol * len_g ** 2))

    def _rterm(self, len_r: Tensor, alpha: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r


class Ewald2d(Ewald):
    """Implement of ewald summation for 2D boundary condition.

    Subclass of the 'Ewald' class, containing formulas for 2d pbc.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary conditoin.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.

    """

    def __init__(self, geometry: Geometry, periodic: Periodic, **kwargs):
        self.length: Tensor = Pbc.get_cell_lengths(geometry.cells)

        # Get the minimal length of non-zero terms
        tem = torch.clone(self.length)
        tem[tem.eq(0)] = 1e5
        param = torch.min(tem, -1).values

        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self) -> Tensor:
        """Returns the default value of alpha."""
        return self.natom ** (1/6) / self.param * np.pi ** 0.5

    def _ewald_reciprocal(self, rr: Tensor, gvec: Tensor,
                          alpha: Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        # Mask of periodic directions
        mask_pd = self.periodic.latvec.ne(0).any(-1)

        # Index to describe non-periodic direction
        index_npd = torch.tensor([0, 1, 2]).repeat(self._n_batch, 1)[~mask_pd]

        # Lengths of lattice vectors of periodic directions
        length_pd = self.length[mask_pd].reshape(self._n_batch, 2)

        g2 = torch.sum(gvec ** 2, -1)
        gg = torch.sqrt(g2)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Vectors of the non-periodic direction for reciprocal sum. Different
        # dircetions can be specified in the batch.
        rr_npe = rr[torch.arange(self._n_batch), :, index_npd]

        # Reciprocal, L
        tem = gg.unsqueeze(-1).transpose(-1, -2) * rr_npe.unsqueeze(-1)

        aa = torch.exp(tem)
        tem2 = gg / (alpha.unsqueeze(-1) * 2.0)

        bb = tem2.unsqueeze(-2) + (alpha.unsqueeze(-1) * rr_npe
                                   ).unsqueeze(-1).repeat_interleave(
                                       tem2.size(-1), -1)

        cc = torch.exp(- tem)
        dd = tem2.unsqueeze(-2) - (alpha.unsqueeze(-1) * rr_npe
                                   ).unsqueeze(-1).repeat_interleave(
                                       tem2.size(-1), -1)

        yyt = aa * torch.erfc(bb) + cc * torch.erfc(dd)
        yy = yyt / gg.unsqueeze(-2)

        # Replace nan values
        yy[yy != yy] = 0
        recl = torch.sum(torch.cos(dot) * yy, -1) * 2.0 * np.pi / (
            length_pd[..., 0] * length_pd[..., 1]).unsqueeze(-1)

        # Reciprocal, 0
        tem3 = torch.exp(- alpha.unsqueeze(-1) ** 2 * rr_npe ** 2
                         ) / alpha.unsqueeze(-1) +\
            (np.pi) ** 0.5 * rr_npe * torch.erf(alpha.unsqueeze(-1) * rr_npe)
        rec0 = tem3 * (- 2.0 * np.pi ** 0.5 / ((
            length_pd[..., 0] * length_pd[..., 1]).unsqueeze(-1)))

        return recl + rec0

    def _ewald_reciprocal_single(self, rr: Tensor, gvec: Tensor, alpha:
                                 Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        # Mask of periodic directions
        mask_pd = self.periodic.latvec.ne(0).any(-1)

        # Index to describe non-periodic direction
        index_npd = torch.tensor([0, 1, 2])[~mask_pd]

        # Lengths of lattice vectors of periodic directions
        length_pd = self.length[mask_pd]

        g2 = torch.sum(gvec ** 2, -1)
        gg = torch.sqrt(g2)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        tem = gg.unsqueeze(-1).transpose(-1, -2) * rr[..., index_npd]
        aa = torch.exp(tem)
        tem2 = gg / (alpha * 2)
        bb = tem2 + alpha * rr[..., index_npd].repeat_interleave(
            tem2.size(-1), -1)
        cc = torch.exp(- tem)
        dd = tem2 - alpha * rr[..., index_npd].repeat_interleave(
            tem2.size(-1), -1)
        yy = (aa * torch.erfc(bb) + cc * torch.erfc(dd)) / gg
        recl = torch.sum(torch.cos(dot) * yy, -1) * 2.0 * np.pi / (
            length_pd[0] * length_pd[1])

        # Reciprocal, 0
        tem3 = torch.exp(- alpha ** 2 * rr[..., index_npd[0]] ** 2) / alpha + \
            (np.pi) ** 0.5 * rr[..., index_npd[0]] * torch.erf(
                alpha * rr[..., index_npd[0]])
        rec0 = tem3 * (- 2.0 * np.pi ** 0.5 / (length_pd[0] * length_pd[1]))

        return recl + rec0

    def _gterm(self, len_g: Tensor, alpha: Tensor, length: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return (torch.erfc(len_g / (alpha * 2)) * 2
                ) / len_g * np.pi / length ** 2

    def _rterm(self, len_r: Tensor, alpha: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r


class Ewald1d(Ewald):
    """Implement of ewald summation for 1D boundary condition.

    Subclass of the 'Ewald' class, containing formulas for 1d pbc.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation
            vectors and neighbour list for periodic boundary conditoin.

    Attributes:
        invrmat: 1/R matrix for the periodic geometry.

    """

    def __init__(self, geometry: Geometry, periodic: Periodic, **kwargs):
        self.length: Tensor = Pbc.get_cell_lengths(geometry.cells)

        # Get the minimal length of non-zero terms
        tem = torch.clone(self.length)
        tem[tem.eq(0)] = 1e5
        param = torch.min(tem, -1).values

        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self) -> Tensor:
        """Returns the default value of alpha."""
        return self.natom ** (1/6) / self.param * np.pi ** 0.5

    def _ewald_reciprocal(self, rr: Tensor, gvec: Tensor,
                          alpha: Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        dd = {'dtype': self._dtype, 'device': self._device}

        # Mask of periodic direction
        mask_pd = self.periodic.latvec.ne(0).any(-1)

        # Index of non-periodic directions
        index_npd = torch.tensor([0, 1, 2]).repeat(
                self._n_batch, 1)[~mask_pd].reshape(self._n_batch, 2)

        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        aa = g2 / (4 * alpha.unsqueeze(-1) ** 2)

        bb = (rr[torch.arange(self._n_batch), :, index_npd[:, 0]] ** 2 + rr[
            torch.arange(self._n_batch), :, index_npd[:, 1]]
            ** 2) * alpha.unsqueeze(-1) ** 2

        # Numerical method to calculate integral value
        xx = torch.linspace(10.0 ** -20, 1.0, 5000,  **dd)
        kk0 = torch.tensor([[[torch.trapz(1.0 / xx * torch.exp(
            -iaa / xx - ibb * xx), xx)
                              for iaa in aa[ibatch]]
                             for ibb in bb[ibatch]]
                            for ibatch in range(self._n_batch)],
                            device=self._device, dtype=self._dtype)

        recl = torch.sum(torch.cos(dot) * kk0, -1) * 2.0 / length.unsqueeze(-1)

        # Reciprocal, 0
        rec0 = torch.zeros_like(bb)
        mask = bb != 0

        rec0[mask] = (- _euler - torch.log(bb[mask])
                      - special.exp1(bb[mask].cpu()).to(self._device)
                      ) / length.unsqueeze(
                          -1).repeat_interleave(bb.size(-1), -1)[mask]

        return recl + rec0

    def _ewald_reciprocal_single(self, rr: Tensor, gvec: Tensor,
                                 alpha: Tensor, length: Tensor) -> Tensor:
        """Calculate the reciprocal part of the Ewald sum."""
        dd = {'dtype': self._dtype, 'device': self._device}

        # Mask of the periodic direction
        mask_pd = self.periodic.latvec.ne(0).any(-1)

        # Index of non-periodic directions
        index_npd = torch.tensor([0, 1, 2])[~mask_pd]

        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        aa = g2 / (4 * alpha ** 2)
        bb = (rr[:, index_npd[0]] ** 2 + rr[:, index_npd[1]] ** 2) * alpha ** 2
        xx = torch.linspace(10.0 ** -20, 1.0, 5000, **dd)  # Numerical method
        kk0 = torch.tensor([[torch.trapz(1.0 / xx * torch.exp(
            - iaa / xx - ibb * xx), xx) for iaa in aa] for ibb in bb],
            device=self._device, dtype=self._dtype)
        recl = torch.sum(torch.cos(dot) * kk0, -1) * 2.0 / length

        # Reciprocal, 0
        rec0 = torch.zeros_like(bb)
        mask_bb = bb != 0
        rec0[mask_bb] = (- _euler - torch.log(bb[mask_bb])
                         - special.exp1(bb[mask_bb].cpu()).to(self._device
                                                              )) / length

        return recl + rec0

    def _gterm(self, len_g: Tensor, alpha: Tensor, length: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return special.exp1((len_g ** 2 / (4 * alpha ** 2)).cpu()).to(
            self._device) / length

    def _rterm(self, len_r: Tensor, alpha: Tensor) -> Tensor:
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r
