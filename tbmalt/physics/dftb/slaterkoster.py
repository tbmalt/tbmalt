# -*- coding: utf-8 -*-
"""Code associated with performing Slater-Koster transformations.

This houses the code responsible for constructing & applying the Slater-Koster
transformations that rotate matrix elements from the reference frame of the
parametrisation set, [0,0,1], into that required by the calculation.
"""
import numpy as np
import torch
from torch import Tensor, stack


# Static module-level constants (used for SK transformation operations)
_SQR3, _SQR6, _SQR10, _SQR15 = np.sqrt(np.array([3., 6., 10., 15.])).tolist()
_HSQR3 = 0.5 * np.sqrt(3.)


def sub_block_rot(l_pair: Tensor, u_vec: Tensor,
                  integrals: Tensor) -> Tensor:
    """Diatomic sub-block rotated into the reference frame of the system.

    This takes the unit distance vector and Slater-Koster integrals between a
    pair of orbitals and constructs the associated diatomic block which has
    been rotated into the reference frame of the system.

    Args:
        l_pair: Azimuthal quantum numbers associated with the orbitals.
        u_vec: Unit distance vector between the atoms.
        integrals: Slater-Koster integrals between the orbitals, in order of
            σ, π, δ, γ, etc.

    Returns:
        block: Diatomic block(s)

    Examples:
        >>> import torch
        >>> from tbmalt.physics.dftb.slaterkoster import sub_block_rot

        # Slater-Koster transformation for single system
        >>> l_pair = torch.tensor([0, 0])  # s-s orbital
        >>> u_vecs = torch.tensor([0.7001, 0.6992, 0.1449])
        >>> integrals = torch.tensor([-0.1610])
        >>> pred = sub_block_rot(l_pair, u_vecs, integrals)
        >>> print(pred)
        tensor([[-0.1610]])

        # Slater-Koster transformation for batch system
        >>> l_pair = torch.tensor([0, 1])  # s-p orbital
        >>> u_vecs  = torch.tensor([
        [0.7001, 0.6992, 0.1449], [0.8199, 0.5235, 0.2317],
        [0.1408, 0.7220, 0.6774], [0.5945, 0.6138, 0.5195],
        [0.5749, 0.6112, 0.5440], [0.4976, 0.5884, 0.6374],
        [0.0075, 0.5677, 0.8232], [0.7632, 0.4264, 0.4856],
        [0.3923, 0.3178, 0.8632], [0.0278, 0.6896, 0.7237]])
        >>> integrals = torch.tensor([
        [-0.1011], [-0.1011], [-0.1011], [-0.1011], [-0.1011],
        [-0.1011], [-0.1011], [-0.1011], [-0.1011], [-0.1011]])
        >>> pred = sub_block_rot(l_pair, u_vecs, integrals)
        >>> print(pred)
        tensor([[[-0.0707, -0.0146, -0.0708]],
                [[-0.0529, -0.0234, -0.0829]],
                [[-0.0730, -0.0685, -0.0142]],
                [[-0.0621, -0.0525, -0.0601]],
                [[-0.0618, -0.0550, -0.0581]],
                [[-0.0595, -0.0644, -0.0503]],
                [[-0.0574, -0.0832, -0.0008]],
                [[-0.0431, -0.0491, -0.0772]],
                [[-0.0321, -0.0873, -0.0397]],
                [[-0.0697, -0.0732, -0.0028]]])

    """
    if u_vec.device != integrals.device:
        raise RuntimeError(  # Better to throw this exception manually
            f'Expected u_vec({u_vec.device}) & integrals({integrals.device}) '
            'to be on the same device!')

    # If smallest is ℓ first the matrix multiplication complexity is reduced
    l1, l2 = int(min(l_pair)), int(max(l_pair))

    # Tensor in which to place the results.
    block = torch.zeros(len(u_vec) if u_vec.ndim > 1 else 1,
                        2 * l1 + 1, 2 * l2 + 1, device=integrals.device)

    # Integral matrix block (in the reference frame of the parameter set)
    i_mat = sub_block_ref(l_pair.sort()[0], integrals)

    # Identify which vectors must use yz type rotations & which must use xy.
    rot_mask = torch.gt(u_vec[..., -2].abs(), u_vec[..., -1].abs())

    # Perform transformation operation (must do yz & xy type rotations)
    for rots, mask in zip((_sk_yz_rots, _sk_xy_rots), (rot_mask, ~rot_mask)):
        if len(u_vec_selected := u_vec[mask].squeeze()) > 0:
            rot_a = rots[l1](u_vec_selected)
            rot_b = rots[l2](u_vec_selected)
            block[mask] = torch.einsum(
                '...ji,...ik,...ck->...jc', rot_a, i_mat[mask], rot_b)

    # The masking operation converts single instances into batches of size 1,
    # therefore a catch is added to undo this.
    if u_vec.dim() == 1:
        block = block.squeeze(1)

    # Transpose if ℓ₁>ℓ₂ and flip the sign as needed.
    if l_pair[0] > l_pair[1]:
        sign = (-1) ** (l1 + l2)
        block = sign * block.transpose(-1, -2)

    return block


def sub_block_ref(l_pair: Tensor, integrals: Tensor):
    """Diatomic sub-block in the Slater-Koster integrals' reference frame.

    This yields the tensor that is multiplied with the transformation matrices
    to produce the diatomic sub-block in the atomic reference frame.

    Args:
        l_pair: Angular momenta of the two systems.
        integrals: Slater-Koster integrals between orbitals with the specified
            angular momenta, in order of σ, π, δ, γ, etc.

    Returns:
        block: Diatomic sub-block in the reference frame of the integrals.

    Notes:
        Each row of ``integrals`` should represent a separate system; i.e.,
        a 3x1 matrix would indicate a batch of size 3, each with one integral.
        Whereas a matrix of size 1x3 or a vector of size 3 would indicate one
        system with three integral values.
    """
    l1, l2 = min(l_pair), max(l_pair)

    # Test for anticipated number of integrals to ensure `integrals` is in the
    # correct shape.
    if (m := integrals.shape[-1]) != (n := l1 + 1):
        raise ValueError(
            f'Expected {n} integrals per-system (l_min={l1}), but found {m}')

    # Generate integral reference frame block; extending its dimensionality if
    # working on multiple systems.
    block = torch.zeros(2 * l1 + 1, 2 * l2 + 1, device=integrals.device)
    if integrals.dim() == 2:
        block = block.expand(len(integrals), -1, -1).clone()

    # Fetch the block's diagonal and assign the integrals to it like so
    #      ┌               ┐
    #      │ i_1, 0.0, 0.0 │  Where i_0 and i_1 are the first and second
    #      │ 0.0, i_0, 0.0 │  integral values respectively.
    #      │ 0.0, 0.0, i_1 │
    #      └               ┘
    # While this method is a little messy it is faster than alternate methods
    diag = block.diagonal(offset=l2 - l1, dim1=-2, dim2=-1)
    size = integrals.shape[-1]
    diag[..., -size:] = integrals
    diag[..., :size - 1] = integrals[..., 1:].flip(-1)
    # Return the results; a transpose s required if l1 > l2
    return block if l1 == l_pair[0] else block.transpose(-1, -2)


#################################
# Slater-Koster Transformations #
#################################
# Note that the internal Slater-Koster transformation functions "_skt_*" are
# able to handle batches of systems, not just one system at a time.
def _rot_yz_s(unit_vector: Tensor) -> Tensor:
    r"""s-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating s-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Notes:
        This function acts as a dummy subroutine, as integrals do not
        require transformation operations. This exists primarily to maintain
        functional consistency.
    """
    # Using `norm()` rather than `ones()` allows for backpropagation
    return torch.linalg.norm(unit_vector, dim=-1).view(
        (-1, *[1]*unit_vector.ndim))


def _rot_xy_s(unit_vector: Tensor) -> Tensor:
    r"""s-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating s-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Notes:
        This function acts as a dummy subroutine, as integrals do not
        require transformation operations. This exists primarily to maintain
        functional consistency.
    """
    # Using `norm()` rather than `ones()` allows for backpropagation
    return torch.linalg.norm(unit_vector, dim=-1).view(
        (-1, *[1] * unit_vector.ndim))


def _rot_yz_p(unit_vector: Tensor) -> Tensor:
    r"""p-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating p-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill-defined
        when z≈1.
    """
    x, y, z = unit_vector.T
    zeros = torch.zeros_like(x)
    alpha = torch.sqrt(1.0 - z * z)
    rot = stack([
        stack([x / alpha, zeros, -y / alpha], -1),
        stack([y, z, x], -1),
        stack([y * z / alpha, -alpha, x * z / alpha], -1)], -1)
    return rot


def _rot_xy_p(unit_vector: Tensor) -> Tensor:
    r"""p-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating p-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill-defined
        when y≈1.
    """
    x, y, z = unit_vector.T
    zeros = torch.zeros_like(x)
    alpha = torch.sqrt(1.0 - y * y)
    rot = stack([
        stack([alpha, -y * z / alpha, -x * y / alpha], -1),
        stack([y, z, x], -1),
        stack([zeros, -x / alpha, z / alpha], -1)], -1)
    return rot


def _rot_yz_d(unit_vector: Tensor) -> Tensor:
    r"""d-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating d-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill-defined
        when z≈1.
    """
    x, y, z = unit_vector.T
    zeros = torch.zeros_like(x)
    a = 1.0 - z * z
    b = torch.sqrt(a)
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    x2 = x * x
    rot = stack([
        stack([-z + 2.0 * x2 * z / a, -x, zeros, y, -2.0 * xyz / a], -1),
        stack([-b + 2.0 * x2 / b, xz / b, zeros, -yz / b, -2.0 * xy / b], -1),
        stack([xy * _SQR3, yz * _SQR3, 1.0 - 1.5 * a, xz * _SQR3,
               _SQR3 * (-0.5 * a + x2)], -1),
        stack([2.0 * xyz / b, -2.0 * y * b + y / b, -_SQR3 * z * b,
               -2.0 * x * b + x / b, -z * b + 2.0 * x2 * z / b], -1),
        stack([-xy + 2.0 * xy / a, -yz, 0.5 * _SQR3 * a, -xz,
               0.5 * a - 1.0 + x2 * (-1.0 + 2.0 / a)], -1)
    ], -1)
    return rot


def _rot_xy_d(unit_vector: Tensor) -> Tensor:
    r"""d-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating d-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill-defined
        when y≈1.
    """
    x, y, z = unit_vector.T
    a = 1.0 - y * y
    b = torch.sqrt(a)
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    z2 = z * z
    rot = stack([
        stack([z, -x, xyz * _SQR3 / a, y * (1 - 2 * z2 / a), -xyz / a], -1),
        stack([x * (2 * b - 1 / b), z * (2 * b - 1.0 / b),
               -y * z2 * _SQR3 / b, -2 * xyz / b, y * (-2 * b + z2 / b)], -1),
        stack([xy * _SQR3, yz * _SQR3, 1.5 * z2 - 0.5, xz * _SQR3,
               0.5 * _SQR3 * (2 * a - z2 - 1)], -1),
        stack([yz / b, -xy / b, -xz * _SQR3 / b, -b + 2 * z2 / b, xz / b], -1),
        stack([xy, yz, _SQR3 * (0.5 * (z2 + 1) - z2 / a),
               xz - 2 * xz / a, a - 0.5 * z2 - 0.5 + z2 / a], -1)
    ], -1)
    return rot


def _rot_yz_f(unit_vector: Tensor) -> Tensor:
    r"""f-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating f-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill-defined
        when z≈1.
    """
    x, y, z = unit_vector.T
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    zeros = torch.zeros_like(x)
    a = 1.0 - z * z
    b = torch.sqrt(a)
    c = b ** 3
    x2 = x * x
    rot = stack([
        stack([
            x * (2.25 * b - 3 * (x2 + 1) / b + 4 * x2 / c),
            _SQR6 * z * (0.5 * b - x2 / b),
            0.25 * _SQR15 * x * b,
            zeros,
            -0.25 * _SQR15 * y * b,
            _SQR6 * xyz / b,
            y * (-0.75 * b + 0.25 * (12 * x2 + 4) / b - 4 * x2 / c)
        ], -1),
        stack([
            _SQR6 * xz * (-1.5 + 2 * x2 / a),
            2 * a - 1 + x2 * (-4 + 2 / a),
            -0.5 * _SQR10 * x * z, zeros,
            0.5 * _SQR10 * y * z,
            xy * (4 - 2 / a),
            _SQR6 * yz * (0.5 - 2 * x2 / a)
        ], -1),
        stack([
            _SQR15 * x * (-0.75 * b + x2 / b),
            _SQR10 * z * (-0.5 * b + x2 / b),
            x * (-1.25 * b + 1 / b),
            zeros,
            y * (1.25 * b - 1 / b),
            -_SQR10 * xyz / b,
            _SQR15 * y * (0.25 * b - x2 / b)
        ], -1),
        stack([
            y * _SQR10 * (-0.25 * a + x2),
            xyz * _SQR15,
            _SQR6 * y * (-1.25 * a + 1),
            z * (-2.5 * a + 1),
            _SQR6 * x * (-1.25 * a + 1),
            _SQR15 * z * (-0.5 * a + x2),
            _SQR10 * x * (-0.75 * a + x2)
        ], -1),
        stack([
            _SQR15 * yz * (-0.25 * b + x2 / b),
            _SQR10 * xy * (-1.5 * b + 1 / b),
            yz * (-3.75 * b + 1 / b),
            _SQR6 * (1.25 * c - b),
            xz * (-3.75 * b + 1 / b),
            _SQR10 * (0.75 * c - 0.25 * (6.0 * x2 + 2) * b + x2 / b),
            _SQR15 * xz * (-0.75 * b + x2 / b)
        ], -1),
        stack([
            _SQR6 * y * (0.25 * a - 0.25 * (4 * x2 + 2) + 2 * x2 / a),
            xyz * (-3 + 2 / a),
            _SQR10 * y * (0.75 * a - 0.5),
            0.5 * _SQR15 * a * z,
            _SQR10 * x * (0.75 * a - 0.5),
            z * (1.5 * a - 0.5 * (6.0 * x2 + 2) + 2 * x2 / a),
            _SQR6 * x * (0.75 * a - 0.25 * (4 * x2 + 6.0) + 2 * x2 / a)
        ], -1),
        stack([
            yz * (0.25 * b - (x2 + 1) / b + 4 * x2 / c),
            _SQR6 * xy * (0.5 * b - 1 / b),
            0.25 * _SQR15 * yz * b,
            -0.25 * _SQR10 * c,
            0.25 * _SQR15 * xz * b,
            _SQR6 * (-0.25 * c + 0.25 * (2 * x2 + 2) * b - x2 / b),
            xz * (0.75 * b - 0.25 * (4 * x2 + 12) / b + 4 * x2 / c)
        ], -1)
    ], -1)
    return rot


def _rot_xy_f(unit_vector: Tensor) -> Tensor:
    r"""f-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating f-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produced simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill-defined
        when y≈1.
    """
    x, y, z = unit_vector.T
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    a = 1.0 - y * y
    b = torch.sqrt(a)
    c = b ** 3
    z2 = z * z
    rot = stack([
        stack([
            c + (-0.75 * z2 - 0.75) * b + 1.5 * z2 / b,
            _SQR6 * xz * (0.5 * b - 1 / b),
            _SQR15 * (0.25 * (z2 + 1) * b - 0.5 * z2 / b),
            _SQR10 * yz * (-0.25 * (z2 + 3) / b + z2 / c),
            _SQR15 * xy * (-0.25 * (z2 + 1) / b + z2 / c),
            _SQR6 * yz * (-0.5 * b + (0.25 * z2 + 0.75) / b - z2 / c),
            xy * (-b + (0.25 * z2 + 0.25) / b - z2 / c)
        ], -1),
        stack([
            _SQR6 * xz * (1 - 0.5 / a),
            -2 * a + 4 * z2 + 1 - 2 * z2 / a,
            _SQR10 * xz * (-1 + 0.5 / a),
            _SQR15 * xy * z2 / a,
            _SQR10 * yz * (1 - 1.5 * z2 / a),
            xy * (2 - 3 * z2 / a),
            _SQR6 * yz * (-1 + 0.5 * z2 / a)
        ], -1),
        stack([
            _SQR15 * (c - 0.75 * (z2 + 1) * b + 0.5 * z2 / b),
            _SQR10 * xz * (1.5 * b - 1 / b),
            (3.75 * z2 - 0.25) * b - 2.5 * z2 / b,
            -0.25 * _SQR6 * yz * (5 * z2 - 1) / b,
            -0.25 * xy * (15 * z2 - 1) / b,
            _SQR10 * yz * (-1.5 * b + (0.75 * z2 + 0.25) / b),
            _SQR15 * xy * (-b + (0.25 * z2 + 0.25) / b)
        ], -1),
        stack([
            _SQR10 * y * (a - 0.75 * z2 - 0.25),
            _SQR15 * xy * z,
            0.25 * _SQR6 * (5 * z2 - 1) * y,
            z * (2.5 * z2 - 1.5),
            _SQR6 * x * (1.25 * z2 - 0.25),
            _SQR15 * z * (a - 0.5 * z2 - 0.5),
            _SQR10 * x * (a - 0.25 * z2 - 0.75)
        ], -1),
        stack([
            0.5 * _SQR15 * xyz / b,
            _SQR10 * y * (-0.5 * b + z2 / b),
            -2.5 * xyz / b,
            -0.25 * _SQR6 * x * (5 * z2 - 1) / b,
            z * (-2.5 * b - 0.25 * (-15 * z2 + 1) / b),
            _SQR10 * x * (-0.5 * b + (0.75 * z2 + 0.25) / b),
            _SQR15 * z * (0.5 * b - (0.25 * z2 + 0.25) / b)
        ], -1),
        stack([
            _SQR6 * y * (a - 0.25 * (3 * z2 + 1) + 0.5 * z2 / a),
            xyz * (3 - 2 / a),
            _SQR10 * y * (0.25 * (3 * z2 + 1) - 0.5 * z2 / a),
            _SQR15 * z * (0.5 * (z2 + 1) - z2 / a),
            _SQR10 * x * (0.75 * z2 + 0.25 - 1.5 * z2 / a),
            z * (3 * a - 1.5 * z2 - 3.5 + 3 * z2 / a),
            _SQR6 * x * (a - 0.25 * z2 - 0.75 + 0.5 * z2 / a)
        ], -1),
        stack([
            1.5 * xyz / b,
            _SQR6 * y * (-0.5 * b + z2 / b),
            -0.5 * _SQR15 * xyz / b,
            _SQR10 * x * (-0.25 * (3 * z2 + 1) / b + z2 / c),
            _SQR15 * z * (-0.5 * b + (0.75 * z2 + 0.75) / b - z2 / c),
            _SQR6 * x * (-0.5 * b + (0.75 * z2 + 0.25) / b - z2 / c),
            z * (1.5 * b - (0.75 * z2 + 0.75) / b + z2 / c)
        ], -1)
    ], -1)

    return rot


_sk_yz_rots = {0: _rot_yz_s, 1: _rot_yz_p, 2: _rot_yz_d, 3: _rot_yz_f}
_sk_xy_rots = {0: _rot_xy_s, 1: _rot_xy_p, 2: _rot_xy_d, 3: _rot_xy_f}
