# -*- coding: utf-8 -*-
"""Code associated with performing Slater-Koster transformations.

This houses the code responsible for constructing & applying the Slater-Koster
transformations that rotate matrix elements from the reference frame of the
parametrisation set, [0,0,1], into that required by the calculation.

Hamiltonian & overlap matrices can be constructed via calls to `sk_hs_matrix`.
"""
import numpy as np
import torch
from torch.nn.functional import normalize
from torch import Tensor, stack
from tbmalt import Geometry, Basis
from tbmalt.common import split_by_size
from tbmalt.common.batch import pack
from tbmalt.ml.skfeeds import SkFeed

# Static module-level constants (used for SK transformation operations)
_SQR3, _SQR6, _SQR10, _SQR15 = np.sqrt(np.array([3., 6., 10., 15.])).tolist()
_HSQR3 = 0.5 * np.sqrt(3.)


def hs_matrix(geometry: Geometry, basis: Basis, sk_feed: SkFeed,
              **kwargs) -> Tensor:
    """Build the Hamiltonian or overlap matrix via Slater-Koster transforms.

    Constructs the Hamiltonian or overlap matrix for the target system(s)
    through the application of Slater-Koster transformations to the integrals
    provided by the ``sk_feed``.

    Arguments:
        geometry: `Geometry` instance associated with the target system(s).
        basis: `Basis` instance associated with the target system(s).
        sk_feed: The Slater-Koster feed entity responsible for providing the
            requisite Slater Koster integrals and on-site terms.

    Keyword Arguments:
        kwargs: `kwargs` are passed into all calls made to the ``sk_feed``
            object's `off_site` & `on_site` methods. This permits additional
            information, such as that needed for environmental dependent feeds,
            to be provided to these methods without having to add custom call
            signatures for each Slater-Koster feed type.

    Returns:
        mat: Hamiltonian or overlap matrices for the target systems.

    Warnings:
        There is no tolerance unit test for this function; one will be created
        once a production grade standard `SkFeed` object has been implemented
        & a set of dummy skf files have been created. However, tests carried
        out during development have shown the method to be within acceptable
        tolerances. Please don't modify this function until a tolerance test
        has been implemented.

    Todo:
        - Create tolerance unit test once a production level standard `SkFeed`
          object has been implemented and a dummy set of skf files have been
          created (which include f-orbitals).

    """
    # Developers Notes
    # ----------------
    # CPU/memory efficiency can be improved here, but such changes have been
    # proposed until a later date. Variables named "*_mat_*" hold data that's
    # used to build masks or is gathered by other masks. Suffixes _f, _s & _a
    # indicate whether a tensor is full, shell-wise or atom-wise resolved.
    # Time permitting a change should be introduced which caches the rotation
    # matrices for rather than recomputing them every time.

    # The device on which the results matrix sits is defined by the geometry
    # object. This choice has been made as such objects are made to be moved
    # between devices and so should always be on the "correct" one.
    mat = torch.zeros(basis.orbital_matrix_shape,  # <- Results matrix
                      device=geometry.positions.device)

    # True of the hamiltonian is square (used for safety checks)
    mat_is_square = mat.shape[-1] == mat.shape[-2]

    # Matrix Initialisation
    l_mat_f = basis.azimuthal_matrix(mask_diag=True, mask_on_site=True)
    l_mat_s = basis.azimuthal_matrix('shell', mask_on_site=True)
    i_mat_s = basis.index_matrix('shell')
    an_mat_a = basis.atomic_number_matrix('atomic')
    sn_mat_s = basis.shell_number_matrix('shell')
    dist_mat_a = geometry.distances
    vec_mat_a = -normalize(geometry.distance_vectors, 2, -1)  # Unit vectors

    # Loop over each azimuthal-pair interaction (max ℓ=3 (f))
    l_pairs = torch.tensor([[i, j] for i in range(4)
                            for j in range(4)
                            if i <= j])

    for l_pair in l_pairs:
        # Mask identifying indices associated with the current l_pair target
        index_mask_s = torch.nonzero((l_mat_s == l_pair).all(dim=-1)).T

        # Ignore duplicate operations in the lower triangle when ℓ₁=ℓ₂
        if l_pair[0] == l_pair[1:] and mat_is_square:
            # If the matrix is not square this will case many problems!
            index_mask_s = index_mask_s.T[index_mask_s[0] < index_mask_s[1]].T

        if len(index_mask_s[0]) == 0:  # Skip if no l_pair blocks are found
            continue

        # Gather shell numbers associated with the selected (masked) orbitals
        shell_pairs = sn_mat_s[[*index_mask_s]]

        # Gather from i_mat_s to get the atom index mask.
        index_mask_a = index_mask_s.clone()  # <- batch agnostic approach
        index_mask_a[-2:] = i_mat_s[[*index_mask_s]].T

        # Gather the atomic numbers, distances, and unit vectors.
        g_anum = an_mat_a[[*index_mask_a]]
        g_dist = dist_mat_a[[*index_mask_a]]
        g_vecs = vec_mat_a[[*index_mask_a]]

        # Get off-site integrals from the sk_feed, passing on any kwargs
        # provided by the user. If the SK-feed is environmentally dependent,
        # then it will need the indices of the atoms; as this data cannot be
        # provided by the user it must be explicitly added to the kwargs here.
        integrals = _gather_off_site(g_anum, shell_pairs, g_dist, sk_feed,
                                     **kwargs, atom_indices=index_mask_a)

        # Make a call to the relevant Slater-Koster function to get the sk-block
        sk_data = sub_block_rot(l_pair, g_vecs, integrals)

        # Multidimensional assigment operations assign their data row-by-row.
        # While this does not pose a problem when dealing with SK blocks which
        # span only a single row (i.e. ss, sp, sd) it causes multi-row SK data
        # (i.e. ps, sd, pp) to be incorrectly parsed; e.g, when attempting to
        # assign two 3x3 blocks [a-i & j-r] to a tensor the desired outcome
        # would be tensor A), however, a more likely outcome is the tensor B).
        # A) ┌                           ┐ B) ┌                           ┐
        #    │ .  .  .  .  .  .  .  .  . │    │ .  .  .  .  .  .  .  .  . │
        #    │ a  b  c  .  .  .  j  k  l │    │ a  b  c  .  .  .  d  e  f │
        #    │ d  e  f  .  .  .  m  n  o │    │ g  h  i  .  .  .  j  k  l │
        #    │ g  h  i  .  .  .  p  q  r │    │ m  n  o  .  .  .  p  q  r │
        #    │ .  .  .  .  .  .  .  .  . │    │ .  .  .  .  .  .  .  .  . │
        #    └                           ┘    └                           ┘
        # To prevent this; the SK block's elements are rearranged by row. To
        # avoid the issues associated with partial row overlap only sk-blocks
        # that are azimuthal minor, e.g. sp, pd, etc. (lowest ℓ first), are
        # are considered. Azimuthal major blocks, ps, dp, etc., are dealt with
        # during the final assignment by flipping the indices.

        # Split SK-data into row-wise slices, flatten, then concatenate.
        groupings = index_mask_s[:-1].unique_consecutive(False, True, 1)[1]
        groups = split_by_size(sk_data, groupings)
        sk_data = torch.cat([g.transpose(1, 0).flatten() for g in groups])

        # Create the full sized index mask and assign the results.
        a_mask = torch.nonzero((l_mat_f == l_pair).all(-1)).T

        # Mask lower triangle like before (as appropriate)
        if l_pair[0] == l_pair[1:] and mat_is_square:
            a_mask = a_mask.T[a_mask[0] < a_mask[1]].T

        mat[[*a_mask]] = sk_data  # (ℓ_1, ℓ_2) blocks, i.e. the row blocks
        mat.transpose(-1, -2)[[*a_mask]] = sk_data  # (ℓ_2, ℓ_1) column-wise

    # Set the onsite terms (diagonal)
    mat.diagonal(0, -2, -1)[:] = _gather_on_site(geometry, basis, sk_feed,
                                                 **kwargs)
    return mat


def _gather_on_site(geometry: Geometry, basis: Basis, sk_feed: SkFeed,
                    **kwargs) -> Tensor:
    """Retrieves on site terms from a target feed in a batch-wise manner.

    This is a convenience function for retrieving on-site terms from an SKFeed
    object.

    Arguments:
        geometry: `Geometry` instance associated with the target system(s).
        basis: `Basis` instance associated with the target system(s).
        sk_feed: The Slater-Koster feed entity responsible for providing the
            requisite Slater Koster integrals and on-site terms.

    Keyword Arguments:
        kwargs: `kwargs` are passed into calls made to the ``sk_feed``
            object's `off_site` method.

    Returns:
        on_site_values: On-site values associated with the specified systems.

    Notes:
        Unlike `_gather_of_site`, this function does not require the keyword
        argument ``atom_indices`` as it can be constructed internally.
    """
    an = geometry.atomic_numbers
    a_shape = basis.atomic_matrix_shape[:-1]
    o_shape = basis.orbital_matrix_shape[:-1]

    # Get the onsite values for all non-padding elements & pass on the indices
    # of the atoms just in case they are needed by the SkFeed
    mask = an.nonzero(as_tuple=True)

    if 'atom_indices' not in kwargs:
        kwargs['atom_indices'] = torch.arange(geometry.n_atoms.max()
                                              ).expand(a_shape)

    os_flat = torch.cat(sk_feed.on_site(atomic_numbers=an[mask], **kwargs))

    # Pack results if necessary (code has no effect on single systems)
    c = torch.unique_consecutive((basis.on_atoms != -1).nonzero().T[0],
                                 return_counts=True)[1]
    return pack(split_by_size(os_flat, c)).view(o_shape)


def _gather_off_site(
        atom_pairs: Tensor, shell_pairs: Tensor, distances: Tensor,
        sk_feed: SkFeed, **kwargs) -> Tensor:

    """Retrieves integrals from a target feed in a batch-wise manner.

    This convenience function mediates the integral retrieval operation by
    splitting requests into batches of like types permitting fast batch-
    wise retrieval.

    Arguments:
        atom_pairs: Atomic numbers of each atom pair.
        shell_pairs: Shell numbers associated with each interaction. Note that
            all shells must correspond to identical azimuthal numbers.
        distances: Distances between the atom pairs.
        sk_feed: The Slater-Koster feed entity responsible for providing the
            requisite Slater Koster integrals and on-site terms.

    Keyword Arguments:
        kwargs: Surplus `kwargs` are passed into calls made to the ``sk_feed``
            object's `off_site` method.
        atom_indices: Tensor: Indices of the atoms for which the integrals are
            being evaluated. For a single system this should be a tensor of
            size 2xN where the first & second row specify the indices of the
            first and second atoms respectively. For a batch of systems an
            extra row is appended to the start specifying which system the
            atom pair is associated with.

    Returns:
        integrals: The relevant integral values evaluated at the specified
            distances.

    Notes:
        Any kwargs specified will be passed through to the `integral_feed`
        during function calls. Integrals can only be evaluated for a single
        azimuthal pair at a time.

    Warnings:
        All shells specified in ``shell_pairs`` must have a common azimuthal
        number / angular momentum. This is because shells with azimuthal
        quantum numbers will return a different number of integrals, which
        will cause size mismatch issues.

    """
    # Block the passing of vectors, which can cause hard to diagnose issues
    if distances.ndim >= 2:
        raise ValueError('Argument "distances" must be a 1d torch.tensor.')

    integrals = None

    # Sort lists so that separate calls are not needed for O-H and H-O
    sorter = atom_pairs.argsort(-1)
    atom_pairs = atom_pairs.gather(-1, sorter)
    shell_pairs = shell_pairs.gather(-1, sorter)

    # Identify all unique [atom|atom|shell|shell] sets.
    as_pairs = torch.cat((atom_pairs, shell_pairs), -1)
    as_pairs_u = as_pairs.unique(dim=0)

    # If "atom_indices" was passed, make sure only the relevant atom indices
    # get passed during each call.
    atom_indices = kwargs.get('atom_indices', None)
    if atom_indices is not None:
        del kwargs['atom_indices']

    # Loop over each of the unique atom_pairs
    for as_pair in as_pairs_u:
        # Construct an index mask for gather & scatter operations
        mask = torch.where((as_pairs == as_pair).all(1))[0]

        # Select the required atom indices (if applicable)
        ai_select = atom_indices.T[mask] if atom_indices is not None else None

        # Retrieve the integrals & assign them to the "integrals" tensor. The
        # SkFeed class requires all arguments to be passed in as keywords.
        off_sites = sk_feed.off_site(
            atom_pair=as_pair[..., :-2], shell_pair=as_pair[..., -2:],
            distances=distances[mask],
            atom_indices=ai_select, **kwargs)

        # The result tensor's shape cannot be *safely* identified prior to the
        # first sk_feed call, thus it must be instantiated in the first loop.
        if integrals is None:
            integrals = torch.zeros((len(as_pairs), off_sites.shape[-1]),
                                    dtype=distances.dtype,
                                    device=distances.device)

        # If shells with differing angular momenta are provided then a shape
        # mismatch error will be raised. However, the message given is not
        # exactly useful thus the exception's message needs to be modified.
        try:
            integrals[mask] = off_sites
        except RuntimeError as e:
            if str(e).startswith('shape mismatch'):
                raise type(e)(
                    f'{e!s}. This could be due to shells with mismatching '
                    'angular momenta being provided.')
    # Return the resulting integrals
    return integrals


def sub_block_rot(l_pair: Tensor, u_vec: Tensor,
                  integrals: Tensor) -> Tensor:
    """Diatomic sub-block rotated into the reference frame of the system.

    This takes the unit distance vector and slater-koster integrals between a
    pair of orbitals and constructs the associated diatomic block which has
    been rotated into the reference frame of the system.

    Args:
        l_pair: Azimuthal quantum numbers associated with the orbitals.
        u_vec: Unit distance vector between the orbitals.
        integrals: Slater-Koster integrals between the orbitals, in order of
            σ, π, δ, γ, etc.

    Returns:
        block: Diatomic block(s)

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
        Each row of ``integrals`` should represent a separate system; i.e.
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
# Note that the internal slater-koster transformation functions "_skt_*" are
# able to handle batches of systems, not just one system at a time.
def _rot_yz_s(unit_vector: Tensor) -> Tensor:
    r"""s-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating s-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Notes:
        This function acts as a dummy subroutine as s integrals do not require
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
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Notes:
        This function acts as a dummy subroutine as s integrals do not require
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
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
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
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
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
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
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
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
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
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
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
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
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
