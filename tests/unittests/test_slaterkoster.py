# -*- coding: utf-8 -*-
"""Unit tests associated with `tbmalt.physics.dftb.slaterkoster`."""
from typing import Tuple, List, Callable, Optional, Union
import numpy as np
from re import findall
import pytest
from numpy.random import choice
import torch
from torch import Tensor
from torch.autograd import gradcheck
from tests.test_utils import fix_seed
from tbmalt.common.batch import pack
from tbmalt.physics.dftb.slaterkoster import (
    _rot_yz_s, _rot_xy_s, _rot_yz_p, _rot_xy_p, _rot_yz_d,
    _rot_xy_d, _rot_yz_f, _rot_xy_f,
    sub_block_ref, sub_block_rot,
    _gather_on_site, _gather_off_site,
    hs_matrix
)
from tbmalt.ml.skfeeds import _SkFeed
from tbmalt import Geometry, Basis

####################
# Helper Functions #
####################
l_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
max_ls = {1: 0, 6: 1, 79: 2, 57: 3}
shell_dict = {1: [0], 6: [0, 1], 79: [0, 1, 2], 57: [0, 1, 2, 3]}


class FoundOffSiteTestKwarg(Exception):
    """Used when evaluating off-site kwarg passthrough."""
    pass


class FoundOnSiteTestKwarg(Exception):
    """Used when evaluating on-site kwarg passthrough."""
    pass


class AtomIndicesIncorrect(Exception):
    """Used to catch errors in atom indexing during calls to off_site."""
    pass


def from_file(path: str, **kwargs) -> Tensor:
    """Reads data from a numpy text file into a `Tensor`.

    This function extracts and returns n-dimensional tensors from a specified
    numpy text file.

    Arguments:
        path: Path to the target numpy text file.
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Returns:
        tensor: Data from the specified file.
    """
    shape = np.array(findall('[0-9]+', open(path).readline()), dtype=int)
    return torch.tensor(np.loadtxt(path).reshape(shape), **kwargs)


def build_geom(*atomic_numbers: int, **kwargs) -> Geometry:
    """Constructs a `Geometry` object for a given number of atoms.

    Takes an arbitrary number of atoms and returns a `Geometry` object. Note
    that the positions are randomly generated.

    Arguments:
        *atomic_numbers: An arbitrary number atomic numbers.
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Returns:
        geometry: `Geometry` object with the desired atoms at random positions.
    """

    return Geometry(torch.tensor(*[atomic_numbers],),
                    torch.rand(len(atomic_numbers), 3, **kwargs))


class _TestSkFeed(_SkFeed):
    """Dummy `SkFeed` instance.

    This is an extremely simple `SkFeed` implementation that is needed for
    testing.
    """
    @fix_seed
    def __init__(self, atoms=None, **kwargs):
        atoms = [1, 6, 79, 57] if atoms is None else atoms

        def n_orbs(atom):
            max_l = max_ls[atom]
            return max_l**2 + 2 * max_l + 1

        self.on_site_values = {atom: torch.rand(n_orbs(atom), **kwargs)
                               for atom in atoms}

        self.off_site_values = {(atom_1, atom_2, l1, l2): torch.rand(l1 + 1, **kwargs)
                                for atom_1 in atoms
                                for atom_2 in atoms
                                for l1 in range(max_ls[atom_1] + 1)
                                for l2 in range(max_ls[atom_2] + 1)
                                if l1 <= l2}

        self.off_site_values.update({(k[1], k[0], k[3], k[2]): v
                                     for k, v in self.off_site_values.items()})

    def off_site(self, atom_pair, shell_pair, distances, **kwargs):
        """This creates and returns dummy off-site Slater-Koster integrals.

        The integral value returned is just the product of the distance and a
        static vector whose length corresponds to the bond order.
        """

        # Used in testing for kwarg passthrough
        if 'test_off_site_kwarg_passthrough' in kwargs:
            raise FoundOffSiteTestKwarg()

        # Used in validating _gather_off_site atom_indices argument parsing
        if (basis := kwargs.get('test_basis', None)) is not None:
            an_mat = basis.atomic_number_matrix(form='atomic')
            check = (an_mat[[*kwargs['atom_indices']]] == atom_pair).all()
            if not check:
                raise AtomIndicesIncorrect()

        sorter = atom_pair.argsort()
        shell_pair = shell_pair[sorter]
        atom_pair = atom_pair[sorter]

        if atom_pair[0] == atom_pair[1] and shell_pair[0] > shell_pair[1]:
            shell_pair = shell_pair.flip(-1)

        return (self.off_site_values[(*atom_pair.tolist(), *shell_pair.tolist())].view(1, -1)
                * distances.view(-1, 1))

    def on_site(self, atomic_numbers, **kwargs):
        """Returns dummy on-site values."""

        # Used in testing for kwarg passthrough
        if 'test_on_site_kwarg_passthrough' in kwargs:
            raise FoundOnSiteTestKwarg()

        return tuple([self.on_site_values[int(i)] for i in atomic_numbers])

    def to(self, *args):
        pass


def molecules(device: torch.device, n: Optional[int] = None
              ) -> Union[
                         Tuple[List[Tensor], List[Tensor]],
                         Tuple[Tensor, Tensor]
                        ]:
    """Atomic numbers & positions of a small selection of molecules.

    This data is used in testing the ``hs_matrix`` function. This contains
    three molecules: i) H2, simple diatomic molecule; ii) CH4, a slightly more
    complex molecules; and iii) CH4+Au2La a contrived system that permits
    testing of all supported orbital interactions.

    Arguments:
        device: The device onto which the position tensors are to be placed.
        n: If only a single system is desired ``n`` can be used to specify the
            index of the desired system. By default, ``n`` is None; meaning
            that all systems are returned.

    Returns:
        atomic_numbers: Atomic numbers of the systems.
        positions: Positions of said systems.
    """
    atomic_numbers = [
        torch.tensor([1, 1]),
        torch.tensor([6,  1,  1,  1,  1]),
        torch.tensor([57, 79, 57, 6, 1, 1, 1, 1])
    ]

    positions = [
        torch.tensor([[0.00,   0.00,  0.00],
                      [0.00,   0.00,  1.40]], device=device),
        torch.tensor([[0.00,   0.00,  0.00],
                      [1.19,   1.19,  1.19],
                      [-1.19, -1.19,  1.19],
                      [1.19,  -1.19, -1.19],
                      [-1.19,  1.19, -1.19]], device=device),
        torch.tensor([[3.78,   3.78,  3.78],
                      [-2.83, -2.83, -2.83],
                      [-3.78,  3.78,  3.78],
                      [0.00,   0.00,  0.00],
                      [1.19,   1.19,  1.19],
                      [-1.19, -1.19,  1.19],
                      [1.19,  -1.19, -1.19],
                      [-1.19,  1.19, -1.19]], device=device)
    ]
    if n is None:
        return atomic_numbers, positions
    else:
        return atomic_numbers[n], positions[n]


def sk_rotation_data(batch: bool = False, **kwargs
                     ) -> Tuple[Callable, Tensor, Tensor]:
    """Slater-Koster rotation matrix data.

    This function returns data required to run and validate the slater koster
    rotation sub-blocks.

    Arguments:
        batch: Controls whether single data-points or a batch thereof should
            be yielded at each iteration. [DEFAULT=False]
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Yields:
        func: Rotation function to be tested.
        unit_vectors: Unit vectors (input data for the functions)
        reference: Reference results.
    """
    skt_functions = [_rot_yz_s, _rot_xy_s, _rot_yz_p, _rot_xy_p, _rot_yz_d,
                     _rot_xy_d, _rot_yz_f, _rot_xy_f]

    # Read in unit vectors & the reference rotation matrices. Then create a mas
    # identifying vectors where |y| > |z|.
    path = 'tests/unittests/data/slaterkoster'
    u_vecs = from_file(f'{path}/unit_vectors.dat', **kwargs)
    r_mats = {ll: from_file(f'{path}/rot_{ll}.dat', **kwargs) for ll in range(4)}
    yz_mask = u_vecs[:, 1].abs() > u_vecs[:, 2].abs()

    for f in skt_functions:
        # Extract azimuthal number and rotation plane from the function's name.
        mask = yz_mask if 'yz' in f.__name__ else ~yz_mask
        s = ... if batch else 0
        yield f, u_vecs[mask][s], r_mats[l_dict[f.__name__[-1]]][mask][s]


def sub_block_ref_data(batch: bool = False, **kwargs
                       ) -> Tuple[Tensor, Tensor, Tensor]:
    """Data for testing the `sub_block_ref` function.

    Yields synthetic data for testing `sub_block_ref`. Synthetic data is used
    as `sub_block_ref` is effectively a fancy diagonal embedding algorithm.

    Arguments:
        batch: Controls whether single data-points or a batch thereof should
            be yielded at each iteration. [DEFAULT=False]
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Yields:
        l_pair: Azimuthal pair (input data).
        integrals: Associated Slater-Koster integrals (input data).
        reference: Reference results.
    """
    dtype = torch.rand(1).dtype

    def to_batch(data):
        """Convert single system into a batch of systems"""
        size = torch.Size((3, *([1] * data[-1].ndim)))
        fac = torch.arange(3, **kwargs).view(size)
        return [torch.stack([i] * 3) * fac for i in data]

    requires_grad = kwargs.pop('requires_grad', False)

    # Azimuthal quantum number pairs. The device on which l_pairs is placed is
    # irrelevant, as it is only ever used for indexing.
    l_pairs = torch.tensor([[i, j] for i in range(4) for j in range(4) if i <= j])

    # Reference block matrices
    ff_m = torch.diag_embed(torch.arange(-3, 4, dtype=dtype, **kwargs).abs() + 1.)
    slices = list(reversed([slice(i, 7 - i) for i in range(4)]))
    results = [torch.atleast_1d(ff_m[r, c].squeeze()).clone()
               for l1, r in enumerate(slices) for c in slices[l1:]]

    # Generate some dummy off-site integral values; ordered as follows:
    #   [ss_i, sp_i, sd_i, sf_i, pp_i, pd_i, pf_i, dd_i, df_i, ff_i]
    integrals = [i for l, n in [(2, 4), (3, 3), (4, 2), (5, 1)] for i in
                 torch.arange(1, l, **kwargs, dtype=dtype).tile(n, 1)]

    if batch:  # Create additional systems if a batch has been requested
        results = to_batch(results)
        integrals = to_batch(integrals)

    # Enable gradient tracking for integrals if requested, this must be done
    # after "batchification" to prevent graph flow issues.
    if requires_grad:
        for i in integrals:
            i.requires_grad = True

    for l_pair, integral, result in zip(l_pairs, integrals, results):
        yield l_pair, integral, result


def sub_block_rot_data(batch: bool = False, **kwargs
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Data for testing the `sub_block_rot` function.

    Yields reference data required to test the `sub_block_rot` function.

    Arguments:
        batch: Controls whether single data-points or a batch thereof should
            be yielded at each iteration. [DEFAULT=False]
        **kwargs: Passed to `torch.tensor` as keyword arguments. Allows for
            device on which the tensor is placed to be controlled.

    Yields:
        l_pair: Azimuthal pair (input data).
        unit_vectors: Unit vectors (input data).
        integrals:  Slater-Koster integrals (input data).
        reference: Reference results.
    """
    path = 'tests/unittests/data/slaterkoster'
    l_pairs = [(l1, l2) for l1 in range(4) for l2 in range(4)]
    u_vecs = from_file(f'{path}/unit_vectors.dat', **kwargs)

    kwargs.pop('requires_grad', None)
    for l_pair in l_pairs:
        integrals = from_file(
            f'{path}/integrals_{min(l_pair)}_{max(l_pair)}.dat', **kwargs)
        ref = from_file(f'{path}/block_rot_{l_pair[0]}_{l_pair[1]}.dat', **kwargs)
        s = ... if batch else 0
        yield torch.tensor(l_pair), u_vecs[s], integrals[s], ref[s]


###########################################
# Slater-Koster Rotation Matrix Functions #
###########################################
def _sk_rotation_matrices_test_helper(batch, **kwargs):
    """Tests the Slater-Koster rotation matrix functions."""
    for func, u_vec, ref in sk_rotation_data(batch, **kwargs):
        pred = func(u_vec)
        check_1 = torch.allclose(pred, ref)
        check_2 = pred.device == kwargs['device']

        name = f'{func.__name__} {"[batch]" if batch else "[single]"}'

        assert check_1, f'Result of {name} exceed tolerance limits'
        assert check_2, f'{name} failed device persistence check'


def test_sk_rotation_matrices_single(device):
    """Runs single system tests of the SK rotation matrix functions."""
    _sk_rotation_matrices_test_helper(False, device=device)


def test_sk_rotation_matrices_batch(device):
    """Runs batch system tests of the SK rotation matrix functions."""
    _sk_rotation_matrices_test_helper(True, device=device)


@pytest.mark.grad
def test_sk_rotation_matrices_grad(device):
    """Checks the gradient stability of the SK rotation matrix functions."""
    # Note this test will always take a long time as the final
    for func, u_vec, _ in sk_rotation_data(
            True, device=device, requires_grad=True):
        name = f'{func.__name__}'
        grad_s = gradcheck(func, (u_vec[0],), raise_exception=False)
        grad_b = gradcheck(func, (u_vec[[0, -1]],), raise_exception=False)
        assert grad_s, f'{name} failed single system gradient stability test'
        assert grad_b, f'{name} failed batch system gradient stability test'


#############################################
# tbmalt.physics.slaterkoster.sub_block_ref #
#############################################
def _sub_block_ref_helper(batch=False, **kwargs):
    """Used when testing the `sub_block_ref` function.

    The `sub_block_ref` function is responsible for constructing the unrotated
    diatomic sub-blocks.
    """
    for l_pair, integrals, res in sub_block_ref_data(batch, **kwargs):
        # Ensure results are within acceptable tolerance when in both
        # azimuthal minor & azimuthal major modes.
        pred_1 = sub_block_ref(l_pair, integrals)
        pred_2 = sub_block_ref(l_pair.flip(-1), integrals)

        res = torch.atleast_2d(res)
        check_1a = torch.allclose(pred_1, res)
        check_1b = torch.allclose(pred_2.transpose(-1, -2), res)
        check_1 = check_1a and check_1b
        check_2 = pred_1.device == kwargs['device']

        form = '[batch]' if batch else '[single]'
        name = str(l_pair.tolist()) + form
        assert check_1, f'Integral matrix tolerances exceeded; {name}'
        assert check_2, f'Device persistence check failed; {name}'


def test_sub_block_ref_single(device):
    """Runs single system tests on the `sub_block_ref` function."""
    _sub_block_ref_helper(False, device=device)


def test_sub_block_ref_batch(device):
    """Runs batch system tests on the `sub_block_ref` function."""
    _sub_block_ref_helper(True, device=device)


@pytest.mark.grad
def test_sub_block_ref_grad(device):
    """Runs gradient stability tests on the `sub_block_ref` function."""
    for l_pair, integrals, res in sub_block_ref_data(
            True, device=device, requires_grad=True):
        grad_s = gradcheck(sub_block_ref, (l_pair, integrals[0],),
                           raise_exception=False)
        grad_b = gradcheck(sub_block_ref, (l_pair, integrals[::3],),
                           raise_exception=False)
        name = str(l_pair.tolist())
        assert grad_s, f'Single system gradient stability test failed on {name}'
        assert grad_b, f'Batch system gradient stability test failed on {name}'


#############################################
# tbmalt.physics.slaterkoster.sub_block_rot #
#############################################
def _sub_block_rot_helper(batch=False, **kwargs):
    """Used in testing the `sub_block_rot` function.

    The `sub_block_rot` function is responsible for constructing the rotated
    Slater-Koster diatomic sub-block from a set of SK-integrals & unit vectors.

    Notes:
         `sub_block_rot` utilises the`sub_block_ref` & rotation matrix
         functions. Thus, this test will likely fail if either of the
         following tests fail:
            - test_sk_rotation_matrices_single
            - test_sub_block_ref_single
    """

    for l_pair, u_vecs, ints, ref in sub_block_rot_data(batch, **kwargs):
        pred = sub_block_rot(l_pair, u_vecs, ints)
        check_1 = torch.allclose(ref, pred)
        check_2 = pred.device == kwargs['device']

        form = '[batch]' if batch else '[single]'
        name = str(l_pair.tolist()) + form

        assert check_1, f'Result of {name} exceed tolerance limits'
        assert check_2, f'{name} failed device persistence check'


def test_sub_block_rot_single(device):
    """Performs a single system test on the `sub_block_rot` function.

    If this test fails, read the doc-string in `_sub_block_rot_helper`."""
    _sub_block_rot_helper(False, device=device)


def test_sub_block_rot_batch(device):
    """Performs a batch system test on the `sub_block_rot` function.

    If this test fails, read the doc-string in `_sub_block_rot_helper`."""
    _sub_block_rot_helper(True, device=device)


@pytest.mark.grad
def test_sub_block_rot_grad(device):
    """Checks the gradient stability of the `sub_block_rot` function.

    If this test fails, read the doc-string in `_sub_block_rot_helper`."""
    for l_pair, u_vecs, ints, ref in sub_block_rot_data(
            True, device=device, requires_grad=True):
        grad_s = gradcheck(sub_block_rot, (l_pair, u_vecs[0], ints[0]),
                           raise_exception=False)
        grad_b = gradcheck(sub_block_rot, (l_pair, u_vecs[::3], ints[::3]),
                           raise_exception=False)

        name = str(l_pair.tolist())
        assert grad_s, f'Single system gradient stability test failed on {name}'
        assert grad_b, f'Batch system gradient stability test failed on {name}'


###############################################
# tbmalt.physics.slaterkoster._gather_on_site #
###############################################
def test_gather_on_site_general(device):
    """Runs general operational tests on `_gather_on_site`.

    This tests that kwargs get passed-through to the `SkFeed` objects
    `on_site` method.
    """
    atomic_numbers, positions = molecules(device, 1)
    geometry = Geometry(atomic_numbers, positions)
    basis = Basis(atomic_numbers, shell_dict)
    sk_feed = _TestSkFeed(atomic_numbers.unique().tolist(), device=device)

    try:
        _gather_on_site(geometry, basis, sk_feed,
                        test_on_site_kwarg_passthrough=None)
        # If an FoundOnSiteTestKwarg exception did not get raised then the
        # kwarg did not get passed-through, so fail the test.
        pytest.fail(
            'kwargs not passed-through to the SkFeed on_site method')
    except FoundOnSiteTestKwarg:
        pass


@fix_seed
def test_gather_on_site_single(device):
    """Runs single system tests on `_gather_on_site`."""
    # Set up the dummy SkFeed
    sk_feed = _TestSkFeed(device=device)

    # Create diatomic geometries with a range of different orbital types
    atoms = [1, 6, 79, 57]
    geoms = [build_geom(a1, a2, device=device)
             for a1 in atoms for a2 in atoms if a1 <= a2]

    for geometry in geoms:
        basis = Basis(geometry.atomic_numbers, shell_dict)
        pred = _gather_on_site(geometry, basis, sk_feed)
        ref = torch.cat(sk_feed.on_site(geometry.atomic_numbers))
        check_1 = torch.allclose(ref, pred)
        check_2 = pred.device == device
        assert check_1, f'Incorrect on-site values returned for {geometry}'
        assert check_2, 'Device persistence check failed'


@fix_seed
def test_gather_on_site_batch(device):
    """Runs batch system tests on `_gather_on_site`."""
    # Set up the dummy SkFeed
    sk_feed = _TestSkFeed(device=device)

    # Create diatomic geometries with a range of different orbital types
    atoms = [1, 6, 79, 57]
    geoms = [build_geom(a1, a2, device=device)
             for a1 in atoms for a2 in atoms if a1 <= a2]
    bases = [Basis(i.atomic_numbers, shell_dict) for i in geoms]

    geometry = Geometry([i.atomic_numbers for i in geoms],
                        [i.positions for i in geoms])
    basis = Basis([i.atomic_numbers for i in geoms], shell_dict)
    pred = _gather_on_site(geometry, basis, sk_feed)
    ref = pack([torch.cat(sk_feed.on_site(i.atomic_numbers))
                for i in geoms])

    check_1 = torch.allclose(ref, pred)
    check_2 = pred.device == device
    assert check_1, f'Incorrect on-site values returned for {geometry}'
    assert check_2, 'Device persistence check failed'


@fix_seed
@pytest.mark.grad
def test_gather_on_site_grad(device):
    """Checks the gradient stability of the `_gather_on_site` function."""

    # Proxy function is needed for calling gradcheck as only torch.tensors
    # can be proved as input arguments.
    def single_proxy(*args):
        return _gather_on_site(geom_s, basis_s, sk_feed_s)

    def batch_proxy(*args):
        return _gather_on_site(geom_b, basis_b, sk_feed_b)

    # Construct the single and batch systems
    sk_feed_s = _TestSkFeed(device=device, requires_grad=True)
    atomic_numbers_s = torch.tensor([57, 57])
    geom_s = Geometry(atomic_numbers_s, torch.rand(2, 3, device=device))
    basis_s = Basis(atomic_numbers_s, shell_dict)

    sk_feed_b = _TestSkFeed(device=device, requires_grad=True)
    atomic_numbers_b = [torch.tensor(i) for i in [[1, 1], [1, 6], [57, 57]]]
    geom_b = Geometry(atomic_numbers_b,
                      [torch.rand(2, 3, device=device) for _ in range(3)])
    basis_b = Basis(atomic_numbers_b, shell_dict)

    # Perform the grad checks
    grad_s = gradcheck(single_proxy,
                       (sk_feed_s.on_site_values[57],),
                       raise_exception=False)

    grad_b = gradcheck(batch_proxy,
                       (*[sk_feed_b.on_site_values[i] for i in [1, 6, 57]],),
                       raise_exception=False)

    assert grad_s, 'Single system gradient stability test failed.'
    assert grad_b, 'Batch system gradient stability test failed.'


################################################
# tbmalt.physics.slaterkoster._gather_off_site #
################################################
def test_gather_off_site_general(device):
    """Runs general operational tests on `_gather_off_site`.

    This tests that i) kwargs get passed-through to the `SkFeed` objects
    `on_site` method, and ii) that the `atom_indices` argument is sliced
    correctly.
    """
    # Feed the geometry in a "geom" and have the function check that the atom
    # pair corresponds to the atom index given

    l_pair = torch.tensor([0, 1])
    basis = Basis(torch.tensor([6, 1, 6, 6, 1, 6]), shell_dict)
    sk_feed = _TestSkFeed(basis.atomic_numbers.unique().tolist(),
                          device=device)

    # The following code is just ripped out of the hs_matrix function
    l_mat_s = basis.azimuthal_matrix('shell', mask_lower=False)
    i_mat_s = basis.index_matrix('shell')
    s_mat_s = basis.shell_number_matrix('shell')
    an_mat_a = basis.atomic_number_matrix('atomic')

    index_mask_b = torch.nonzero((l_mat_s == l_pair).all(dim=-1)).T
    shells = s_mat_s[[*index_mask_b]]
    index_mask_a = i_mat_s[[*index_mask_b]].T
    g_anum = an_mat_a[[*index_mask_a]]
    dists = torch.rand(len(g_anum), device=device)

    # Check for kwarg passthrough
    try:
        _gather_off_site(g_anum, shells, dists, sk_feed,
                         test_off_site_kwarg_passthrough=None)
        pytest.fail(
            'kwargs not passed-through to the SkFeed off_site method')
    except FoundOffSiteTestKwarg:
        pass

    # Ensure atom_indices are passed through correctly (a special check is
    # required as _gather_off_site manipulates the tensor).
    try:
        _gather_off_site(g_anum, shells, dists, sk_feed,
                         atom_indices=index_mask_a)
    except AtomIndicesIncorrect:
        pytest.fail('Atom indices passed to off_site were incorrect')


@fix_seed
def test_gather_off_site(device):
    """Runs test on the `_gather_off_site` function.

    Note that no special tests are needed for batch vs single system mode here
    as there is no difference between the two.
    """

    def get_interactions(l_pair, n):
        """Function for randomly generating interactions."""
        # Identify what atoms can be used for the first and second atoms
        atom_1_choice = [k for k, v in max_ls.items() if v >= l_pair[0]]
        atom_2_choice = [k for k, v in max_ls.items() if v >= l_pair[1]]

        # Select some atom pairs
        atom_pairs = torch.tensor([
            choice(atom_1_choice, n),
            choice(atom_2_choice, n)],
            device=device).T

        # Create some distances
        distances = torch.rand(n, device=device)

        return atom_pairs, distances

    # Instantiate the SkFeed object
    sk_feed = _TestSkFeed(device=device)

    # Perform tests using different azimuthal pairs.
    for l_pair in torch.tensor([[i, j] for i in range(4) for j in range(4)], device=device):
        atom_pairs, distances = get_interactions(l_pair, 20)
        shells = l_pair.expand(20, -1).clone()

        # Flip some elements of atom_pairs & shells to test the sorting code
        atom_pairs = torch.vstack((atom_pairs[:-10], atom_pairs[-10:].flip(-1)))
        shells = torch.vstack((shells[:-10], shells[-10:].flip(-1)))

        pred = _gather_off_site(atom_pairs, shells, distances, sk_feed)
        ref = torch.cat([sk_feed.off_site(i, j, k)
                         for i, j, k in zip(atom_pairs, shells, distances)], dim=0)
        check_1 = torch.allclose(pred, ref)
        check_2 = pred.device == device

        assert check_1, f'Incorrect values returned for l-pair {l_pair}'
        assert check_2, 'Device persistence check failed'


@fix_seed
@pytest.mark.grad
def test_gather_off_site_grad(device):
    """Checks gradient stability of the `_gather_off_site` function."""

    def proxy(*args):
        return _gather_off_site(atom_pairs, shell_pairs, distances, sk_feed)

    # Instantiate the SkFeed object
    sk_feed = _TestSkFeed(device=device, requires_grad=True)
    l_pair = torch.tensor([1, 3])
    atom_pairs = torch.full((6, 2), 57)
    shell_pairs = l_pair.expand(6, -1)
    distances = torch.rand(6, device=device)

    grad = gradcheck(proxy, (sk_feed.off_site_values[(57, 57, 1, 3)],),
                     raise_exception=False)

    assert grad, 'Gradient stability check failed'


#########################################
# tbmalt.physics.slaterkoster.hs_matrix #
#########################################
def test_hs_matrix_general(device):
    sk_feed = _TestSkFeed([1], device=device)
    atomic_numbers, positions = molecules(device=device)
    geometry = Geometry(atomic_numbers[0], positions[0])
    basis = Basis(atomic_numbers[0], shell_dict)

    try:
        hs_matrix(geometry, basis, sk_feed,
                  test_off_site_kwarg_passthrough=None)
        pytest.fail(
            'kwargs not passed-through to the SkFeed off_site method')
    except FoundOffSiteTestKwarg:
        pass

    try:
        hs_matrix(geometry, basis, sk_feed,
                  test_on_site_kwarg_passthrough=None)
        pytest.fail(
            'kwargs not passed-through to the SkFeed on_site method')
    except FoundOnSiteTestKwarg:
        pass


@fix_seed
@pytest.mark.skip(reason="Cannot test until a SkFeed entity is implemented")
def test_hs_matrix_single(device):
    sk_feed = _TestSkFeed(device=device)
    for atomic_numbers, positions in zip(*molecules(device=device)):
        geometry = Geometry(atomic_numbers, positions)
        basis = Basis(atomic_numbers, shell_dict)
        res = hs_matrix(geometry, basis, sk_feed)

        # Tolerance threshold tests are not implemented, so just fail here
        check_1 = False
        check_2 = res.device == device

        assert check_1, 'Results are outside of permitted tolerance thresholds'
        assert check_2, 'Device persistence check failed'


@fix_seed
@pytest.mark.skip(reason="Cannot test until a SkFeed entity is implemented")
def test_hs_matrix_batch(device):
    sk_feed = _TestSkFeed(device=device)

    atomic_numbers, positions = molecules(device=device)
    geometry = Geometry(atomic_numbers, positions)
    basis = Basis(atomic_numbers, shell_dict)
    res = hs_matrix(geometry, basis, sk_feed)

    # Tolerance threshold tests are not implemented, so just fail here
    check_1 = False
    check_2 = res.device == device

    assert check_1, 'Results are outside of permitted tolerance thresholds'
    assert check_2, 'Device persistence check failed'


@fix_seed
@pytest.mark.grad
def test_hs_matrix_grad(device):
    """

    Warnings:
        This gradient check can take a **VERY, VERY LONG TIME** if great care
        is not taken to limit the number of input variables. Therefore, tests
        are only performed on H2 and CH4, change at your own peril!
    """

    def proxy(geometry_in, basis_in, sk_feed_in, *args):
        """Proxy function is needed to enable gradcheck to operate properly"""
        return hs_matrix(geometry_in, basis_in, sk_feed_in)

    # Make sure the SkFeed object only creates data from H & C otherwise the
    # calculation will take an inordinate amount time.
    sk_feed = _TestSkFeed([1, 6], device=device, requires_grad=True)
    atomic_numbers, positions = molecules(device=device)

    # Identify what variables the gradient will be calculated with respect to.
    args = (*sk_feed.off_site_values.values(),
            *sk_feed.on_site_values.values())

    geom_s = Geometry(atomic_numbers[1], positions[1])
    basis_s = Basis(atomic_numbers[1], shell_dict)
    grad_s = gradcheck(proxy, (geom_s, basis_s, sk_feed, *args),
                       raise_exception=False)

    geom_b = Geometry(atomic_numbers[:-1], positions[:-1])
    basis_b = Basis(atomic_numbers[:-1], shell_dict)
    grad_b = gradcheck(proxy, (geom_b, basis_b, sk_feed, *args),
                       raise_exception=False)

    assert grad_s, 'Single system gradient stability test failed.'
    assert grad_b, 'Batch system gradient stability test failed.'
