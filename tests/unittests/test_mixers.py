# -*- coding: utf-8 -*-
"""Performs tests on mixers present in the tbmalt.common.maths.mixers module.

For each mixer three tests are expected to be performed:

Test 1) General Operation:
    This should test all of a mixer's general functionality; e.g. it should
    make sure that the `cull`, `reset`, etc. work as intended and that the
    `step_number` and `converged` properties return the expected results.
    This should effectively test everything other than the actual mixing
    operation.

Test 2) Convergence:
    Here the actual ability of a mixer to converge a system should be tested.
    This should be done by passing an instance of the mixer to be tested to
    the `convergence` function. This will ensure that a mixer does converge
    and that it converges to the same result independent of batch_mode and
    zero padded packing. This will not only test that a function can converge
    but also that it can converge to the correct answer.

Test 3) Gradient:
    As the name suggests this will test the gradient stability of the mixer.


Warnings:
    The tests contained within this module have dependencies on other TBMaLT
    modules; `tbmalt.common.maths` and `tbmalt.common.batch`. Without access
    to the `tbmalt.common.maths.eighb` function, no gradchecks can be run on
    the mixers using the `faux_SCC` or `faux_SCF` functions. Due to this, any
    errors in the `eighb` function will cause these test to fail. As such, the
    `test_batch` and `test_maths` unit tests should be run first.

    The general test function ``func`` can be used during debugging to
    decouple the mixer tests from the maths module.

"""
import pytest
import torch
from torch.autograd import gradcheck
from tbmalt.common.maths.mixers import Simple, Anderson
from tests.test_utils import fix_seed, clean_zero_padding
from tbmalt.common.maths import sym, eighb
from tbmalt.common.batch import pack

Tensor = torch.Tensor


####################
# Helper Functions #
####################

@fix_seed
def gen_systems(device, sizes):
    """Generates a batch of fake system for faux-SCC/SCF convergence testing.

    Returns variables needed to conduct faux-SCC/SCF cycles on a batch of
    systems. Note; variables have no underling physical basis as they are
    randomly generated.

    Returns:
        A tuple containing the:
            hamiltonian matrices
            overlap matrices
            gamma matrices
            neutral_charge vectors

    """
    H, S, G, q0 = [], [], [], []
    for size in sizes:
        for i in [H, S, G]:
            mat = torch.rand(size, size, device=device)
            mat = (mat + mat.T) / 2
            i.append(mat @ mat)  # Make positive semi-definite
        _q0 = torch.rand(size, device=device)
        q0.append((_q0 / _q0.sum()) * int(size / 2))
    # Zero pad pack data into single tensors and return them
    return pack(H), pack(S), pack(G), pack(q0)


def faux_SCF(F_in, neutral_charges, H, S, G):
    """A faux self-consistent field cycle.

    This function approximates a self consistent field cycle. It takes a Fock
    matrix as an input and returns an updated one.
    """
    C = eighb(F_in, S, eigenvectors=True)[1]
    new_charges = torch.sum((C @ C.transpose(-2, -1)) * S, -1)
    if H.dim() == 3:
        esp = torch.einsum('bn,bnm->bm', (new_charges - neutral_charges), G)
        esp_matrix = torch.unsqueeze(esp, 1) + torch.unsqueeze(esp, 2)

    else:
        esp = torch.einsum('n,nm->m', (new_charges - neutral_charges), G)
        esp_matrix = torch.unsqueeze(esp, 1) + esp

    F_out = H + 0.5 * S * esp_matrix
    return F_out


def faux_SCC(charges_in, neutral_charges, H, S, G):
    """A faux self-consistent charge cycle.

    This function approximates a self consistent charge cycle. It takes a
    charge vector as an input and returns an updated one.
    """
    # Takes an old vector (charges) and returns a new one
    if H.dim() == 3:
        esp = torch.einsum('bn,bnm->bm', (charges_in - neutral_charges), G)
        esp_matrix = torch.unsqueeze(esp, 1) + torch.unsqueeze(esp, 2)

    else:
        esp = torch.einsum('n,nm->m', (charges_in - neutral_charges), G)
        esp_matrix = torch.unsqueeze(esp, 1) + esp

    F = H + 0.5 * S * esp_matrix
    C = eighb(F, S, eigenvectors=True)[1]
    charges_out = torch.sum((C @ C.transpose(-2, -1)) * S, -1)
    return charges_out


def func(x):
    """Non-linear convergence test function.

    This function, although simple, is commonly used to test the validity of
    mixing algorithms. A initial ``x`` vector of ones should be used, & should
    eventually converge to zero. However, it should be noted that directly
    calling this function cyclically without the aid of a mixing algorithm
    will result in divergent behaviour.
    """
    d = torch.tensor([3., 2., 1.5, 1.0, 0.5], dtype=x.dtype, device=x.device)
    c = 0.01
    return x + (-d * x - c * x**3)


def cycle(mixer, target, function, arguments=None, n=200):
    """Performs a self-consistency cycle using a specified mixer & function.

    Will check class is operating on the correct device.

    Arguments:
        mixer: Mixer class instance.
        target: Vector or matrix that is to be mixed.
        function: Function to be cycled, e.g. ``faux_SCC``.
        arguments: Tuple holding any other arguments that must be passed into
            ``function``.
        n: Number of mixing cycles to perform.

    Returns:
        converged: Tensor of booleans indicating convergence status.
        mixed: The final mixed system

    """
    arguments = () if arguments is None else arguments
    device = target.device
    n_first = int(n / 2)
    n_second = n - n_first

    # "x_old" will only be provided for the first few cycles
    for _ in range(n_first):
        target = mixer(function(target, *arguments), target)
    # After that it will have to rely on its internal tracking
    for _ in range(n_second):
        target = mixer(function(target, *arguments))

    # Find floating point tensors that have been placed on the wrong device.
    tensors = [k for k, v in mixer.__dict__.items() if isinstance(v, Tensor)
               and v.dtype.is_floating_point and v.device != device]
    cls = mixer.__class__.__name__
    msg = (f'{cls}: The following tensors were placed on the wrong device\n'
           f'\t{", ".join(tensors)}')

    # Assert everything is on the correct device
    assert len(tensors) == 0, msg
    assert device == target.device, f'{cls} device persistence check failed'

    converged = mixer.converged
    mixer.reset()
    return converged, target


@pytest.mark.filterwarnings("ignore:Tolerance value")
def general(mixer, device):
    """Tests some of the mixer's general operational functionality.

    This tests the basic operational functionality of a mixer. It should be
    noted that this is not a comprehensive test. This is intended to catch
    basic operating errors. Once this function is complete it will call the
    ``mixer.reset()`` function. It is important to check that the mixer has
    actually be reset. Each check must be done individually for each mixer
    as it is a highly mixer specific task.

    Tests:
        1. Tolerance condition's warning and clip subroutines.
        2. Mixing can be performed & returns the correct shape.
        3. Step number is incremented.
        4. Delta values are returned correctly.
        5. Convergence property returns expected result.
        6. Cull operation runs as anticipated.
        7. Reset function works as expected.

    Args:
        mixer: The mixer to test
        device: the device to run on

    """
    name = mixer.__class__.__name__
    a = torch.ones(5, 5, 5, device=device)
    a_copy = a.clone()
    mixer._is_batch = True

    # Checks 1 & 2
    mixer.tolerance = 1E-20
    with pytest.warns(UserWarning, match='Tolerance*'):
        for _ in range(10):
            a = mixer(func(a), a)
            chk_2 = a.shape == a_copy.shape
            assert chk_2, f'{name} Input & output shapes do not match'

    chk_1 = mixer.tolerance >= torch.finfo(a.dtype).resolution
    assert chk_1, f'{name}.tolerance was not downgraded'

    # Check 3
    chk_3 = mixer.step_number != 0
    assert chk_3, f'{name}.step_number was not incremented'

    # Check 4
    b = a
    a = mixer(func(a), a)
    chk_4a = mixer.delta.shape == a_copy.shape
    chk_4b = torch.allclose(a - b, mixer.delta)
    assert chk_4a, f'{name}.delta has an incorrect shape'
    assert chk_4b, f'{name}.delta values are not correct'

    # Check 5
    converged = mixer.converged
    chk_5a = converged.shape == a_copy.shape[0:1]
    chk_5b = converged.any() == False  # Should not have converged yet
    a = mixer(a, a)
    chk_5c = mixer.converged.all() == True
    assert chk_5a, f'{name}.converged has an incorrect shape'
    assert chk_5b, f'{name}.converged should all be False'
    assert chk_5c, f'{name}.converged should all be True'

    # Check 6
    cull_list = torch.tensor([True, False, True, False, True], device=device)
    mixer.cull(cull_list)
    # Next mixer call should crash if cull was not implemented correctly.
    a = mixer(func(a[~cull_list]), a[~cull_list])
    chk_6 = a.shape == a_copy[~cull_list].shape
    assert chk_6, f'{name} returned an unexpected shape after cull operation'

    # Check 7
    # This only performs the reset operation to catch any fatal errors.
    # Additional checks must be performed in the mixer specific function.
    mixer.reset()


def convergence(mixer, device):
    """Tests a mixer's convergence functionality.

    This function is designed to check that a mixer can:
        1. Converge a simple function to the correct solution.
        2. Converge a set of more complex systems.
        3. Return results correctly in batch and non-batch modes.
        4. Ensure zero-padded packing does not effect the result.
        5. Carries out work on the correct device.

    It should be noted that test 2 only checks that the systems do converge
    not that the converge to a specific result. It is the job of test 1 to
    ensure that the correct result is obtained, this prevents a mixer from
    passing that, for example, just returns the input.

    Arguments:
        mixer: Mixer class instance.
        device: The device on which to run the calculation.
    """
    name = mixer.__class__.__name__
    H, S, G, q0 = gen_systems(device, [10, 6, 4])  # Generate test data

    # CHECK 1:
    mixer._is_batch = False
    _, res_0 = cycle(mixer, torch.ones(5, device=device), func, None, 400)
    converged = torch.allclose(torch.zeros(5, device=device), res_0)
    assert converged, f'{name}: Failed to converge to correct result'

    # CHECK 2:
    # Check convergence of:
    #   1) A single vector
    conv_1, _ = cycle(mixer, q0[0], faux_SCC, (q0[0], H[0], S[0], G[0]))
    #   2) A single matrix
    conv_2, res_1 = cycle(mixer, H[-1], faux_SCF, (q0[-1], H[-1], S[-1], G[-1]))
    mixer._is_batch = True
    #   3) A batch a of vectors
    conv_3, _ = cycle(mixer, q0, faux_SCC, (q0, H, S, G))
    #   4) A of batch of a matrices
    conv_4, res_2 = cycle(mixer, H, faux_SCF, (q0, H, S, G))


    assert conv_1.all(), f'{name} Failed to converge single vector (faux_SCC)'
    assert conv_2.all(), f'{name} Failed to converge single matrix (faux_SCF)'
    assert conv_3.all(), f'{name} Failed to converge vector batch (faux_SCC)'
    assert conv_4.all(), f'{name} Failed to converge matrix batch (faux_SCF)'

    # CHECK 3:
    # Ensure that batch & non-batch runs yield the same result
    similar =  torch.allclose(res_1, res_2[-1])
    assert similar, f'{name}: Batch operations return different results'

    # CHECK 4:
    # Check zero-padded packing does not adversely affect the final result
    # Ensure the same answer is given with padding as without padding.
    s = (-1, slice(0,4), slice(0,4))
    _, res_3 = cycle(mixer, H[s], faux_SCF, (q0[-1, :4,], H[s], S[s], G[s]))
    similar = torch.allclose(res_3, res_1[:4, :4],)

    assert similar, f'{name}: Zero padded packing returns different results'
    # Check 5 is carried out continuously by the cycle function.


def gradient(mixer, device):
    def mixer_proxy(H, S, G, q0, sizes, mixer):

        # Proxy is needed to clean the perturbations in the padding values
        # caused by the gradcheck operation. Otherwise gradcheck will fail.
        H = clean_zero_padding(H, sizes)
        S = clean_zero_padding(S, sizes)
        G = clean_zero_padding(G, sizes)

        # Tensors must also be symmetrised.
        H, S, G = sym(H), sym(S), sym(G)

        # F = H
        q_new = q0
        for _ in range(15):
            # F = mixer(faux_SCF(F, q0, H, S, G), F)
            q_new = mixer(faux_SCC(q_new, q0, H, S, G), q_new)

        # Reset mixer before gradchecks next call
        mixer.reset()

        # return F
        return q_new

    sizes = torch.tensor([5, 3], device=device)
    # Generate test data
    H, S, G, q0 = gen_systems(device, sizes)

    # Enable gradient tracking
    # H.requires_grad = True
    # S.requires_grad = True
    # G.requires_grad = True
    q0.requires_grad = True

    grad_is_safe = gradcheck(mixer_proxy, (H, S, G, q0, sizes, mixer),
                             raise_exception=False)

    cls = mixer.__class__.__name__
    assert grad_is_safe, f'{cls} failed gradient check'


#####################################
# TBMaLT.common.maths.mixers.Simple #
#####################################
def test_simple_general(device):
    mixer = Simple(is_batch=True)
    general(mixer, device)
    # Mixer should have been reset; check the reset was carried out correctly
    reset = all([mixer.step_number == 0,
                 mixer._delta is None,
                 mixer._x_old is None])

    assert reset, 'Reset operation was incomplete'


def test_simple_convergence(device):
    mixer = Simple(is_batch=False, tolerance=1E-6)
    mixer.mix_param = 0.1
    convergence(mixer, device)


@pytest.mark.grad
def test_simple_grad(device):
    mixer = Simple(is_batch=True, tolerance=1E-6)
    mixer.mix_param = 0.1
    gradient(mixer, device)


#######################################
# TBMaLT.common.maths.mixers.Anderson #
#######################################
def test_anderson_general(device):
    mixer = Anderson(is_batch=True)
    general(mixer, device)
    # Mixer should have been reset; check the reset was carried out correctly
    reset = all([mixer.step_number == 0,
                 mixer._delta is None,
                 mixer._x_hist is None,
                 mixer._f is None,
                 mixer._shape_in is None,
                 mixer._shape_out is None])

    assert reset, 'Reset operation was incomplete'


def test_anderson_convergence(device):
    mixer = Anderson(is_batch=False, tolerance=1E-6)
    mixer.mix_param = 0.1
    convergence(mixer, device)


@pytest.mark.grad
def test_anderson_grad(device):
    mixer = Anderson(is_batch=True, tolerance=1E-6)
    mixer.mix_param = 0.1
    gradient(mixer, device)
