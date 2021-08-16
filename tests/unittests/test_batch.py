"""Performs tests on functions present in the tbmalt.common.batch module"""
import torch
from torch.autograd import gradcheck
import pytest
import numpy as np
from tbmalt.common import batch
from tests.test_utils import *


@fix_seed
def test_pack(device):
    """Sanity test of batch packing operation."""
    # Generate matrix list
    sizes = torch.randint(2, 8, (10,))
    matrices = [torch.rand(i, i, device=device) for i in sizes]
    # Pack matrices into a single tensor
    packed = batch.pack(matrices)
    # Construct a numpy equivalent
    max_size = max(packed.shape[1:])
    ref = np.stack(
        np.array([np.pad(i.sft(), (0, max_size-len(i))) for i in matrices]))
    equivalent = np.all((packed.sft() - ref) < 1E-12)
    same_device = packed.device == device

    assert equivalent, 'Check pack method against numpy'
    assert same_device, 'Device persistence check (packed tensor)'

    # Check that the mask is correct
    *_, mask = batch.pack([
        torch.rand(1, device=device),
        torch.rand(2, device=device),
        torch.rand(3, device=device)],
        return_mask=True)

    ref_mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                            dtype=torch.bool, device=device)

    same_device_mask = mask.device == device
    eq = torch.all(torch.eq(mask, ref_mask))

    assert eq, 'Mask yielded an unexpected result'
    assert same_device_mask, 'Device persistence check (mask)'


@pytest.mark.grad
@fix_seed
def test_pack_grad(device):
    """Gradient stability test of batch packing operation."""
    sizes = torch.randint(2, 6, (3,))
    tensors = [torch.rand(i, i, device=device, requires_grad=True) for i in sizes]

    def proxy(*args):
        # Proxy function is used to prevent an undiagnosed error from occurring.
        return batch.pack(list(args))

    grad_is_safe = gradcheck(proxy, tensors, raise_exception=False)
    assert grad_is_safe, 'Gradient stability test'


@fix_seed
def test_sort(device):
    """Ensures that the ``psort`` and ``pargsort`` methods work as intended.

    Notes:
        A separate check is not needed for the ``pargsort`` method as ``psort``
        just wraps around it.
    """

    # Test on with multiple different dimensions
    for d in range(1, 4):
        tensors = [torch.rand((*[i] * d,), device=device) for i in
                   np.random.randint(3, 10, (10,))]

        packed, mask = batch.pack(tensors, return_mask=True)

        pred = batch.psort(packed, mask).values
        ref = batch.pack([i.sort().values for i in tensors])

        check_1 = (pred == ref).all()
        assert check_1, 'Values were incorrectly sorted'

        check_2 = pred.device == device
        assert check_2, 'Device persistence check failed'
