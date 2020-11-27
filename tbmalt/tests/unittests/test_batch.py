"""Performs tests on functions present in the tbmalt.common.batch module"""
import torch
from torch.autograd import gradcheck
import pytest
import numpy as np
from tbmalt.common import batch
from tbmalt.tests.test_utils import *


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
    assert same_device, 'Device persistence check'


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
