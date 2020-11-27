# -*- coding: utf-8 -*-
r"""This contains a collection of general utilities used then running PyTests.

This module should be imported as:
    from tbmalt.tests.test_utils import *

This ensures that the default dtype and autograd anomaly detection settings
are all inherited.
"""

import numpy as np
import torch
import functools

# Default must be set to float64 otherwise gradcheck will not function
torch.set_default_dtype(torch.float64)
# This will track for any anomalys in the gradient
torch.autograd.set_detect_anomaly(True)


def fix_seed(func):
    """Sets torch's & numpy's random number generator seed.

    Fixing the random number generator's seed maintains consistency between
    tests by ensuring that the same test data is used every time. If this is
    not done it can make debugging problems very difficult.

    Arguments:
        func (function):
            The function that is being wrapped.

    Returns:
        wrapped (function):
            The wrapped function.
    """
    # Use functools.wraps to maintain the original function's docstring
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # Set both numpy's and pytorch's seed to zero
        np.random.seed(0)
        torch.manual_seed(0)

        # Call the function and return its result
        return func(*args, **kwargs)

    # Return the wapped function
    return wrapper


def clean_zero_padding(m, sizes):
    """Removes perturbations induced in the zero padding values by gradcheck.

    When performing gradient stability tests via PyTorch's gradcheck function
    small perturbations are induced in the input data. However, problems are
    encountered when these perturbations occur in the padding values. These
    values should always be zero, and so the test is not truly representative.
    Furthermore, this can even prevent certain tests from running. Thus this
    function serves to remove such perturbations in a gradient safe manner.

    Note that this is intended to operate on 3D matrices where. Specifically a
    batch of square matrices.

    Arguments:
        m (torch.Tensor):
            The tensor whose padding is to be cleaned.
        sizes (torch.Tensor):
            The true sizes of the tensors.

    Returns:
        cleaned (torch.Tensor):
            Cleaned tensor.

    Notes:
        This is only intended for 2D matrices packed into a 3D tensor.
    """

    # First identify the maximum tensor size
    max_size = torch.max(sizes)

    # Build a mask that is True anywhere that the tensor should be zero, i.e.
    # True for regions of the tensor that should be zero padded.
    mask_1d = (
            (torch.arange(max_size) - sizes.unsqueeze(1)) >= 0
    ).repeat(max_size, 1, 1)

    # This, rather round about, approach to generating and applying the masks
    # must be used as some PyTorch operations like masked_scatter do not seem
    # to function correctly
    mask_full = torch.zeros(*m.shape).bool()
    mask_full[mask_1d.permute(1, 2, 0)] = True
    mask_full[mask_1d.transpose(0, 1)] = True

    # Create and apply the subtraction mask
    temp = torch.zeros_like(m, device=m.device)
    temp[mask_full] = m[mask_full]
    cleaned = m - temp

    return cleaned

def __sft(self):
    """Aliese for calling .cpu().numpy()"""
    return self.cpu().numpy()


torch.Tensor.sft = __sft
