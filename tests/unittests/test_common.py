# -*- coding: utf-8 -*-
"""Unit tests associated with `tbmalt.common.__init__`."""
import numpy as np
import torch
from tests.test_utils import fix_seed
from tbmalt.common import split_by_size


@fix_seed
def test_split_by_size(device):
    """Tests functionality of `split_by_size`.

    Device and gradient tests are not needed as the results is just a view of
    the input. Single and batch test are also not carried out as they are not
    applicable to this function.

    Notes:
        The `split_by_size` function will eventually get phased out once the
        pytorch `split_with_sizes` function is fully supported in the main
        pytorch branch.
    """
    for _ in range(10):
        a = torch.rand(10, 10, 10, device=device)
        indices = list(np.random.choice(range(1, 4), 2, replace=False))
        indices.append(int(10 - sum(indices)))
        dim = int(torch.randint(0, 3, (1,))[0])
        ref = torch.split_with_sizes(a, indices, dim=dim)
        prd = split_by_size(a, torch.tensor(indices, device=device), dim=dim)

        # Check that the split operation proceeded as anticipated
        check_1 = all([torch.allclose(i, j) for i, j in zip(prd, ref)])

        assert check_1, 'Tensor split operation failed'
