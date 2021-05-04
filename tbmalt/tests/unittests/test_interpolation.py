"""Performs tests on functions in the tbmalt.common.maths.interpolator."""
import torch
import numpy as np
from torch.autograd import gradcheck
import pytest
from tbmalt.tests.test_utils import *
from tbmalt.common.maths.interpolator import PolyInterpU
torch.set_default_dtype(torch.float64)

data = np.loadtxt('data/HH.dat')


###################################
# TBMaLT.common.maths.PolyInterpU #
###################################
def test_polyinterpu_single(device):
    """Test single distances interpolation in Hamiltonian."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, 9]).to(device)
    ref = -2.7051061568285979E-002
    fit = PolyInterpU(xa, yb)
    pred = fit(torch.tensor([4.3463697737315234], device=device))

    assert abs(ref - pred) < 1E-14


def test_polyinterpu_batch(device):
    """Test multi distance interpolations in Hamiltonian."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, 9]).to(device)
    fit = PolyInterpU(xa, yb)
    pred = fit(torch.tensor(
        [4.3463697737315234, 8.1258217508893704], device=device))
    ref = torch.tensor(
        [-2.7051061568285979E-002, -7.9892794322938818E-005], device=device)

    assert (abs(ref - pred) < 1E-14).all()


def test_polyinterpu_tail(device):
    """Test the smooth Hamiltonian to zero code in the tail."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, 9]).to(device)
    fit = PolyInterpU(xa, yb)
    pred = fit(torch.tensor([10.015547739468293], device=device))
    ref = torch.tensor([1.2296664801642019E-005], device=device)

    assert (abs(ref - pred) < 1E-11).all()


@pytest.mark.grad
def test_polyinterpu_grad(device):
    """Gradient evaluation of polynomial interpolation."""
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                     device=device)
    y = torch.rand(10, device=device)
    fit = PolyInterpU(x, y)
    xi = torch.tensor([0.6], requires_grad=True, device=device)
    grad_is_safe = gradcheck(fit, xi, raise_exception=False)

    assert grad_is_safe, 'Gradient stability test'
