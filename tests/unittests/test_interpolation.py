"""Performs tests on functions in the tbmalt.common.maths.interpolator."""
import torch
from torch.autograd import gradcheck
from scipy.interpolate import CubicSpline as SciCubSpl
import pytest
from tests.test_utils import *
from tbmalt.common.maths.interpolation import PolyInterpU, CubicSpline
torch.set_default_dtype(torch.float64)

data = np.loadtxt('tests/unittests/data/polyinterp/HH.dat')


###################################
# TBMaLT.common.maths.PolyInterpU #
###################################
def test_polyinterpu_single(device):
    """Test single distances interpolation in Hamiltonian."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, 9]).to(device)
    ref = -2.7051061568285979E-002
    fit = PolyInterpU(xa, yb, n_interp=8, n_interp_r=4)
    pred = fit(torch.tensor([4.3463697737315234], device=device))

    assert abs(ref - pred) < 1E-14, 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred.device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Check n_interp
    ref2 = -2.705744877076714E-02
    fit.n_interp, fit.n_interp_r = 3, 2
    pred = fit(torch.tensor([4.3463697737315234], device=device))

    assert abs(ref2 - pred) < 1E-14, 'tolerance check'


def test_polyinterpu_batch(device):
    """Test multi distance interpolations in Hamiltonian."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, 9]).to(device)
    fit = PolyInterpU(xa, yb)
    pred = fit(torch.tensor(
        [4.3463697737315234, 8.1258217508893704], device=device))
    ref = torch.tensor(
        [-2.7051061568285979E-002, -7.9892794322938818E-005], device=device)

    assert (abs(ref - pred) < 1E-14).all(), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred.device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Check n_interp
    ref2 = torch.tensor(
        [-2.705744877076714E-02, -8.008294135339589E-05], device=device)
    fit.n_interp, fit.n_interp_r = 3, 2
    pred = fit(torch.tensor(
        [4.3463697737315234, 8.1258217508893704], device=device))

    assert (abs(ref2 - pred) < 1E-14).all(), 'tolerance check'


def test_polyinterpu_tail(device):
    """Test the smooth Hamiltonian to zero code in the tail."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, 9]).to(device)
    fit = PolyInterpU(xa, yb)
    pred = fit(torch.tensor([10.015547739468293], device=device))
    ref = torch.tensor([1.2296664801642019E-005], device=device)

    assert (abs(ref - pred) < 1E-11).all(), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred.device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Check tail
    fit.tail = 0.6
    pred2 = fit(torch.tensor([10.015547739468293], device=device))
    ref2 = torch.tensor([9.760620707717307E-06], device=device)

    assert (abs(ref2 - pred2) < 1E-12).all(), 'tolerance check'

    # Check delta_r
    fit.tail, fit.delta_r = 1.0, 1E-4
    pred3 = fit(torch.tensor([10.015547739468293], device=device))
    ref3 = torch.tensor([1.229666474639370E-05], device=device)

    assert (abs(ref3 - pred3) < 1E-13).all(), 'tolerance check'


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


###################################
# TBMaLT.common.maths.CubicSpline #
###################################
def test_spline_cubic(device):
    """Test distances interpolation in SK tables."""
    # Test interpolation with one dimension yy value
    xa0 = torch.linspace(0.2, 10, 50, device=device)
    yb0 = torch.from_numpy(data[:, 9]).to(device)

    fit0 = CubicSpline(xa0, yb0)
    pred0 = fit0(torch.tensor([2.0, 2.5, 3.5, 4.9, 5.5, 6.2], device=device))

    cs0 = SciCubSpl(xa0.cpu(), yb0.cpu())
    ref0 = torch.from_numpy(cs0(np.array([2.0, 2.5, 3.5, 4.9, 5.5, 6.2])))

    assert torch.allclose(ref0, pred0.cpu()), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred0.device == xa0.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Test interpolation with two dimension yy value
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, [9, 19]]).to(device)

    fit = CubicSpline(xa, yb)
    pred = fit(torch.tensor([2.0, 2.5, 3.5, 4.9, 5.5, 6.2], device=device))

    cs = SciCubSpl(xa.cpu(), yb.cpu())
    ref = torch.from_numpy(cs(torch.tensor([2.0, 2.5, 3.5, 4.9, 5.5, 6.2])))

    assert torch.allclose(ref, pred.cpu()), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred.device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'


def test_cubic_spline_tail(device):
    """Test the smooth Hamiltonian to zero code in the tail."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, 9]).to(device)
    fit = CubicSpline(xa, yb)
    pred = fit(torch.tensor([10.015547739468293], device=device))
    ref = torch.tensor([1.2296664801642019E-005], device=device)

    assert (abs(ref - pred) < 1E-11).all(), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred.device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Check tail
    fit.tail = 0.6
    pred2 = fit(torch.tensor([10.015547739468293], device=device))
    ref2 = torch.tensor([9.760620707717307E-06], device=device)

    assert (abs(ref2 - pred2) < 1E-12).all(), 'tolerance check'

    # Check delta_r
    fit.tail, fit.delta_r = 1.0, 1E-4
    pred3 = fit(torch.tensor([10.015547739468293], device=device))
    ref3 = torch.tensor([1.229666474639370E-05], device=device)

    assert (abs(ref3 - pred3) < 1E-13).all(), 'tolerance check'


@pytest.mark.grad
def test_cubic_spline_grad(device):
    """Gradient evaluation of cubic spline interpolation."""
    xa0 = torch.linspace(0.2, 10, 50, device=device)
    yb0 = torch.from_numpy(data[:, 9]).to(device)
    fit = CubicSpline(xa0, yb0)
    xi = torch.tensor([0.6], requires_grad=True, device=device)
    grad_is_safe = gradcheck(fit, xi, raise_exception=False)

    assert grad_is_safe, 'Gradient stability test'

    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = torch.from_numpy(data[:, [9, 19]]).to(device)
    fit = CubicSpline(xa, yb)
    xi = torch.tensor([0.6], requires_grad=True, device=device)
    grad_is_safe = gradcheck(fit, xi, raise_exception=False)

    assert grad_is_safe, 'Gradient stability test'