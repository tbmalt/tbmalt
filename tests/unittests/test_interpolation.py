"""Performs tests on functions in the tbmalt.common.maths.interpolator."""
import torch
from torch.autograd import gradcheck
from torch.nn import Parameter
from scipy.interpolate import CubicSpline as SciCubSpl
import pytest
from tests.test_utils import *
from tbmalt.common.maths.interpolation import PolyInterpU, CubicSpline, BicubInterpSpl
from tbmalt import Geometry, OrbitalInfo
from tbmalt.common.batch import bT
torch.set_default_dtype(torch.float64)

data = np.loadtxt('tests/unittests/data/polyinterp/HH.dat')

# The data is for C-C SS0 Hamiltonian integrals at distances 3.5 and
# 7.0 Bohr, with nine compression radii grid points: 2.5, 3., 3.5, 4.,
# 4.5, 5., 6., 8., 10.
radii = torch.tensor([2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])
bi_data = torch.tensor([
    [[-0.12041970309907, -0.12607062197295, -0.13040827432332,
      -0.13379929249464, -0.13648979667355, -0.13865182283744,
      -0.14185168564134, -0.14560389706176, -0.14758877731604],
     [-0.12607062197295, -0.13165395551599, -0.13594427270972,
      -0.13930083715157, -0.14196546863873, -0.14410756940051,
      -0.14727934804605, -0.15100092324407, -0.15297102269216],
     [-0.13040827432332, -0.13594427270972, -0.14020046890345,
      -0.14353164228322, -0.14617687490005, -0.14830383498558,
      -0.15145387625500, -0.15515124646351, -0.15710939243317],
     [-0.13379929249464, -0.13930083715157, -0.14353164228322,
      -0.14684363199158, -0.14947400310512, -0.15158924015339,
      -0.15472232979364, -0.15840050906303, -0.16034903766059],
     [-0.13648979667355, -0.14196546863873, -0.14617687490005,
      -0.14947400310512, -0.15209276473078, -0.15419880720868,
      -0.15731845434416, -0.16098131854589, -0.16292210496253],
     [-0.13865182283744, -0.14410756940051, -0.14830383498558,
      -0.15158924015339, -0.15419880720868, -0.15629749116700,
      -0.15940636863209, -0.16305688707348, -0.16499139096909],
     [-0.14185168564134, -0.14727934804605, -0.15145387625500,
      -0.15472232979364, -0.15731845434416, -0.15940636863209,
      -0.16249940631833, -0.16613163527834, -0.16805677960142],
     [-0.14560389706176, -0.15100092324407, -0.15515124646351,
      -0.15840050906303, -0.16098131854589, -0.16305688707348,
      -0.16613163527834, -0.16974258224665, -0.17165677626688],
     [-0.14758877731604, -0.15297102269216, -0.15710939243317,
      -0.16034903766059, -0.16292210496253, -0.16499139096909,
      -0.16805677960142, -0.17165677626688, -0.17356527661582]],

    [[-0.00048887396938, -0.00068719172255, -0.00089615572813,
      -0.00110822000503, -0.00131713930768, -0.00151833881397,
      -0.00188705458906, -0.00247207283481, -0.00288176055265],
     [-0.00068719172255, -0.00091593762483, -0.00114870522275,
      -0.00137929526334, -0.00160265376523, -0.00181516696465,
      -0.00219992581120, -0.00280222336060, -0.00322014165558],
     [-0.00089615572813, -0.00114870522275, -0.00139947437076,
      -0.00164371478637, -0.00187749384896, -0.00209803190495,
      -0.00249393682743, -0.00310785939940, -0.00353111535357],
     [-0.00110822000503, -0.00137929526334, -0.00164371478637,
      -0.00189809578565, -0.00213948678193, -0.00236580381585,
      -0.00276959728663, -0.00339152297036, -0.00381833426462],
     [-0.00131713930768, -0.00160265376523, -0.00187749384896,
      -0.00213948678193, -0.00238651181973, -0.00261705496987,
      -0.00302653138580, -0.00365408158252, -0.00408331689105],
     [-0.00151833881397, -0.00181516696465, -0.00209803190495,
      -0.00236580381585, -0.00261705496987, -0.00285073412371,
      -0.00326437240183, -0.00389594965820, -0.00432687020740],
     [-0.00188705458906, -0.00219992581120, -0.00249393682743,
      -0.00276959728663, -0.00302653138580, -0.00326437240183,
      -0.00368343868575, -0.00432011450148, -0.00475308374702],
     [-0.00247207283481, -0.00280222336060, -0.00310785939940,
      -0.00339152297036, -0.00365408158252, -0.00389594965820,
      -0.00432011450148, -0.00496132473729, -0.00539597364693],
     [-0.00288176055265, -0.00322014165558, -0.00353111535357,
      -0.00381833426462, -0.00408331689105, -0.00432687020740,
      -0.00475308374702, -0.00539597364693, -0.00583116366768]]])


###################################
# TBMaLT.common.maths.BicubInterp #
###################################
def test_bicubinterp_single(device):
    """Test single distances interpolation in Hamiltonian."""
    # 1. choose the distance 3.5 Bohr
    bi_interp1 = BicubInterpSpl(radii.to(device), bi_data[0].to(device))
    data11 = bi_interp1(torch.tensor([3.75, 3.75], device=device))
    assert torch.abs(torch.tensor([-0.14372390269510], device=device)
                     - data11).lt(1e-3), 'tolerance error'

    data12 = bi_interp1(torch.tensor([7.0, 7.0], device=device))
    assert torch.abs(torch.tensor([-0.16674101544150], device=device)
                     - data12).lt(1e-3), 'tolerance error'

    # 1. choose the distance 7.0 Bohr
    bi_interp2 = BicubInterpSpl(radii.to(device), bi_data[1].to(device))
    data21 = bi_interp2(torch.tensor([3.75, 3.75], device=device))
    assert torch.abs(torch.tensor([-0.00164882038419], device=device)
                     - data21).lt(1e-4), 'tolerance error'

    data22 = bi_interp2(torch.tensor([7.0, 7.0], device=device))
    assert torch.abs(torch.tensor([-0.00438283226086], device=device)
                     - data22).lt(1e-4), 'tolerance error'


def test_bicubinterp_batch(device):
    """Test batch distances interpolation in Hamiltonian."""
    bi_interp = BicubInterpSpl(radii.to(device), bi_data.to(device))
    data = bi_interp(torch.tensor([[3.75, 3.75],
                                   [7.0, 7.0]], device=device))
    ref = torch.tensor([[-0.14372390269510, -0.00438283226086]], device=device)

    assert torch.abs(ref - data).lt(1e-3).all(), 'tolerance error'


@pytest.mark.grad
def test_bicubinterp_grad(device):
    """Gradient evaluation of BicubInterp interpolation."""
    fit = BicubInterpSpl(radii.to(device), bi_data[0].to(device))
    xi = torch.tensor([3.75, 3.75], requires_grad=True, device=device)

    grad_is_safe = gradcheck(fit, xi, raise_exception=False)

    assert grad_is_safe, 'Gradient stability test'


###################################
# TBMaLT.common.maths.PolyInterpU #
###################################
def test_polyinterpu_single(device):
    """Test single distances interpolation in Hamiltonian."""
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = Parameter(
        torch.from_numpy(data[:, 9]).to(device), requires_grad=False)
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
    yb = Parameter(
        torch.from_numpy(data[:, 9]).to(device), requires_grad=False)
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

    # Reference data is generated by running DFTB+ calculations
    ref = torch.tensor([1.334466532678998e-05, 1.334274113885803e-05, 1.334466534542479e-05], device=device)
    tol = {"rtol": 1E-12, "atol": 1E-12}

    xt = torch.tensor([10.015547739468293], device=device)

    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = Parameter(
        torch.from_numpy(data[:, 9]).to(device), requires_grad=False)
    fit = PolyInterpU(xa, yb)

    check_1 = torch.allclose(ref[0], fit(xt), **tol)

    assert check_1, 'tail region tolerance check failed'

    # Check device: Device persistence check
    check_dev = fit(xt).device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Check tail
    fit.tail = 0.6

    check_2 = torch.allclose(ref[1], fit(xt), **tol)
    assert check_2, 'tail region out of tolerance when tail distance modified'

    # Check delta_r
    fit.tail, fit.delta_r = 1.0, 1E-4

    check_3 = torch.allclose(ref[2], fit(xt), **tol)
    assert check_3, 'tail region out of tolerance when displacement distance modified'

@pytest.mark.grad
def test_polyinterpu_grad(device):
    """Gradient evaluation of polynomial interpolation."""

    x_knots = torch.linspace(0.2, 10, 50, device=device)
    x_target = torch.tensor([0.6], device=device)
    spline_proxy = lambda y: PolyInterpU(x_knots, y)(x_target)

    y_knots = Parameter(torch.from_numpy(data[:, 9]).to(device))
    grad_is_safe = gradcheck(spline_proxy, y_knots, raise_exception=False)
    assert grad_is_safe, 'Gradient stability test failed'

    y_knots = Parameter(torch.from_numpy(data[:, [9, 19]]).to(device))
    grad_is_safe = gradcheck(spline_proxy, y_knots, raise_exception=False)
    assert grad_is_safe, 'Gradient stability test failed'


###################################
# TBMaLT.common.maths.CubicSpline #
###################################
def test_spline_cubic(device):
    """Test distances interpolation in SK tables."""
    # Test interpolation with one dimension yy value
    xa0 = torch.linspace(0.2, 10, 50, device=device)
    yb0 = Parameter(torch.from_numpy(data[:, 9]).to(device),
                    requires_grad=False)

    fit0 = CubicSpline(xa0, yb0)
    pred0 = fit0(
        torch.tensor([2.0, 2.5, 3.5, 4.9, 5.5, 6.2], device=device))

    cs0 = SciCubSpl(xa0.cpu(), yb0.cpu())
    ref0 = torch.from_numpy(cs0(np.array([2.0, 2.5, 3.5, 4.9, 5.5, 6.2])))

    assert torch.allclose(ref0, pred0.cpu(), atol=1e-6), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred0.device == xa0.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Test interpolation with two dimension yy value
    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = Parameter(torch.from_numpy(data[:, [9, 19]]).to(device),
                   requires_grad=False)

    fit = CubicSpline(xa, yb)
    pred = fit(
        torch.tensor([2.0, 2.5, 3.5, 4.9, 5.5, 6.2], device=device))

    cs = SciCubSpl(xa.cpu(), yb.cpu())
    ref = torch.from_numpy(cs(torch.tensor([2.0, 2.5, 3.5, 4.9, 5.5, 6.2])))

    assert torch.allclose(ref, pred.cpu(), atol=1e-5), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred.device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'


def test_cubic_spline_tail(device):
    """Test the smooth Hamiltonian to zero code in the tail."""

    distance = torch.tensor([10.015547739468293], device=device)

    xa = torch.linspace(0.2, 10, 50, device=device)
    yb = Parameter(torch.from_numpy(data[:, 9]).to(device),
                   requires_grad=False)
    fit = CubicSpline(xa, yb)
    pred = fit(distance)
    ref = torch.tensor([1.33409107588738e-05], device=device)

    assert (abs(ref - pred) < 1E-11).all(), 'tolerance check'

    # Check device: Device persistence check
    check_dev = pred.device == xa.device
    assert check_dev, 'Device of prediction is not consistent with input'

    # Check tail
    fit.tail = 0.6
    pred2 = fit(distance)
    ref2 = torch.tensor([1.3334001762933034E-005], device=device)
    assert (abs(ref2 - pred2) < 1E-11).all(), 'tolerance check'



@pytest.mark.grad
def test_cubic_spline_grad(device):
    """Gradient evaluation of cubic spline interpolation."""

    x_knots = torch.linspace(0.2, 10, 50, device=device)
    x_target = torch.tensor([0.6], device=device)
    spline_proxy = lambda y: CubicSpline(x_knots, y)(x_target)

    y_knots = Parameter(torch.from_numpy(data[:, 9]).to(device))
    grad_is_safe = gradcheck(spline_proxy, y_knots, raise_exception=False)
    assert grad_is_safe, 'Gradient stability test failed'

    y_knots = Parameter(torch.from_numpy(data[:, [9, 19]]).to(device))
    grad_is_safe = gradcheck(spline_proxy, y_knots, raise_exception=False)
    assert grad_is_safe, 'Gradient stability test failed'

