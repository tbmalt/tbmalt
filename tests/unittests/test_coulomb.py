"""Ewald summation unit-tests.

Here the functionality of the Coulomb module
is tested."""

import torch
import pytest
from tbmalt import Geometry
from tbmalt.physics.dftb.coulomb import Ewald3d, build_coulomb_matrix
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)


def test_get_alpha(device):
    """Test optimizing alpha for the Ewald sum."""
    latvec = torch.tensor([[2., 0., 0.], [0., 4., 0.], [0., 0., 2.]],
                          device=device)
    positions = torch.tensor([[0., 0., 0.], [0., 2., 0.]], device=device)
    numbers = torch.tensor([1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    periodic = system.periodic
    coulomb = Ewald3d(system, periodic, method='search')

    # Check the tolerance and device
    check1 = torch.max(abs(coulomb.alpha - alpha_from_dftbplus.to(device))
                       ) < 1E-14
    check2 = coulomb.alpha.device == device

    assert check1, 'Searching for alpha value failed (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_get_alpha_batch(device):
    """Test optimizing alphas in batch calculation."""
    latvec = [torch.tensor([[3., 3., 0.], [0., 3., 3.], [3., 0., 3.]],
                           device=device),
              torch.tensor([[5., 5., 0.], [0., 5., 5.], [5., 0., 5.]],
                           device=device),
              torch.tensor([[3., -3., 3.], [-3., 3., 3.], [3., 3., -3.]],
                           device=device)]
    positions = [torch.tensor([[0., 0., 0.], [0., 2., 0.]], device=device),
                 torch.tensor([[0., 0., 0.], [0., 2., 0.]], device=device),
                 torch.tensor([[0., 0., 0.], [0., 2., 0.]], device=device)]
    numbers = [torch.tensor([1, 1], device=device),
               torch.tensor([1, 1], device=device),
               torch.tensor([1, 1], device=device)]
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    periodic = system.periodic
    coulomb = Ewald3d(system, periodic, method='search')

    # Check the tolerance and device
    check1 = torch.max(abs(coulomb.alpha - alpha_batch_from_dftbplus.to(device))
                       ) < 1E-14
    check2 = coulomb.alpha.device == device

    assert check1, 'Searching  for alpha value failed (batch).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_3d(device):
    """Test Ewald summation for ch4 with 3d pbc."""
    latvec = torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]],
                          device=device)
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
        [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system, method='search')

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_ch4_from_dftbplus.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 3d pbc is out of tolerance (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_3d_batch(device):
    """Test batch calculation of Ewald summation for 3d pbc."""
    latvec = [torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]],
                           device=device),
              torch.tensor([[1., 1., 0.], [0., 3., 3.], [2., 0., 2.]],
                           device=device),
              torch.tensor([[4., 0., 0.], [0., 5., 0.], [0., 0., 6.]],
                           device=device)]
    positions = [torch.tensor([
             [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
             [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device),
         torch.tensor([[0., 0., 0.], [0., 2., 0.]], device=device),
         torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056],
                       [2.244, 0.660, 0.778]], device=device)]
    numbers = [torch.tensor([6, 1, 1, 1, 1], device=device),
               torch.tensor([1, 1], device=device),
               torch.tensor([1, 8, 1], device=device)]
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system, method='search')

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_batch_3d.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 3d pbc is out of tolerance (batch).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_2d_ch4(device):
    """Test Ewald summation for ch4 with 2d pbc."""
    latvec = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 0.]],
                          device=device)
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
        [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_2d_ch4.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 2d pbc is out of tolerance (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_2d_h2(device):
    """Test Ewald summation for h2 with 2d pbc."""
    latvec = torch.tensor([[4., 0., 0.], [0., 0., 0.], [0., 0., 5.]],
                          device=device)
    positions = torch.tensor([[0., 0., 0.], [0., 0., 2.]], device=device)
    numbers = torch.tensor([1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_2d_h2.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 2d pbc is out of tolerance (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_2d_h2o(device):
    """Test Ewald summation for h2o with 2d pbc."""
    latvec = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 0.]],
                          device=device)
    positions = torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056],
                              [2.244, 0.660, 0.778]], device=device)
    numbers = torch.tensor([1, 8, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_2d_h2o.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 2d pbc is out of tolerance (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_2d_batch(device):
    """Test batch calculation of Ewald summation for 2d pbc."""
    latvec = [torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 0.]],
                           device=device),
              torch.tensor([[4., 0., 0.], [0., 0., 0.], [0., 0., 5.]],
                           device=device),
              torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 0.]],
                           device=device)]
    positions = [torch.tensor([
             [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
             [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device),
         torch.tensor([[0., 0., 0.], [0., 0., 2.]], device=device),
         torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056],
                       [2.244, 0.660, 0.778]], device=device)]
    numbers = [torch.tensor([6, 1, 1, 1, 1], device=device),
               torch.tensor([1, 1], device=device),
               torch.tensor([1, 8, 1], device=device)]
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_batch_2d.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 2d pbc is out of tolerance (batch).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_1d_ch4(device):
    """Test Ewald summation for ch4 with 1d pbc."""
    latvec = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 4.]],
                          device=device)
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
        [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_1d_ch4.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 1d pbc is out of tolerance (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_1d_h2(device):
    """Test Ewald summation for h2 with 1d pbc."""
    latvec = torch.tensor([[4., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                          device=device)
    positions = torch.tensor([[0., 0., 0.], [2., 0., 0.]], device=device)
    numbers = torch.tensor([1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_1d_h2.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 1d pbc is out of tolerance (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_1d_h2o(device):
    """Test Ewald summation for h2o with 1d pbc."""
    latvec = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 6.]],
                          device=device)
    positions = torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056],
                              [2.244, 0.660, 0.778]], device=device)
    numbers = torch.tensor([1, 8, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_1d_h2o.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 1d pbc is out of tolerance (single).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_1d_batch(device):
    """Test batch calculation of Ewald summation for 1d pbc."""
    latvec = [torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 4.]],
                           device=device),
              torch.tensor([[4., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                           device=device),
              torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 6.]],
                           device=device)]
    positions = [torch.tensor([
             [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
             [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device),
         torch.tensor([[0., 0., 0.], [2., 0., 0.]], device=device),
         torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056],
                       [2.244, 0.660, 0.778]], device=device)]
    numbers = [torch.tensor([6, 1, 1, 1, 1], device=device),
               torch.tensor([1, 1], device=device),
               torch.tensor([1, 8, 1], device=device)]
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Check the tolerance and device
    check1 = torch.max(abs(invrmat - invr_batch_1d.to(device))
                       ) < 1E-15
    check2 = invrmat.device == device

    assert check1, 'Ewald summation for 1d pbc is out of tolerance (batchg).'
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_3d_convergence(device):
    """Test the convergence of 3d Ewald summation."""
    latvec = torch.tensor([[4., 4., 0.], [0., 4., 4.], [4., 0., 4.]],
                          device=device)
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
        [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system, method='search')

    # Increase the positions among atoms twice
    system2 = Geometry(numbers, positions * 2, latvec * 2, units='a',
                       cutoff=cutoff)
    invrmat2 = build_coulomb_matrix(system2, method='search')

    # Check the convergence of 3d ewald summation
    check1 = torch.max(abs(invrmat - 2 * invrmat2)
                       ) < 1E-15
    assert check1, '3d Ewald summation does not converge.'

    # Check the device
    check2a = invrmat.device == device
    check2b = invrmat2.device == device
    check2 = (check2a and check2b)
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_2d_convergence(device):
    """Test the convergence of 2d Ewald summation."""
    latvec = torch.tensor([[4., 0., 0.], [0., 5., 0.], [0., 0., 0.]],
                          device=device)
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
        [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Increase the positions among atoms twice
    system2 = Geometry(numbers, positions * 2, latvec * 2, units='a',
                       cutoff=cutoff)
    invrmat2 = build_coulomb_matrix(system2)

    # Check the convergence of 3d ewald summation
    check1 = torch.max(abs(invrmat - 2 * invrmat2)
                       ) < 1E-15
    assert check1, '2d Ewald summation does not converge.'

    # Check the device
    check2a = invrmat.device == device
    check2b = invrmat2.device == device
    check2 = (check2a and check2b)
    assert check2, 'Coulomb returned on incorrect device.'


def test_coulomb_1d_convergence(device):
    """Test the convergence of 1d Ewald summation."""
    latvec = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 5.]],
                          device=device)
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6],
        [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]], device=device)
    numbers = torch.tensor([6, 1, 1, 1, 1], device=device)
    cutoff = torch.tensor([9.98], device=device)
    system = Geometry(numbers, positions, latvec, units='a', cutoff=cutoff)
    invrmat = build_coulomb_matrix(system)

    # Increase the positions among atoms twice
    system2 = Geometry(numbers, positions * 2, latvec * 2, units='a',
                       cutoff=cutoff)
    invrmat2 = build_coulomb_matrix(system2)

    # Check the convergence of 3d ewald summation
    check1 = torch.max(abs(invrmat - 2 * invrmat2)
                       ) < 1E-15
    assert check1, '1d Ewald summation does not converge.'

    # Check the device
    check2a = invrmat.device == device
    check2b = invrmat2.device == device
    check2 = (check2a and check2b)
    assert check2, 'Coulomb returned on incorrect device.'


# Alpha values from dftb+
alpha_from_dftbplus = torch.tensor([0.47513600000000000])
alpha_batch_from_dftbplus = torch.tensor(
    [0.41943039999999998, 0.20971519999999999, 0.29360127999999996])

# 1/R matrix from dftb+
invr_ch4_from_dftbplus = torch.tensor(
    [[-0.30327559498234302, 0.21535637403937258,
      0.21535637403937263, 0.21535637403937263, 0.21535637403937263],
     [0.21535637403937258, -0.30327559498234302,
      0.14990319370389693, 0.14990319370389693, 0.14990319370389676],
     [0.21535637403937263, 0.14990319370389693,
      -0.30327559498234302, 0.03400749583503342, 0.03400749583503349],
     [0.21535637403937263, 0.14990319370389693,
      0.03400749583503342, -0.30327559498234302, 0.03400749583503349],
     [0.21535637403937263, 0.14990319370389676,
      0.03400749583503349, 0.03400749583503349, -0.30327559498234302]])
invr_h2_from_dftbplus = torch.tensor(
    [[-0.49711993867470139, -0.20209287655651773],
     [-0.20209287655651773, -0.49711993867470139]])
invr_h2o_from_dftbplus = torch.tensor(
    [[-0.29116782550225667, 0.26371910503769314, 0.07981412460933784],
     [0.26371910503769314, -0.29116782550225667, 0.24773719874691910],
     [0.07981412460933784, 0.24773719874691910, -0.29116782550225667]])
invr_batch_3d = pack([invr_ch4_from_dftbplus,
                      invr_h2_from_dftbplus, invr_h2o_from_dftbplus])

# 1/R matrix for 2D pbc
invr_2d_ch4 = torch.tensor(
    [[-0.51598286520064041, -0.00769099112260654, -0.00769099112260663,
      -0.00769099112260663, -0.00769099112260663],
     [-0.00769099112260654, -0.51598286520064041, -0.04521522653847165,
      -0.04521522653847165, -0.12685977438654669],
     [-0.00769099112260663, -0.04521522653847168, -0.51598286520064041,
      -0.15539720865233969, -0.23287419605890902],
     [-0.00769099112260663, -0.04521522653847168, -0.15539720865233969,
      -0.51598286520064041, -0.23287419605890902],
     [-0.00769099112260663, -0.12685977438654669, -0.23287419605890902,
      -0.23287419605890902, -0.51598286520064041]])
invr_2d_h2 = torch.tensor(
    [[-0.45773317749163400, -0.15130277857350258],
     [-0.15130277857350258, -0.45773317749163400]])
invr_2d_h2o = torch.tensor(
    [[-0.41278629216492951, 0.13154396754549280, -0.06577126607959843],
     [0.13154396754549280, -0.41278629216492951, 0.11574945826572122],
     [-0.06577126607959843, 0.11574945826572122, -0.41278629216492951]])
invr_batch_2d = pack([invr_2d_ch4, invr_2d_h2, invr_2d_h2o])

# 1/R matrix for 1D pbc
invr_1d_ch4 = torch.tensor(
    [[-0.32945092309706636, 0.17928296859309856, 0.17928296859309845,
      0.17928296859309845, 0.17928296859309847],
     [0.17928296859309856, -0.32945092309706636, 0.09799482882684556,
      0.09799482882684556, 0.14259058801379723],
     [0.17928296859309845, 0.09799482882684556, -0.32945092309706636,
      -0.04334203356695324, -0.00707370344507031],
     [0.17928296859309845, 0.09799482882684556, -0.04334203356695324,
      -0.32945092309706636, -0.00707370344507031],
     [0.17928296859309847, 0.14259058801379720, -0.00707370344507031,
      -0.00707370344507031, -0.32945092309706636]])
invr_1d_h2 = torch.tensor(
    [[-0.28904423903899029, 0.07775347890792746],
     [0.07775347890792746, -0.28904423903899029]])
invr_1d_h2o = torch.tensor(
    [[-0.20461632104458385, 0.32712524879190535, 0.13001230561312524],
     [0.32712524879190535, -0.20461632104458385, 0.33128252827369753],
     [0.13001230561312524, 0.33128252827369753, -0.20461632104458385]])
invr_batch_1d = pack([invr_1d_ch4, invr_1d_h2, invr_1d_h2o])
