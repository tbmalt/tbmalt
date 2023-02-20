"""
Test the xTB adapter.

The eigenvalues (`eig_values`) can only loosely be compared between DFTB2 and
GFN1 because of the different Coulomb potentials in the SCC. Comparison for
DFTB1 does not work because the xTB Hamiltonian always contains contributions
from the potential, even in the first diagonalization/iteration.
"""

from typing import Any, Dict, List
from math import sqrt
import numpy as np
import pytest
import torch

from tbmalt import Geometry, Basis
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb1, Dftb2
from tbmalt.physics.dftb.feeds import HubbardFeed
from tbmalt.common.batch import pack

try:
    from tbmalt.physics.xtb.feeds import (
        Gfn1HamiltonianFeed,
        Gfn1OverlapFeed,
        Gfn1OccupationFeed,
    )

    import external.dxtb.origin.dxtb as dxtb

    GFN1_XTB = dxtb.param.GFN1_XTB
    get_elem_angular = dxtb.param.get_elem_angular
except ModuleNotFoundError:
    dxtb = None


# fixture
from tests.test_utils import skf_file

Sample = Dict[str, torch.Tensor]
Samples = Dict[str, Sample]

samples: Samples = {
    "H2": {
        "numbers": torch.tensor([1, 1]),
        "positions": torch.tensor(
            [
                [0.00000000000000, 0.00000000000000, -0.70252931147690],
                [0.00000000000000, 0.00000000000000, 0.70252931147690],
            ],
        ),
        "n_electrons": torch.tensor(+2.000000000000000e00),
        "occupancy": torch.tensor(
            [
                +2.000000000000000e00,  # Ha 1s
                +0.000000000000000e00,  # Ha 2s
                +0.000000000000000e00,  # Hb 1s
                +0.000000000000000e00,  # Hb 2s
            ]
        ),
        "eig_values": torch.tensor(
            [
                -0.5292992018169639,
                -0.0928771453310359,
                -0.0746810722966491,
                +0.2626817508870521,
            ]
        ),
        "q_final_atomic": torch.tensor(
            [+1.000000000000000e00, +1.000000000000000e00],
        ),
    },
    "H2O": {
        "numbers": torch.tensor([8, 1, 1]),
        "positions": torch.tensor(
            [
                [+0.00000000000000, +0.00000000000000, -0.74288549752983],
                [-1.43472674945442, +0.00000000000000, +0.37144274876492],
                [+1.43472674945442, +0.00000000000000, +0.37144274876492],
            ]
        ),
        "n_electrons": torch.tensor(+8.000000000000000e00),
        "occupancy": torch.tensor(
            [
                +2.000000000000000e00,
                +2.000000000000000e00,
                +2.000000000000000e00,
                +2.000000000000000e00,
                +0.000000000000000e00,
                +0.000000000000000e00,  # Ha 2s
                +0.000000000000000e00,
                +0.000000000000000e00,  # Hb 2s
            ],
        ),
        "eig_values": torch.tensor(
            [
                -0.7583696084426055,
                -0.6107120737172073,
                -0.5490847898506864,
                -0.5003664436148365,
                -0.1584280237407449,
                -0.0505465495653054,
                0.3287767336257804,
                0.4590338766558852,
            ]
        ),
        "q_final_atomic": torch.tensor(
            [6.58558984371061, 0.70720507814469, 0.70720507814469],
        ),
    },
    "CH4": {
        "numbers": torch.tensor([6, 1, 1, 1, 1]),
        "positions": torch.tensor(
            [
                [+0.00000000000000, +0.00000000000000, +0.00000000000000],
                [-1.19077691784446, -1.19077691784446, -1.19077691784446],
                [+1.19077691784446, +1.19077691784446, -1.19077691784446],
                [+1.19077691784446, -1.19077691784446, +1.19077691784446],
                [-1.19077691784446, +1.19077691784446, +1.19077691784446],
            ]
        ),
        "n_electrons": torch.tensor(+8.000000000000000e00),
        "occupancy": torch.tensor(
            [
                +2.000000000000000e00,
                +2.000000000000000e00,
                +2.000000000000000e00,
                +2.000000000000000e00,
                +5.668228834912260e-41,
                +0.000000000000000e00,  # Ha 2s
                +0.000000000000000e00,
                +0.000000000000000e00,  # Hb 2s
                +0.000000000000000e00,
                +0.000000000000000e00,  # Hc 2s
                +0.000000000000000e00,
                +0.000000000000000e00,  # Hd 2s
            ],
        ),
        "eig_values": torch.tensor(
            [
                -0.6264201514179448,
                -0.5126791699511010,
                -0.5126791699511006,
                -0.5126791699511004,
                -0.1457497020613727,
                -0.0595220365483580,
                -0.0595220365483578,
                -0.0595220365483578,
                0.2652692832864244,
                0.2652692832864249,
                0.2652692832864250,
                0.5064174209157685,
            ],
        ),
        "q_final_atomic": torch.tensor(
            [
                4.30537894059011,
                0.92365526485247,
                0.92365526485247,
                0.92365526485247,
                0.92365526485247,
            ],
        ),
    },
}

sample_list = list(samples.keys())
kwargs = {"filling_scheme": "fermi", "filling_temp": 0.0036749324}

ref_overlap = np.load("tests/unittests/data/xtb/overlap.npz")
ref_h0 = np.load("tests/unittests/data/xtb/h0.npz")


###########
# UTILITY #
###########


def load_from_npz(
    npzfile: Any, name: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Get torch tensor from npz file

    Parameters
    ----------
    npzfile : Any
        Loaded npz file.
    name : str
        Name of the tensor in the npz file.
    dtype : torch.dtype
        Data type of the tensor.

    Returns
    -------
    Tensor
        Tensor from the npz file.
    """
    name = name.replace("-", "").lower()
    return torch.from_numpy(npzfile[name]).type(dtype).to(device)


##############
# TEST FEEDS #
##############


@pytest.mark.skipif(dxtb is None, reason="dxtb not yet available")
@pytest.mark.parametrize("name", sample_list)
def test_feed_single(device: torch.device, name: str) -> None:
    dtype = torch.get_default_dtype()
    tol = sqrt(torch.finfo(dtype).eps) * 10

    # create geometry
    sample = samples[name]
    positions = sample["positions"].type(dtype).to(device)
    numbers = sample["numbers"].to(device)
    geometry = Geometry(numbers, positions)

    # load refs
    ref_s = load_from_npz(ref_overlap, name, dtype, device)
    ref_h = load_from_npz(ref_h0, name, dtype, device)

    # create integral feed and get matrix
    s_feed = Gfn1OverlapFeed(GFN1_XTB, dtype, device)
    s = s_feed.matrix(geometry)
    h_feed = Gfn1HamiltonianFeed(GFN1_XTB, dtype, device)
    h = h_feed.matrix(geometry, s)

    assert torch.allclose(s, s.mT, atol=tol)
    assert torch.allclose(s, ref_s, atol=tol)

    assert torch.allclose(h, h.mT, atol=tol)
    assert torch.allclose(h, ref_h, atol=tol)


###################
# TEST CALCULATOR #
###################


def dftb_checker(
    calc_dftb: Calculator,
    name: List[str],
    sample: List[Sample],
    dtype: torch.dtype,
    device: torch.device,
):
    default_tol = sqrt(torch.finfo(dtype).eps) * 10

    def check_allclose(i, atol=default_tol, rtol=default_tol):
        predicted = getattr(calc_dftb, i)
        ref = pack([s[i] for s in sample]).type(dtype).to(device)
        is_close = torch.allclose(predicted, ref, atol=atol, rtol=rtol)
        assert is_close, f"Attribute {i} is in error for system {name}"
        if isinstance(predicted, torch.Tensor):
            device_check = predicted.device == calc_dftb.device
            assert device_check, f"Attribute {i} was returned on the wrong device"

    check_allclose("n_electrons")

    nel = getattr(calc_dftb, "n_electrons")
    assert pytest.approx(getattr(calc_dftb, "q_final").sum(-1)) == nel
    assert pytest.approx(getattr(calc_dftb, "q_final_atomic").sum(-1)) == nel
    assert pytest.approx(getattr(calc_dftb, "q_delta_atomic").sum(-1)) == 0.0

    if isinstance(calc_dftb, Dftb1):
        check_allclose("occupancy")
    elif isinstance(calc_dftb, Dftb1):
        # loose comparison with converged GFN1 orbital energies
        check_allclose("eig_values", 0.1)

        # loose comparison to charges from DFTB Hamiltonian
        check_allclose("q_final_atomic", 0.1, 0.1)
    else:
        assert False


@pytest.mark.skipif(dxtb is None, reason="dxtb not yet available")
@pytest.mark.parametrize("name", sample_list)
def test_dftb1_single(device: torch.device, name: str, skf_file) -> None:
    dtype = torch.get_default_dtype()

    # create geometry
    sample = samples[name]
    positions = sample["positions"].type(dtype).to(device)
    numbers = sample["numbers"].to(device)
    geometry = Geometry(numbers, positions)

    # H has a second s-function in GFN1-xTB!!
    basis = Basis(numbers, get_elem_angular(GFN1_XTB.element))

    # create (integral) feeds
    h_feed = Gfn1HamiltonianFeed(GFN1_XTB, dtype, device)
    s_feed = Gfn1OverlapFeed(GFN1_XTB, dtype, device)
    o_feed = Gfn1OccupationFeed(GFN1_XTB, dtype, device)

    # init calculator and trigger the calculation
    calc_dftb1 = Dftb1(h_feed, s_feed, o_feed, **kwargs)
    _ = calc_dftb1(geometry, basis)

    dftb_checker(calc_dftb1, [name], [sample], dtype, device)


@pytest.mark.skipif(dxtb is None, reason="dxtb not yet available")
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_dftb1_batch(device: torch.device, name1: str, name2: str, skf_file) -> None:
    dtype = torch.get_default_dtype()

    # create geometry
    s1, s2 = samples[name1], samples[name2]
    positions = pack([s1["positions"], s2["positions"]]).type(dtype).to(device)
    numbers = pack([s1["numbers"], s2["numbers"]]).to(device)
    geometry = Geometry(numbers, positions)

    # H has a second s-function in GFN1-xTB!!
    basis = Basis(numbers, get_elem_angular(GFN1_XTB.element))

    # create (integral) feeds
    h_feed = Gfn1HamiltonianFeed(GFN1_XTB, dtype, device)
    s_feed = Gfn1OverlapFeed(GFN1_XTB, dtype, device)
    o_feed = Gfn1OccupationFeed(GFN1_XTB, dtype, device)

    # init calculator and trigger the calculation
    calc_dftb1 = Dftb1(h_feed, s_feed, o_feed, **kwargs)
    _ = calc_dftb1(geometry, basis)

    dftb_checker(calc_dftb1, [name1, name2], [s1, s2], dtype, device)


@pytest.mark.skipif(dxtb is None, reason="dxtb not yet available")
@pytest.mark.parametrize("name", sample_list)
def test_dftb2_single(device: torch.device, name: str, skf_file) -> None:
    dtype = torch.get_default_dtype()

    # create geometry
    sample = samples[name]
    positions = sample["positions"].type(dtype).to(device)
    numbers = sample["numbers"].to(device)
    species = torch.unique(numbers).tolist()
    geometry = Geometry(numbers, positions)

    # H has a second s-function in GFN1-xTB!!
    basis = Basis(numbers, {1: [0, 0], 6: [0, 1], 8: [0, 1]})

    # create (integral) feeds
    h_feed = Gfn1HamiltonianFeed(GFN1_XTB, dtype, device)
    s_feed = Gfn1OverlapFeed(GFN1_XTB, dtype, device)
    o_feed = Gfn1OccupationFeed(GFN1_XTB, dtype, device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    # init calculator and trigger the calculation
    calc_dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
    _ = calc_dftb2(geometry, basis)

    dftb_checker(calc_dftb2, [name], [sample], dtype, device)


@pytest.mark.skipif(dxtb is None, reason="dxtb not yet available")
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_dftb2_batch(device: torch.device, name1: str, name2: str, skf_file) -> None:
    dtype = torch.get_default_dtype()

    # create geometry
    s1, s2 = samples[name1], samples[name2]
    positions = pack([s1["positions"], s2["positions"]]).type(dtype).to(device)
    numbers = pack([s1["numbers"], s2["numbers"]]).to(device)
    species = torch.unique(numbers[numbers != 0]).tolist()
    geometry = Geometry(numbers, positions)

    # H has a second s-function in GFN1-xTB!!
    basis = Basis(numbers, get_elem_angular(GFN1_XTB.element))

    # create (integral) feeds
    h_feed = Gfn1HamiltonianFeed(GFN1_XTB, dtype, device)
    s_feed = Gfn1OverlapFeed(GFN1_XTB, dtype, device)
    o_feed = Gfn1OccupationFeed(GFN1_XTB, dtype, device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)

    # init calculator and trigger the calculation
    calc_dftb2 = Dftb2(h_feed, s_feed, o_feed, u_feed, **kwargs)
    _ = calc_dftb2(geometry, basis)

    dftb_checker(calc_dftb2, [name1, name2], [s1, s2], dtype, device)
