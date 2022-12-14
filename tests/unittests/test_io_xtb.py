# This file is part of tbmalt.
# SPDX-Identifier: LGPL-3.0-or-later

import pytest

import tbmalt.io.xtb
import tomli as toml


def test_builtin_gfn1():
    from tbmalt.io.xtb.gfn1 import GFN1_XTB as par

    assert par.meta.name == "GFN1-xTB"
    assert par.meta.version == 1

    assert par.dispersion is not None
    assert par.repulsion is not None
    assert par.charge is not None
    assert par.multipole is None
    assert par.halogen is not None
    assert par.thirdorder is not None

    assert par.hamiltonian.xtb.cn == "exp"
    assert "sp" in par.hamiltonian.xtb.shell
    assert pytest.approx(par.hamiltonian.xtb.wexp) == 0.0
    assert pytest.approx(par.hamiltonian.xtb.kpol) == 2.85
    assert pytest.approx(par.hamiltonian.xtb.enscale) == -7.0e-3

    assert "Te" in par.element


def test_param_minimal():
    data = """
    [hamiltonian.xtb]
    wexp = 5.0000000000000000E-01
    kpol = 2.0000000000000000E+00
    enscale = 2.0000000000000000E-02
    cn = "gfn"
    shell = {ss=1.85, pp=2.23, dd=2.23, sd=2.00, pd=2}
    [element.H]
    shells = [ "1s" ]
    levels = [ -1.0707210999999999E+01 ]
    slater = [ 1.2300000000000000E+00 ]
    ngauss = [ 3 ]
    refocc = [ 1.0000000000000000E+00 ]
    shpoly = [ -9.5361800000000000E-03 ]
    kcn = [ -5.0000000000000003E-02 ]
    gam = 4.0577099999999999E-01
    lgam = [ 1.0000000000000000E+00 ]
    gam3 = 8.0000000000000016E-02
    zeff = 1.1053880000000000E+00
    arep = 2.2137169999999999E+00
    xbond = 0.0000000000000000E+00
    dkernel = 5.5638889999999996E-02
    qkernel = 2.7430999999999999E-04
    mprad = 1.3999999999999999E+00
    mpvcn = 1.0000000000000000E+00
    en = 2.2000000000000002E+00
    [element.C]
    shells = [ "2s", "2p" ]
    levels = [ -1.3970922000000002E+01, -1.0063292000000001E+01 ]
    slater = [ 2.0964320000000001E+00, 1.8000000000000000E+00 ]
    ngauss = [ 4, 4 ]
    refocc = [ 1.0000000000000000E+00, 3.0000000000000000E+00 ]
    shpoly = [ -2.2943210000000002E-02, -2.7110200000000002E-03 ]
    kcn = [ -1.0214400000000000E-02, 1.6165700000000002E-02 ]
    gam = 5.3801500000000002E-01
    lgam = [ 1.0000000000000000E+00, 1.1056357999999999E+00 ]
    gam3 = 1.5000000000000002E-01
    zeff = 4.2310780000000001E+00
    arep = 1.2476550000000000E+00
    xbond = 0.0000000000000000E+00
    dkernel = -4.1167399999999998E-03
    qkernel = 2.1358300000000000E-03
    mprad = 3.0000000000000000E+00
    mpvcn = 3.0000000000000000E+00
    en = 2.5499999999999998E+00
    """

    par = tbmalt.io.xtb.Param(**toml.loads(data))

    assert "H" in par.element
    assert "C" in par.element
