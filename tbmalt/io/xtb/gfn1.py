# This file is part of tbmalt.
# SPDX-Identifier: LGPL-3.0-or-later

"""
Self-consistent extended tight binding Hamiltonian with isotropic second order
electrostatic contributions and third order on-site contributions.

Geometry, frequency and non-covalent interactions parametrisation for elements up to Z=86.

Cite as:
S. Grimme, C. Bannwarth, P. Shushkov,
*J. Chem. Theory Comput.*, 2017, 13, 1989-2009.
DOI: `10.1021/acs.jctc.7b00118 <https://dx.doi.org/10.1021/acs.jctc.7b00118>`_
"""

from .base import Param
import os.path as op
import tomli as toml

with open(op.join(op.dirname(__file__), "gfn1-xtb.toml"), "rb") as fd:
    GFN1_XTB = Param(**toml.load(fd))
