# -*- coding: utf-8 -*-
"""
Definition of the global core Hamiltonian parameters.

The core Hamiltonian is rescaling the shell-blocks of the overlap integrals
formed over the basis set by the average of the atomic self-energies and an
additional distance dependent function formed from the element parametrization.
"""

from typing import Dict, Optional
from pydantic import BaseModel


class XTBHamiltonian(BaseModel):
    """Core Hamiltonian formation parameters.

    Global parameters for the formation of the core Hamiltonian from the overlap
    integrals. Contains the required atomic & shell dependent scaling parameters
    to obtain the off-site scaling functions independent of the self-energy and
    the distance polynomial.
    """

    shell: Dict[str, float]
    """Shell-pair dependent scaling factor for off-site blocks"""
    kpair: Dict[str, float] = {}
    """Atom-pair dependent scaling factor for off-site valence blocks"""
    enscale: float
    """Electronegativity scaling factor for off-site valence blocks"""
    wexp: float
    """Exponent of the orbital exponent dependent off-site scaling factor"""
    cn: Optional[str]
    """Local environment descriptor for shifting the atomic self-energies"""
    kpol: float = 2.0
    """Scaling factor for polarization functions"""


class Hamiltonian(BaseModel):
    """Possible Hamiltonian parametrizations.

    Notes:
        Currently only the xTB Hamiltonian is supported.
    """

    xtb: XTBHamiltonian
    """Data for the xTB Hamiltonian"""
