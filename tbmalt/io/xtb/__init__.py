# This file is part of tbmalt.
# SPDX-Identifier: LGPL-3.0-or-later
"""
This module defines the parametrization of the extended tight-binding Hamiltonian.

The structure of the parametrization is adapted from the `tblite`_ library and
separates the species-specific parameter records from the general interactions
included in the method.

.. _tblite: https://tblite.readthedocs.io
"""

from .base import Param
