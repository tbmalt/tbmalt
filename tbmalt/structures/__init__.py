# -*- coding: utf-8 -*-
"""A module for holding data structures and any associated code.

The `tbmalt.structures` module contains all generic data structure classes,
i.e. those python classes which act primarily as data containers. The TBMaLT
project uses classes for data storage rather than dictionaries as they offer
greater functionality and their contents are more consistent.

All data structure classes are directly accessible from the top level TBMaLT
namespace, e.g.

.. code-block:: python

    # Use this
    from tbmalt import Geometry
    # Rather than this
    from tbmalt.structures.geometry import Geometry

"""
