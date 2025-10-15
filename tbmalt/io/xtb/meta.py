# -*- coding: utf-8 -*-
"""Meta data associated with a parametrization.

This is primarily used for data format identification.
"""

from typing import Optional
try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel


class Meta(BaseModel):
    """Representation of the meta data for a parametrization."""

    name: Optional[str]
    """Name of the represented method"""
    reference: Optional[str]
    """References relevant for the parametrization records"""
    version: int = 0
    """Version of the represented method"""
