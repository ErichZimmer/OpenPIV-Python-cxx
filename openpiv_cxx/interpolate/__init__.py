"""
=============
Interpolation
=============

Gridded 2D Interpolation
========================
    bilinear2D - 2D bilinear interpolation

Gridded 2D Mapping
==================
    taylor_expansion2D - 2D Taylor expansions with finite differences
    whittaker2D - 2D Whittaker-Shannon mapping
"""
from .interpolate import *

__all__ = [s for s in dir() if not s.startswith('_')]