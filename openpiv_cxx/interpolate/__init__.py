"""
=============
Interpolation
=============

Gridded 2D Interpolation
========================

    bilinear2D - 2D bilinear interpolation
    whittaker2D - 2D Whittaker-Shannon interpolation
"""

from .interpolate import whittaker2D, bilinear2D

__all__ = [
    "bilinear2D",
    "whittaker2D"
]