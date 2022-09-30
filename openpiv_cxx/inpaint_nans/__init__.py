"""
============
Inpaint NaNs
============

Replace NaNs
============
    replace_outliers - iteratively replace NaNs

"""
from ._inpaint_nans import *

__all__ = [s for s in dir() if not s.startswith("_")]
