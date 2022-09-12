"""
=============
Input Checker
=============

2D Input Checkers
=================
    check_nd - check dimensions of arrays

"""
from ._input_checker import *

__all__ = [s for s in dir() if not s.startswith("_")]
