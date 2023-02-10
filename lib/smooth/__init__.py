"""
===========================
Post-Processing (Smoothing)
===========================

This module contains various function for vector field smoothing.

Smoothing Filters
===============

   smooth_gaussian - Gaussian smoothing
"""

import numpy as np
from ._gaussian_smooth import *

__all__ = [s for s in dir() if not s.startswith("_")]
