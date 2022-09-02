"""
===========================
Post-Processing (Smoothing)
===========================

Smoothing Filters
===============

   smoothn - Spline filter and interpolator (if SciPy is present)
"""

try:
    import scipy
    from .smoothn import smoothn
except ImportError:
    raise ImportError(
        "Could not import scipy as package is not found"
    )

__all__ = ["smoothn"]