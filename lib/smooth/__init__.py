"""
===========================
Post-Processing (Smoothing)
===========================

Smoothing Filters
===============

   smoothn - Spline filter and interpolator (needs scipy and matplotlib)
"""

try:
    import scipy

    del scipy

    # scipy was found
    from ._smoothn import smoothn
except ImportError:
    raise ImportError("Could not import scipy as package is not found")

__all__ = ["smoothn"]
