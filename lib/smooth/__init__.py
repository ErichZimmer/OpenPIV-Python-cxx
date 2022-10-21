"""
===========================
Post-Processing (Smoothing)
===========================

Smoothing Filters
===============

   smoothn - Spline filter and interpolator (needs scipy and matplotlib)
"""

import numpy as np


__all__ = []


HAS_SCIPY = False
try:
    import scipy
    del scipy

    HAS_SCIPY=True
except ImportError:
    raise ImportError(
        "Could not import scipy as package is not found"
    )

if HAS_SCIPY:
    from ._smoothn import smoothn as _smth

    def smooth_spline(u, s=None, **kwargs):
        if not isinstance(u, np.ma.MaskedArray):
            u = np.ma.masked_array(u, mask=np.ma.nomask)

        return _smth(u, s=s, **kwargs)
    
    __all__ += ["smooth_spline"]
