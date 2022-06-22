"""
============================
Post-Processing (Validation)
============================

Correlation-Based
=================

   sig2noise_val - Validate signal-to-noise or peak height
   
   
Vector-Based
============

   global_val - Validate vectors within a certain velocity range
   global_std - Validate vectors within a standard deviation threshold
   global_z_score - Validate vector within a certain z-score threshold
"""

from .correlation_based import *
from .vector_based import *

__all__ = [s for s in dir() if not s.startswith('_')]