"""
==============
PIV processing
==============

This module contains various functions for PIV processing.

Grid Generation
===============
   get_coordinates - Get x and y coordinates of a square interrogation window
   get_field_shape - Get number of rows and columns of a grid
   get_rect_coordinates - Get x and y coordinates of a rectangular interrogation window

Correlation
===========
    correlation_to_displacement - Obtain displacements from correlation matrixes
    fft_correlate_images - Cross correlate two images to obtain a correlation matrix
    
"""
from ._pyprocess import *

__all__ = [s for s in dir() if not s.startswith("_")]
