"""
OpenPIV-Python-cxx: A Python PIV package with a c++ backend
===========================================================

Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import openpiv_cxx.process``.

 filters      --- Filter PIV images and vectors
 inpaint_nans --- Replace NaNs in vector field
 interpolate  --- Interpolate and map images
 openpiv      --- OpenPIV-Python interface
 process      --- Correlation and subpixel approximation
 smooth       --- Smooth vector fields
 tools        --- Image and vector tools
 validate     --- Validate vector fields
 windef       --- Window deformation algorithms
"""

try:
    import openpiv_cxx.__set_path__
except ImportError as e:
    msg = """Error importing openpiv_cxx: you cannot import openpiv_cxx while
    being in openpiv_cxx source directory. Please exit out of openpiv_cxx resources
    and relaunch your Python interpreter."""
    raise ImportError(msg) from e

submodules = [
    "filters",
    "inpaint_nans",
    "interpolate",
    "openpiv",
    "process",
    "smooth",
    "tools",
    "validate",
    "windef",
]

__all__ = submodules


def __dir__():
    return __all__


import importlib as _importlib


def __gatattr__(name):
    if name in submodules:
        return _importlib.import_module(f"openpiv_cxx.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'openpic_cxx' has no attribue '{name}'")
