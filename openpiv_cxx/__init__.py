try:
    import openpiv_cxx.__set_path__
except ImportError as e:
    msg = """Error importing openpiv_cxx: you cannot import openpiv_cxx while
    being in openpiv_cxx source directory. Please exit out of openpiv_cxx resources
    and relaunch your Python interpreter."""
    raise ImportError(msg) from e;
    
submodules = [
    "process",
    "spatial_filters",
    "smooth"
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
            raise AttributeError(
                f"Module 'openpic_cxx' has no attribue '{name}'"
            )