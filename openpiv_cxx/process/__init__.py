try:
    from ._libs import _process
    del _process
except ImportError:
    raise ValueError("Could not locate either _libs folder or _process wrapper.")
    
from .pyprocess import *