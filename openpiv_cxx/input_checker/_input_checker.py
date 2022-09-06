import numpy as np


__all__ = [
    "check_nd"
]


def check_nd(ndim = 2, **kwargs):
    for arg in kwargs:
        arr = kwargs[arg]
        
        if np.ndim(arr) != ndim:
            raise ValueError(
                f"{arg} is not a {ndim}D array"
            )