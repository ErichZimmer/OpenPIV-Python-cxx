from typing import Optional
from numpy import ndarray
from openpiv_cxx.input_checker import check_nd as _check

import numpy as np


__all__ = [
    "save",
    "transform_coordinates",
    "uniform_scaling"
]


def save(
    filename: str,
    fmt: str = "%8.4f", 
    delimiter: str = "\t",
    **kwargs
) -> None:
    """Save flow field to an ascii file.
    
    Parameters
    ----------
    filename : string
        The path of the file where to save the flow field.
    fmt : string
        A format string. See documentation of numpy.savetxt
        for more details.
    delimiter : string
        Character separating columns.
    
    **kwargs : ndarray
        Components to save.
    
    Examples
    --------
    openpiv_cxx.tools.save(
        'dummy.txt', 
        x = x,
        y = y,
        u = u, 
        v = v
    )
    
    """
    den = []
    header = ""
    
    # extract components
    for arg in kwargs:
        arr = kwargs[arg]
        
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(0.)
            
        den.append(arr)
        header += arg + delimiter
    
    # build output array
    out = np.vstack([m.flatten() for m in den])

    # save data to file.
    np.savetxt(
        filename,
        out.T,
        fmt=fmt,
        delimiter=delimiter,
        header=header
    )

    
def transform_coordinates(
    x: ndarray, 
    y: ndarray, 
    u: ndarray, 
    v: ndarray
) -> tuple( [ndarray] * 4 ):
    """Set origin of flow field.
    
    Converts coordinate systems from/to the image based / physical based.

    Parameters
    ----------
    x, y : ndarray
        2D arrays of x/y coordinates.
    u, v : ndarray
        2D arrays of u/v components.
    
    Returns
    -------
    x, y : ndarray
        2D arrays of x/y coordinates.
    u, v : ndarray
        2D arrays of u/v components.
    
    """
    
    _check(ndim = 2,
        x = x, y = y,
        u = u, v = v
    )
    
    y = y[::-1, :]
    v *= -1
    
    return x, y, u, v


def uniform_scaling(
    x: ndarray, 
    y: ndarray, 
    u: ndarray, 
    v: ndarray,
    scaling_factor: float
) -> tuple( [ndarray] * 4 ):
    """Apply an uniform scaling.

    Parameters
    ----------
    x, y : ndarray
        2D arrays of x/y coordinates.
    u, v : ndarray
        2D arrays of u/v components.
    scaling_factor : float
        The image scaling factor in pixels per meter.

    Returns
    -------
    x, y : ndarray
        2D arrays of scaled x/y coordinates.
    u, v : ndarray
        2D arrays of scaled u/v components.
        
    """
    return (
        x / scaling_factor,
        y / scaling_factor,
        u / scaling_factor,
        v / scaling_factor,
    )