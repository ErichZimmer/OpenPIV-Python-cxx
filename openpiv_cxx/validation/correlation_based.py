"""
Module for validating Particle Image Velocimetry (PIV) data (vectors) to attain a higher quality vector field.
"""
# Note: typing.Union doesn't work with np.ndarray
from typing import Optional

import numpy as np

__all__ = [
    "sig2noise_val",
]


###############################################################################
# These functions are implemented in Python as they are already "fast enough" #
###############################################################################


def sig2noise_val(
    u: np.ndarray, 
    v: np.ndarray, 
    s2n: np.ndarray,
    w: Optional[np.ndarray] = None, 
    threshold: float = 1.05, 
    mask: Optional[np.ndarray] = None, 
    convention: str = "openpiv"
) -> np.ndarray:
    """Eliminate spurious vectors from cross-correlation signal to noise ratio.

    Replace spurious vectors with zero if signal to noise ratio
    is below a specified threshold.


    Parameters
    ----------
    u : 2d or 3d np.ndarray
        A two or three dimensional array containing the u velocity component.

    v : 2d or 3d np.ndarray
        A two or three dimensional array containing the v velocity component.

    s2n : 2d np.ndarray
        A two or three dimensional array containing the value  of the signal to
        noise ratio from cross-correlation function.
        
    w : 2d or 3d np.ndarray
        A two or three dimensional array containing the w (in z-direction)
        velocity component.

    threshold: float
        The signal to noise ratio threshold value.
    
    mask : 2d np.ndarray
        A two dimensional array containing the flags for invalid vectors.
        
    convention: "string"
        Which flag convention to use.


    Returns
    -------
    u : 2d np.ndarray
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    v : 2d np.ndarray
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    
    w : 2d or 3d  np.ndarray
        Optional, a two or three dimensional array containing the w
        (in z-direction) velocity component, where spurious vectors
        have been replaced by NaN if convention is set to 'openpiv'.

    mask : boolean or int 2d np.ndarray 
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid. 


    References
    ----------
    R. D. Keane and R. J. Adrian, Measurement Science & Technology, 1990,
        1, 1202-1215.

    """
    ind = s2n < threshold

    if convention == "openpiv":
        u[ind] = np.nan
        v[ind] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind] = True
        
        if isinstance(w, np.ndarray):
            w[ind] = np.nan
            return u, v, w, mask

        return u, v, mask

    else:
        if isinstance(mask, np.ndarray):
            mask[ind] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind] = 1

        return mask 