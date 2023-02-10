from openpiv_cxx.input_checker import check_nd as _check

import numpy as np


__all__ = [
    "sig2noise_val",
]


###############################################################################
# These functions are implemented in Python as they are already "fast enough" #
###############################################################################


def sig2noise_val(
    s2n,
    threshold=1.05,
):
    """Eliminate spurious vectors from cross-correlation signal to noise ratio.

    Replace spurious vectors with zero if signal to noise ratio
    is below a specified threshold.

    Parameters
    ----------
    s2n : ndarray
        A two dimensional array containing the value  of the signal to
        noise ratio from cross-correlation function.

    threshold: float
        The signal to noise ratio threshold value.

    Returns
    -------
    mask : ndarray
        A boolean or integer array where elemtents that equal 0 are valid
        and equal 1 are invalid.

    References
    ----------
    R. D. Keane and R. J. Adrian, Measurement Science & Technology, 1990, 1, 1202-1215.

    """
    
    
    if threshold < 0:
        raise ValueError(
            "Threshold can not be less than zero"
        )
        
    ind = s2n < threshold

    mask = np.zeros_like(s2n, dtype=int)
    mask[ind != 0] = 1

    return mask
