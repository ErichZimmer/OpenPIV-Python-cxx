from openpiv_cxx.input_checker import check_nd as _check
from openpiv_cxx.filters import gaussian_kernel

import numpy as np
from mahotas import convolve


__all__ = [
    "smooth_gaussian"
]


Float = np.float64


def smooth_gaussian(
    u,
    sigma=1.0,
    truncate=2,
    half_width=None
):
    """Gaussian low pass filter.
    
    A 2D gaussian filter via convolution.

    Parameters
    ----------
    u : 2D float64 array
        A two dimensional array containing pixel intenensities.
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at specified standard deviations.
    half_width : int, optional
        Half width of gaussian kernel. Sigma and truncation asre ignored when
        using this parameter.

    Returns
    -------
    u_smth : 2D float64 array
        A two dimensional smoothed array.

    """
    
    _check(ndim=2, u=u)
    
    if np.count_nonzero(np.isnan(u)) > 0:
        raise ValueError(
            "Could not smooth vector field due to NaNs"
        )
    
    if u.dtype != Float:
        u = u.astype(Float, copy=False)
    
    # get kernel
    g_kernel = gaussian_kernel(
        sigma,
        truncate,
        half_width,
        ndim=2
    )
    
    # 2D convolution via mahotas convolve
    return convolve(
        u,
        g_kernel,
        mode="reflect"
    )