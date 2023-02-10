import numpy as np


__all__ = [
    "gaussian_kernel"
]


Float = np.float64


def _gaussian_kernel_size_1D(half_width=1):
    """A normalized 1D Gaussian kernel

    Parameters
    ----------
    half_width : int
        The half width of the kernel. Kernel
        has shape 2*half_width + 1 (default half_width = 1, i.e.
        a 3 x 3 Gaussian kernel).

    Returns
    -------
    kernel : 1d float32 array
        A 1D gaussian kernel.

    Examples
    --------

    >>> from openpiv_cxx.filter import _gaussian_kernel_size_1D
    >>> _gaussian_kernel_size_1D(1)
    array([0.21194156, 0.57611688, 0.21194156])

    """
    half_width = float(half_width)
    # grid for gaussian kernel
    x = np.mgrid[
        -half_width : half_width + 1
    ].astype(Float, copy=False)
    # compute gaussian kernel
    k = np.exp(-(x**2 / half_width)).astype(Float, copy=False)
    # normalize kernel
    np.divide(k, np.sum(k), out = k)
    
    return k


def _gaussian_kernel_sigma_1D(sigma, truncate=4.0):
    """Gaussian that truncates at the given number of standard deviations.

    Parameters
    ----------
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at this many standard deviations.

    Returns
    -------
    kernel : 1D float32 array
        A 1D gaussian kernel.

    """
    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)
    # get grid for gaussian kernel
    x = np.mgrid[
        -radius : radius + 1
    ].astype(Float, copy=False)
    sigma = sigma**2
    # compute gaussian kernel
    k = 2 * np.exp(-0.5 * (x**2) / sigma).astype(Float, copy=False)
    # normalize gaussian kernel
    np.divide(k, np.sum(k), out = k)

    return k


def _gaussian_kernel_size_2D(half_width=1):
    """A normalized 2D Gaussian kernel

    Parameters
    ----------
    half_width : int
        The half width of the kernel. Kernel
        has shape 2*half_width + 1 (default half_width = 1, i.e.
        a 3 x 3 Gaussian kernel).

    Returns
    -------
    kernel : 2D float32 array
        A 2D gaussian kernel.

    Examples
    --------

    >>> from openpiv_cxx.filters import _gaussian_kernel_size_2D
    >>> _gaussian_kernel_size_2D(1)
    array([[ 0.04491922,  0.12210311,  0.04491922],
       [ 0.12210311,  0.33191066,  0.12210311],
       [ 0.04491922,  0.12210311,  0.04491922]])

    """
    # since the gaussian kernel is separable, we can use np.outer
    return np.outer(
        _gaussian_kernel_size_1D(half_width),
        _gaussian_kernel_size_1D(half_width)
    )


def _gaussian_kernel_sigma_2D(sigma, truncate=4.0):
    """Gaussian that truncates at the given number of standard deviations.

    Parameters
    ----------
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at this many standard deviations.

    Returns
    -------
    kernel : 2D float32 array
        A 2D gaussian kernel.

    """
    # since the gaussian kernel is separable, we can use np.outer
    return np.outer(
        _gaussian_kernel_sigma_1D(sigma, truncate),
        _gaussian_kernel_sigma_1D(sigma, truncate)
    )


def gaussian_kernel(sigma=1.0, truncate=4.0, half_width=None, ndim=2):
    """Gaussian kernel of specified width, sigma, and dimensions.

    Parameters
    ----------
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : int
        If set, will truncate the kernel at specified standard deviations.
    half_width : int, optional
        The half width of the kernel. If specified, create a normalized
        Gaussian kernel of N*2+1 x N*2+1 size (ignores sigma).
    ndim : int
        Number of dimensions of the kernel. Currently supports 1D 
        and 2D kernels.

    Returns
    -------
    kernel : 1D or 2D float32 array
        A 1D or 2D gaussian kernel.

    """
    # If the user specifies half_width, then use it.
    if isinstance(half_width, int):
        if ndim == 1:
            return _gaussian_kernel_size_1D(half_width)
        else:
            return _gaussian_kernel_size_2D(half_width)
    else:
        if ndim == 1:
            return _gaussian_kernel_sigma_1D(sigma, truncate)
        else:
            return _gaussian_kernel_sigma_2D(sigma, truncate)
