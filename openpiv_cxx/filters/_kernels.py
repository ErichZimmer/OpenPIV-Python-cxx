import numpy as np


__all__ = ["gaussian_kernel"]


def _gaussian_kernel1(half_width=1):
    """A normalized 2D Gaussian kernel

    Parameters
    ----------
    half_width : int
        The half width of the kernel. Kernel
        has shape 2*half_width + 1 (default half_width = 1, i.e.
        a 3 x 3 Gaussian kernel).

    Returns
    -------
    kernel : ndarray
        A 2D gaussian kernel.

    Examples
    --------

    >>> from openpiv_cxx.filter import _gaussian_kernel1
    >>> _gaussian_kernel1(1)
    array([[ 0.04491922,  0.12210311,  0.04491922],
       [ 0.12210311,  0.33191066,  0.12210311],
       [ 0.04491922,  0.12210311,  0.04491922]])

    """
    half_width = float(half_width)
    x, y = np.mgrid[-half_width : half_width + 1, -half_width : half_width + 1].astype(
        float
    )
    g = np.exp(-(x**2 / half_width + y**2 / half_width))
    return g / g.sum()


def _gaussian_kernel2(sigma, truncate=4.0):
    """Gaussian that truncates at the given number of standard deviations.

    Parameters
    ----------
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at this many standard deviations.

    Returns
    -------
    kernel : ndarray
        A 2D gaussian kernel.

    """
    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(float)
    sigma = sigma**2

    k = 2 * np.exp(-0.5 * (x**2 + y**2) / sigma)
    k = k / np.sum(k)

    return k


def gaussian_kernel(half_width=None, sigma=1.0, truncate=4.0):
    """Gaussian kernel of specified width and sigma.

    Parameters
    ----------
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : int
        If set, will truncate the kernel at specified standard deviations.
    half_width : int, optional
        The half width of the kernel. If specified, create a normalized
        Gaussian kernel of N*2+1 x N*2+1 size (ignores sigma).

    Returns
    -------
    kernel : ndarray
        A 2D gaussian kernel.

    """
    # If the user specifies half_width, then use it.
    if isinstance(half_width, int):
        return _gaussian_kernel1(half_width)
    else:
        return _gaussian_kernel2(sigma, truncate)
