from numpy import ndarray
from openpiv_cxx.input_checker import check_nd as _check
from ._bilinear2D import _bilinear2D

import numpy as np


__all__ = ["bilinear2D"]


def bilinear2D(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    xi: ndarray,
    yi: ndarray,
    keep_dtype: bool = False,
) -> ndarray:
    """2-D bilinear interpolation over a regular grid.

    Parameters
    ----------
    x, y: ndarray
        The `x` and `y`-coordinates in strictly ascending order
        which to evaluate the interpolated values.
    z: ndarray
        A two dimensionional array.
    xi, yi: ndarray
        The `x` and `y`-coordinates at which to evaluate the interpolated
        values.
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.

    Returns
    -------
    z_deform : ndarray
        The interpolated array.

    References
    ----------
    [1] Wikipedia contributors. (2022, April 10). Bilinear interpolation. In Wikipedia, The Free
    Encyclopedia. Retrieved April 1, 2022, from https://en.wikipedia.org/wiki/Bilinear_interpolation

    """
    _check(ndim=1, x=x, y=y, xi=xi, yi=yi)

    _check(ndim=2, z=z)

    # store original dtype
    orig_dtype = z.dtype

    # convert to 64 bit float for interpolation
    if orig_dtype != "float64":
        z = z.astype("float64")

    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x must be strictly increasing")

    if not np.all(np.diff(y) > 0.0):
        raise ValueError("y must be strictly increasing")

    if x.dtype != "int64":
        x = x.astype("int64")

    if y.dtype != "int64":
        y = y.astype("int64")

    if xi.dtype != "float64":
        xi = xi.astype("float64")

    if yi.dtype != "float64":
        yi = yi.astype("float64")

    z_interp = _bilinear2D(x, y, z, xi, yi)

    # cast to original dtype if needed
    if orig_dtype != "float64" and keep_dtype == True:
        z_interp = z_interp.astype(orig_dtype)

    return z_interp
