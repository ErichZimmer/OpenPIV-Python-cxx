from openpiv_cxx.input_checker import check_nd as _check
from ._bilinear2D_cpp import _bilinear2D

import numpy as np


__all__ = ["bilinear2D"]


Float = np.float64
Int = np.int32


def bilinear2D(x, y, z, xi, yi, keep_dtype=False):
    """2-D bilinear interpolation over a regular grid.

    Parameters
    ----------
    x, y : 2D int64 array
        The coordinates, in strictly ascending order,
        which to evaluate the interpolated values.
    z : 2D float64 array
        A two dimensionional array.
    xi, yi : 2D float64 array
        The coordinates at which to evaluate the interpolated
        values.
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.

    Returns
    -------
    z_deform : 2D float64 array
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
    if orig_dtype != Float:
        z = z.astype(Float)

    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x must be strictly increasing")

    if not np.all(np.diff(y) > 0.0):
        raise ValueError("y must be strictly increasing")

    if x.dtype != Int:
        x = x.astype(Int)

    if y.dtype != Int:
        y = y.astype(Int)

    if xi.dtype != Float:
        xi = xi.astype(Float)

    if yi.dtype != Float:
        yi = yi.astype(Float)

    z_interp = _bilinear2D(x, y, z, xi, yi)

    # cast to original dtype if needed
    if orig_dtype != Float and keep_dtype == True:
        z_interp = z_interp.astype(orig_dtype)

    return z_interp
