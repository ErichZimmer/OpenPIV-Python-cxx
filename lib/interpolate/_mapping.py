from openpiv_cxx.input_checker import check_nd as _check
from ._whittaker2D_cpp import _whittaker2D
from ._taylor_expansion2D_cpp import _taylor_expansion2D

import numpy as np


__all__ = ["taylor_expansion2D", "whittaker2D"]


Float = np.float64
Int = np.int64


def taylor_expansion2D(image, yi, xi, order=1, keep_dtype=False):
    """Taylor expansion mapping

    Perform Taylor expansions with finite differences mapping over a 2D
    regular mesh.

    Parameters
    ----------
    image : 2D float64 array
        A two dimensionional array containing grey levels of an image.
    xi, yi : 2D float64 array
        The x/y-coordinates at which to evaluate the interpolated
        values of shape (rows, cols).
    order : int
        The order of the interpolation.
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.

    Returns
    -------
    img_deform : 2D float64 array
        The interpolated image frame.

    """
    _check(ndim=2, image=image, xi=xi, yi=yi)

    # store original dtype
    orig_dtype = image.dtype

    # convert to 64 bit float for interpolation
    if orig_dtype != Float:
        image = image.astype(Float)

    if xi.dtype != Float:
        xi = xi.astype(Float)

    if yi.dtype != Float:
        yi = yi.astype(Float)

    if order not in [1, 3, 5, 7]:
        raise ValueError(
            f"Order {order} is not supported. Supported interpolation "
            + "orders are 1, 3, 5, and 7"
        )

    img_deform = _taylor_expansion2D(image, yi, xi, int(order))

    # cast to original dtype if needed
    if orig_dtype != Float and keep_dtype == True:
        if "int" in str(orig_dtype):
            img_deform = np.round(img_deform, 0)

        img_deform = img_deform.astype(orig_dtype)

    return img_deform


def whittaker2D(image, yi, xi, radius=3, keep_dtype=False):  # optimal radius is 3 and 5
    """Whittaker-Shanon (sinc) mapping

    Perform Whittaker-Shannon mapping over a 2D regular mesh.

    Parameters
    ----------
    image : 2D float64 array
        A two dimensionional array containing grey levels of an image.
    xi, yi : 2D float64 array
        The x/y-coordinates at which to evaluate the interpolated
        values of shape (rows, cols).
    radius : int
        The radius of the Whittaker interpolation kernel.
        Optimal radii is 3 and 5.
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.

    Returns
    -------
    img_deform : 2D float64 array
        The interpolated image frame.

    References
    ----------
    [1] Wikipedia contributors. (2022, March 18). Whittaker-Shannon
        interpolation formula. In Wikipedia, The Free
        Encyclopedia. Retrieved June 30, 2022, from
        https://en.wikipedia.org/w/index.php?title=Whittaker%E2%80%93Shannon_interpolation_formula&oldid=1077909297

    """
    _check(ndim=2, image=image, xi=xi, yi=yi)

    # store original dtype
    orig_dtype = image.dtype

    # convert to 64 bit float for interpolation
    if orig_dtype != Float:
        image = image.astype(Float)

    if xi.dtype != Float:
        xi = xi.astype(Float)

    if yi.dtype != Float:
        yi = yi.astype(Float)

    if radius < 1:
        raise ValueError("Radius < 1 is not supported")

    img_deform = _whittaker2D(image, yi, xi, int(radius))

    # cast to original dtype if needed
    if orig_dtype != Float and keep_dtype == True:
        if "int" in str(orig_dtype):
            img_deform = np.round(img_deform, 0)

        img_deform = img_deform.astype(orig_dtype)

    return img_deform
