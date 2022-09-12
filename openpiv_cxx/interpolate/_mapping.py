from numpy import ndarray
from openpiv_cxx.input_checker import check_nd as _check
from ._whittaker2D import _whittaker2D
from ._taylor_expansion2D import _taylor_expansion2D

import numpy as np


__all__ = ["taylor_expansion2D", "whittaker2D"]


def taylor_expansion2D(
    image: ndarray, yi: ndarray, xi: ndarray, order: int = 1, keep_dtype: bool = False
) -> ndarray:
    """Taylor expansion mapping

    Perform Taylor expansions with finite differences mapping over a 2D
    regular mesh.

    Parameters
    ----------
    image: ndarray
        A two dimensionional array containing grey levels of an image.
    xi, yi: ndarray
        The x/y-coordinates at which to evaluate the interpolated
        values of shape (rows, cols).
    order: int
        The order of the interpolation.
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.

    Returns
    -------
    img_deform : ndarray
        The interpolated image frame.

    """
    _check(ndim=2, image=image, xi=xi, yi=yi)

    # store original dtype
    orig_dtype = image.dtype

    # convert to 64 bit float for interpolation
    if orig_dtype != "float64":
        image = image.astype("float64")

    if xi.dtype != "float64":
        xi = xi.astype("float64")

    if yi.dtype != "float64":
        yi = yi.astype("float64")

    if order not in [1, 3, 5, 7]:
        raise ValueError(
            f"Order {order} is not supported. Supported interpolation " +
            "orders are 1, 3, 5, and 7"
        )

    img_deform = _taylor_expansion2D(image, yi, xi, int(order))

    # cast to original dtype if needed
    if orig_dtype != "float64" and keep_dtype == True:
        if "int" in str(orig_dtype):
            img_deform = np.round(img_deform, 0)
            
        img_deform = img_deform.astype(orig_dtype)

    return img_deform


def whittaker2D(
    image: ndarray,
    yi: ndarray,
    xi: ndarray,
    radius: int = 3,  # optinal radius is 3 and 5
    keep_dtype: bool = False,
) -> ndarray:
    """Whittaker-Shanon (sinc) mapping

    Perform Whittaker-Shannon mapping over a 2D regular mesh.

    Parameters
    ----------
    image: ndarray
        A two dimensionional array containing grey levels of an image.
    xi, yi: ndarray
        The x/y-coordinates at which to evaluate the interpolated
        values of shape (rows, cols).
    radius: int
        The radius of the Whittaker interpolation kernel.
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.

    Returns
    -------
    img_deform : ndarray
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
    if orig_dtype != "float64":
        image = image.astype("float64")

    if xi.dtype != "float64":
        xi = xi.astype("float64")

    if yi.dtype != "float64":
        yi = yi.astype("float64")

    if radius < 1:
        raise ValueError(
            "Radius < 1 is not supported"
        )
    
    img_deform = _whittaker2D(image, yi, xi, int(radius))

    # cast to original dtype if needed
    if orig_dtype != "float64" and keep_dtype == True:
        if "int" in str(orig_dtype):
            img_deform = np.round(img_deform, 0)
            
        img_deform = img_deform.astype(orig_dtype)

    return img_deform
