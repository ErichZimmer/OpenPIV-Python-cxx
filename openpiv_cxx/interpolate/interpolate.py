import numpy as np
from ._bilinear2D import _bilinear2D
from ._whittaker2D import _whittaker2D
from ._taylor_expansion2D import _taylor_expansion2D

"""This module contains a c++ implementation of basic interpolation routines."""

__all__ = [
    "bilinear2D",
    "taylor_expansion2D",
    "whittaker2D"
]


def bilinear2D(
    y: np.ndarray, 
    x: np.ndarray, 
    z: np.ndarray,
    yi: np.ndarray, 
    xi: np.ndarray,
    keep_dtype: bool = False
) -> np.ndarray:
    """Perform bilinear interpolation over a 2D regular grid.


    Parameters
    ----------
    y, x: 1D or 2D np.ndarray
        The `x` and `y`-coordinates in strictly ascending order 
        which to evaluate the interpolated values.
    
    z: 2D np.ndarray
        A two dimensionional array.
    
    yi, xi: 1D or None
        The `x` and `y`-coordinates at which to evaluate the interpolated
        values.

    keep_dtype : bool
        Whether to cast the output to the original dtype or not.
    
    Returns
    -------
    z_deform : 2D np.ndarray
        The interpolated array.
    
    
    References
    ----------
    [1] Wikipedia contributors. (2022, April 10). Bilinear interpolation. In Wikipedia, The Free
        Encyclopedia. Retrieved April 1, 2022, from
        https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    # store original dtype
    orig_dtype = z.dtype
    
    # convert to 64 bit float for interpolation
    if orig_dtype != "float64":
        z = z.astype("float64")
        
    if not np.all(np.diff(x) > 0.0):
        raise ValueError(
            "x must be strictly increasing"
        )

    if not np.all(np.diff(y) > 0.0):
        raise ValueError(
            "y must be strictly increasing"
        )

    if x.dtype != "int64":
        x = x.astype("int64")

    if y.dtype != "int64":
        y = y.astype("int64")

    if xi.dtype != "float64":
        xi = xi.astype("float64")

    if yi.dtype != "float64":
        yi = yi.astype("float64")

    z_interp = _bilinear2D(
        x, 
        y,
        z,
        xi,
        yi
    )
    
    # cast to original dtype if needed
    if orig_dtype != "float64" and keep_dtype == True:
        z_interp = z_interp.astype(orig_dtype)
                              
    return z_interp


def taylor_expansion2D(
    image: np.ndarray, 
    yi: np.ndarray, 
    xi: np.ndarray,
    order: int = 1,
    keep_dtype: bool = False
) -> np.ndarray:
    """Perform Taylor expansions with finite differences mapping over a 2D regular mesh.


    Parameters
    ----------
    image: 2D np.ndarray
        A two dimensionional array containing grey levels of an image.

    yi, xi: 2D np.ndarray
        The `y` and `x`-coordinates at which to evaluate the interpolated values of shape (rows, cols).

    order: int
        The order of the interpolation.
    
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.


    Returns
    -------
    img_deform : 2D np.ndarray
        The interpolated image frame.
    """
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
            f"Order {order} is not supported. Supported interpolation \
            orders are 1, 3, 5, and 7"
        )
        
    img_deform = _taylor_expansion2D(
        image,
        yi,
        xi,
        int(order)
    )
    
    # cast to original dtype if needed
    if orig_dtype != "float64" and keep_dtype == True:
        img_deform = img_deform.astype(orig_dtype)
                              
    return img_deform


def whittaker2D(
    image: np.ndarray, 
    yi: np.ndarray, 
    xi: np.ndarray,
    radius: int = 3, # optinal radius is 3 and 5
    keep_dtype: bool = False
) -> np.ndarray:
    """Perform Whittaker-Shannon mapping over a 2D regular mesh.


    Parameters
    ----------
    image: 2D np.ndarray
        A two dimensionional array containing grey levels of an image.

    yi, xi: 2D np.ndarray
        The `y` and `x`-coordinates at which to evaluate the interpolated values of shape (rows, cols).

    radius: int
        The radius of the Whittaker interpolation kernel.
    
    keep_dtype : bool
        Whether to cast the output to the original dtype or not.


    Returns
    -------
    img_deform : 2D np.ndarray
        The interpolated image frame.
    
    
    References
    ----------
    [1] Wikipedia contributors. (2022, March 18). Whittaker-Shannon
        interpolation formula. In Wikipedia, The Free
        Encyclopedia. Retrieved June 30, 2022, from
        https://en.wikipedia.org/w/index.php?title=Whittaker%E2%80%93Shannon_interpolation_formula&oldid=1077909297
    """
    # store original dtype
    orig_dtype = image.dtype
    
    # convert to 64 bit float for interpolation
    if orig_dtype != "float64":
        image = image.astype("float64")
 
    if xi.dtype != "float64":
        xi = xi.astype("float64")

    if yi.dtype != "float64":
        yi = yi.astype("float64")

    img_deform = _whittaker2D(
        image,
        yi,
        xi,
        int(radius)
    )
    
    # cast to original dtype if needed
    if orig_dtype != "float64" and keep_dtype == True:
        img_deform = img_deform.astype(orig_dtype)
                              
    return img_deform