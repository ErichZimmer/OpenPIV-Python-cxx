import numpy as np
from ._bilinear2D import _bilinear2D
from ._whittaker2D import _whittaker2D
"""This module contains a c++ implementation of basic interpolation routines."""

__all__ = [
    "bilinear2D",
    "whittaker2D"
]


def bilinear2D(y, x, z, yi = None, xi = None):
    """Perform bilinear interpolation over a 2D regular grid.


    Parameters
    ----------
    y, x: 1D or 2D np.ndarray
        The `x` and `y`-coordinates in strictly ascending order or 
        meshgrids hich to evaluate the interpolated values of shape (rows, cols).
    
    z: 2D np.ndarray
        A two dimensionional array.
    
    yi, xi: 1D or None
        The `x` and `y`-coordinates at which to evaluate the interpolated
        values.


    Returns
    -------
    z_deform : 2D np.ndarray
        The interpolated array.
    
    
    References
    ----------
    [1] Wikipedia contributors. (2022, April 10). Bilinear interpolation. In Wikipedia, The Free
        Encyclopedia. Retrieved June 1, 2022, from
        https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    # store original dtype
    orig_dtype = z.dtype
    
    # convert to 64 bit float for interpolation
    if orig_dtype != "float64":
        z = z.astype("float64")
        
    if isinstance(yi, np.ndarray):
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
    else:
        raise ValueError(
            "Interpolation in (z, yq, xq) format is not yet supported"
        )
    
    # cast to original dtype if needed
    if orig_dtype != "float64":
        z_interp = z_interp.astype(orig_dtype)
                              
    return z_interp


def whittaker2D(image, yi, xi, radius=1):
    """Perform Whittaker-Shannon interpolation over a 2D regular grid


    Parameters
    ----------
    image: 2D np.ndarray
        A two dimensionional array containing grey levels of an image.

    yi, xi: 2D np.ndarray
        The `y` and `x`-coordinates at which to evaluate the interpolated values of shape (rows, cols).

    radius: int
        The radius of the Whittaker interpolation kernel.


    Returns
    -------
    img_deform : 2D np.ndarray
        The interpolated image frame.
    
    
    References
    ----------
    [1] Wikipedia contributors. (2022, March 18). Whittaker-Shannon interpolation formula. In Wikipedia, The Free
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
    if orig_dtype != "float64":
        img_deform = img_deform.astype(orig_dtype)
                              
    return img_deform
