from imageio import imread as _imread, imsave as _imsave

import numpy as np


__all__ = ["imread", "imsave", "negative", "rgb2gray"]


def rgb2gray(rgb):
    """Convert RGB to grayscale

    Parameters
    ----------
    rgb : ndarray
        2D RGB image

    Returns
    -------

    img : ndarray
        2D grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def imread(filename, flatten=False):
    """Read an image

    Read an image file into a numpy array using imageio.imread.

    Parameters
    ----------
    filename : str
        The absolute path of the image file.
    flatten :  bool
        True if the image is RGB color or False (default) if greyscale.

    Returns
    -------
    frame : ndarray
        A 2-dimensional numpy array with grey levels.

    Examples
    --------
    >>> image = openpiv_cxx.tools.imread( 'image.bmp' )
    >>> print image.shape
        (1280, 1024)

    """
    im = _imread(filename)
    if np.ndim(im) > 2:
        im = rgb2gray(im)

    return im


def imsave(filename, arr) -> None:
    """Write an image.

    Write an 8-bit image file from a numpy array using imageio.imread

    Parameters
    ----------
    filename :  string
        The absolute path of the image file that will be created.

    arr : ndarray
        A 2D numpy array with grey levels.

    """

    if np.ndim(arr) > 2:
        arr = rgb2gray(arr)

    if np.amin(arr) < 0:
        arr -= arr.min()

    if np.amax(arr) > 255:
        arr /= arr.max()
        arr *= 255

    if filename.endswith("tif"):
        _imsave(filename, arr, format="TIFF")
    else:
        _imsave(filename, arr)


def negative(image):
    """Return the negative of an 8-bit image

    Parameter
    ----------
    image : ndarray
        2D array of grey levels.

    Returns
    -------
    (255-image) : 2D array of grey levels.

    """
    return 255 - image
