from openpiv_cxx.input_checker import check_nd as _check
from ._spatial_filters_cpp import (
    _intensity_cap,
    _threshold_binarization,
    _lowpass_filter,
    _highpass_filter
)
from ._kernels import gaussian_kernel

import numpy as np


__all__ = [
    "contrast_stretch",
    "gaussian_filter",
    "highpass_filter",
    "intensity_cap",
    "sobel_filter",
    "sobel_h_filter",
    "sobel_v_filter",
    "threshold_binarization",    
    "variance_normalization_filter"
]


kernel_size_error = "kernel_size must be an odd number"


def intensity_cap(img, std_mult = 2.0, keep_dtype = True):
    """Set pixels above threshold to threshold.

    Set pixels higher than calculated threshold to said threshold.
    The threshold value is calcualted by img_mean + std_mult * img_std.

    Parameters
    ----------
    img: ndarray
        A two dimensional array containing pixel intenensities.
    std_mult: float
        Lower values yields a lower threshold.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    img: ndarray
        A filtered two dimensional array of the input image.

    """
    _check(ndim=2, img=img)

    # store original image dtype
    img_dtype = img.dtype

    if img_dtype != "float32":
        img = img.astype("float32")

    new_img = _intensity_cap(img, float(std_mult))

    if img_dtype != "float32" and keep_dtype == True:
        new_img = new_img.astype(img_dtype)

    return new_img


def threshold_binarization(img, threshold = 0.5, keep_dtype = True):
    """A threshold binarization filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    kernel_size : int
        nxn size of the convolution kernel.
    sigma : float
        sigma of nxn gaussian convolution kernel.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img)

    # store original image dtype
    img_dtype = img.dtype
    max_ = img.max()

    # make sure array is float32
    if img.dtype != "float32":
        img = img.astype("float32")

    # normalize array if pixel intensities are greater than 1
    if max_ > 1:
        img = img.astype("float32")
        img /= max_

    # extract filtered image
    new_img = _threshold_binarization(img, float(threshold))

    # if the image wasn't normalized beforehand, return original range
    if max_ > 1:
        new_img *= max_

    if img_dtype != "float32" and keep_dtype == True:
        new_img = new_img.astype(img_dtype)

    return new_img


def _convolve_kernel(img, kernel, pad_type = "reflect", cval = 0.0):
    """A simple sliding convolution filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    kernel : ndarray
        2D square kernel to convolve (must be odd).
    pad_type : str
        Type of padding used on borders of image.
    cval : float
        If padding is constant, pad with user selected constant.

    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img, kernel=kernel)
    
    # make sure kernel is square
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(
            "Kernel must be square"
        )
    
    kernel_size = kernel.shape[0]

    if kernel_size % 2 != 1:
        raise ValueError(kernel_size_error)

    # pad array by kernel half size
    pad = int(kernel_size / 2)
    if pad_type == "constant":
        buffer1 = np.pad(img, pad, mode=pad_type, constant_values = cval)
    else:
        buffer1 = np.pad(img, pad, mode=pad_type)

    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # make sure array is float32
    if kernel.dtype != "float32":
        kernel = kernel.astype("float32")
        
    # extract filtered image
    new_img = _lowpass_filter(buffer1, kernel)

    # remove padding
    new_img = new_img[pad : buffer1.shape[0] - pad, pad : buffer1.shape[1] - pad]

    return new_img


def gaussian_filter(img, sigma = 1.0, truncate = 2.0, keep_dtype = False):
    """A simple sliding window gaussian low pass filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at specified standard deviations.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img)

    # store original image dtype
    img_dtype = img.dtype
    max_ = img.max()

    if img_dtype != "float32":
        img = img.astype("float32")
    
    if max_ > 1:
        img /= max_

    # get Gaussian kernel
    kernel = gaussian_kernel(sigma=sigma, truncate=truncate)

    # extract filtered image
    new_img = _convolve_kernel(img, kernel)

    # if the image wasn't normalized beforehand, return original range
    if max_ > 1:
        new_img *= max_
    
    if img_dtype != "float32" and keep_dtype == True:
        new_img = new_img.astype(img_dtype)

    return new_img


def highpass_filter(
    img, sigma = 1.0, truncate = 2.0, clip_at_zero = True, keep_dtype = False
):
    """A simple sliding window gaussian high pass filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at specified standard deviations.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
    """
    _check(ndim=2, img=img)

    # get Gaussian kernel
    kernel = gaussian_kernel(sigma=sigma, truncate=truncate).astype("float32")
    kernel_size = kernel.shape[0]
    
    # store original image data
    img_dtype = img.dtype
    max_ = img.max()

    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = np.pad(img, pad, mode="reflect")

    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")

    # normalize array if pixel intensities are greater than 1
    if max_ > 1:
        buffer1 = buffer1.astype("float32")
        buffer1 /= max_

    # extract filtered image
    new_img = _highpass_filter(buffer1, kernel, bool(clip_at_zero))

    # remove padding
    new_img = new_img[pad : buffer1.shape[0] - pad, pad : buffer1.shape[1] - pad]

    # if the image wasn't normalized beforehand, return original range
    if max_ > 1:
        new_img *= max_

    if img_dtype != "float32" and keep_dtype == True:
        new_img = new_img.astype(img_dtype)

    return new_img


def variance_normalization_filter(
    img,
    sigma1 = 1.0,
    sigma2 = 2.0,
    truncate = 2.0,
    clip_at_zero = True,
    keep_dtype = False
    
):
    """A simple gaussian variance normalization filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    sigma1 : float
        Sigma of nxn gaussian convolution kernel.
    sigma2 : float
        Sigma of nxn gaussian convolution kernel.
    truncate : float
        Truncate the kernel at specified standard deviations.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    den : ndarray
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img)
    
    # store original image data
    img_dtype = img.dtype
    max_ = img.max()
    
    if img_dtype != "float32":
        img = img.astype("float32")
    
    if max_ > 1:
        img /= max_
    
    high_pass = highpass_filter(img, sigma1, truncate, False)
    
    den = gaussian_filter(high_pass * high_pass, sigma2, truncate)
    
    den = np.sqrt(den)
    
    new_img = np.divide( # stops image from being all black
        high_pass, den,
        where = (den != 0.0)
    )  
    
    if clip_at_zero == True: 
        new_img[new_img < 0] = 0 
        
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min())
    
    if max_ > 1:
        new_img *= max_
    
    if img_dtype != "float32" and keep_dtype == True:
        new_img = den.astype(img_dtype)
    
    return new_img


def sobel_filter(
    img,
    keep_dtype = False
):
    """A simple sobel filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
    
    Notes
    -----
    
    We use the following two kernels::
    
      1   2   1
      0   0   0
     -1  -2  -1
     
     
      1   0  -1
      2   0  -2
      1   0  -1
    
    """
    _check(ndim=2, img=img)
    
    # horizontal
    kernel1 = np.asarray([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype = "float32")
    
    # vertical
    kernel2 = np.asarray([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ], dtype = "float32")
    
    # store original image data
    img_dtype = img.dtype
    max_ = img.max()
    
    if img_dtype != "float32":
        img = img.astype("float32")
    
    if max_ > 1:
        img /= max_
    
    out1 = _convolve_kernel(img, kernel1)
    out2 = _convolve_kernel(img, kernel2)
    
    new_img = np.sqrt(np.square(out1) + np.square(out2))
    
    if max_ > 1:
        new_img *= max_
    
    if img_dtype != "float32" and keep_dtype == True:
        new_img = new_img.astype(img_dtype)
    
    return new_img


def sobel_h_filter(
    img,
    keep_dtype = False
):
    """A simple horizontal sobel filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
    
    Notes
    -----
    
    We use the following kernel::
     
      1   2   1
      0   0   0
     -1  -2  -1
    
    """
    _check(ndim=2, img=img)
    
    kernel = np.asarray([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype = "float32")
    
    # store original image data
    img_dtype = img.dtype
    max_ = img.max()
    
    if img_dtype != "float32":
        img = img.astype("float32")
    
    if max_ > 1:
        img /= max_
    
    new_img = _convolve_kernel(img, kernel)
    
    if max_ > 1:
        new_img *= max_
    
    if img_dtype != "float32" and keep_dtype == True:
        new_img = new_img.astype(img_dtype)
    
    return new_img


def sobel_v_filter(
    img,
    keep_dtype = False
):
    """A simple vertical sobel filter.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
    
    Notes
    -----
    
    We use the following kernel::
     
      1   0  -1
      2   0  -2
      1   0  -1
    
    """
    _check(ndim=2, img=img)
    
    kernel = np.asarray([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ], dtype = "float32")
    
    # store original image data
    img_dtype = img.dtype
    max_ = img.max()
    
    if img_dtype != "float32":
        img = img.astype("float32")
    
    if max_ > 1:
        img /= max_
    
    new_img = _convolve_kernel(img, kernel)
    
    if max_ > 1:
        new_img *= max_
    
    if img_dtype != "float32" and keep_dtype == True:
        new_img = new_img.astype(img_dtype)
    
    return new_img


def contrast_stretch(
    img, lower_limit = 2.0, upper_limit = 98.0
):
    """Simple percentile-based contrast stretching.

    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    lower_limit: int
        Lower percentile limit.
    upper_limit: int
        Upper percentile limit.

    Returns
    -------
    img: ndarray
        A filtered two dimensional array of the input image

    """
    _check(ndim=2, img=img)

    if lower_limit < 0:
        lower_limit = 0
    if upper_limit > 100:
        upper_limit = 100

    img_dtype = img.dtype
    img_max = img.max()

    if img_dtype != "float32":
        img = img.astype("float32")

    if img_max > 1:
        img /= img_max

    lower = np.percentile(img, lower_limit)
    upper = np.percentile(img, upper_limit)

    img_max = img.max()
    img_min = img.min()

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    stretch_min = 0
    if img_max < 2:
        stretch_max = 1
    elif img_max < 2**8:
        stretch_max = 2**8 - 1
    elif img_max < 2**10:
        stretch_max = 2**10 - 1
    elif img_max < 2**12:
        stretch_max = 2**12 - 1
    elif img_max < 2**14:
        stretch_max = 2**14 - 1
    else:
        stretch_max = 2**16 - 1

    img *= stretch_max

    return img.astype(img_dtype)