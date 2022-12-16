from openpiv_cxx.input_checker import check_nd as _check
from ._kernels import gaussian_kernel
from ._spatial_filters_cpp import (
    _intensity_cap,
    _convolve2D
)

import numpy as np


__all__ = [
    "contrast_stretch",
    "convolve_2D_sep",
    "gaussian_filter",
    "highpass_filter",
    "intensity_cap",
    "sobel_filter",
    "threshold_binarization",    
    "variance_normalization_filter"
]


Float = np.float32
Int = np.int32

kernel_size_error = "kernel_size must be an odd number"


def contrast_stretch(
    img,
    lower_limit = 2.0, 
    upper_limit = 98.0
):
    """Percentile-based contrast stretching.
    
    Percentile-based contrast stretching similar to MATLAB's imadjust.
    Assumes array is normalized to [0..1].

    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    lower_limit: int
        Lower percentile limit.
    upper_limit: int
        Upper percentile limit.

    Returns
    -------
    img: 2D float32 array
        A filtered two dimensional array of the input image

    """
    _check(ndim=2, img=img)

    if lower_limit < 0:
        lower_limit = 0
    if upper_limit > 100:
        upper_limit = 100

    lower = np.percentile(img, lower_limit)
    upper = np.percentile(img, upper_limit)

    img_max = img.max()
    img_min = img.min()

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    return img


def convolve_2D_sep(
    img,
    kernel_h,
    kernel_v
):
    """2D convolution by 1D kernels
    
    2D convolution by 1D kernels for horizontal and vertical axes.
    The convolution operates in O(2N) insteas of O(N*M), so it is usually
    much faster. Borders are treated using the following scheme: dcba|abcd|dcba.
    
    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    kernel_h : 1D float32 array
        A one dimensional array containing kernel values for the x-axis.
    kernel_v : 1D float32 array
        A one dimensional array containing kernel values for the y-axis.

    Returns
    -------
    new_img : 2D float32 array
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img)
    _check(ndim=1, kernel_h=kernel_h, kernel_v=kernel_v)
    
    if kernel_h.size != kernel_v.size:
        raise ValueError(
            f"kernel_h (size={kernel_h}) must be same size as" +
            f"kernel_v (size={kernel_v})"
        )
    
    # img, kernel_h, and kernel_v are automatically float32 in c++
    return _convolve2D(
        img,
        kernel_h,
        kernel_v
    )


def gaussian_filter(
    img, 
    sigma = 1.0,
    truncate = 2.0
):
    """Gaussian low pass filter.
    
    A simple gaussian filter via convolution of two 1D kernels.
    Assumes array is normalized to [0..1].

    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at specified standard deviations.

    Returns
    -------
    new_img : 2D float32 array
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img)

    # get 1D kernel
    kernel = gaussian_kernel(
        sigma=sigma,
        truncate=truncate,
        ndim=1
    )
    
    # filter image
    new_img = convolve_2D_sep(
        img,
        kernel,
        kernel
    )

    return new_img


def highpass_filter(
    img, 
    sigma = 1.0, 
    truncate = 2.0,
    clip_at_zero = True,
):
    """Gaussian high pass filter.

    A simple gaussian filter via convolution of two 1D kernels is
    subtracted from original image. If clip_at_zero is set (recommended),
    negative pixel values are set to zero, removing noise.
    Assumes array is normalized to [0..1].
    
    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    sigma : float
        Standard deviation of gaussian kernel.
    truncate : float
        Truncate the kernel at specified standard deviations.
    clip_at_zero : bool
        Clip negative pixel intenensities.

    Returns
    -------
    new_img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    """
    _check(ndim=2, img=img)

    # extract filtered image
    new_img = gaussian_filter(
        img,
        sigma,
        truncate
    )
    
    new_img = img - new_img
    
    if clip_at_zero == True:
        new_img[new_img < 0] = 0

    return new_img


def intensity_cap(
    img, 
    std_mult = 2.0
):
    """Set pixels above calculated threshold to threshold.

    Set pixels higher than calculated threshold to said threshold.
    The threshold value is calcualted by img_mean + std_mult * img_std.
    Assumes array is normalized to [0..1].

    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    std_mult : float
        Lower values yields a lower threshold.

    Returns
    -------
    img : 2D float32 array
        A filtered two dimensional array of the input image.

    """
    _check(ndim=2, img=img)

    # store original image dtype
    img_dtype = img.dtype

    if img_dtype != Float:
        img = img.astype(Float)

    # cpp version was found to be faster than NumPy
    new_img = _intensity_cap(img, float(std_mult))

    return new_img


def sobel_filter(
    img
):
    """Simple Sobel filter
    
    Simple 2D Sobel filter by convolution of 1D arrays.
    
    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    
    Returns
    -------
    new_img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    
    """
    
    sobel_h = np.array([-1, 0, 1], dtype = Float)
    sobel_v = np.array([ 1, 2, 1], dtype = Float)
    sobel_h /= 3
    sobel_v /= 3
    
    img1 = convolve_2D_sep(
        img,
        sobel_h,
        sobel_v
    )
    
    # Use this as the output image to avoid more temporary arrays
    new_img = convolve_2D_sep(
        img,
        sobel_v,
        sobel_h
    )
    
    # consolidate the two images
    np.hypot(img1, new_img, out = new_img)
    
    return new_img


def variance_normalization_filter(
    img,
    sigma1 = 1.0,
    sigma2 = 2.0,
    truncate = 2.0,
    clip_at_zero = True,
):
    """Gaussian variance normalization filter.
    
    Gaussian variance normalization via two gaussian filters.
    Assumes array is normalized to [0..1].

    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    sigma1 : float
        Sigma of nxn gaussian convolution kernel.
    sigma2 : float
        Sigma of nxn gaussian convolution kernel.
    truncate : float
        Truncate the kernel at specified standard deviations.
    clip_at_zero : bool
        Clip negative pixel intenensities.

    Returns
    -------
    new_img : 2D float32 array
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img)
    
    high_pass = highpass_filter(
        img, 
        sigma1, 
        truncate, 
        False, 
    )
    
    den = gaussian_filter(
        high_pass * high_pass, 
        sigma2, 
        truncate, 
    )
    
    np.sqrt(den, out = den)
    
    new_img = np.divide( # stops image from being all black
        high_pass, den,
        where = (den != 0.0)
    )  
    
    if clip_at_zero == True: 
        new_img[new_img < 0] = 0 
        
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min())
    
    return new_img
    

def threshold_binarization(
    img, 
    threshold = 0.5,
    out = None
):
    """Threshold binarization filter.
    
    A simple threshold binarization filter.
    Assumes array is normalized to [0..1].

    Parameters
    ----------
    img : 2D float32 array
        A two dimensional array containing pixel intenensities.
    kernel_size : int
        nxn size of the convolution kernel.
    sigma : float
        sigma of nxn gaussian convolution kernel.
    keep_dtype : bool
        Cast output to original dtype.

    Returns
    -------
    new_img : 2D float32 array
        A two dimensional array containing pixel intenensities.

    """
    _check(ndim=2, img=img)

    # extract filtered image
    img[img < threshold] = 0
    img[img > threshold] = 1

    return img