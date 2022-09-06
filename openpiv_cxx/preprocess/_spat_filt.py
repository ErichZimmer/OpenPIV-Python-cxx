from numpy import pad as pad_array, percentile, clip, ndarray
from openpiv_cxx.input_checker import check_nd as _check
from ._spatial_filters import _intensity_cap, _threshold_binarization,\
    _lowpass_filter, _highpass_filter, _local_variance_normalization,\
    _gaussian_kernel

__all__ = [
    "intensity_cap",
    "threshold_binarization",
    "gaussian_filter",
    "highpass_filter",
    "variance_normalization_filter",
    "contrast_stretch"
]

kernel_size_error = "kernel_size must be an odd number"
    
def intensity_cap(
    img: ndarray, 
    std_mult: int = 2.0
) -> ndarray:
    """Set pixels above threshold to threshold.
    
    Set pixels higher than calculated threshold to said threshold. 
    The threshold value is calcualted by img_mean + std_mult * img_std.
    
    Parameters
    ----------
    img: ndarray
        A two dimensional array containing pixel intenensities.
    std_mult: float
        Lower values yields a lower threshold
    
    Returns
    -------
    img: ndarray
        A filtered two dimensional array of the input image
        
    """
    _check(ndim = 2,
        img = img
    )
    
    # store original image dtype
    img_dtype = img.dtype
    
    if img_dtype != "float32":
        img = img.astype("float32")
        
    new_img = _intensity_cap(
        img,
        float(std_mult)
    )
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    return new_img


def threshold_binarization(
    img: ndarray, 
    threshold: float = 0.5
) -> ndarray:
    """A threshold binarization filter.
    
    Parameters
    ----------
    img : ndarray 
        A two dimensional array containing pixel intenensities.
    kernel_size : int
        nxn size of the convolution kernel
    sigma : float
        sigma of nxn gaussian convolution kernel
        
    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
    
    """
    _check(ndim = 2,
        img = img
    )
    
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
    new_img = _threshold_binarization(
        img,
        float(threshold)
    )
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    return new_img


def gaussian_filter(
    img: ndarray, 
    kernel_size: int = 3, 
    sigma: float = 1.0
) -> ndarray:
    """A simple sliding window gaussian low pass filter.
    
    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    kernel_size : int
        nxn size of the convolution kernel
    sigma : float
        Sigma of nxn gaussian convolution kernel
        
    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
    
    """
    _check(ndim = 2,
        img = img
    )
    
    if kernel_size % 2 != 1:
        raise Exception(kernel_size_error)
        
    # store original image dtype
    img_dtype = img.dtype
    max_ = img.max()
    flip = False
    
    # check if image needs to be flipped
    if img.shape[0] > img.shape[1]:
        flip = True
        img = img.T
    
    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = pad_array(img, pad, mode = "reflect")
    
    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # normalize array if pixel intensities are greater than 1
    if max_ > 1: 
        buffer1 = buffer1.astype("float32")
        buffer1 /= max_
    
    kernel = _gaussian_kernel(kernel_size, sigma)
    
    # extract filtered image
    new_img = _lowpass_filter(
        buffer1,
        kernel
    )
    
    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    if flip == True:
        new_img = new_img.T
        
    return new_img


def highpass_filter(
    img: ndarray, 
    kernel_size: int = 3, 
    sigma: float = 1.0, 
    clip_at_zero: bool = True
) -> ndarray:
    """A simple sliding window gaussian high pass filter.
    
    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    kernel_size : int
        nxn size of the convolution kernel.
    sigma : float
        Sigma of nxn gaussian convolution kernel.
        
    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
    """
    _check(ndim = 2,
        img = img
    )
    
    if kernel_size % 2 != 1:
        raise Exception(kernel_size_error)
        
    # store original image dtype
    img_dtype = img.dtype
    max_ = img.max()
    flip = False
    
    # check if image needs to be flipped
    if img.shape[0] > img.shape[1]:
        flip = True
        img = img.T
    
    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = pad_array(img, pad, mode = "reflect")
    
    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # normalize array if pixel intensities are greater than 1
    if max_ > 1: 
        buffer1 = buffer1.astype("float32")
        buffer1 /= max_
    
    kernel = _gaussian_kernel(kernel_size, sigma)
    
    # extract filtered image
    new_img = _gaussian_highpass_filter(
        buffer1,
        kernel,
        bool(clip_at_zero)
    )
    
    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_

    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    if flip == True:
        new_img = new_img.T
        
    return new_img


def variance_normalization_filter(
    img: ndarray, 
    kernel_size: int = 3, 
    sigma1: float = 1.0, 
    sigma2: float = 1.0, 
    clip_at_zero: bool = True
) -> ndarray:
    """A simple gaussian variance normalization filter.
    
    Parameters
    ----------
    img : ndarray
        A two dimensional array containing pixel intenensities.
    kernel_size : int
        nxn size of the convolution kernel.
    sigma1 : float
        Sigma of nxn gaussian convolution kernel.
    sigma2 : float
        Sigma of nxn gaussian convolution kernel.
        
    Returns
    -------
    new_img : ndarray
        A two dimensional array containing pixel intenensities.
        
    """
    _check(ndim = 2,
        img = img
    )
    
    if kernel_size % 2 != 1:
        raise Exception(kernel_size_error)
        
    # store original image dtype
    img_dtype = img.dtype
    max_ = img.max()
    flip = False
    
    # check if image needs to be flipped
    if img.shape[0] > img.shape[1]:
        flip = True
        img = img.T
    
    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = pad_array(img, pad, mode = "reflect")
    
    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # normalize array if pixel intensities are greater than 1
    if max_ > 1: 
        buffer1 = buffer1.astype("float32")
        buffer1 /= max_
        
    # extract filtered image
    new_img = _local_variance_normalization(
        buffer1, 
        kernel_size,
        float(sigma1), float(sigma2),
        bool(clip_at_zero)
    )
    
    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    if flip == True:
        new_img = new_img.T
        
    return new_img


def contrast_stretch(
    img: ndarray, 
    lower_limit: float = 2.0, 
    upper_limit: float = 98.0
) -> ndarray:
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
    _check(ndim = 2,
        img = img
    )
    
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

    lower = percentile(img, lower_limit)
    upper = percentile(img, upper_limit)

    img_max = img.max()
    img_min = img.min()

    img = clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    stretch_min = 0
    if img_max < 2:
        stretch_max = 1
    elif img_max < 2**8 - 1:
        stretch_max = 2**8 - 1
    elif img_max < 2**10 - 1:
        stretch_max = 2**10 - 1
    elif img_max < 2**12 - 1:
        stretch_max = 2**12 - 1
    elif img_max < 2**14 - 1:
        stretch_max = 2**14 - 1
    else:
        stretch_max = 2**16 - 1

    img *= stretch_max

    return img.astype(img_dtype)