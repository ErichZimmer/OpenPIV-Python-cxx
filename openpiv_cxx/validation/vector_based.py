"""
Module for validating Particle Image Velocimetry (PIV) data (vectors) to attain a higher quality vector field.
"""
# Note: typing.Union doesn't work with np.ndarray
from typing import Optional

import numpy as np

from ._validation import _difference_test, _local_median_test, \
    _noarmalized_local_median_test


__all__ = [
    "global_val",
    "global_std",
    "local_difference",
    "local_median",
    "normalized_local_median",
]


##############################################################################
# These functions are implemented in NumPy as they are already "fast enough" #
##############################################################################


def global_val(
    u: np.ndarray,
    v: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    u_thresholds: tuple( [float, float] ), 
    v_thresholds: tuple( [float, float] ), 
    convention: str = "openpiv"
) -> np.ndarray:
    """Eliminate spurious vectors with a global threshold.

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with Nan (Not a Number) if at
    least one of the two velocity components is out of a specified global
    range.


    Parameters
    ----------
    u : 2d np.ndarray
        A two dimensional array containing the u velocity component.

    v : 2d np.ndarray
        A two dimensional array containing the v velocity component.

    mask : 2d np.ndarray (optional)
        A two dimensional array containing the flags for invalid vectors.
        
    u_thresholds: two elements tuple
        u_thresholds = (u_min, u_max). If u < u_min or u > u_max
        the vector is treated as an outlier.

    v_thresholds: two elements tuple
        v_thresholds = (v_min, v_max). If v < v_min or v > v_max
        the vector is treated as an outlier.
        
    convention: "string"
        Which flag convention to use.


    Returns
    -------
    u : 2d np.ndarray
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    v : 2d np.ndarray
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    mask : boolean or 2D np.ndarray of dtype int
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid. 

    """

    np.warnings.filterwarnings("ignore")

    ind = np.logical_or(
        np.logical_or(u < u_thresholds[0], u > u_thresholds[1]),
        np.logical_or(v < v_thresholds[0], v > v_thresholds[1]),
    )
    
    if convention == "openpiv":
        u[ind] = np.nan
        v[ind] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind] = True

        return u, v, mask
    
    else:
        if isinstance(mask, np.ndarray):
            mask[ind] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind] = 1
            
        return mask


def global_std(
    u: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    std_threshold: float = 3.0, 
    convention: str = "openpiv"
) -> np.ndarray:
    """Eliminate spurious vectors with a global threshold defined by the
    standard deviation

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with NaN (Not a Number) if at least
    one of the two velocity components is out of a specified global range.


    Parameters
    ----------
    u : 2d masked np.ndarray
        A two dimensional array containing the u velocity component.

    v : 2d masked np.ndarray
        A two dimensional array containing the v velocity component.

    mask : 2d np.ndarray (optional)
        A two dimensional array containing the flags for invalid vectors.
        
    std_threshold: float
        If the length of the vector (actually the sum of squared components) is
        larger than std_threshold times standard deviation of the flow field,
        then the vector is treated as an outlier. [default = 3]
        
    convention: "string"
        Which flag convention to use.


    Returns
    -------
    u : 2d np.ndarray
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    v : 2d np.ndarray
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    mask : boolean or 2D np.ndarray of dtype int
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid. 

    """
    # both previous nans and masked regions are not 
    # participating in the magnitude comparison

    # create nan filled arrays where masks
    # if u,v, are non-masked, ma.copy() adds false masks
    
    if isinstance(u, np.ma.MaskedArray):
        tmpu = np.ma.copy(u).filled(np.nan)
        tmpv = np.ma.copy(v).filled(np.nan)
    else:
        tmpu = np.copy(u)
        tmpv = np.copy(v)

    ind = np.logical_or(np.abs(tmpu - np.nanmean(tmpu)) > std_threshold * np.nanstd(tmpu), 
                        np.abs(tmpv - np.nanmean(tmpv)) > std_threshold * np.nanstd(tmpv))

    if np.all(ind): # if all is True, something is really wrong
        print('Warning! probably a uniform shift data, do not use this filter')
        ind = ~ind

    if convention == "openpiv":
        u[ind] = np.nan
        v[ind] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind] = True
        
        return u, v, mask
    
    else:
        if isinstance(mask, np.ndarray):
            mask[ind] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind] = 1
        
        return mask
    

##############################################################################
#      These functions are implemented in c++ and wrapped with pybind11      #
##############################################################################


def local_difference(
    u: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 3.0, 
    convention: str = "openpiv"
) -> np.ndarray:
    """Eliminate spurious vectors with a local threshold.

    This validation method tests for the local consistency of the data
    and outliers vector are replaced with NaN (Not a Number) if there
    are greater than 4 vector differences that exceed the threshold limit.
    See reference for more details.


    Parameters
    ----------
    u : 2d masked np.ndarray
        A two dimensional array containing the u velocity component.

    v : 2d masked np.ndarray
        A two dimensional array containing the v velocity component.

    mask : 2d np.ndarray (optional)
        A two dimensional array containing the flags for invalid vectors.
    
    threshold: float
        Threshold for u and v component.
        
    convention: "string"
        Which flag convention to use.


    Returns
    -------
    u : 2d np.ndarray
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    v : 2d np.ndarray
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    mask : boolean or 2D np.ndarray of dtype int
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid. 

    """
    # pad array by kernel half size
    buffer_u = np.pad(u, 1, mode = "constant", constant_values = np.nan)
    buffer_v = np.pad(v, 1, mode = "constant", constant_values = np.nan)
    
    # make sure array is float64
    if buffer_u.dtype != "float64":
        buffer_u = buffer_u.astype("float64")
        
    if buffer_v.dtype != "float64":
        buffer_v = buffer_v.astype("float64")
    
    ind = np.zeros(buffer_u.shape, dtype = int)
    
    _difference_test(
        buffer_u,
        buffer_v,
        ind,
        threshold
    )
    
    slices = (
        slice(1, ind.shape[0]-1),
        slice(1, ind.shape[1]-1)
    )
    
    ind = ind[slices]

    if convention == "openpiv":
        u[ind != 0] = np.nan
        v[ind != 0] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind != 0] = True

        return u, v, mask
    
    else:
        if isinstance(mask, np.ndarray):
            mask[ind != 0] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind != 0] = 1
        
        return mask
    

def local_median(
    u: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 3.0, 
    threshold_dummy: Optional[float] = None, # used only for compatability
    size: int = 1,
    kernel_min_size: int = 0,
    convention: str = "openpiv"
) -> np.ndarray:
    """Eliminate spurious vectors with a local median threshold.

    This validation method tests for the spatial consistency of the data.
    Vectors are classified as outliers and replaced with Nan (Not a Number) if
    the absolute difference with the local median is greater than a user
    specified threshold. The median is computed for both velocity components.


    Parameters
    ----------
    u : 2d masked np.ndarray
        A two dimensional array containing the u velocity component.

    v : 2d masked np.ndarray
        A two dimensional array containing the v velocity component.

    mask : 2d np.ndarray (optional)
        A two dimensional array containing the flags for invalid vectors.
    
    threshold: float
        Threshold for u and v component.
    
    size : int
        The radius of the median kernel.
    
    kernel_min_size : int
        The minimum amount of non-NaN values in a kernel. If less, the kernel
        is marked as invalid due to not enough valid points.
    
    convention: "string"
        Which flag convention to use.


    Returns
    -------
    u : 2d np.ndarray
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    v : 2d np.ndarray
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    mask : boolean or 2D np.ndarray of dtype int
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid. 

    """
    # pad array by kernel half size
    buffer_u = np.pad(u, size, mode = "constant", constant_values = np.nan)
    buffer_v = np.pad(v, size, mode = "constant", constant_values = np.nan)
    
    # make sure array is float64
    if buffer_u.dtype != "float64":
        buffer_u = buffer_u.astype("float64")
        
    if buffer_v.dtype != "float64":
        buffer_v = buffer_v.astype("float64")
    
    ind = np.zeros(buffer_u.shape, dtype = int)
    
    _local_median_test(
        buffer_u,
        buffer_v,
        ind,
        threshold,
        int(size),
        int(kernel_min_size)
    )
    
    slices = (
        slice(size, ind.shape[0]-size),
        slice(size, ind.shape[1]-size)
    )
    
    ind = ind[slices]
    
    if convention == "openpiv":
        u[ind != 0] = np.nan
        v[ind != 0] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind != 0] = True

        return u, v, mask
    
    else:
        if isinstance(mask, np.ndarray):
            mask[ind != 0] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind != 0] = 1
        
        return mask
    

def local_median(
    u: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 3.0, 
    threshold_dummy: Optional[float] = None, # used only for compatability
    size: int = 1,
    kernel_min_size: int = 0,
    eps: float = 0.1,
    convention: str = "openpiv"
) -> np.ndarray:
    """Eliminate spurious vectors with a local normalized median threshold.

    This validation method tests for the spatial consistency of the data.
    Vectors are classified as outliers and replaced with Nan (Not a Number) if
    the absolute difference with the local median is greater than a user
    specified threshold. The median is computed for both velocity components.


    Parameters
    ----------
    u : 2d masked np.ndarray
        A two dimensional array containing the u velocity component.

    v : 2d masked np.ndarray
        A two dimensional array containing the v velocity component.

    mask : 2d np.ndarray (optional)
        A two dimensional array containing the flags for invalid vectors.
    
    threshold: float
        Threshold for u and v component.
    
    size : int
        The radius of the median kernel.
    
    kernel_min_size : int
        The minimum amount of non-NaN values in a kernel. If less, the kernel
        is marked as invalid due to not enough valid points.
    
    eps : float
        Epsilon.
        
    convention: "string"
        Which flag convention to use.


    Returns
    -------
    u : 2d np.ndarray
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    v : 2d np.ndarray
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.

    mask : boolean or 2D np.ndarray of dtype int
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid. 

    """
    # pad array by kernel half size
    buffer_u = np.pad(u, size, mode = "constant", constant_values = np.nan)
    buffer_v = np.pad(v, size, mode = "constant", constant_values = np.nan)
    
    # make sure array is float64
    if buffer_u.dtype != "float64":
        buffer_u = buffer_u.astype("float64")
        
    if buffer_v.dtype != "float64":
        buffer_v = buffer_v.astype("float64")
    
    ind = np.zeros(buffer_u.shape, dtype = int)
    
    _normalized_local_median_test(
        buffer_u,
        buffer_v,
        ind,
        threshold,
        int(size),
        float(eps)
        int(kernel_min_size)
    )
    
    slices = (
        slice(size, ind.shape[0]-size),
        slice(size, ind.shape[1]-size)
    )
    
    ind = ind[slices]
    
    if convention == "openpiv":
        u[ind != 0] = np.nan
        v[ind != 0] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind != 0] = True

        return u, v, mask
    
    else:
        if isinstance(mask, np.ndarray):
            mask[ind != 0] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind != 0] = 1
        
        return mask


###############################################################################
#                   Aliases for compatability with OpenPIV-Python             #
###############################################################################

local_median_val = local_median