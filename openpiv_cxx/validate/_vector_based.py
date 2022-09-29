from openpiv_cxx.input_checker import check_nd as _check
from ._validation_cpp import (
    _difference_test,
    _local_median_test,
    _normalized_local_median_test,
)

import numpy as np


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
    u,
    v,
    u_thresholds=[-10, 10],
    v_thresholds=[-10, 10],
    mask=None,
    convention="openpiv",
):
    """Eliminate spurious vectors with a global threshold.

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with Nan (Not a Number) if at
    least one of the two velocity components is out of a specified global
    range.

    Parameters
    ----------
    u : ndarray
        A two dimensional array containing the u velocity component.
    v : ndarray
        A two dimensional array containing the v velocity component.
    mask : ndarray, optional
        A two dimensional array containing the flags for invalid vectors.
    u_thresholds: tuple
        u_thresholds = (u_min, u_max). If u < u_min or u > u_max
        the vector is treated as an outlier.
    v_thresholds: tuple
        v_thresholds = (v_min, v_max). If v < v_min or v > v_max
        the vector is treated as an outlier.
    convention: str
        Which flag convention to use.

    Returns
    -------
    u : ndarray
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    v : ndarray
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    mask : ndarray
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid.

    """

    np.warnings.filterwarnings("ignore")

    _check(ndim=2, u=u, v=v)

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
            if mask.shape != ind.shape:
                raise ValueError("mask shape must be same as u/v shape")
            mask[ind] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind] = 1

        return mask


def global_std(u, v, mask=None, std_threshold=3.0, convention="openpiv"):
    """Eliminate spurious vectors with a global threshold defined by the
    standard deviation

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with NaN (Not a Number) if at least
    one of the two velocity components is out of a specified global range.

    Parameters
    ----------
    u : ndarray
        A two dimensional array containing the u velocity component.
    v : ndarray
        A two dimensional array containing the v velocity component.
    mask : ndarray, optional
        A two dimensional array containing the flags for invalid vectors.
    std_threshold : float
        If the length of the vector (actually the sum of squared components) is
        larger than std_threshold times standard deviation of the flow field,
        then the vector is treated as an outlier.
    convention : str
        Which flag convention to use.

    Returns
    -------
    u : ndarray, optional
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    v : ndarray, optional
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    mask : ndarray
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid.

    """

    # both previous nans and masked regions are not
    # participating in the magnitude comparison

    # create nan filled arrays where masks
    # if u,v, are non-masked, ma.copy() adds false masks

    _check(ndim=2, u=u, v=v)

    if isinstance(u, np.ma.MaskedArray):
        tmpu = np.ma.copy(u).filled(np.nan)
        tmpv = np.ma.copy(v).filled(np.nan)
    else:
        tmpu = np.copy(u)
        tmpv = np.copy(v)

    ind = np.logical_or(
        np.abs(tmpu - np.nanmean(tmpu)) > std_threshold * np.nanstd(tmpu),
        np.abs(tmpv - np.nanmean(tmpv)) > std_threshold * np.nanstd(tmpv),
    )

    if np.all(ind):  # if all is True, something is really wrong
        print("Warning! probably a uniform shift data, do not use this filter")
        ind = ~ind

    if convention == "openpiv":
        u[ind] = np.nan
        v[ind] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind] = True

        return u, v, mask

    else:
        if isinstance(mask, np.ndarray):
            if mask.shape != ind.shape:
                raise ValueError("mask shape must be same as u/v shape")
            mask[ind] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind] = 1

        return mask


##############################################################################
#      These functions are implemented in c++ and wrapped with pybind11      #
##############################################################################


def local_difference(
    u,
    v,
    mask=None,
    threshold_u=3.0,
    threshold_v=3.0,
    threshold=None,
    convention="openpiv",
):
    """Eliminate spurious vectors with a local threshold.

    This validation method tests for the local consistency of the data
    and outliers vector are replaced with NaN (Not a Number) if there
    are greater than 4 vector differences that exceed the threshold limit.
    See reference for more details.

    Parameters
    ----------
    u : ndarray
        A two dimensional array containing the u velocity component.
    v : ndarray
        A two dimensional array containing the v velocity component.
    mask : ndarray, optional
        A two dimensional array containing the flags for invalid vectors.
    threshold_u : float
        Threshold for u component.
    threshold_v : float
        Threshold for v component.
    threshold : float, optional
        Set thresholds for both u and v components.
    convention : str
        Which flag convention to use.

    Returns
    -------
    u : ndarray, optional
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    v : ndarray, optional
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    mask : ndarray
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid.

    """
    _check(ndim=2, u=u, v=v)

    if isinstance(mask, np.ndarray):
        _check(ndim=2, mask=mask)

    if threshold != None:
        threshold_u = threshold_v = threshold

    # pad array by kernel half size (kernel size is 3x3)
    buffer_u = np.pad(u, 1, mode="constant", constant_values=np.nan)
    buffer_v = np.pad(v, 1, mode="constant", constant_values=np.nan)

    # make sure array is float64
    if buffer_u.dtype != "float64":
        buffer_u = buffer_u.astype("float64")

    if buffer_v.dtype != "float64":
        buffer_v = buffer_v.astype("float64")

    ind = _difference_test(buffer_u, buffer_v, float(threshold_u), float(threshold_v))

    slices = (slice(1, ind.shape[0] - 1), slice(1, ind.shape[1] - 1))

    ind = ind[slices]

    if convention == "openpiv":
        u[ind != 0] = np.nan
        v[ind != 0] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind != 0] = True

        return u, v, mask

    else:
        if isinstance(mask, np.ndarray):
            if mask.shape != ind.shape:
                raise ValueError("mask shape must be same as u/v shape")
            mask[ind != 0] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind != 0] = 1

        return mask


def local_median(
    u,
    v,
    mask=None,
    threshold_u=3.0,
    threshold_v=3.0,
    threshold=None,
    size=2,
    kernel_min_size=1,
    convention="openpiv",
):
    """Eliminate spurious vectors with a local median threshold.

    This validation method tests for the spatial consistency of the data.
    Vectors are classified as outliers and replaced with Nan (Not a Number) if
    the absolute difference with the local median is greater than a user
    specified threshold. The median is computed for both velocity components.

    Parameters
    ----------
    u : ndarray
        A two dimensional array containing the u velocity component.
    v : ndarray
        A two dimensional array containing the v velocity component.
    mask : ndarray, optional
        A two dimensional array containing the flags for invalid vectors.
    threshold_u : float
        Threshold for u component.
    threshold_v : float
        Threshold for v component.
    threshold : float, optional
        Set thresholds for both u and v components.
    size : int
        The radius of the median kernel.
    kernel_min_size : int
        The minimum amount of non-NaN values in a kernel. If less, the kernel
        is marked as invalid due to not enough valid points.
    convention : str
        Which flag convention to use.

    Returns
    -------
    u : ndarray, optional
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    v : ndarray, optional
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    mask : ndarray
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid.

    """
    _check(ndim=2, u=u, v=v)

    if isinstance(mask, np.ndarray):
        _check(ndim=2, mask=mask)

    if threshold != None:
        threshold_u = threshold_v = threshold

    # pad array by kernel half size
    buffer_u = np.pad(u, size, mode="constant", constant_values=np.nan)
    buffer_v = np.pad(v, size, mode="constant", constant_values=np.nan)

    # make sure array is float64
    if buffer_u.dtype != "float64":
        buffer_u = buffer_u.astype("float64")

    if buffer_v.dtype != "float64":
        buffer_v = buffer_v.astype("float64")

    ind = _local_median_test(
        buffer_u, buffer_v, threshold_u, threshold_v, int(size), int(kernel_min_size)
    )

    slices = (slice(size, ind.shape[0] - size), slice(size, ind.shape[1] - size))

    ind = ind[slices]

    if convention == "openpiv":
        u[ind != 0] = np.nan
        v[ind != 0] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind != 0] = True

        return u, v, mask

    else:
        if isinstance(mask, np.ndarray):
            if mask.shape != ind.shape:
                raise ValueError("mask shape must be same as u/v shape")
            mask[ind != 0] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind != 0] = 1

        return mask


def normalized_local_median(
    u,
    v,
    mask=None,
    threshold_u=3.0,
    threshold_v=3.0,
    threshold=None,
    size=2,
    kernel_min_size=1,
    eps=0.1,
    convention="openpiv",
):
    """Eliminate spurious vectors with a local normalized median threshold.

    This validation method tests for the spatial consistency of the data.
    Vectors are classified as outliers and replaced with Nan (Not a Number) if
    the absolute difference with the local median is greater than a user
    specified threshold. The median is computed for both velocity components.

    Parameters
    ----------
    u : ndarray
        A two dimensional array containing the u velocity component.
    v : ndarray
        A two dimensional array containing the v velocity component.
    mask : ndarray, optional
        A two dimensional array containing the flags for invalid vectors.
    threshold_u : float
        Threshold for u component.
    threshold_v : float
        Threshold for v component.
    threshold : float, optional
        Set thresholds for both u and v components.
    size : int
        The radius of the median kernel.
    kernel_min_size : int
        The minimum amount of non-NaN values in a kernel. If less, the kernel
        is marked as invalid due to not enough valid points.
    eps : float
        Epsilon, should remain bewteen 0.1 and 0.2.
    convention: str
        Which flag convention to use.

    Returns
    -------
    u : ndarray, optional
        A two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    v : ndarray, optional
        A two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN if convention
        is set to 'openpiv'.
    mask : ndarray
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid.

    """
    _check(ndim=2, u=u, v=v)

    if isinstance(mask, np.ndarray):
        _check(ndim=2, mask=mask)

    if threshold != None:
        threshold_u = threshold_v = threshold

    # pad array by kernel half size
    buffer_u = np.pad(u, size, mode="constant", constant_values=np.nan)
    buffer_v = np.pad(v, size, mode="constant", constant_values=np.nan)

    # make sure array is float64
    if buffer_u.dtype != "float64":
        buffer_u = buffer_u.astype("float64")

    if buffer_v.dtype != "float64":
        buffer_v = buffer_v.astype("float64")

    ind = _normalized_local_median_test(
        buffer_u,
        buffer_v,
        threshold_u,
        threshold_v,
        int(size),
        float(eps),
        int(kernel_min_size),
    )

    slices = (slice(size, ind.shape[0] - size), slice(size, ind.shape[1] - size))

    ind = ind[slices]

    if convention == "openpiv":
        u[ind != 0] = np.nan
        v[ind != 0] = np.nan

        mask = np.zeros_like(u, dtype=bool)
        mask[ind != 0] = True

        return u, v, mask

    else:
        if isinstance(mask, np.ndarray):
            if mask.shape != ind.shape:
                raise ValueError("mask shape must be same as u/v shape")
            mask[ind != 0] = 1
        else:
            mask = np.zeros_like(u, dtype=int)
            mask[ind != 0] = 1

        return mask
