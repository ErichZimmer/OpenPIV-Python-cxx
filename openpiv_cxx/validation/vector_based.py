"""
Module for validating Particle Image Velocimetry (PIV) data (vectors) to attain a higher quality vector field.
"""
import numpy as np

__all__ = [
    "global_val",
    "global_std",
    "global_z_score"
]


###############################################################################
# These functions are implemented in Python as they are already "fast enough" #
###############################################################################


def global_val(u, v, u_thresholds, v_thresholds, mask = None, convention = "openpiv"):
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

    u_thresholds: two elements tuple
        u_thresholds = (u_min, u_max). If u < u_min or u > u_max
        the vector is treated as an outlier.

    v_thresholds: two elements tuple
        v_thresholds = (v_min, v_max). If v < v_min or v > v_max
        the vector is treated as an outlier.
    
    mask : 2d np.ndarray
        A two dimensional array containing the flags for invalid vectors.
        
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

    mask : boolean or int 2d np.ndarray 
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


def global_std(u, v, std_threshold=5, mask = None, convention = "openpiv"):
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

    std_threshold: float
        If the length of the vector (actually the sum of squared components) is
        larger than std_threshold times standard deviation of the flow field,
        then the vector is treated as an outlier. [default = 3]
    
    mask : 2d np.ndarray
        A two dimensional array containing the flags for invalid vectors.
        
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

    mask : boolean or int 2d np.ndarray 
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


def global_z_score(u, v, std_threshold=5, mask = None, convention = "openpiv"):
    """Eliminate spurious vectors with a global threshold.

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with NaN (Not a Number) if at least
    one of the two velocity components is out of a specified global range.


    Parameters
    ----------
    u : 2d masked np.ndarray
        A two dimensional array containing the u velocity component.

    v : 2d masked np.ndarray
        A two dimensional array containing the v velocity component.

    std_threshold: float
        If the length of the vector (actually the sum of squared components) is
        larger than std_threshold times standard deviation of the flow field,
        then the vector is treated as an outlier. [default = 3]
    
    mask : 2d np.ndarray
        A two dimensional array containing the flags for invalid vectors.
        
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

    mask : boolean or int 2d np.ndarray 
        A boolean or integer array where elemtents that = 0 are valid
        and 1 = invalid. 

    """
    # both previous nans and masked regions are not 
    # participating in the magnitude comparison

    # create nan filled arrays where masks
    # if u,v, are non-masked, ma.copy() adds false masks
    
    if isinstance(u, np.ma.MaskedArray):
        velocity_magnitude = np.hypot(u.filled(np.nan), v.filled(np.nan))    
    else:
        velocity_magnitude = np.hypot(u, v)

    ind = threshold < np.abs(((velocity_magnitude - np.nanmean(velocity_magnitude)) / np.nanstd(velocity_magnitude)))

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
    

###############################################################################
#                   These functions are implemented in c++                    #
###############################################################################