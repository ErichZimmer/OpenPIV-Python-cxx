from numpy import array
from numpy.ma import MaskedArray
from openpiv_cxx import process as piv_proc
from openpiv_cxx.interpolate import bilinear2D, whittaker2D
from .window_deformation import deform_windows, create_deformation_field

__all__ = [
    "first_pass",
    "multipass_img_deform"
]

def first_pass(
    frame_a, 
    frame_b,
    window_size,
    overlap,
    correlation_method="circular",
    normalized_correlation=False,
):
    """
    First pass of the PIV evaluation.
    This function does the PIV evaluation of the first pass. It returns
    the coordinates of the interrogation window centres, the displacment
    u and v for each interrogation window as well as the mask which indicates
    wether the displacement vector was interpolated or not.
    
    
    Parameters
    ----------
    frame_a : 2D np.ndarray
        The first image.
        
    frame_b : 2D np.ndarray
        The second image.
        
    window_size : int
         The size of the interrogation window.
         
    overlap : int
        The overlap of the interrogation window, typically it is window_size/2.
    
    
    Returns
    -------
    x : 2D np.array
        Array containg the x coordinates of the interrogation window centres.
        
    y : 2D np.array
        Array containg the y coordinates of the interrogation window centres.
        
    u : 2D np.array
        Array containing the u displacement for every interrogation window.
        
    u : 2D np.array
        Array containing the v displacement for every interrogation window.
    """

    #     if do_sig2noise is False or iterations != 1:
    #         sig2noise_method = None  # this indicates to get out nans
    
    if normalized_correlation == True:
        algorithm = 'ncc'
    else:
        algorithm = 'scc'
    
    cmatrix = piv_proc.fft_correlate_images(
        frame_a,
        frame_b,
        window_size,
        overlap,
        algorithm,
        correlation_method,
        thread_count = 1
    )
    
    field_shape = piv_proc.get_field_shape(
            frame_a.shape,
            window_size,
            overlap
        )
    
    u, v, peakHeight, s2n = piv_proc.correlation_to_displacement(
        cmatrix,
        field_shape[0],
        field_shape[1],
        limit_peak_search = False,
        thread_count = 1
    )

    x, y = piv_proc.get_rect_coordinates(
        frame_a.shape,
        window_size,
        overlap
    )

    return x, y, u, v, s2n


def multipass_img_deform(
    frame_a,
    frame_b,
    x_old,
    y_old,
    u_old,
    v_old,
    window_size,
    overlap,
    correlation_method="circular",
    normalized_correlation=False,
    deformation_method="symmetric",
    radius=2
):
    """
    Multi pass of the PIV evaluation.
    This function does the PIV evaluation of the second and other passes.
    It returns the coordinates of the interrogation window centres,
    the displacement u, v for each interrogation window as well as
    the signal to noise ratio array (which is filled with NaNs if opted out).
    
    
    Parameters
    ----------
    frame_a : 2D np.ndarray
        The first image.

    frame_b : 2D np.ndarray
        The second image.

    window_size : tuple of ints
         The size of the interrogation window.

    overlap : tuple of ints
        The overlap of the interrogation window, e.g. window_size/2.

    x_old : 2D np.ndarray
        The x coordinates of the vector field of the previous pass.

    y_old : 2D np.ndarray
        The y coordinates of the vector field of the previous pass.

    u_old : 2D np.ndarray
        The u displacement of the vector field of the previous pass
        in case of the image mask - u_old and v_old are MaskedArrays.

    v_old : 2D np.ndarray
        The v displacement of the vector field of the previous pass.

    radius : int
        The order of the spline interpolation used for the image deformation.


    Returns
    -------
    x : 2D np.array
        Array containg the x coordinates of the interrogation window centres.

    y : 2D np.array
        Array containg the y coordinates of the interrogation window centres.

    u : 2D np.array
        Array containing the u displacement for every interrogation window.

    u : 2D np.array
        Array containing the v displacement for every interrogation window.

    s2n : 2D np.array 
        Array consisting of signal to noise ratio values.
    """
    x, y = piv_proc.get_rect_coordinates(
        frame_a.shape,
        window_size,
        overlap
    )
    
    # The interpolation function dont like meshgrids as input.
    # plus the coordinate system for y is now from top to bottom
    # and RectBivariateSpline wants an increasing set

    y_old = y_old[:, 0]
    x_old = x_old[0, :]

    y_int = y[:, 0]
    x_int = x[0, :]

    # interpolating the displacements from the old grid onto the new grid
    # y befor x because of numpy works row major
    if isinstance(u_old, MaskedArray):
        u_old = u_old.filled(0.)
        v_old = v_old.filled(0.)
        
    u_pre = bilinear2D(y_old, x_old, u_old, y_int, x_int)
    v_pre = bilinear2D(y_old, x_old, v_old, y_int, x_int)

    x_new, y_new, ut, vt = create_deformation_field(
        frame_a, 
        x, y,
        u_pre, v_pre
    )
    
    if deformation_method == "second image":
        frame_b = whittaker2D(
            frame_b.astype("float64"), # so we don't lose any data
            y_new - vt,
            x_new + ut,
            radius
        )
        
    elif deformation_method == "symmetric":
        frame_a = whittaker2D(
            frame_a.astype("float64"), 
            y_new - vt / 2,
            x_new - ut / 2,
            radius
        )
        frame_b = whittaker2D(
            frame_b.astype("float64"), 
            y_new + vt / 2,
            x_new + ut / 2,
            radius
        )
    
    else:
        raise ValueError(
            "Deformation method not supported"
        )
        
    x, y, u, v, s2n = first_pass(
        frame_a,
        frame_b,
        window_size,
        overlap,
        correlation_method,
        normalized_correlation
    )

    u += u_pre
    v += v_pre
    
    return x, y, u, v, s2n