from numpy import array, ndarray
from numpy.ma import MaskedArray
from openpiv_cxx import process as piv_proc
from openpiv_cxx.interpolate import bilinear2D, whittaker2D
from .window_deformation import deform_windows, create_deformation_field

__all__ = [
    "first_pass",
    "multipass_img_deform"
]

def first_pass(
    frame_a: ndarray, 
    frame_b: ndarray,
    window_size: int = 32,
    overlap: int = 16,
    correlation_method: str = "circular"
) -> tuple( [[ndarray] * 5] ):
    """
    First pass of the PIV evaluation.
    This function does the PIV evaluation of the first pass. It returns
    the coordinates of the interrogation window centres, the displacment
    u and v for each interrogation window as well as the mask which indicates
    wether the displacement vector was interpolated or not.
    
    
    Parameters
    ----------
    frame_a : 2D ndarray
        The first image.
        
    frame_b : 2D ndarray
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
    
    s2n : 2D np.array 
        Array consisting of signal to noise ratio values.
    """
    
    cmatrix = piv_proc.fft_correlate_images(
        frame_a,
        frame_b,
        window_size,
        overlap,
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
    frame_a: ndarray,
    frame_b: ndarray,
    x_old: ndarray,
    y_old: ndarray,
    u_old: ndarray,
    v_old: ndarray,
    window_size: int,
    overlap: int,
    correlation_method: str = "circular",
    deformation_method: str = "symmetric",
    deformation_algorithm: str = "taylor expansions",
    order: int = 1,
    radius: int = 2
)-> tuple( [[ndarray] * 5] ):
    """
    Multi pass of the PIV evaluation.
    This function does the PIV evaluation of the second and other passes.
    It returns the coordinates of the interrogation window centres,
    the displacement u, v for each interrogation window as well as
    the signal to noise ratio array.
    
    
    Parameters
    ----------
    frame_a : 2D ndarray
        The first image.

    frame_b : 2D ndarray
        The second image.

    window_size : tuple of ints
         The size of the interrogation window.

    overlap : tuple of ints
        The overlap of the interrogation window, e.g. window_size/2.

    x_old : 2D ndarray
        The x coordinates of the vector field of the previous pass.

    y_old : 2D ndarray
        The y coordinates of the vector field of the previous pass.

    u_old : 2D ndarray
        The u displacement of the vector field of the previous pass
        in case of the image mask - u_old and v_old are MaskedArrays.

    v_old : 2D ndarray
        The v displacement of the vector field of the previous pass.
        
    correlation_method : str
        Type of correlation to use where linear is zero padded to
        2N*2M (must remain power of 2 unless FFTW is used).
    
    deformation_algorithm : str
        Type of deformation to use.
        
    deformation_method : str
        Order/type of deformation to use.
        
    order : scalar
        The order of the Taylor expansions interpolation kernel.
        
    radius : scalar
        The radius of the Whittaker-Shannon interpolation kernel.   


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
    
    if deformation_method == "symmetric":
        deform_order = 2
    elif deformation_method == "second image":
        deform_order = 1
    else:
        raise ValueError(
            f"Deformation method {deformation_method} not supported"
        )
    
    frame_a, frame_b = deform_windows(
        frame_a.astype("float64"), # force the result to be float64
        frame_b.astype("float64"),
        x, y, 
        u_pre, v_pre,
        order = order,
        radius = radius,
        deformation_method = deformation_algorithm,
        deformation_order = deform_order
    )
        
    x, y, u, v, s2n = first_pass(
        frame_a,
        frame_b,
        window_size,
        overlap,
        correlation_method
    )

    u += u_pre
    v += v_pre
    
    return x, y, u, v, s2n