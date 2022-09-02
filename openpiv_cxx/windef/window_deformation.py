from numpy import arange, meshgrid, ndarray
from openpiv_cxx.interpolate import bilinear2D, whittaker2D, taylor_expansion2D


__all__ = [
    "create_deformation_field",
    "deform_windows"
]

def create_deformation_field(
    frame: ndarray, 
    x: ndarray,
    y: ndarray, 
    u: ndarray, 
    v: ndarray
) -> tuple( [[ndarray] * 4] ):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.
    
    
    Parameters
    ----------
    frame : 2d ndarray, dtype=np.int32
        An two dimensions array of integers containing grey levels of
        the first frame.
    
    x : 2D ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    
    y : 2D ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    
    u : 2D ndarray
        A two dimensional array containing the u velocity component,
        in pixels/seconds.
    
    v : 2D ndarray
        A two dimensional array containing the v velocity component,
        in pixels/seconds.
    
    
    Returns
    -------
        x,y : new grid (after meshgrid)
        u,v : deformation field
    """
    y1 = y[:, 0]  # extract first coloumn from meshgrid
    x1 = x[0, :]  # extract first row from meshgrid
    side_x = arange(frame.shape[1])  # extract the image grid
    side_y = arange(frame.shape[0])

    # interpolating displacements onto a new meshgrid
    ut = bilinear2D(y1, x1, u, side_y, side_x)
    vt = bilinear2D(y1, x1, v, side_y, side_x)

    x, y = meshgrid(side_x, side_y)

    return x, y, ut, vt


def deform_windows(
    frame_a: ndarray,
    frame_b: ndarray,
    x: ndarray, 
    y: ndarray, 
    u: ndarray, 
    v: ndarray, 
    deformation_method: str = "whittaker-shanon",
    order: int  = 1,
    radius: int = 1,
    deformation_order: int = 1
) -> tuple( [ndarray, ndarray] ):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid. Currently, two deformation algorithms are
    implemented, Whittaker-Shanon (sinc) and Taylor expansions with finite
    differences. Taylor expansions interpolation is usually much faster and 
    provides good results.
    
    
    Parameters
    ----------
    frame_a : 2D ndarray, dtype=np.int32
        A two dimensions array of integers containing grey levels of
        the first frame.
        
    frame_b : 2D ndarray, dtype=np.int32
        A two dimensions array of integers containing grey levels of
        the second frame.
    
    x : 2D ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    
    y : 2D ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    
    u : 2D ndarray
        A two dimensional array containing the u velocity component,
        in pixels/seconds.
    
    v : 2D ndarray
        A two dimensional array containing the v velocity component,
        in pixels/seconds.
        
    order : scalar
        The order of the Taylor expansions interpolation kernel.
        
    radius : scalar
        The radius of the Whittaker-Shannon interpolation kernel.
    
    deformation_method : str
        Type of deformation to use.
        
    deformation_order : scalar
        Order of deformation to use where '1' deforms the second image
        and '2' deforms both image symetrically.

    
    Returns
    -------
    frame_def_a, frame_def_b : 2D ndarray
        A deformed image based on the meshgrid and displacements of the
        previous pass

    """
    
    if deformation_order not in [1, 2]:
        raise ValueError(
            f"Deformation order {deformation_order} not supported.\n" +
             "Supported orders are 1 or 2"
        )
    
    x_new, y_new, ut, vt = create_deformation_field(
        frame_a,
        x, y,
        u, v
    )
    
    if deformation_method.lower() == "whittaker-shanon":
        deform_algo = whittaker2D
        
    elif deformation_method.lower() == "taylor expansions":
        deform_algo = taylor_expansion2D
        
    else:
        raise ValueError(
            f"Deformation method {deformation_method} not supported. \n\
             Supported algorithms are 'whittaker-shanon' and 'taylor expansions'"
        )
    
    
    if deformation_order == 1:
        frame_def_a = frame_a.copy()

        frame_def_b = deform_algo(
            frame_b, 
            y_new - vt, 
            x_new + ut, 
            order
        )
    
    else:
        frame_def_a = deform_algo(
            frame_a, 
            y_new - vt / 2, 
            x_new - ut / 2, 
            order
        )
        
        frame_def_b = deform_algo(
            frame_b, 
            y_new + vt / 2, 
            x_new + ut / 2, 
            order
        )

    return frame_def_a, frame_def_b