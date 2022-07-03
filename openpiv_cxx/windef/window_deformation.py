from numpy import arange, meshgrid
from openpiv_cxx.interpolate import bilinear2D, whittaker2D


__all__ = [
    "create_deformation_field",
    "deform_windows"
]

def create_deformation_field(
    frame, 
    x, y, 
    u, v
):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.
    
    
    Parameters
    ----------
    frame : 2d np.ndarray, dtype=np.int32
        An two dimensions array of integers containing grey levels of
        the first frame.
    
    x : 2D np.ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    
    y : 2D np.ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    
    u : 2D np.ndarray
        A two dimensional array containing the u velocity component,
        in pixels/seconds.
    
    v : 2D np.ndarray
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
    frame_a,
    frame_b,
    x, 
    y, 
    u, 
    v, 
    radius=1,
    deformation_order = 1
):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.
    
    
    Parameters
    ----------
    frame_a : 2D np.ndarray, dtype=np.int32
        A two dimensions array of integers containing grey levels of
        the first frame.
        
    frame_b : 2D np.ndarray, dtype=np.int32
        A two dimensions array of integers containing grey levels of
        the second frame.
    
    x : 2D np.ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    
    y : 2D np.ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    
    u : 2D np.ndarray
        A two dimensional array containing the u velocity component,
        in pixels/seconds.
    
    v : 2D np.ndarray
        A two dimensional array containing the v velocity component,
        in pixels/seconds.
    
    radius : scalar
        The radius of the Whittaker-Shannon interpolation kernel.
        
    deformation_order : scalar
        Type of deformation to use where '1' deforms the second image
        and '2' deforms both image symetrically.

    
    Returns
    -------
    frame_def_a, frame_def_b : 2D np.ndarray
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
    
    if deformation_order == 1:
        frame_def_a = frame_a

        frame_def_b = whittaker2D(
            frame_b, 
            y_new - vt, 
            x_new + ut, 
            radius
        )
    
    else:
        frame_def_a = whittaker2D(
            frame_a, 
            y_new - vt / 2, 
            x_new - ut / 2, 
            radius
        )
        
        frame_def_b = whittaker2D(
            frame_b, 
            y_new + vt / 2, 
            x_new + ut / 2, 
            radius
        )

    return frame_def_a, frame_def_b