from numpy import arange, meshgrid
from openpiv_cxx.interpolate import bilinear2D, whittaker2D, taylor_expansion2D
from openpiv_cxx.input_checker import check_nd as _check


__all__ = ["create_deformation_field", "deform_windows"]


def create_deformation_field(frame, x, y, u, v):
    """Create a deformation field

    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.

    Parameters
    ----------
    frame : ndarray
        A two dimensional array of integers containing grey levels of
        an image.
    x : ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    u : ndarray
        A two dimensional array containing the u velocity component,
        in pixels/seconds.
    v : ndarray
        A two dimensional array containing the v velocity component,
        in pixels/seconds.

    Returns
    -------
        x, y : ndarray
            A new grid with same dimensions as frame
        u, v : ndarray
            Deformation field u/v components


    """
    _check(ndim=2, frame=frame, x=x, y=y, u=u, v=v)

    y1 = y[:, 0]  # extract first coloumn from meshgrid
    x1 = x[0, :]  # extract first row from meshgrid
    side_x = arange(frame.shape[1])  # extract the image grid
    side_y = arange(frame.shape[0])

    # interpolating displacements onto a new meshgrid
    ut = bilinear2D(x1, y1, u, side_x, side_y)
    vt = bilinear2D(x1, y1, v, side_x, side_y)

    x, y = meshgrid(side_x, side_y)

    return x, y, ut, vt


def deform_windows(
    frame_a,
    frame_b,
    x,
    y,
    u,
    v,
    deformation_method="whittaker-shanon",
    order=1,
    radius=1,
    deformation_order=1,
):
    """Deform images by interpolation

    Deform an image pair by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid. Currently, two deformation algorithms are
    implemented, Whittaker-Shanon (sinc) and Taylor expansions with finite
    differences. Taylor expansions interpolation is usually much faster and
    provides good results.

    Parameters
    ----------
    frame_a : ndarray
        A two dimensional array of integers containing grey levels of
        the first frame.
    frame_b : ndarray
        A two dimensional array of integers containing grey levels of
        the second frame.
    x : ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    u : ndarray
        A two dimensional array containing the u velocity component,
        in pixels/seconds.
    v : ndarray
        A two dimensional array containing the v velocity component,
        in pixels/seconds.
    order : scalar
        The order of the Taylor expansions interpolation kernel.
    radius : scalar
        The radius of the Whittaker-Shannon interpolation kernel.
        Optimal radii are 3 and 5.
    deformation_method : str
        Type of deformation to use.
    deformation_order : scalar
        Order of deformation to use where '1' deforms the second image
        and '2' deforms both images symetrically.

    Returns
    -------
    frame_def_a, frame_def_b : ndarray
        Deformed images based on the meshgrid and displacements of the
        previous pass

    """
    _check(ndim=2, frame_a=frame_a, frame_b=frame_b, x=x, y=y, u=u, v=v)

    if deformation_order not in [1, 2]:
        raise ValueError(
            f"Deformation order {deformation_order} not supported.\n"
            + "Supported orders are 1 or 2"
        )

    x_new, y_new, ut, vt = create_deformation_field(frame_a, x, y, u, v)

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

        frame_def_b = deform_algo(frame_b, y_new + vt, x_new + ut, order)

    else:
        frame_def_a = deform_algo(frame_a, y_new - vt / 2, x_new - ut / 2, order)

        frame_def_b = deform_algo(frame_b, y_new + vt / 2, x_new + ut / 2, order)

    return frame_def_a, frame_def_b
