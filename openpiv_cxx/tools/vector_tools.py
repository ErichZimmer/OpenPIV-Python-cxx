import numpy as np

__all__ = [
    "save",
    "transform_coordinates",
    "uniform_scaling"
]


def save(x, y, u, v, flag, filename, fmt="%8.4f", delimiter="\t"):
    """Save flow field to an ascii file.

    Parameters
    ----------
    x : 2D np.ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2D np.ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    u : 2D np.ndarray
        A two dimensional array containing the u velocity components,
        in pixels/seconds.

    v : 2D np.ndarray
        A two dimensional array containing the v velocity components,
        in pixels/seconds.

    flag : 2D np.ndarray
        A two dimensional boolean array where elements with the convention
        0 = valid, 1 = invalid, 2 = interpolated.

    filename : string
        The path of the file where to save the flow field.

    fmt : string
        A format string. See documentation of numpy.savetxt
        for more details.

    delimiter : string
        Character separating columns.

    Examples
    --------

    openpiv_cxx.tools.save( x, y, u, v, 'field_001.txt', fmt='%6.3f',
                        delimiter='\t')

    """
    if isinstance(u, np.ma.MaskedArray):
        u = u.filled(0.)
        v = v.filled(0.)

    # build output array
    out = np.vstack([m.flatten() for m in [x, y, u, v, flag]])

    # save data to file.
    np.savetxt(
        filename,
        out.T,
        fmt=fmt,
        delimiter=delimiter,
        header="x"
        + delimiter
        + "y"
        + delimiter
        + "u"
        + delimiter
        + "v"
        + delimiter
        + "flag",
    )

    
def transform_coordinates(x, y, u, v):
    """ Converts coordinate systems from/to the image based / physical based 
    
    Input/Output: x,y,u,v

        image based is 0,0 top left, x = columns to the right, y = rows downwards
        and so u,v 

        physical or right hand one is that leads to the positive vorticity with 
        the 0,0 origin at bottom left to be counterclockwise
    
    """
    y = y[::-1, :]
    v *= -1
    return x, y, u, v


def uniform_scaling(x, y, u, v, scaling_factor):
    """
    Apply an uniform scaling.

    Parameters
    ----------
    x : 2D np.ndarray
    y : 2D np.ndarray
    u : 2D np.ndarray
    v : 2D np.ndarray
    scaling_factor : float
        The image scaling factor in pixels per meter.

    Return
    ----------
    x : 2D np.ndarray
    y : 2D np.ndarray
    u : 2D np.ndarray
    v : 2D np.ndarray
    """
    return (
        x / scaling_factor,
        y / scaling_factor,
        u / scaling_factor,
        v / scaling_factor,
    )