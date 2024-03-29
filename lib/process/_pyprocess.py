from openpiv_cxx.input_checker import check_nd as _check
from . import _process_cpp as _proc

import numpy as np


__all__ = [
    "get_field_shape",
    "get_coordinates",
    "get_rect_coordinates",
    "fft_correlate_images",
    "correlation_to_displacement",
]


def get_field_shape(image_size, window_size, overlap):
    """Get vector field shape.

    Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.


    Parameters
    ----------
    image_size : tuple
        A two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape.
    window_size : tuple
        The size of the (square) interrogation window.
    overlap : tuple
        The number of pixel by which two adjacent interrogation
        windows overlap.

    Returns
    -------
    field_shape : tuple
        The shape of the resulting flow field.

    """
    field_shape = (np.array(image_size) - np.array(window_size)) // (
        np.array(window_size) - np.array(overlap)
    ) + 1

    return field_shape


def get_coordinates(image_size, window_size=32, overlap=16):
    """Get x/y coordinates of vector field.

    Compute the x, y coordinates of the centers of the interrogation windows.
    The origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom

    Parameters
    ----------
    image_size : tuple
        A two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    window_size : int
        The size of the (square) interrogation window.
    overlap : int
        The number of pixel by which two adjacent interrogation
        windows overlap.

    Returns
    -------
    x : ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size, window_size, overlap)

    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = np.arange(field_shape[1]) * (window_size - overlap) + (window_size) / 2.0
    # note the rows in field_shape[0]
    y = np.arange(field_shape[0]) * (window_size - overlap) + (window_size) / 2.0

    return np.meshgrid(x, y)


def get_rect_coordinates(
    image_size,
    window_size=32,
    overlap=16,
):
    """Same as get_coordinates, except for rectangualr windows.

    Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom


    Parameters
    ----------
    image_size : tuple
        A two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    window_size : int | tuple
        The size of the (square) interrogation window.
    overlap : int | tuple
        The number of pixel by which two adjacent interrogation
        windows overlap.

    Returns
    -------
    x : ndarray
        A two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    """
    if isinstance(window_size, (tuple, list)) == False:
        window_size = [window_size, window_size]

    if isinstance(overlap, (tuple, list)) == False:
        overlap = [overlap, overlap]

    _, y = get_coordinates(image_size, window_size[0], overlap[0])
    x, _ = get_coordinates(image_size, window_size[1], overlap[1])

    return np.meshgrid(x[0, :], y[:, 0])


def fft_correlate_images(
    image_a,
    image_b,
    window_size=32,
    overlap=16,
    correlation_method="circular",
    thread_count=1,
):
    """Standard FFT based cross-correlation of two images.

    Parameters
    ----------
    frame_a : ndarray
        A two dimensionional array containing grey levels of the first frame.
    frame_b : ndarray
        A two dimensionional array containing grey levels of the second frame.
    window_size : int
        The size of the (square) interrogation window, [default: 32 pix].
    overlap : int
        The number of pixels by which two adjacent windows overlap,
        [default: 16 pix].
    correlation_method : str
        Which correlation method to use where 'circular' is periodic
        (e.g. not padded) and 'linear' is padded to size 2*window_size.
    thread_count : int
        The number of threads to use with values < 1 automatically setting thread_count
        to the maximum of concurrent threads - 1, [default: 1].

    Returns
    -------
    corr : ndarray
        A three dimensional array with axis 0 being the two dimensional correlation matrix
        of an interrogation window.

    """
    _check(ndim=2, image_a=image_a, image_b=image_b)

    if correlation_method not in ["circular", "linear"]:
        raise ValueError(f"Unsupported correlation method: {correlation_method}.")

    if image_a.dtype != "float64":
        image_a = image_a.astype("float64")

    if image_b.dtype != "float64":
        image_b = image_b.astype("float64")

    if correlation_method == "circular":
        correlation_method = 0  # circular
    else:
        correlation_method = 1  # linear

    return _proc._img2corr_standard(
        image_a,
        image_b,
        int(window_size),
        int(overlap),
        correlation_method,
        int(thread_count),
    )


def correlation_to_displacement(
    corr,
    n_rows=None,
    n_cols=None,
    kernel="2x3",
    limit_peak_search=True,
    thread_count=1,
    return_type="first_peak",
):
    """Standard subpixel estimation.

    Parameters
    ----------
    corr : 3D ndarray
        A three dimensional array with axis 0 being the two dimensional correlation matrix
        of an interrogation window.
    n_rows, n_cols : int, optional
        Number of rows and columns of the vector field being evaluated, output of
        get_field_shape.
    kernel : str
        Type of kernel used to find the subpixel peak. Kernels with '2xN' are 2 1D subpixel
        estimations that are 'N' elements wide.
    limit_peak_search : bool
        Limit peak search area to a quarter of the size of the interrogation window if the
        width and height of the interrogation window is greater than 12.
    thread_count : int
        The number of threads to use with values < 1 automatically setting thread_count
        to the maximum of concurrent threads - 1.
    return_type : str
        Which peak data to return.

    Returns
    -------
    u, v : ndarray
        2D array of displacements in pixels/dt.
    peakHeight : ndarray
        2D array of correlation peak heights
    peak2peak : ndarray
        2D array of signal-to-noise ratios.
    u2, v2 : ndarray, optional
        2D array of displacements in pixels/dt for a second peak.
    u3, v3 : ndarray, optional
        2D array of displacements in pixels/dt for a third peak.

    """
    _check(ndim=3, corr=corr)

    if kernel not in ["2x3"]:
        raise ValueError(f"Unsupported peak search method method: {kernel}.")

    if return_type not in ["first_peak", "second_peak", "third_peak", "all_peaks"]:
        raise ValueError(
            f"Unsupported return type: {return_type}. \nSupported "
            + "peak types are 'first_peak', 'second_peak', 'third_peak', 'all_peaks'"
        )

    if kernel == "2x3":
        kernel = 0

    if limit_peak_search == True:
        limit_peak_search = 1
    else:
        limit_peak_search = 0

    if n_rows == None or n_cols == None:
        shape = (8, -1)
    else:
        shape = (8, n_rows, n_cols)

    if return_type == "first_peak":
        return_type = 1
    elif return_type == "second_peak":
        return_type = 2
    elif return_type == "third_peak":
        return_type = 3
    else:
        return_type = 0

    if limit_peak_search == True and corr.shape[1] >= 12 and corr.shape[1] >= 12:
        corr_slice = (  # use the one-quarter rule for limited peak search
            slice(0, corr.shape[0]),
            slice(
                corr.shape[1] // 2 - corr.shape[1] // 4,
                corr.shape[1] // 2 + corr.shape[1] // 4,
            ),
            slice(
                corr.shape[2] // 2 - corr.shape[2] // 4,
                corr.shape[2] // 2 + corr.shape[2] // 4,
            ),
        )
    else:
        corr_slice = (
            slice(0, corr.shape[0]),
            slice(0, corr.shape[1]),
            slice(0, corr.shape[2]),
        )

    u1, v1, peakHeight, peak2peak, u2, v2, u3, v3 = _proc._corr2vec(
        corr[corr_slice], kernel, limit_peak_search, int(thread_count), return_type
    ).reshape(shape)

    if return_type == 1:
        return u1, v1, peakHeight, peak2peak
    elif return_type == 2:
        return u2, v2, peakHeight, peak2peak
    elif return_type == 3:
        return u3, v3, peakHeight, peak2peak
    else:
        return (u1, v1, peakHeight, peak2peak, u2, v2, u3, v3)
