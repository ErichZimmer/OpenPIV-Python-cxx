import _process as _proc
import numpy as np

"""This module contains a c++ implementation of the basic
cross-correlation algorithm for PIV image processing."""

def get_field_shape(image_size, window_size, overlap):
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.


    Parameters
    ----------
    image_size: two elements tuple
        A two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape.

    window_size : tuple
        The size of the (square) interrogation window.

    Overlap: tuple
        The number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    field_shape : two elements tuple
        The shape of the resulting flow field
    """
    field_shape = (np.array(image_size) - np.array(window_size)) // (
        np.array(window_size) - np.array(overlap)
    ) + 1
    
    return field_shape


def get_coordinates(image_size, window_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom


    Parameters
    ----------
    image_size: two elements tuple
        A two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    window_size : int
        The size of the (square) interrogation window.

    overlap: int 
        The number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(
        image_size,
        window_size,
        overlap
    )

    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = (
        np.arange(field_shape[1]) * (window_size - overlap)
        + (window_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[0]) * (window_size - overlap)
        + (window_size) / 2.0
    )

    return np.meshgrid(x, y)


def get_rect_coordinates(frame_a, window_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom


    Parameters
    ----------
    image_size: two elements tuple
        A two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    window_size : int
        The size of the (square) interrogation window, [default: 32 pix].

    overlap: int 
        The number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        A two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    """
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]
    _, y = get_coordinates(frame_a, window_size[0], overlap[0])
    x, _ = get_coordinates(frame_a, window_size[1], overlap[1])
    
    return np.meshgrid(x[0,:], y[:,0])


def fft_correlate_images(
    image_a,
    image_b,
    window_size = 32
    overlap = 16,
    thread_count = 1,
    thread_execution = "bulk-pool"
):
     """ Standard FFT based cross-correlation of two images. 


    Parameters
    ----------
    frame_a : 2d np.ndarray
        A two dimensionional array containing grey levels of the first frame.

    frame_b : 2d np.ndarray
        A two dimensionional array containing grey levels of the second frame.

    window_size : int
        The size of the (square) interrogation window, [default: 32 pix].

    overlap : int
        The number of pixels by which two adjacent windows overlap,
        [default: 16 pix].
    
    thread_count : int
        The number of threads to use with values < 1 automatically setting thread_count
        to the maximum of concurrent threads - 1, [default: 1].
        
    thread_execution : string {'pool', 'bulk-pool'}
        How the thread-pool is used, [default: 'pool'].
    
    
    Returns
    -------
    corr : 3d np.ndarray
        A three dimensional array with axis 0 being the two dimensional correlation matrix
        of an interrogation window.
        
    """
        
    if thread_execution not in ["pool", "bulk-pool"]:
        raise ValueError(f"Unsupported thread initializer methed. {thread_execution}")
    
    if thread_execution == "pool":
        thread_execution = 0
    else:
        thread_execution = 1
    
    return _proc.img2corr(
        image_a,
        image_b,
        int(window_size),
        int(overlap),
        int(thread_count),
        thread_execution
    )