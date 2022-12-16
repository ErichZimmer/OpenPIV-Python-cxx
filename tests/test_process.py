import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal,
                           assert_)
from os.path import join
from openpiv_cxx.tools import imread
from openpiv_cxx import process


Frame_a = imread(join(__file__, '..', '..', 'synthetic_tests', 'vel_magnitude', 'vel_48a.bmp'))
Frame_b = imread(join(__file__, '..', '..', 'synthetic_tests', 'vel_magnitude', 'vel_48b.bmp'))

shift_u = 0.0
shift_v = 1.5

def test_fft_correlate_images_wrong_inputs():
    frame_a = np.random.rand(32, 32)
    frame_b = np.random.rand(32, 32)
    frame_b2 = np.random.rand(32)

    with pytest.raises(ValueError):
        # image miss-fit
        out = process.fft_correlate_images(frame_a, frame_b2)

        # wrong correlation method
        out = process.fft_correlate_images(
            frame_a, frame_b, correlation_method="wrong_method"
        )

    with pytest.raises(RuntimeError):  # error raised by wrapper
        # image shape is different
        frame_b3 = np.random.rand(32, 64)
        out = process.fft_correlate_images(frame_a, frame_b3, window_size=64)

        # window size larger than image
        out = process.fft_correlate_images(frame_a, frame_b, window_size=64)

        # overlap larger than window size
        out = process.fft_correlate_images(frame_a, frame_b, window_size=32, overlap=64)

        # overlap = 0
        out = process.fft_correlate_images(frame_a, frame_b, overlap=0)

        # non power of 2 window size
        out = process.fft_correlate_images(frame_a, frame_b, window_size=33)

        # window size = 0
        out = process.fft_correlate_images(frame_a, frame_b, window_size=0)


def test_fft_correlate_images_01():
    frame_a, frame_b = Frame_a.copy(), Frame_b.copy()

    corr = process.fft_correlate_images(frame_a, frame_b)[0]

    max_idx = np.unravel_index(np.argmax(corr), corr.shape)

    assert_equal(max_idx[0], corr.shape[0] // 2 + 1)
    assert_equal(max_idx[1], corr.shape[1] // 2)


def test_fft_correlate_images_02() -> None:
    frame_a, frame_b = Frame_a.copy(), Frame_b.copy()

    corr = process.fft_correlate_images(frame_a, frame_b, correlation_method="linear")[
        0
    ]

    max_idx = np.unravel_index(np.argmax(corr), corr.shape)

    assert_equal(max_idx[0], corr.shape[0] // 2 + 1)
    assert_equal(max_idx[1], corr.shape[1] // 2)


def test_correlation_to_displacement_wrong_inputs():
    corr = np.random.rand(32, 32, 32)
    corr2 = np.random.rand(32, 32)

    with pytest.raises(ValueError):
        # corr is not 3D array
        out = process.correlation_to_displacement(corr2)

        # wrong estimation algo
        out = process.correlation_to_displacement(
            corr,
            kernel="wrong kernel"
        )

        # wrong return type
        out = process.correlation_to_displacement(
            corr,
            return_type="wrong return"
        )


def test_correlation_to_displacement_01():
    width = height = 32
    corr = np.random.rand(width, height)
    
    peak = np.array(
        [[0.04491923, 0.12210312, 0.04491923],
         [0.12210312, 0.3319107 , 0.12210312],
         [0.04491923, 0.12210312, 0.04491923]],
        dtype = np.float64
    ) * 6
    
    peakVal = peak[1,1]
    
    peak1 = (16, 16) # middle
    peak2 = (1,  1)  # top left
    peak3 = (29, 29) # bottom right
    
    corr[peak1[0] - 1 : peak1[0] + 2,
         peak1[1] - 1 : peak1[1] + 2] = peak * 3
    
    corr[peak2[0] - 1 : peak2[0] + 2,
         peak2[1] - 1 : peak2[1] + 2] = peak * 2
    
    corr[peak3[0] - 1 : peak3[0] + 2,
         peak3[1] - 1 : peak3[1] + 2] = peak * 1
    
    u1, v1, ph, p2p, u2, v2, u3, v3 = process.correlation_to_displacement(
        corr[np.newaxis, :, :],
        limit_peak_search=False,
        return_type="all peaks"
    )
    
    # first peak
    assert_equal(
        u1[0], 
        peak1[0] - width // 2
    )
    assert_equal(
        v1[0], 
        peak1[1] - height // 2
    )
    
    # second peak
    assert_equal(
        u2[0], 
        peak2[0] - width // 2,
    )
    assert_equal(
        u2[0],
        peak2[1] - height // 2,
    )
    
    # third peak
    assert_equal(
        u3[0], 
        peak3[0] - width // 2,
    )
    assert_equal(
        u3[0], 
        peak3[1] - height // 2,
    )
    
    # s2n 
    assert_equal(
        p2p[0],
        3 / 2,

    )
    
    # peak height
    assert_equal(
        ph[0], 
        peakVal * 3,
    )
    
        
def test_correlation_to_displacement_02():
    frame_a, frame_b = Frame_a.copy(), Frame_b.copy()

    corr = process.fft_correlate_images(
        frame_a, 
        frame_b,
        correlation_method="linear"
    )

    u, v, _, _ = process.correlation_to_displacement(corr)

    assert_(np.nanmean(np.abs(u - shift_u)) < 0.05)
    assert_(np.nanmean(np.abs(v - shift_v)) < 0.05)
