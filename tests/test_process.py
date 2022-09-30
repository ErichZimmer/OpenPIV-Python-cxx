import numpy as np
import pytest

from openpiv_cxx import process

try:
    from scipy.ndimage import shift
except ImportError as e:
    raise (e)


shift_u = -3.25
shift_v = 2.25


def create_img_pair(
    image_size: tuple([int, int]) = (32, 32), u: float = shift_u, v: float = shift_v
):
    frame_a = np.random.rand(image_size[0], image_size[1])
    frame_b = shift(frame_a, (v, u), mode="wrap")

    frame_a *= 255
    frame_b *= 255

    return frame_a.astype(int), frame_b.astype(int)


def test_fft_correlate_images_wrong_inputs() -> None:
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


def test_fft_correlate_images_01() -> None:
    frame_a, frame_b = create_img_pair()

    corr = process.fft_correlate_images(frame_a, frame_b)[0]

    max_idx = np.unravel_index(np.argmax(corr), corr.shape)

    assert max_idx[0] == corr.shape[0] // 2 + 2
    assert max_idx[1] == corr.shape[1] // 2 - 3


def test_fft_correlate_images_02() -> None:
    frame_a, frame_b = create_img_pair()

    corr = process.fft_correlate_images(frame_a, frame_b, correlation_method="linear")[
        0
    ]

    max_idx = np.unravel_index(np.argmax(corr), corr.shape)

    assert max_idx[0] == corr.shape[0] // 2 + 2
    assert max_idx[1] == corr.shape[1] // 2 - 3


def test_correlation_to_displacement_wrong_inputs() -> None:
    corr = np.random.rand(32, 32, 32)
    corr2 = np.random.rand(32, 32)

    with pytest.raises(ValueError):
        # corr is not 3D array
        out = process.correlation_to_displacement(corr2)

        # wrong estimation algo
        out = process.correlation_to_displacement(corr, kernel="wrong_kernel")

        # wrong return type
        out = process.correlation_to_displacement(corr, return_type="wrong_returnT")


def test_correlation_to_displacement() -> None:
    frame_a, frame_b = create_img_pair((256, 256))

    corr = process.fft_correlate_images(frame_a, frame_b, correlation_method="linear")

    u, v, _, _ = process.correlation_to_displacement(corr)

    assert np.mean(np.abs(u - shift_u)) < 0.2
    assert np.mean(np.abs(v - shift_v)) < 0.2
