from numpy.testing import assert_array_almost_equal

import numpy as np
import pytest

from openpiv_cxx import interpolate


def test_bilinear2D_wrong_Z_dim():
    Z = np.random.rand(256)  # should be 2D
    X = np.random.rand(32)
    Y = np.random.rand(32)

    xq = np.arange(256)
    yq = np.arange(256)

    with pytest.raises(ValueError):
        out = interpolate.bilinear2D(X, Y, Z, xq, yq)


def test_bilinear2D_wrong_XY_dim():
    Z = np.random.rand(256, 256)
    X = np.random.rand(32)
    Y = np.random.rand(32, 32)  # should be 1D

    xq = np.arange(256)
    yq = np.arange(256, 256)  # should be 1D

    with pytest.raises(ValueError):
        out = interpolate.bilinear2D(X, Y, Z, xq, yq)


def test_bilinear2D_not_ascending():
    Z = np.random.rand(64, 64)
    X = np.linspace(0, 256, 64)
    Y = np.linspace(0, 256, 64)

    X[32] = X[35]  # no longer increasing

    xq = np.arange(256)
    yq = np.arange(256)

    with pytest.raises(ValueError):
        out = interpolate.bilinear2D(X, Y, Z, xq, yq)


def test_bilinear2D_Z_X_mismatch():
    Z = np.random.rand(64, 64)
    X = np.linspace(0, 256, 70)
    Y = np.linspace(0, 256, 64)

    xq = np.arange(256)
    yq = np.arange(256)

    with pytest.raises(RuntimeError):  # error generated by wrapper
        out = interpolate.bilinear2D(X, Y, Z, xq, yq)


# def test_bilinear2D():
#    Z = np.zeros([5, 5], dtype = "float64")
#    X, Y = np.linspace(5)
#
#    xq = np.array([2.0, 2.0], dtype = "float64")
#    yq = np.array([1.5, 2.5], dtype = "float64")
#
#    out = interpolate.bilinear2D(
#        X, Y,
#        Z,
#        xq, yq
#    )
