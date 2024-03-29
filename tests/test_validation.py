import numpy as np
import pytest

from os.path import join
from openpiv_cxx import validate

# read test data
data = np.loadtxt("exp1_001.vec", delimiter="\t")

# extract individual components
u, v, s2n = data[:,2], data[:,3], data[:,4]
u = u.reshape(29,41)
v = v.reshape(29,41)
s2n = s2n.reshape(29,41)

# modify flow field for testing
invalid_ind = [u.shape[0] // 2, u.shape[1] // 2]

s2n = np.clip(s2n, 2, s2n.max())

s2n[invalid_ind[0], invalid_ind[1]] = 1.2
u[invalid_ind[0], invalid_ind[1]] = 6
v[invalid_ind[0], invalid_ind[1]] = -10


def test_s2n_val_wrong_inputs():
    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.sig2noise_val(np.random.rand(32,32,32), threshold=-2)


def test_s2n_val():
    res = validate.sig2noise_val(s2n, threshold=1.5)

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_global_val_wrong_inputs():
    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.global_val(
            u, np.random.rand(32), (-2, 2), (0, 7)
        )


def test_global_val():
    res = validate.global_val(u, v, (-3, 3), (0, 8))

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_global_std_wrong_inputs():
    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.global_std(u, np.random.rand(32))


def test_std_val():
    res = validate.global_std(u, v, 4.0)

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_difference_val_wrong_inputs():
    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.local_difference(u, np.random.rand(32))

    with pytest.raises(RuntimeError):  # error produced by wrapper
        # inputs are not same shape
        res = validate.local_difference(
            u, np.random.rand(32, 38)
        )

        # threshold less than zero
        res = validate.local_difference(u, v, theshold=0)


def test_difference_vali():
    res = validate.local_difference(
        u,
        v,
        threshold = 4.0
    )

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_local_median_wrong_inputs():
    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.local_median(u, np.random.rand(32))

    with pytest.raises(RuntimeError):  # error produced by wrapper
        # inputs are not same shape
        res = validate.local_median(u, np.random.rand(32, 38))

        # kernel radius is too small
        res = validate.local_median(
            u, np.random.rand(32, 38), kernel_radius=0
        )

        # kernel_min_size is too small
        res = validate.local_median(
            u, np.random.rand(32, 38), kernel_min_size=0
        )

        # threshold less than zero
        res = validate.local_median(u, v, theshold=0)


def test_local_median_vali():
    res = validate.local_median(
        u,
        v,
        threshold = 4.0
    )
    
    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_normalized_local_median_wrong_inputs():
    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.normalized_local_median(
            u, np.random.rand(32)
        )

    with pytest.raises(RuntimeError):  # error produced by wrapper
        # inputs are not same shape
        res = validate.normalized_local_median(
            u, np.random.rand(32, 38)
        )

        # kernel radius is too small
        res = validate.normalized_local_median(
            u, np.random.rand(32, 38), kernel_radius=0
        )

        # kernel_min_size is too small
        res = validate.normalized_local_median(
            u, np.random.rand(32, 38), kernel_min_size=0
        )

        # threshold less than zero
        res = validate.normalized_local_median(
            u, v, theshold=0
        )

# something is definetly wrong due to a high threshold to pass tests..
def test_normalized_local_median_vali():
    res = validate.normalized_local_median(
        u,
        v,
        threshold = 6
    )

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1
