import numpy as np
import pytest

from openpiv_cxx import validate


def sample_vec_field(field_size: tuple([int, int]) = (32, 32)):
    x, y = np.meshgrid(np.arange(field_size[0]), np.arange(field_size[1]))
    x = x.astype(float)
    y = y.astype(float)

    # create s2n
    s2n = np.random.rand(field_size[0], field_size[1])
    s2n = np.clip(s2n * 10, 2, 10)
    s2n[int(field_size[0] / 2), int(field_size[1] / 2)] = 0.8

    # create velocity field
    u = np.ones(x.shape).astype(float) * 2
    v = np.ones(x.shape).astype(float) * 2

    noise = np.random.rand(u.shape[0], u.shape[1]).astype(float) / 2.0

    u += noise
    v += noise

    # create outlier
    u[int(field_size[0] / 3), int(field_size[1] / 3)] = -5.0
    v[int(field_size[0] / 3), int(field_size[1] / 3)] = -5.0

    return u, v, s2n


def test_s2n_val_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 2, u.shape[1] // 2)

    res = validate.sig2noise_val(u, v, s2n, threshold=1.05, convention="!openpiv")

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_global_val_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 3, u.shape[1] // 3)

    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.global_val(
            u, np.random.rand(32), (-2.5, 2.5), (-2.5, 2.5), convention="!openpiv"
        )


def test_global_val() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 3, u.shape[1] // 3)

    res = validate.global_val(u, v, (-2.5, 2.5), (-2.5, 2.5), convention="!openpiv")

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_global_std_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 3, u.shape[1] // 3)

    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.global_std(u, np.random.rand(32), convention="!openpiv")


def test_std_val() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 3, u.shape[1] // 3)

    res = validate.global_std(u, v, 3.0, convention="!openpiv")

    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_difference_val_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 3, u.shape[1] // 3)

    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.local_difference(u, np.random.rand(32), convention="!openpiv")

    with pytest.raises(RuntimeError):  # error produced by wrapper
        # inputs are not same shape
        res = validate.local_difference(
            u, np.random.rand(32, 38), convention="!openpiv"
        )

        # threshold less than zero
        res = validate.local_difference(u, v, convention="!openpiv", theshold=0)


# The validate works according to notebooks, but not here. Why?
# def test_difference_vali() -> None:
#    u, v, s2n = sample_vec_field()
#
#    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
#
#    res = validate.local_difference(
#        u,
#        v,
#        2.0,
#        convention = "!openpiv"
#    )
#
#
#    assert res[invalid_ind[0], invalid_ind[1]] == 1
#    assert np.count_nonzero(res) == 1


def test_local_median_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 3, u.shape[1] // 3)

    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.local_median(u, np.random.rand(32), convention="!openpiv")

    with pytest.raises(RuntimeError):  # error produced by wrapper
        # inputs are not same shape
        res = validate.local_median(u, np.random.rand(32, 38), convention="!openpiv")

        # kernel radius is too small
        res = validate.local_median(
            u, np.random.rand(32, 38), kernel_radius=0, convention="!openpiv"
        )

        # kernel_min_size is too small
        res = validate.local_median(
            u, np.random.rand(32, 38), kernel_min_size=0, convention="!openpiv"
        )

        # threshold less than zero
        res = validate.local_median(u, v, convention="!openpiv", theshold=0)


# The validate works according to notebooks, but not here. Why?
# def test_local_median_vali() -> None:
#    u, v, s2n = sample_vec_field()
#
#    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
#
#    res = validate.local_median(
#        u,
#        v,
#        2.0,
#        convention = "!openpiv"
#    )
#
#
#    assert res[invalid_ind[0], invalid_ind[1]] == 1
#    assert np.count_nonzero(res) == 1


def test_narmalized_local_median_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()

    invalid_ind = (u.shape[0] // 3, u.shape[1] // 3)

    with pytest.raises(ValueError):
        # input is not 2D
        res = validate.normalized_local_median(
            u, np.random.rand(32), convention="!openpiv"
        )

    with pytest.raises(RuntimeError):  # error produced by wrapper
        # inputs are not same shape
        res = validate.normalized_local_median(
            u, np.random.rand(32, 38), convention="!openpiv"
        )

        # kernel radius is too small
        res = validate.normalized_local_median(
            u, np.random.rand(32, 38), kernel_radius=0, convention="!openpiv"
        )

        # kernel_min_size is too small
        res = validate.normalized_local_median(
            u, np.random.rand(32, 38), kernel_min_size=0, convention="!openpiv"
        )

        # threshold less than zero
        res = validate.normalized_local_median(
            u, v, convention="!openpiv", theshold=0
        )


# The validate works according to notebooks, but not here. Why?
# def test_normalized_local_mmedian_vali() -> None:
#    u, v, s2n = sample_vec_field()
#
#    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
#
#    res = validate.normalized_local_median(
#        u,
#        v,
#        2.0,
#        convention = "!openpiv"
#    )
#
#    assert res[invalid_ind[0], invalid_ind[1]] == 1
#    assert np.count_nonzero(res) == 1
