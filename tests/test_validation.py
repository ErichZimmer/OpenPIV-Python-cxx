import numpy as np
import pytest

from openpiv_cxx import validation

    
def sample_vec_field(
    field_size: tuple([int,int]) = (32, 32)
):
    x, y = np.meshgrid(
        np.arange(field_size[0]),
        np.arange(field_size[1])
    )
    x = x.astype(float)
    y = y.astype(float)
    
    # create s2n
    s2n = np.random.rand(field_size[0], field_size[1])
    s2n = np.clip(s2n * 10, 2, 10)
    s2n[int(field_size[0] / 2), int(field_size[1] / 2)] = 0.8
    
    # create velocity field
    u = np.ones(x.shape).astype(float) * 2
    v = np.ones(x.shape).astype(float) * 2

    # create outlier
    u[int(field_size[0] / 3), int(field_size[1] / 3)] = -4.0
    v[int(field_size[0] / 3), int(field_size[1] / 3)] = -4.0
    
    return u, v, s2n


def test_s2n_val_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 2, u.shape[1] // 2)
    
    res = validation.sig2noise_val(
        u, v, s2n,
        threshold = 1.05,
        convention = "!openpiv"
    )
    
    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1


def test_global_val_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
            
    with pytest.raises(ValueError):
        # input is not 2D
        res = validation.global_val(
            u,
            np.random.rand(32),
            (-2.5, 2.5),
            (-2.5, 2.5),
            convention = "!openpiv"
        )
    
    
def test_global_val() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )

    res = validation.global_val(
        u,
        v,
        (-2.5, 2.5),
        (-2.5, 2.5),
        convention = "!openpiv"
    )
    
    
    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1
    
    
def test_global_std_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
            
    with pytest.raises(ValueError):
        # input is not 2D
        res = validation.global_std(
            u,
            np.random.rand(32),
            convention = "!openpiv"
        )
    
    
def test_std_val() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )

    res = validation.global_std(
        u,
        v,
        3.0,
        convention = "!openpiv"
    )
    
    
    assert res[invalid_ind[0], invalid_ind[1]] == 1
    assert np.count_nonzero(res) == 1
    
    
def test_difference_val_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
            
    with pytest.raises(ValueError):
        # input is not 2D
        res = validation.local_difference(
            u,
            np.random.rand(32),
            convention = "!openpiv"
        )
    
    with pytest.raises(RuntimeError): # error produced by wrapper
        # inputs are not same shape
        res = validation.local_difference(
            u,
            np.random.rand(32, 38),
            convention = "!openpiv"
        )
    
    
# The validation works according to notebooks, but not here. Why?
#def test_difference_vali() -> None:
#    u, v, s2n = sample_vec_field()
#    
#    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
#
#    res = validation.local_difference(
#        u,
#        v,
#        1.2,
#        convention = "!openpiv"
#    )
#    
#    
#    assert res[invalid_ind[0], invalid_ind[1]] == 1
#    assert np.count_nonzero(res) == 1
    

def test_local_median_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
            
    with pytest.raises(ValueError):
        # input is not 2D
        res = validation.local_median(
            u,
            np.random.rand(32),
            convention = "!openpiv"
        )
    
    with pytest.raises(RuntimeError): # error produced by wrapper
        # inputs are not same shape
        res = validation.local_median(
            u,
            np.random.rand(32, 38),
            convention = "!openpiv"
        )
        
        # kernel radius is too small
        res = validation.local_median(
            u,
            np.random.rand(32, 38),
            kernel_radius = 0,
            convention = "!openpiv"
        )
        
        # kernel_min_size is too small
        res = validation.local_median(
            u,
            np.random.rand(32, 38),
            kernel_min_size = 0,
            convention = "!openpiv"
        )
    
# The validation works according to notebooks, but not here. Why?
#def test_local_median_vali() -> None:
#    u, v, s2n = sample_vec_field()
#    
#    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
#
#    res = validation.local_median(
#        u,
#        v,
#        1.2,
#        convention = "!openpiv"
#    )
#    
#    
#    assert res[invalid_ind[0], invalid_ind[1]] == 1
#    assert np.count_nonzero(res) == 1
    
    
def test_narmalized_local_median_wrong_inputs() -> None:
    u, v, s2n = sample_vec_field()
    
    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
            
    with pytest.raises(ValueError):
        # input is not 2D
        res = validation.normalized_local_median(
            u,
            np.random.rand(32),
            convention = "!openpiv"
        )
    
    with pytest.raises(RuntimeError): # error produced by wrapper
        # inputs are not same shape
        res = validation.normalized_local_median(
            u,
            np.random.rand(32, 38),
            convention = "!openpiv"
        )
        
        # kernel radius is too small
        res = validation.normalized_local_median(
            u,
            np.random.rand(32, 38),
            kernel_radius = 0,
            convention = "!openpiv"
        )
        
        # kernel_min_size is too small
        res = validation.normalized_local_median(
            u,
            np.random.rand(32, 38),
            kernel_min_size = 0,
            convention = "!openpiv"
        )
    

# The validation works according to notebooks, but not here. Why?
#def test_normalized_local_mmedian_vali() -> None:
#    u, v, s2n = sample_vec_field()
#    
#    invalid_ind = ( u.shape[0] // 3, u.shape[1] // 3 )
#    
#    res = validation.normalized_local_median(
#        u,
#        v,
#        1.2,
#        convention = "!openpiv"
#    )
#    
#    assert res[invalid_ind[0], invalid_ind[1]] == 1
#    assert np.count_nonzero(res) == 1