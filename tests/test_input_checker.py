import numpy as np
import pytest

from openpiv_cxx import input_checker


def test_check_nd_warning():
    arr = np.random.rand(32, 32)
    
    with pytest.raises(ValueError):
        input_checker.check_nd(ndim = 1, arr = arr)


def test_check_nd():
    arr = np.random.rand(32, 32)
    
    input_checker.check_nd(ndim = 2,  arr = arr)