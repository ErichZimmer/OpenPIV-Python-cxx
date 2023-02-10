import numpy as np
import pytest

from os.path import join
from openpiv_cxx import smooth


# read test data
data = np.loadtxt('exp1_001.vec', delimiter="\t")

# extract individual components
u = data[:,2]
u = u.reshape(29,41)


def test_smooth_gaussian_wrong_inputs():
    dummy = np.random.rand(32,32,32)
    
    with pytest.raises(ValueError):
        # Not 2D
        u_smth = smooth.smooth_gaussian(
            dummy
        )
        
        # test for NaNs
        u2 = u.copy()
        u2[-3, -3] = np.nan
        
        u_smth = smooth.smooth_gaussian(
            u2
        )


def test_smooth_gaussian_01():
    u_smth = smooth.smooth_gaussian(
        u
    )


def test_smooth_gaussian_02():
    u_smth = smooth.smooth_gaussian(
        u,
        sigma=1.0,
        truncate=2.0
    )
    
    
def test_smooth_gaussian_03():
    u_smth = smooth.smooth_gaussian(
        u,
        half_width=1
    )
