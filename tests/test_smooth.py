import numpy as np
import pytest

from os.path import join
from openpiv_cxx import smooth


# read test data
data = np.loadtxt('exp1_001.vec', delimiter="\t")

# extract individual components
u = data[:,2]
u = u.reshape(29,41)



def test_smooth_spline_no_mask():
    u_smth = smooth.smooth_spline(u, s=1)


def test_smooth_spline_blank_mask():
    u_masked = np.ma.MaskedArray(u, mask=np.ma.nomask)
    
    u_smth = smooth.smooth_spline(u_masked, s=1)
