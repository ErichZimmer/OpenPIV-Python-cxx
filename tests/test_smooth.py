import numpy as np
import pytest

from os.path import join
from openpiv_cxx import smooth

# read test data
data = np.loadtxt('exp1_001.vec', delimiter="\t")

# extract individual components
u, v, s2n = data[:,2], data[:,3], data[:,4]
u = u.reshape(29,41)
v = v.reshape(29,41)
s2n = s2n.reshape(29,41)



def test_smoothn_no_mask():
    u_smth = smooth.smooth_spline(u, s=1)
    v_smth = smooth.smooth_spline(v, s=1)


def test_smoothn_blank_mask():
    u_masked = np.ma.MaskedArray(u, mask=np.ma.nomask)
    v_masked = np.ma.MaskedArray(v, mask=np.ma.nomask)
    
    u_smth = smooth.smooth_spline(u_masked, s=1)
    v_smth = smooth.smooth_spline(v_masked, s=1)