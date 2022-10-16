import numpy as np
import pytest

from os.path import join

from openpiv_cxx.tools import imread, imsave
from openpiv_cxx import filters
from openpiv_cxx.filters import _kernels

path_to_img = join(__file__, '..', '..', 'synthetic_tests', 'vel_magnitude', 'vel_0a.bmp')


def test_contrast_stretch():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.contrast_stretch(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.contrast_stretch(invalid_img)


def test_intensity_cap():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.intensity_cap(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.intensity_cap(invalid_img)

        
def test_threshold_binarization():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.threshold_binarization(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.threshold_binarization(invalid_img)


def test_gaussian_filter():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.gaussian_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.gaussian_filter(invalid_img)


def test_highpass_filter():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.highpass_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.highpass_filter(invalid_img)


def test_variance_normalization_filter():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.variance_normalization_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.variance_normalization_filter(invalid_img)


def test_sobel_filter():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.sobel_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.sobel_filter(invalid_img)


def test_sobel_h_filter():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.sobel_h_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.sobel_h_filter(invalid_img)


def test_sobel_v_filter():
    img = imread(path_to_img)
    
    # use default values
    new_img = filters.sobel_v_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # input is not 2D
        new_img = filters.sobel_v_filter(invalid_img)
