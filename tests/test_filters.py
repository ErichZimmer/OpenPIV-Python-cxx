import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           assert_)
from os.path import join

from openpiv_cxx.tools import imread, imsave
from openpiv_cxx import filters


Float = np.float32

path_to_img = join(__file__, '..', '..', 'synthetic_tests', 'vel_magnitude', 'vel_0a.bmp')


def normalize_image(img):
    img = Float(img)
    return img / img.max()


def test_gaussian_kernel_01():
    kernel1D = filters.gaussian_kernel(ndim=1)
    kernel2D = filters.gaussian_kernel(ndim=2)
    
    kernel1D_to_2D = np.outer(
        kernel1D,
        kernel1D
    )
    
    assert_(kernel1D.ndim, 1)
    assert_(kernel2D.ndim, 2)
    assert_array_almost_equal(kernel2D, kernel1D_to_2D)


def test_gaussian_kernel_02():
    kernel1D_1 = filters.gaussian_kernel(
        sigma=0.5, 
        truncate=3,
        ndim=1
    )
    
    kernel1D_2 = filters.gaussian_kernel(
        half_width=2, 
        ndim=1
    )
    
    assert_equal(kernel1D_1.size, 5)
    assert_equal(kernel1D_2.size, 5)

    
def test_gaussian_kernel_03():
    kernel1D = filters.gaussian_kernel(
        half_width=1,
        ndim=1
    )
    
    kernel2D = filters.gaussian_kernel(
        half_width=1,
        ndim=2
    )
    
    expected1D = np.array(
        [0.21194157, 0.5761169 , 0.21194157],
        dtype=Float
    )
    
    expected2D = np.array(
        [[0.04491923, 0.12210312, 0.04491923],
         [0.12210312, 0.3319107 , 0.12210312],
         [0.04491923, 0.12210312, 0.04491923]],
        dtype=Float
    )
    
    assert_allclose(
        kernel1D,
        expected1D
    )
    
    assert_allclose(
        kernel2D,
        expected2D
    )


def test_convolve2D_sep_invalid_input():
    kernel1D = np.ones(3, dtype=Float)
    kernel2D = np.ones((3,3), dtype=Float)
    
    # test for errors
    with pytest.raises(ValueError):
        img = np.random.rand(32,32)
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # image is not 2D
        new_img = filters.convolve_2D_sep(
            invalid_img,
            kernel1D,
            kernel1D
        )
        
        # kernels are not same size
        new_img = filters.convolve_2D_sep(
            img,
            kernel1D,
            np.ones(5, dtype=Float)
        )
        
        # kernels are not 1D
        new_img = filters.convolve_2D_sep(
            img,
            kernel2D,
            kernel2D
        )

        
def test_convolve2D_sep_01():
    arr = np.array(([1,1], [1,1]), dtype=Float)
    kernel1D = np.array([2], dtype=Float)
    
    output = filters.convolve_2D_sep(
        arr,
        kernel1D,
        kernel1D
    )
    
    expected = np.array(
        [[4, 4],
         [4, 4]], 
        dtype=Float
    )
    
    assert_array_almost_equal(
        output, 
        expected,
        decimal=0
    )
    

def test_convolve2D_sep_02():
    arr = np.array(([1,1], [1,1]), dtype=Float)
    kernel1D = np.array([2], dtype=Float)
    
    output = filters.convolve_2D_sep(
        arr,
        kernel1D,
        kernel1D
    )
    
    expected = np.array(
        [[4, 4],
         [4, 4]], 
        dtype=Float
    )
    
    assert_array_almost_equal(
        output, 
        expected,
        decimal=0
    )


def test_convolve2D_sep_03():
    arr = np.array(([1,2], [1,2]), dtype=Float)
    kernel1D = np.array([2], dtype=Float)
    
    output = filters.convolve_2D_sep(
        arr,
        kernel1D,
        kernel1D
    )
    
    expected = np.array(
        [[4, 8],
         [4, 8]], 
        dtype=Float
    )
    
    assert_array_almost_equal(
        output, 
        expected,
        decimal=0
    )


def test_contrast_stretch_invalid_input():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.contrast_stretch(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # image is not 2D
        new_img = filters.contrast_stretch(invalid_img)

        
def test_contrast_stretch_01():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.contrast_stretch(img)
    
    assert_equal(new_img.min(), 0.0)
    assert_equal(new_img.max(), 1.0)

    
def test_gaussian_filter_invalid_input():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.gaussian_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # image is not 2D
        new_img = filters.gaussian_filter(invalid_img)


def test_gaussian_filter_01():
    arr = np.array(
        [[1, 2, 3],
         [2, 4, 6]],
        dtype = Float
    )
    
    output = filters.gaussian_filter(
        arr,
        sigma=0.00001
    )
    
    assert_allclose(
        arr,
        output
    )


def test_gaussian_filter_02():
    arr = np.arange(128*128).astype(Float)
    arr.shape = (128, 128)
    
    output = filters.gaussian_filter(
        arr,
        sigma=1
    )
    
    assert_equal(
        arr.shape,
        output.shape
    )
    


def test_gaussian_filter_03():
    arr = np.arange(128*128).astype(Float)
    arr.shape = (128, 128)
    
    output = filters.gaussian_filter(
        arr,
        sigma=1
    )
    
    assert_equal(
        arr.shape,
        output.shape
    )
    
    assert_almost_equal(
        arr.sum(),
        output.sum(),
        decimal=0
    )
    
    
def test_highpass_filter_invalid_input():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.highpass_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # image is not 2D
        new_img = filters.highpass_filter(invalid_img)
    
    
def test_highpass_filter_01():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.highpass_filter(img)
    
    assert_equal(
        img.shape,
        new_img.shape
    )
    
    
def test_intensity_cap_invalid_input():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.intensity_cap(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # image is not 2D
        new_img = filters.intensity_cap(invalid_img)
    
    
def test_intensity_cap_01():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.intensity_cap(img)
    
    assert_equal(
        img.shape,
        new_img.shape
    )


def test_intensity_cap_02():
    img = normalize_image( imread(path_to_img) )
    
    std_mult = 2.0
    
    new_img = filters.intensity_cap(img, std_mult=std_mult)
    
    expected = img.mean() + std_mult*img.std()
    
    assert_almost_equal(
        new_img.max(),
        expected,
        decimal=3
    )


def test_threshold_binarization_invalid_input():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.threshold_binarization(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # image is not 2D
        new_img = filters.threshold_binarization(invalid_img)


def test_threshold_binarization_01():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.threshold_binarization(img)
    
    assert_equal(
        img.shape,
        new_img.shape
    )
        
        
def test_variance_normalization_filter_invalid_input():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.variance_normalization_filter(img)
    
    # test for errors
    with pytest.raises(ValueError):
        invalid_img = np.random.rand(32,32,32) # for non-2d case
        
        # image is not 2D
        new_img = filters.variance_normalization_filter(invalid_img)
    
    
def test_variance_normalization_filter_01():
    img = normalize_image( imread(path_to_img) )
    
    # use default values
    new_img = filters.variance_normalization_filter(img)
    
    assert_equal(
        img.shape,
        new_img.shape
    )