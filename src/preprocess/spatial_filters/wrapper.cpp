#include <cmath>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "constants.h"
#include "kernels.h"
#include "filters.h"

using imgDtype = constants::imgDtype;

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
#pragma warning(disable: 4244)

py::array_t<imgDtype> gaussian_kernel_wrapper(
    int kernel_size,
    imgDtype sigma
){
    // check input dimensions
    if ( kernel_size < 2 )
        throw std::runtime_error("kernel_size must be 3 or larger");

    if ( (kernel_size % 2) == 0 )
        throw std::runtime_error("kernel_size must be odd");

    auto GKernel = kernels::gaussian(kernel_size, sigma);

    std::size_t kernel_t = static_cast<std::size_t>(kernel_size);

    // return 2-D NumPy array  
    std::size_t                ndim   = 2;
    std::vector<std::size_t> shape   = { kernel_t, kernel_t };
    std::vector<std::size_t> strides = {
        sizeof(imgDtype)*kernel_t * kernel_t, 
        sizeof(imgDtype)*kernel_t,
        sizeof(imgDtype) 
    };

    // return 3-D NumPy array
    return py::array(py::buffer_info(
        GKernel.data(),                            /* data as contiguous array  */
        sizeof(imgDtype),                          /* size of one scalar        */
        py::format_descriptor<imgDtype>::format(), /* data type                 */
        ndim,                                      /* number of dimensions      */
        shape,                                     /* shape of the matrix       */
        strides                                    /* strides for each axis     */
    ));
    
}


py::array_t<imgDtype> intensity_cap_wrapper(
    py::array_t<imgDtype> input,
    imgDtype std_mult = 2.f
){
    // check input dimensions
    if ( input.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    auto buf1 = input.request();

    int N = input.shape(0), M = input.shape(1);

    py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
    auto buf2 = result.request();

    imgDtype* ptr_out = (imgDtype*) buf2.ptr;

    // call pure C++ function
    intensity_cap_filter(
        ptr_out,
        N*M, 
        std_mult
    );

    result.resize( {N,M} );

    return result;
}


py::array_t<imgDtype> intensity_binarize_wrapper(
    py::array_t<imgDtype> input,
    imgDtype threshold = 0.5
){
    // check input dimensions
    if ( input.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    auto buf1 = input.request();

    int N = input.shape(0), M = input.shape(1);

    py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
    auto buf2 = result.request();

    imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
    imgDtype* ptr_out = (imgDtype*) buf2.ptr;

    // call pure C++ function
    binarize_filter(
        ptr_out,
        ptr_in,
        N*M, 
        threshold
    );

    result.resize( {N,M} );

    return result;
}


py::array_t<imgDtype> low_pass_filter_wrapper(
    py::array_t<imgDtype> input,
    py::array_t<imgDtype> np_kernel
){
    // check input dimensions
    if ( input.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    auto buf1 = input.request();

    int N = input.shape(0), M = input.shape(1);

    py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
    auto buf2 = result.request();

    imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
    imgDtype* ptr_out = (imgDtype*) buf2.ptr;

    std::vector<imgDtype> GKernel(np_kernel.size());
    std::memcpy(
        GKernel.data(),
        np_kernel.data(),
        np_kernel.size()*sizeof(imgDtype)
    );

    int kernel_size = np_kernel.shape(0);

    // call pure C++ function
    apply_kernel_lowpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size
    );

    result.resize( {N,M} );

    return result;
}


py::array_t<imgDtype> high_pass_filter_wrapper(
    py::array_t<imgDtype> input,
    py::array_t<imgDtype> np_kernel,
    py::bool_ clip_at_zero = false
){
    // check input dimensions
    if ( input.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    auto buf1 = input.request();

    int N = input.shape(0), M = input.shape(1);

    py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
    auto buf2 = result.request();

    imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
    imgDtype* ptr_out = (imgDtype*) buf2.ptr;

    std::vector<imgDtype> GKernel(np_kernel.size());
    std::memcpy(
        GKernel.data(),
        np_kernel.data(),
        np_kernel.size()*sizeof(imgDtype)
    );
    
    int kernel_size = np_kernel.shape(0);

    // call pure C++ function
    apply_kernel_highpass(
        ptr_out,
        ptr_in,
        GKernel,
        N, M, 
        kernel_size,
        clip_at_zero
    );

    result.resize( {N,M} );

    return result;
}


py::array_t<imgDtype> local_variance_norm_wrapper(
    py::array_t<imgDtype> input,
    int kernel_size = 3,
    imgDtype sigma1 = 2,
    imgDtype sigma2 = 2,
    py::bool_ clip_at_zero = false
){
    // check input dimensions
    if ( input.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    auto buf1 = input.request();

    int N = input.shape(0), M = input.shape(1);

    py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
    auto buf2 = result.request();

    py::array_t<imgDtype> temp_buffer = py::array_t<imgDtype>(buf1.size);
    auto buf3 = temp_buffer.request();

    imgDtype* ptr_out = (imgDtype*) buf2.ptr;
    imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
    imgDtype* ptr_buf = (imgDtype*) buf3.ptr;

    // call pure C++ function
    local_variance_norm(
        ptr_out,
        ptr_in,
        ptr_buf,
        N, M, 
        kernel_size,
        sigma1,
        sigma2,
        clip_at_zero
    );

    result.resize( {N,M} );

    return result;
}


#pragma warning(default: 4244)

PYBIND11_MODULE(_spatial_filters,m) {
    m.doc() = "Python interface for filters written in c++.";

    m.def("_gaussian_kernel",
        &gaussian_kernel_wrapper,
        "Generate a 2D gaussian kernel",
        py::arg("kernel_size") = 3,
        py::arg("sigma") = 2.f
    );

    m.def("_intensity_cap", 
        &intensity_cap_wrapper,
        "Apply an intensity cap filter to a 2D array",
        py::arg("input"),
        py::arg("std_mult") = 2.f
    );

    m.def("_threshold_binarization", 
        &intensity_binarize_wrapper,
        "Apply an binarization filter to a 2D array",
        py::arg("input"),
        py::arg("threshold") = 0.5f
    );

    m.def("_lowpass_filter", 
        &low_pass_filter_wrapper,
        "Apply a low pass filter to a 2D array",
        py::arg("input"),
        py::arg("np_kernel")
    );

    m.def("_highpass_filter", 
        &high_pass_filter_wrapper,
        "Apply a high pass filter to a 2D array",
        py::arg("input"),
        py::arg("np_kernel"),
        py::arg("clip_at_zero") = false
    );

    m.def("_local_variance_normalization", 
        &local_variance_norm_wrapper, 
        "Apply a local variance normalization filter to a 2D array",
        py::arg("input"),
        py::arg("kernel_size") = 7, 
        py::arg("sigma1") = 2,
        py::arg("sigma2") = 2,
        py::arg("clip_at_zero") = false
    );
}
