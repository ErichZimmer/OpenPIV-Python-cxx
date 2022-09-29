// std
#include <cmath>
#include <vector>
#include <iomanip>

// pybind
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// filters
#include "constants.h"
#include "kernels.h"
#include "filters.h"


using imgDtype = constants::imgDtype;


// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
#pragma warning(disable: 4244)


py::array_t<imgDtype> intensity_cap_wrapper(
    py::array_t<imgDtype>& input,
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
    py::array_t<imgDtype>& input,
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
    py::array_t<imgDtype>& input,
    py::array_t<imgDtype>& np_kernel
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
    py::array_t<imgDtype>& input,
    py::array_t<imgDtype>& np_kernel,
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


#pragma warning(default: 4244)

PYBIND11_MODULE(_spatial_filters_cpp, m) {
    m.doc() = "Python interface for filters written in c++.";

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
}
