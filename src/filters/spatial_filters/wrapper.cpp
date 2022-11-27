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
#include "filters.h"


using imgDtype = constants::imgDtype;


// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
#pragma warning(disable: 4244)


py::array_t<imgDtype> intensity_cap_wrapper(
    py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& input,
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


py::array_t<imgDtype> convolve2D_wrapper(
    const py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& input,
    const py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_kernel_h,
    const py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_kernel_v
){
    // check input dimensions
    if ( input.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");
    
    if ( np_kernel_h.ndim() != 1 )
        throw std::runtime_error("Horizontal kernel should be 1-D NumPy array");
    
    if ( np_kernel_v.ndim() != 1 )
        throw std::runtime_error("Vertical kernel should be 1-D NumPy array");
    
    if ( !(np_kernel_h.size() % 2) || !(np_kernel_v.size() % 2) )
        throw std::runtime_error("Vertical and horizontal kernel sizes must be odd");
    
    if ( np_kernel_h.size() != np_kernel_v.size() )
        throw std::runtime_error("Vertical and horizontal kernel sizes must be same size");

    // get image shape and kernel size
    int N = input.shape(0), M = input.shape(1);
    int kernel_size = np_kernel_h.size();

    // get input buffer
    auto in_buf = input.request();

    // allocate result array (should we do this in python side?)
    py::array_t<imgDtype> result = py::array_t<imgDtype>(in_buf.size);
    auto out_buf = result.request();

    // get kernel buffers
    auto kern_h_buf = np_kernel_h.request();
    auto kern_v_buf = np_kernel_v.request();
    
    // get raw pointers
    imgDtype* ptr_in  = (imgDtype*) in_buf.ptr;
    imgDtype* ptr_out = (imgDtype*) out_buf.ptr;
    imgDtype* ptr_kern_h = (imgDtype*) kern_h_buf.ptr;
    imgDtype* ptr_kern_v = (imgDtype*) kern_v_buf.ptr;

    // call pure C++ function
    convolve2D(
      ptr_in,
      ptr_out,
      M,
      N, 
      ptr_kern_h,
      ptr_kern_v,
      kernel_size
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

    m.def("_convolve2D", 
        &convolve2D_wrapper,
        "Convolve an array with a separatable kernel",
        py::arg("input"),
        py::arg("np_kernel_h"),
        py::arg("np_kernel_v")
    );
    
}
