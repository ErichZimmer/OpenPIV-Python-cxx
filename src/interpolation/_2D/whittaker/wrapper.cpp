#include <cmath>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "whittaker.h"

using imgDtype = double;

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
#pragma warning(disable: 4244)

py::array_t<imgDtype> whittaker_interp_wrapper(
    py::array_t<imgDtype> Z,
    py::array_t<imgDtype> X,
    py::array_t<imgDtype> Y,
    int radius = 3
){
    // check input dimensions
    if ( X.ndim() != 2 )
        throw std::runtime_error("X should be 2-D NumPy array");
    
    if ( Y.ndim() != 2 )
        throw std::runtime_error("Y should be 2-D NumPy array");
    
    if ( Z.ndim() != 2 )
        throw std::runtime_error("Z should be 2-D NumPy array");
    
    if ( X.size() != Z.size() )
        throw std::runtime_error("X/Z array size mismatch");
    
    if ( Y.size() != Z.size() )
        throw std::runtime_error("Y/Z array size mismatch");
    
    int N = Z.shape()[0], M = Z.shape()[1];

    py::array_t<imgDtype> result(N*M);
    
    auto buf_X   = X.request();
    auto buf_Y   = Y.request();
    auto buf_Z   = Z.request();
    auto buf_res = result.request();
    
    imgDtype* ptr_X   = (imgDtype*) buf_X.ptr;
    imgDtype* ptr_Y   = (imgDtype*) buf_Y.ptr;
    imgDtype* ptr_Z   = (imgDtype*) buf_Z.ptr;
    imgDtype* ptr_res = (imgDtype*) buf_res.ptr;
    
    // call pure C++ function
    whittaker2D(
        ptr_X,
        ptr_Y,
        ptr_Z,
        ptr_res,
        N, M,
        radius
    );
    
    result.resize( {N, M} );
    
    return result;
}


#pragma warning(default: 4244)

PYBIND11_MODULE(_whittaker2D_cpp, m) {
    m.doc() = "Python interface for interpolation functions written in c++.";
   
    m.def("_whittaker2D", 
        &whittaker_interp_wrapper,
        "Whittaker interpolation of a 2D grid.",
        py::arg("X"),
        py::arg("Y"),
        py::arg("Z"),
        py::arg("radius")
    );

}
