#include <cmath>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "bilinear.h"

using imgDtype = double;

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
#pragma warning(disable: 4244)

/*
int find_index_wrapper(
    py::array arr, 
    double x
){
    if ( arr.ndim() != 1 )
        throw std::runtime_error("Input should be 1-D NumPy array");
        
    int upperBound = arr.size() - 1;
    
    auto buf_arr = arr.request();
    
    double* ptr_arr = (double*) buf_arr.ptr;
    
    return find_index(
        ptr_arr,
        x,
        upperBound
    );
}  
*/

py::array_t<imgDtype> bilinear_interp_wrapper(
    py::array_t<int> X,
    py::array_t<int> Y,
    py::array_t<imgDtype> Z,
    py::array_t<imgDtype> xi,
    py::array_t<imgDtype> yi
){
    // check input dimensions
    if ( X.ndim() != 1 )
        throw std::runtime_error("X should be 1-D NumPy array");

    if ( Y.ndim() != 1 )
        throw std::runtime_error("Y should be 1-D NumPy array");

    if ( Z.ndim() != 2 )
        throw std::runtime_error("Z should be 2-D NumPy array");

    if ( xi.ndim() != 1 )
        throw std::runtime_error("xi should be 1-D NumPy array");

    if ( yi.ndim() != 1 )
        throw std::runtime_error("yi should be 1-D NumPy array");

    if ( X.shape()[0] != Z.shape()[1] )
        throw std::runtime_error("X/Z array size mismatch");

    if ( Y.shape()[0] != Z.shape()[0] )
        throw std::runtime_error("Y/Z array size mismatch");

    int N = xi.shape()[0], M = yi.shape()[0];

    py::array_t<imgDtype> result(N*M);

    auto buf_X   = X.request();
    auto buf_Y   = Y.request();
    auto buf_Z   = Z.request();
    auto buf_xi  = xi.request();
    auto buf_yi  = yi.request();
    auto buf_res = result.request();

    int* ptr_X        = (int*) buf_X.ptr;
    int* ptr_Y        = (int*) buf_Y.ptr;
    imgDtype* ptr_Z   = (imgDtype*) buf_Z.ptr;
    imgDtype* ptr_xi  = (imgDtype*) buf_xi.ptr;
    imgDtype* ptr_yi  = (imgDtype*) buf_yi.ptr;
    imgDtype* ptr_res = (imgDtype*) buf_res.ptr;

    // call pure C++ function
    bilinear2D(
        ptr_X,
        ptr_Y,
        ptr_Z,
        ptr_xi,
        ptr_yi,
        ptr_res,
        N, M,
        Z.shape()[1],
        X.shape()[0] - 1,
        Y.shape()[0] - 1
    );

    result.resize( {M, N} );

    return result;
}


#pragma warning(default: 4244)

PYBIND11_MODULE(_bilinear2D, m) {
    m.doc() = "Python interface for interpolation functions written in c++.";

    m.def("_bilinear2D", 
        &bilinear_interp_wrapper,
        "Bilinear interpolation of a 2D grid.",
        py::arg("X"),
        py::arg("Y"),
        py::arg("Z"),
        py::arg("xi"),
        py::arg("yi")
    );
/*    
    m.def("find_index", 
        &find_index_wrapper,
        "Find index on a 1D array.",
        py::arg("arr"),
        py::arg("x")
    );
*/
}
