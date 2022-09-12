// std
#include <cmath>
#include <vector>
#include <iomanip>

// pybind
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// validation
#include "vector_based.h"

using imgDtype = double;

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
#pragma warning(disable: 4244)

py::object difference_test2D_wrapper(
    py::array_t<imgDtype> u,
    py::array_t<imgDtype> v,
    imgDtype threshold
){
    // check input dimensions
    if ( u.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");
        
    if ( u.size() != v.size() )
        throw std::runtime_error("Input sizes should be the same");

    int N = u.shape(0), M = u.shape(1);

    py::array_t<int> mask( N*M );

    auto buf_u = u.request();
    auto buf_v = v.request();
    auto buf_m = mask.request();

    imgDtype* ptr_u  = (imgDtype*) buf_u.ptr;
    imgDtype* ptr_v  = (imgDtype*) buf_v.ptr;
    int* ptr_m  = (int*) buf_m.ptr;

    // call pure C++ function
    difference_test2D(
        ptr_u,
        ptr_v,
        ptr_m,
        threshold,
        N, M
    );

    mask.resize( {N,M} );

    return mask;
}


py::object local_median_wrapper(
    py::array_t<imgDtype> u,
    py::array_t<imgDtype> v,
    imgDtype threshold,
    int kernel_radius,
    int kernel_min_size
){
    // check input dimensions
    if ( u.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");
        
    if ( u.size() != v.size() )
        throw std::runtime_error("Input sizes should be the same");

    if ( kernel_radius < 1 )
        throw std::runtime_error("Kernel radius must be larger than 0");

    if ( kernel_min_size < 1 )
        throw std::runtime_error("Kernel size flag must be larger than 0");

    int N = u.shape(0), M = u.shape(1);

    py::array_t<int> mask( N*M );

    auto buf_u = u.request();
    auto buf_v = v.request();
    auto buf_m = mask.request();

    imgDtype* ptr_u      = (imgDtype*) buf_u.ptr;
    imgDtype* ptr_v      = (imgDtype*) buf_v.ptr;
    int* ptr_mask        = (int*) buf_m.ptr;

    // call pure C++ function
    local_median_test(
        ptr_u,
        ptr_v,
        ptr_mask,
        threshold,
        N, M, 
        kernel_radius,
        kernel_min_size
    );

    mask.resize( {N,M} );

    return mask;
}


py::object normalized_local_median_wrapper(
    py::array_t<imgDtype> u,
    py::array_t<imgDtype> v,
    imgDtype threshold,
    int kernel_radius,
    double eps,
    int kernel_min_size
){
    // check input dimensions
    if ( u.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");
        
    if ( u.size() != v.size() )
        throw std::runtime_error("Input sizes should be the same");

    if ( kernel_radius < 1 )
        throw std::runtime_error("Kernel radius must be larger than 0");

    if ( kernel_min_size < 0 )
        throw std::runtime_error("Kernel size flag must be >= 0");

    int N = u.shape(0), M = u.shape(1);

    py::array_t<int> mask( N*M );

    auto buf_u = u.request();
    auto buf_v = v.request();
    auto buf_m = mask.request();

    imgDtype* ptr_u      = (imgDtype*) buf_u.ptr;
    imgDtype* ptr_v      = (imgDtype*) buf_v.ptr;
    int* ptr_mask        = (int*) buf_m.ptr;

    // call pure C++ function
    normalized_local_median_test(
        ptr_u,
        ptr_v,
        ptr_mask,
        threshold,
        N, M, 
        kernel_radius,
        eps,
        kernel_min_size
    );

    mask.resize( {N,M} );

    return mask;
}


double test_median_wrapper(
    py::array_t<imgDtype> arr
){       
    auto buf_arr = arr.request();

    int N_M = buf_arr.size;

    imgDtype* ptr_arr  = (imgDtype*) buf_arr.ptr;

    // call pure C++ function
    double median = test_median(
        ptr_arr,
        N_M
    );

    //std::cout << "Median is: " << median << std::endl;

    return median;
}


#pragma warning(default: 4244)

PYBIND11_MODULE(_validation, m) {
    m.doc() = "Python interface for validation tests written in c++.";
   
    m.def("_difference_test", 
       &difference_test2D_wrapper,
       "Test 8 points around a vector to see if it's invalid",
       py::arg("u"),
       py::arg("v"),
       py::arg("threshold") = 2.0
    );

    m.def("_local_median_test", 
       &local_median_wrapper,
       "Test 8 points around a vector to see if it's invalid",
       py::arg("u"),
       py::arg("v"),
       py::arg("threshold") = 2.0,
       py::arg("kernel_radius") = 1,
       py::arg("kernel_min_size") = 0
    );

    m.def("_normalized_local_median_test", 
       &normalized_local_median_wrapper,
       "Test 8 points around a vector to see if it's invalid",
       py::arg("u"),
       py::arg("v"),
       py::arg("threshold") = 2.0,
       py::arg("kernel_radius") = 1,
       py::arg("eps") = 0.1,
       py::arg("kernel_min_size") = 0
    );

    m.def("_find_median_test", 
       &test_median_wrapper,
       "Find median of finite values"
    );
}
