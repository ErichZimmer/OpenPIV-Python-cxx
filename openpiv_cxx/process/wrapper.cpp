// std
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <cmath>

// utils
#include "threadpool.hpp"

// openpiv
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// wrapper functions
#include "cc_standard.h"
#include "subpixel.h"

namespace py = pybind11;
using namespace openpiv;

// ----------------
// Python interface
// ----------------

// wrap C++ function with NumPy array IO
py::array_t<double> fft_correlate_images_standard_wrapper( // big function name lol
    py::array_t<double, py::array::c_style | py::array::forcecast> np_img_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> np_img_b,
    int window_size,
    int overlap,
    int thread_count,
    int thread_execution
){
    // check inputs
    if ( np_img_a.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    if ( np_img_a.size() != np_img_b.size() )
        throw std::runtime_error("Inputs should have same sizes");

    if ( window_size < 1)
        throw std::runtime_error("Interrogation window sizes can not be smaller than 1");
    
    if ( overlap < 1)
        throw std::runtime_error("Overlap can not be smaller than 1");
        
    if (overlap > window_size)
        throw std::runtime_error("Overlap sizes can not be larger than interrogation window sizes");
        
    // cast ints to proper dtype
    uint32_t window_size_t = static_cast<uint32_t>(window_size);
    uint32_t overlap_t = static_cast<uint32_t>(overlap);
    
    // allocate std::vector (to pass to the C++ function)
    core::gf_image img_a(
        static_cast<uint32_t>(np_img_a.shape()[0]), 
        static_cast<uint32_t>(np_img_a.shape()[1])
    );
    core::gf_image img_b(
        static_cast<uint32_t>(np_img_b.shape()[0]), 
        static_cast<uint32_t>(np_img_b.shape()[1])
    );
    
    // copy py::array -> std::vector
    std::memcpy(img_a.data(),np_img_a.data(),np_img_a.size()*sizeof(double));
    std::memcpy(img_b.data(),np_img_b.data(),np_img_b.size()*sizeof(double));
    
    // call pure C++ function
    auto result = process_images_standard(
        img_a,
        img_b,
        window_size_t,
        overlap_t,
        thread_count,
        thread_execution
    );
    
    // return 3-D NumPy array  
    uint32_t window_num = result.size() / (window_size * window_size);
    uint32_t stride_3d = window_size * window_size;
    uint32_t stride_2d = window_size;
    
    ssize_t              ndim    = 3;
    std::vector<ssize_t> shape   = { window_num, stride_2d, stride_2d };
    std::vector<ssize_t> strides = {
        sizeof(double)*stride_3d, 
        sizeof(double)*stride_2d,
        sizeof(double) 
    };

    // return 3-D NumPy array
    return py::array(py::buffer_info(
        result.data(),                           /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    ));
}

py::array_t<double> find_subpixel_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> np_cmatrix,
    int search_method,
    int thread_count
){
    // check inputs
    if ( np_cmatrix.ndim() != 3 )
        throw std::runtime_error("Input should be 3-D NumPy array");

    auto np_buff = np_cmatrix.request();
   
   
    uint32_t wY = np_buff.shape[1];
    uint32_t wX = np_buff.shape[2];
    uint32_t stride_3d = wX * wY;
    uint32_t maxStep = np_buff.size / stride_3d;
    auto stride_2d = core::size{wY, wX};
    
    // get cmatrix pointer
    double* cmatrix = (double*) np_buff.ptr;
    
    // get result array pointer
    std::vector<ssize_t> dims{ 4, np_buff.shape[0] };
    py::array_t<double> py_result( maxStep * 4 );
    
    auto buf_res = py_result.request();
    double* result_ptr  = (double*) buf_res.ptr;
    
    // call pure  C++ function
    process_cmatrix_2x3(
        cmatrix,
        result_ptr,
        maxStep,
        stride_3d,
        stride_2d,
        thread_count
    );

    py_result.resize( dims );
   
    return py_result;
}



// wrap as Python module
PYBIND11_MODULE(_process,m)
{
    m.doc() = "pybind11 wrapper of main openpivcore functions";

    m.def("img2corr_standard", &fft_correlate_images_standard_wrapper, "Correlate two imaged via FFT cross-correlation");
    m.def("corr2vec", &find_subpixel_wrapper, "Extract displacement and peak information from correlation matrixes");
}