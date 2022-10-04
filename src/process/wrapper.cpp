// std
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <vector>

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// select appropriate header file
#ifdef USE_FFTW
    #include "fftw_correlation.h"
#else
    #include "openpiv_correlation.h"
#endif

#include "cc_subpixel.h"

namespace py = pybind11;
using namespace openpiv;

// ----------------
// Python interface
// ----------------

#pragma warning(disable: 4244)


// wrap C++ function with NumPy array IO
py::array_t<double> fft_correlate_window_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_a,
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_b
){
    // check inputs
    if ( np_img_a.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    if ( np_img_a.size() != np_img_b.size() )
        throw std::runtime_error("Inputs should have same sizes");

    std::vector<double> result = process_window(
            np_img_a,
            np_img_b
        );

    std::vector<std::size_t> field_shape(2);
    field_shape[0] = static_cast<std::size_t>(np_img_a.shape(1));
    field_shape[1] = static_cast<std::size_t>(np_img_a.shape(0));
    

    // return 2-D NumPy array  
    std::size_t              ndim    = 2;
    std::vector<std::size_t> shape   = { field_shape[0], field_shape[1] };
    std::vector<std::size_t> strides = {
        static_cast<std::size_t>(sizeof(double)) * field_shape[0] * field_shape[1], 
        static_cast<std::size_t>(sizeof(double)) * field_shape[1],
        static_cast<std::size_t>(sizeof(double))
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


py::array_t<double> fft_correlate_images_standard_wrapper( // big function name lol
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_a,
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_b,
    int window_size,
    int overlap,
    int correlation_method,
    int thread_count
){
    // check inputs
    if ( np_img_a.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    if ( np_img_a.size() != np_img_b.size() )
        throw std::runtime_error("Inputs should have same sizes");

    if ( window_size < 1 )
        throw std::runtime_error("Interrogation window sizes can not be smaller than 1");
    
    if ( overlap < 1 )
        throw std::runtime_error("Overlap can not be smaller than 1");
        
    if (overlap > window_size)
        throw std::runtime_error("Overlap sizes can not be larger than interrogation window sizes");

    // cast ints to proper dtype
    std::uint32_t window_size_t = static_cast<std::uint32_t>(window_size);
    std::uint32_t overlap_t = static_cast<std::uint32_t>(overlap);

    std::vector<double> result = process_images_standard(
            np_img_a,
            np_img_b,
            window_size_t,
            overlap_t,
            correlation_method,
            thread_count
        );

    // return 3-D NumPy array  
    std::size_t window_num = result.size() / (window_size * window_size);
    std::size_t stride_3d = window_size * window_size;
    std::size_t stride_2d = window_size;

    std::size_t              ndim    = 3;
    std::vector<std::size_t> shape   = { window_num, stride_2d, stride_2d };
    std::vector<std::size_t> strides = {
        static_cast<std::size_t>(sizeof(double))*stride_3d, 
        static_cast<std::size_t>(sizeof(double))*stride_2d,
        static_cast<std::size_t>(sizeof(double))
    };

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
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_cmatrix,
    int search_method,
    int limit_peak_search,
    int num_peaks,
    int thread_count
){
    // check inputs
    if ( np_cmatrix.ndim() != 3 )
        throw std::runtime_error("Input should be 3-D NumPy array");

    auto np_buff = np_cmatrix.request();

    std::uint32_t window_size_y = np_buff.shape[1];
    std::uint32_t window_size_x = np_buff.shape[2];

    std::uint32_t stride_3d = window_size_y * window_size_x;
    std::uint32_t maxStep = np_buff.size / stride_3d;
    std::vector<std::uint32_t> stride_2d(2);
    stride_2d[0] = window_size_y;
    stride_2d[1] = window_size_x;

    // get cmatrix pointer
    double* cmatrix_ptr = (double*) np_buff.ptr;

    // get result array pointer
    std::vector<std::size_t> dims{
        8,
        static_cast<std::size_t>(np_buff.shape[0])
    };
    py::array_t<double> py_result( maxStep * 8 );

    auto buf_res = py_result.request();
    double* result_ptr  = (double*) buf_res.ptr;

    // call pure  C++ function
    process_cmatrix_2x3(
        cmatrix_ptr,
        result_ptr,
        maxStep,
        stride_3d,
        stride_2d,
        limit_peak_search,
        num_peaks,
        thread_count
    );

    py_result.resize( dims );

    return py_result;
}

#pragma warning(default: 4244)

// wrap as Python module
PYBIND11_MODULE(_process_cpp, m)
{
    m.doc() = "pybind11 wrapper of main openpivcore functions";
    m.def("_img2corr_iw", &fft_correlate_window_wrapper, "Correlate two interrogation windows for testing");
    m.def("_img2corr_standard", &fft_correlate_images_standard_wrapper, "Correlate two images");
    m.def("_corr2vec", &find_subpixel_wrapper, "Extract displacement and peak information from correlation matrixes");
}