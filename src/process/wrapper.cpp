// std
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <vector>

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// openpiv
#include "core/image.h"

// functions to wrap
#include "cc_subpixel.h"
#include "openpiv_correlation.h"

// utils
#include "openpiv_utils.h"
#include "constants.h"

namespace py = pybind11;
using namespace openpiv;
using imgDtype = constants::imgDtype;


// ----------------
// Python interface
// ----------------

#pragma warning(disable: 4244)


// wrap C++ function with NumPy array IO
py::array_t<imgDtype> fft_correlate_window_wrapper(
    py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_img_a,
    py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_img_b
){
    // check inputs
    if ( np_img_a.ndim() != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    if ( np_img_a.size() != np_img_b.size() )
        throw std::runtime_error("Inputs should have same sizes");

    core::image<core::g<imgDtype>> img_a{ convert_image(np_img_a) };
    core::image<core::g<imgDtype>> img_b{ convert_image(np_img_b) };

    std::vector<imgDtype> result = process_window(
        img_a,
        img_b
    );

    std::vector<std::size_t> field_shape(2);
    field_shape[0] = static_cast<std::size_t>(np_img_a.shape(0));
    field_shape[1] = static_cast<std::size_t>(np_img_a.shape(1));

    // return 2-D NumPy array  
    std::size_t              ndim    = 2;
    std::vector<std::size_t> shape   = { field_shape[0], field_shape[1] };
    std::vector<std::size_t> strides = {
        static_cast<std::size_t>(sizeof(imgDtype)) * field_shape[1],
        static_cast<std::size_t>(sizeof(imgDtype))
    };

    // return 3-D NumPy array
    return py::array(py::buffer_info(
        result.data(),                           /* data as contiguous array  */
        sizeof(imgDtype),                           /* size of one scalar        */
        py::format_descriptor<imgDtype>::format(),  /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    ));
}


py::array_t<imgDtype> fft_correlate_images_standard_wrapper( // big function name lol
    py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_img_a,
    py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_img_b,
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

    core::image<core::g<imgDtype>> img_a{ convert_image(np_img_a) };
    core::image<core::g<imgDtype>> img_b{ convert_image(np_img_b) };

    std::vector<imgDtype> result = process_images_standard(
        img_a,
        img_b,
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
        static_cast<std::size_t>(sizeof(imgDtype))*stride_3d, 
        static_cast<std::size_t>(sizeof(imgDtype))*stride_2d,
        static_cast<std::size_t>(sizeof(imgDtype))
    };

    return py::array(py::buffer_info(
        result.data(),                           /* data as contiguous array  */
        sizeof(imgDtype),                           /* size of one scalar        */
        py::format_descriptor<imgDtype>::format(),  /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    ));
}


py::array_t<imgDtype> find_subpixel_wrapper(
    py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_cmatrix,
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
    imgDtype* cmatrix_ptr = (imgDtype*) np_buff.ptr;

    // get result array pointer
    std::vector<std::size_t> dims{
        8,
        static_cast<std::size_t>(np_buff.shape[0])
    };
    py::array_t<imgDtype> py_result( maxStep * 8 );

    auto buf_res = py_result.request();
    imgDtype* result_ptr  = (imgDtype*) buf_res.ptr;

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
    m.def("_img2corr_iw_cross", &fft_correlate_window_wrapper, "Correlate two interrogation windows for testing");
    m.def("_img2corr_standard", &fft_correlate_images_standard_wrapper, "Correlate two images");
    m.def("_corr2vec", &find_subpixel_wrapper, "Extract displacement and peak information from correlation matrixes");
}