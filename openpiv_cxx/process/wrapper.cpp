
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
#include "algos/fft.h"
#include "core/enumerate.h"
#include "core/grid.h"
#include "core/image.h"
#include "core/image_utils.h"
#include "core/stream_utils.h"
#include "core/vector.h"

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace openpiv;

// forward declarations
std::string get_execution_type(int);

std::vector<double> process_images_standard(
    core::gf_image& img_a,
    core::gf_image& img_b,
    uint32_t size = 32,
    uint32_t overlap_size = 16,
    int threads = 0,
    int thread_execution = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));
    std::string execution = get_execution_type(thread_execution);
    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    
    if (threads < 1)
        uint32_t thread_count = static_cast<uint32_t>(threads);

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( ia );
    auto correlator = &algos::FFT::cross_correlate_real<core::image, core::g_f>;

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b, 
        &fft, 
        &correlator
     ]( size_t i, const core::rect& ia )
     {
        auto view_a{ core::extract( img_a, ia ) };
        auto view_b{ core::extract( img_b, ia ) };

        // prepare & correlate
        core::gf_image output{ (fft.*correlator)( view_a, view_b ) };
        std::copy(output.begin(), output.end(), cmatrix.begin() + i * cmatrix_stride);
     };

    // check execution
    if ( execution == "pool" )
    {
        ThreadPool pool( thread_count );

        size_t i = 0;
        for ( const auto& ia : grid )
        {
            pool.enqueue( [i, ia, &processor](){ processor(i, ia); } );
            ++i;
        }
    }
    else if ( execution == "bulk-pool" )
    {
        ThreadPool pool( thread_count );

        // - split the grid into thread_count chunks
        // - wrap each chunk into a processing for loop and push to thread

        // ensure we don't miss grid locations due to rounding
        size_t chunk_size = grid.size()/thread_count;
        std::vector<size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = grid.size() - (thread_count-1)*chunk_size;


        size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &grid, &processor]() {
                    for ( size_t j=i; j<i + chunk_size_; ++j )
                        processor(j, grid[j]);
                } );
            i += chunk_size_;
        }
    }
    return cmatrix;
};

std::string get_execution_type(int execution_type)
{
    if (execution_type == 0)
        return "pool";
    else
        return "bulk-pool";
}

std::vector<double> process_cmatrix(
    std::vector<double> cmatrix,
    uint32_t stride_2d,
    core::size stride_1d,
    uint16_t num_peaks,
    uint16_t peak_radius
){
    auto corrCut = core::gf_image(stride_1d);
    uint32_t maxStep = cmatrix.size() / stride_2d;
    
    // overide user input
    num_peaks = 2; 
    peak_radius = 1;
    
    std::vector<uint32_t> center{ peak_radius, peak_radius };
    
    // allocate array for results (couldn't pass list of arrays)
    std::vector<double> results(maxStep * 4);
    uint32_t U =  maxStep * 0;
    uint32_t V = maxStep * 1;
    uint32_t PH = maxStep * 2;
    uint32_t P2P = maxStep * 3;
    
    for (uint32_t step = 0; step < maxStep; ++step)
    {
        std::copy(
            cmatrix.begin() + (step  * stride_2d),
            cmatrix.begin() + ((step + 1) * stride_2d),
            corrCut.begin()
        );
        core::peaks_t<core::g_f> peaks = core::find_peaks( corrCut, num_peaks, peak_radius );
        
        // sub-pixel fitting
        if ( peaks.size() != num_peaks )
        {
            results[step + U] = NAN;
            results[step + V] = NAN;
            results[step + PH] = NAN;
            results[step + P2P] = NAN;
            continue;
        }
            
        core::point2<double> uv;
        uv = core::fit_simple_gaussian( peaks[0] );
        
        results[step + U]  = uv[0] - stride_1d.width()/2;
        results[step + V]  = uv[1] - stride_1d.height()/2;
        results[step + PH] = peaks[0][ {center[0], center[1]} ];

        if ( peaks[1][ {center[0], center[1]} ] > 0 )
            results[step + P2P] = peaks[0][ {center[0], center[1]} ] / peaks[1][ {center[0], center[1]} ];
    }

    return results;
}

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
    uint16_t num_peaks,
    uint32_t peak_radius
){
    // check inputs
    if ( np_cmatrix.ndim() != 3 )
        throw std::runtime_error("Input should be 3-D NumPy array");

    if ( num_peaks < 2 )
        throw std::runtime_error("Not enough peaks to work with");

    if ( peak_radius < 1)
        throw std::runtime_error("Peak radius can not be smaller than 1");
    
    auto np_buff = np_cmatrix.request();
   
   
    uint32_t wY = np_buff.shape[1];
    uint32_t wX = np_buff.shape[2];
    uint32_t stride_3d = wX * wY;
    
    auto stride_2d = core::size{wY, wX};
    
    std::vector<double> cmatrix(np_cmatrix.size());
    std::memcpy(cmatrix.data(),np_cmatrix.data(),np_cmatrix.size()*sizeof(double));
    
    // call pure C++ function
    auto result = process_cmatrix(
        cmatrix,
        stride_3d,
        stride_2d,
        num_peaks,
        peak_radius
    );
    
    std::vector<ssize_t> dims{ 4, np_buff.shape[0] };
    py::array_t<double> py_result( result.size() );
    
    // std::memcpy(py_result.data(),result.data(),result.size()*sizeof(double));
    auto buf_res = py_result.request();
    double* ptr_in  = (double*) buf_res.ptr;
    
    for (int i = 0; i < result.size(); ++i)
        ptr_in[i] = result[i];
    
    py_result.resize( dims );
   
    return py_result;
}



// wrap as Python module
PYBIND11_MODULE(_process,m)
{
    m.doc() = "pybind11 wrapper of main openpivcore functions";

    m.def("img2corr", &fft_correlate_images_standard_wrapper, "Correlate two imaged via FFT cross-correlation");
    m.def("corr2vec", &find_subpixel_wrapper, "Extract displacement and peak information from correlation matrixes");
}