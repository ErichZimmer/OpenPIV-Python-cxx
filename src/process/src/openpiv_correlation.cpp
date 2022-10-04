/*
Due to Link Error 2005 on core::generate_cartesian_grid,
all correlation functions are placed in one file for now.
*/

/*
~~~~~~~~~~~~~~~~~
Table of Contents
~~~~~~~~~~~~~~~~~
1:   comments
17:  includes
47:  standard cross-correlation of one interrogation window
72:  standard cross-correlation
178: auto-correlation
*/

#include "openpiv_correlation.h"

// std
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>

// utils
#include "threadpool.hpp"
#include "openpiv_utils.h"

// openpiv
#include "algos/fft.h"
#include "core/enumerate.h"
#include "core/grid.h"
#include "core/image_utils.h"
#include "core/stream_utils.h"
#include "core/vector.h"

using namespace openpiv;


// standard cross-correlation
std::vector<double> process_window(
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_a,
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_b
){
    core::gf_image img_a{ convert_image(np_img_a) };
    core::gf_image img_b{ convert_image(np_img_b) };
    
    auto fft_size = core::size{ img_a.width(), img_a.height() };

    auto fft = algos::FFT( fft_size );

    auto correlator = &algos::FFT::cross_correlate_real<core::image, core::g_f>;
    
    core::gf_image output{ (fft.*correlator)( img_a, img_b ) };

    std::vector<double> cmatrix( img_a.pixel_count() );
    
    for (std::size_t i = 0; i < img_a.pixel_count(); ++i)
        cmatrix[i] = output[i];

    return cmatrix;
};


// Normalozed cross-correlation
std::vector<double> process_images_standard(
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_a,
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_b,
    std::uint32_t size = 32,
    std::uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    core::gf_image img_a{ convert_image(np_img_a) };
    core::gf_image img_b{ convert_image(np_img_b) };

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // padding
    auto paddedWindow = core::size{size, size};
    if (correlation_method != 0)
        paddedWindow = core::size{size * 2, size * 2}; // pad windows by 2N

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( paddedWindow );
    auto correlator = &algos::FFT::cross_correlate_real<core::image, core::g_f>;

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b,
        &paddedWindow,
        &fft,
        &correlator
     ]( std::size_t i, const core::rect& ia, 
        core::gf_image& view_a, core::gf_image& view_b,
        core::gf_image& output)
     {
        auto mean_stdA = mean_std(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());
        auto mean_stdB = mean_std(img_b, ia.bottom(), ia.top(), ia.left(), ia.right());

        double norm = mean_stdA[1] * mean_stdB[1] * static_cast<double>(paddedWindow.area() * ia.area());

        placeIntoPadded(img_a, view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), mean_stdA[0]);
        placeIntoPadded(img_b, view_b, ia.bottom(), ia.top(), ia.left(), ia.right(), mean_stdB[0]);

        // prepare & correlate
        output = (fft.*correlator)( view_a, view_b );

        // normalize output
        applyScalarToImage(output, norm, paddedWindow.area());

        placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i); 
     };

    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        // - split the grid into thread_count chunks
        // - wrap each chunk into a processing for loop and push to thread

        // ensure we don't miss grid locations due to rounding
        std::size_t chunk_size = grid.size() / thread_count;
        std::vector<size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = grid.size() - (thread_count-1)*chunk_size;

        std::size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &grid, &processor, &paddedWindow]() {
                    core::gf_image view_a{ paddedWindow.height(), paddedWindow.width() };
                    core::gf_image view_b{ paddedWindow.height(), paddedWindow.width() };
                    core::gf_image output{ paddedWindow.height(), paddedWindow.width() };
                    
                    for ( std::size_t j=i; j<i + chunk_size_; ++j )
                        processor(j, grid[j], view_a, view_b, output);
                } );
            i += chunk_size_;
        }
    }
    else
    {
        core::gf_image view_a{ paddedWindow.height(), paddedWindow.width() };
        core::gf_image view_b{ paddedWindow.height(), paddedWindow.width() };
        core::gf_image output{ paddedWindow.height(), paddedWindow.width() };

        for (std::size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i], view_a, view_b, output);
    }

    return cmatrix;
}


// autocorrelation
std::vector<double> process_images_auto(
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img_a,
    std::uint32_t size = 32,
    std::uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    core::gf_image img_a{ convert_image(np_img_a) };
    
    // create a grid for processing
    auto ia = core::size{size, size};
    std::vector<core::rect> grid = core::generate_cartesian_grid( img_a.size(), ia, overlap );

    // padding
    auto paddedWindow = core::size{size, size};
    if (correlation_method != 0)
        paddedWindow = core::size{size * 2, size * 2}; // pad windows by 2N

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( paddedWindow );
    auto correlator = &algos::FFT::auto_correlate<core::image, core::g_f>;

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &paddedWindow,
        &fft,
        &correlator
     ]( std::size_t i, const core::rect& ia )
     {
        auto mean_stdA = mean_std(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());

        double norm = mean_stdA[1] * paddedWindow.area();
        core::gf_image view_a{ ia.width(), ia.height() };

        placeIntoPadded(img_a, view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), mean_stdA[0]);

        // prepare & correlate
        core::gf_image output{ (fft.*correlator)( view_a ) };

        // normalize output
        applyScalarToImage(output, norm, paddedWindow.area());

        placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i);
     };

    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        // - split the grid into thread_count chunks
        // - wrap each chunk into a processing for loop and push to thread

        // ensure we don't miss grid locations due to rounding
        std::size_t chunk_size = grid.size()/thread_count;
        std::vector<std::size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = grid.size() - (thread_count-1)*chunk_size;

        std::size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &grid, &processor]() {
                    for ( std::size_t j=i; j<i + chunk_size_; ++j )
                        processor(j, grid[j]);
                } );
            i += chunk_size_;
        }
    }
    else
    {
        for (std::size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }

    return cmatrix;
};


// error correlation correction (ecc) cross-correlation
/*
std::vector<double> process_images_ecc(
    core::gf_image& img_a,
    core::gf_image& img_b,
    uint32_t size = 32,
    uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // field shape
    uint32_t field_shape_x = img_a.width();
    uint32_t field_shape_y = img_a.height();
    
    // padding
    auto paddedWindow = core::size{size, size};
    if (correlation_method != 0)
        paddedWindow = core::size{size * 2, size * 2}; // pad windows by 2N

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( paddedWindow );
    auto correlator = &algos::FFT::cross_correlate_real<core::image, core::g_f>;

    auto process_row = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b,
        &grid,
        &paddedWindow,
        &fft,
        &correlator,
        filed_shape_x,
        filed_shape_y,
        overlap
     ]( size_t row )
    {
        core::gf_image output_old(paddedWindow.heihgt(), paddedWindow.width());
        
        for (size_t i = 0; i < field_shape_x; ++i)
        {
            auto ia = grid[row * filed_shape_y + i];

            auto mean_stdA = mean_std( img_a, ia.bottom(), ia.top(), ia.left(), ia.right() );
            auto mean_stdB = mean_std( img_b, ia.bottom(), ia.top(), ia.left(), ia.right() );

            double norm = mean_stdA[1] * mean_stdB[1] * paddedWindow.area();

            auto view_a { placeIntoPadded(img_a, paddedWindow, ia, mean_stdA[0]) };
            auto view_b { placeIntoPadded(img_b, paddedWindow, ia, mean_stdB[0]) };

            // prepare & correlate
            core::gf_image output{ (fft.*correlator)( view_a, view_b ) };

            if (i == 0)
            {
                // normalize output
                applyScalarToImage(output, norm, paddedWindow.area());

                placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i);

                output_old.swap(output);
            }

            else
            {
                output = output * output_old;
                
                // normalize output
                applyScalarToImage(output, norm, paddedWindow.area());

                placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i);

                output_old.swap(output);
            }
        ]
     };

    if (thread_count > 1)
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
    else
    {
        for (size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }

    return cmatrix;
}
*/